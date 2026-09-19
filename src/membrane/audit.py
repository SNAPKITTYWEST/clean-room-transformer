"""Append-only, hash-chained, Ed25519-signed audit log.

Structure
---------
The log is JSON Lines. Each line is one entry:

    {"seq": 0, "prev": "<64 hex>", "hash": "<64 hex>", "sig": "<128 hex>",
     "record": {...}}

    hash = SHA-256(canonical_json({"seq":…, "prev":…, "record":{…}}))
    sig  = Ed25519(hash_bytes)

Any edit, reorder, deletion or insertion breaks either the hash recomputation
or the linkage to the previous entry, and forging a replacement requires the
signing key. `verify_log()` reports the first failure and its reason.

What this does and does not give you
------------------------------------
It gives *tamper evidence*: an offline verifier holding only the public key can
tell that a log is unmodified and complete. It does not give tamper
*prevention*: a process that can rewrite the file can truncate it and, if it
also holds the signing key, re-sign a shorter history. Defences against that
are operational (append-only mount / `chattr +a`, WORM storage, off-host
replication of the head hash, a key the runtime cannot read) and are the
deployment's responsibility, not this module's. See README "Honest scoping".

Writes happen on a background thread so that the inference path is not blocked
by hashing, signing or disk I/O.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

GENESIS_HASH = "0" * 64

# ---------------------------------------------------------------------------
# hashing / canonical form
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN/Infinity.

    `allow_nan=False` matters here: a record carrying a NaN statistic must not
    be silently encoded as the non-standard `NaN` token, because a verifier in
    another language may not parse it back to the same value.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entry_hash(seq: int, prev: str, record: dict[str, Any]) -> str:
    import hashlib

    payload = canonical_json({"seq": seq, "prev": prev, "record": record})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def generate_signing_key() -> Ed25519PrivateKey:
    """Fresh random signing key. Use this in production."""
    return Ed25519PrivateKey.generate()


def signing_key_from_seed(seed: bytes) -> Ed25519PrivateKey:
    """TEST ONLY: derive a signing key from fixed bytes for reproducible logs.

    Never use a derived-from-constant key outside tests; anyone who knows the
    seed can forge the entire chain.
    """
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be exactly 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(seed)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


@dataclass
class AuditLog:
    """Non-blocking writer over a hash-chained JSONL file.

    `append()` only enqueues: the hash, signature and disk write happen on the
    writer thread. `flush()` waits for the queue to drain and re-raises any
    error the writer hit, so a failure cannot pass unnoticed.
    """

    path: Path
    signing_key: Ed25519PrivateKey
    fsync_each: bool = False
    _queue: queue.Queue = field(default_factory=queue.Queue, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _seq: int = field(default=0, init=False)
    _head: str = field(default=GENESIS_HASH, init=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(
                f"{self.path} already exists and is non-empty; the log is "
                "append-only and is never reopened for rewriting"
            )
        self.path.touch()
        self._thread = threading.Thread(
            target=self._drain, name="audit-writer", daemon=True
        )
        self._thread.start()

    # -- producer side --------------------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("audit log is closed")
        if self._error is not None:
            raise RuntimeError("audit writer failed") from self._error
        self._queue.put(dict(record))

    def flush(self) -> None:
        self._queue.join()
        if self._error is not None:
            raise RuntimeError("audit writer failed") from self._error

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def head(self) -> str:
        """Hash of the most recent entry written (the chain head)."""
        return self._head

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.signing_key.public_key()

    def public_key_hex(self) -> str:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )

        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writer thread --------------------------------------------------

    def _drain(self) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    self._queue.task_done()
                    return
                try:
                    self._write_one(handle, item)
                except BaseException as exc:  # recorded, re-raised on flush()
                    if self._error is None:
                        self._error = exc
                finally:
                    self._queue.task_done()

    def _write_one(self, handle: Any, record: dict[str, Any]) -> None:
        record.setdefault("ts_ns", time.time_ns())
        digest = entry_hash(self._seq, self._head, record)
        signature = self.signing_key.sign(bytes.fromhex(digest)).hex()
        line = canonical_json(
            {
                "seq": self._seq,
                "prev": self._head,
                "hash": digest,
                "sig": signature,
                "record": record,
            }
        )
        handle.write(line + "\n")
        handle.flush()
        if self.fsync_each:
            os.fsync(handle.fileno())
        self._seq += 1
        self._head = digest


_SENTINEL = object()


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    ok: bool
    entries: int
    head: str
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def read_entries(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no + 1}: malformed JSON: {exc}") from exc


def verify_log(
    path: str | Path,
    public_key: Ed25519PublicKey,
    expected_head: str | None = None,
) -> VerificationReport:
    """Recompute the chain and check every signature.

    Pass `expected_head` (obtained out of band) to also detect truncation of
    the tail, which linkage alone cannot reveal.
    """
    failures: list[str] = []
    prev = GENESIS_HASH
    count = 0

    for index, entry in enumerate(read_entries(path)):
        missing = {"seq", "prev", "hash", "sig", "record"} - entry.keys()
        if missing:
            failures.append(f"entry {index}: missing fields {sorted(missing)}")
            break

        if entry["seq"] != index:
            failures.append(
                f"entry {index}: seq is {entry['seq']}, expected {index} "
                "(reordering or deletion)"
            )
            break
        if entry["prev"] != prev:
            failures.append(
                f"entry {index}: prev {entry['prev'][:12]}… does not link to "
                f"{prev[:12]}… (broken chain)"
            )
            break

        recomputed = entry_hash(entry["seq"], entry["prev"], entry["record"])
        if recomputed != entry["hash"]:
            failures.append(
                f"entry {index}: hash mismatch (record was modified after signing)"
            )
            break

        try:
            public_key.verify(bytes.fromhex(entry["sig"]), bytes.fromhex(entry["hash"]))
        except (InvalidSignature, ValueError):
            failures.append(f"entry {index}: invalid signature")
            break

        prev = entry["hash"]
        count += 1

    if expected_head is not None and not failures and prev != expected_head:
        failures.append(
            f"head is {prev[:12]}… but {expected_head[:12]}… was expected "
            "(log truncated or incomplete)"
        )

    return VerificationReport(ok=not failures, entries=count, head=prev, failures=failures)
