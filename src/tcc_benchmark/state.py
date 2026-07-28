"""Persistência atômica e estado retomável de execuções de benchmark.

O módulo não depende do restante da aplicação de propósito: tanto o ``runner``
quanto ferramentas de inspeção podem ler/escrever os manifestos sem carregar
TensorFlow ou um adaptador de dataset.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


MANIFEST_FILE = "manifest.json"
STATE_EVENTS_FILE = "state_events.jsonl"
VALID_RUN_STATUSES = frozenset({"pending", "running", "interrupted", "completed", "failed"})

_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"pending", "running", "failed", "interrupted"}),
    "running": frozenset({"running", "interrupted", "completed", "failed"}),
    "interrupted": frozenset({"interrupted", "running", "failed"}),
    "failed": frozenset({"failed", "running"}),
    "completed": frozenset({"completed"}),
}


class StateError(RuntimeError):
    """Base error for persisted run state."""


class ManifestCompatibilityError(StateError):
    """Raised when a saved run is resumed with a different configuration."""


class InvalidStateTransition(StateError):
    """Raised for an unsafe run status transition."""


def utc_now() -> str:
    """Return an ISO-8601 timestamp with UTC timezone information."""

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _json_ready(value: Any) -> Any:
    """Convert common scientific/configuration values into stable JSON values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_ready(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        # JSON has no portable representation for NaN/Infinity.  A telemetry
        # collector should preserve the sample with a null value instead of
        # failing an otherwise healthy training run.
        return value if math.isfinite(value) else None

    # NumPy scalar/array and enum support without importing either dependency.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_ready(to_list())
        except (TypeError, ValueError):
            pass
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not isinstance(value, (str, int, float, bool)):
        return _json_ready(enum_value)
    return value


def canonical_json(value: Any) -> str:
    """Serialize a configuration deterministically for fingerprints/manifests."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def config_fingerprint(config: Any) -> str:
    """SHA-256 of the canonical, JSON-ready run configuration."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "unknown"


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """The four dimensions that uniquely identify a planned benchmark run."""

    dataset: str
    normalization: str
    balance_mode: str
    seed: int

    @property
    def run_id(self) -> str:
        return stable_run_id(self.dataset, self.normalization, self.balance_mode, self.seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "normalization": self.normalization,
            "balance_mode": self.balance_mode,
            "seed": int(self.seed),
        }


def stable_run_id(dataset: str, normalization: str, balance_mode: str, seed: int) -> str:
    """Return the readable, deterministic ID used for a run directory.

    Deliberately excludes the configuration fingerprint.  A changed
    configuration must fail resume validation rather than silently creating a
    second directory for the same experiment cell.
    """

    return "__".join(
        (_slug(dataset), _slug(normalization), _slug(balance_mode), f"seed-{int(seed)}")
    )


@dataclasses.dataclass(frozen=True)
class RunPaths:
    """Conventional locations below a single stable run directory."""

    root: Path
    manifest: Path
    status: Path
    checkpoints: Path
    logs: Path
    artifacts: Path
    telemetry: Path

    @classmethod
    def from_root(cls, run_dir: str | Path) -> "RunPaths":
        root = Path(run_dir)
        return cls(
            root=root,
            manifest=root / MANIFEST_FILE,
            status=root / "status.json",
            checkpoints=root / "checkpoints",
            logs=root / "logs",
            artifacts=root / "artifacts",
            telemetry=root / "telemetry",
        )

    def ensure(self) -> "RunPaths":
        for directory in (self.root, self.checkpoints, self.logs, self.artifacts, self.telemetry):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def _write_status_snapshot(paths: RunPaths, manifest: Mapping[str, Any]) -> Path:
    """Write a compact status file alongside the full immutable manifest.

    Consumers such as a dashboard can poll this small JSON document without
    parsing the full configuration.  The manifest remains the source of truth.
    """

    payload = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "attempt": manifest.get("attempt", 0),
        "updated_at": manifest.get("updated_at"),
        "status_updated_at": manifest.get("status_updated_at"),
        "detail": manifest.get("status_detail"),
        "error": manifest.get("error"),
    }
    return atomic_write_json(paths.status, payload)


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def append_lock(
    target: str | Path,
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.05,
    stale_after_seconds: float = 600.0,
) -> Iterator[None]:
    """Serialize appends/manifest updates across threads and local processes.

    A tiny ``.lock`` file created with ``O_EXCL`` works on Windows and Linux
    without requiring ``portalocker``.  Stale locks from a killed process are
    removed conservatively after ten minutes by default.
    """

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_key, threading.Lock())

    acquired_file_lock = False
    lock_path = path.with_name(f".{path.name}.lock")
    deadline = time.monotonic() + timeout_seconds
    with thread_lock:
        while not acquired_file_lock:
            try:
                descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(canonical_json({"pid": os.getpid(), "created_at": utc_now()}))
                    handle.flush()
                    os.fsync(handle.fileno())
                acquired_file_lock = True
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > stale_after_seconds:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Tempo excedido ao aguardar lock: {lock_path}")
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            if acquired_file_lock:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    # A future append will either wait or clean a stale lock.
                    pass


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace ``path`` only after its full content reaches disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write structured data atomically, UTF-8 encoded and human-readable."""

    rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)
    return atomic_write_text(path, f"{rendered}\n")


def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Read JSON, returning ``default`` only when the file is absent."""

    source = Path(path)
    if not source.exists():
        return default
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one durable JSONL record without interleaving concurrent writers."""

    destination = Path(path)
    line = canonical_json(record) + "\n"
    with append_lock(destination):
        with destination.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL stream and identify malformed lines explicitly."""

    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateError(f"JSONL inválido em {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise StateError(f"Registro JSONL em {source}:{line_number} não é um objeto")
            records.append(record)
    return records


def append_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Append rows to a CSV with a single immutable header and an append lock."""

    rendered_rows = [{str(key): _json_ready(value) for key, value in row.items()} for row in rows]
    destination = Path(path)
    if not rendered_rows:
        return destination
    requested_fields = list(fieldnames or rendered_rows[0].keys())
    if not requested_fields:
        raise ValueError("fieldnames não pode ser vazio")

    with append_lock(destination):
        existing_fields: list[str] | None = None
        if destination.exists() and destination.stat().st_size:
            with destination.open("r", encoding="utf-8", newline="") as existing:
                existing_fields = next(csv.reader(existing), None)
            if existing_fields != requested_fields:
                raise StateError(
                    f"Cabeçalho incompatível ao anexar {destination}: esperado {existing_fields}, recebido {requested_fields}"
                )
        for row in rendered_rows:
            extra_fields = set(row) - set(requested_fields)
            if extra_fields:
                raise StateError(f"Campos CSV não declarados para {destination}: {sorted(extra_fields)}")
        with destination.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=requested_fields, extrasaction="raise")
            if existing_fields is None:
                writer.writeheader()
            writer.writerows(rendered_rows)
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def make_run_manifest(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    identity: RunIdentity | Mapping[str, Any] | None = None,
    split_fingerprint: str | None = None,
    status: str = "pending",
) -> dict[str, Any]:
    """Create the canonical initial manifest for a run.

    ``identity`` can be omitted when the configuration carries the four fields
    (``dataset``, ``normalization``, ``balance_mode``, ``seed``).
    """

    if status not in VALID_RUN_STATUSES:
        raise ValueError(f"Status inválido: {status}")
    config_copy = _json_ready(dict(config))
    if identity is None:
        try:
            identity_obj = RunIdentity(
                dataset=str(config_copy["dataset"]),
                normalization=str(config_copy["normalization"]),
                balance_mode=str(config_copy["balance_mode"]),
                seed=int(config_copy["seed"]),
            )
        except KeyError as exc:
            raise ValueError("config precisa incluir dataset, normalization, balance_mode e seed") from exc
    elif isinstance(identity, RunIdentity):
        identity_obj = identity
    else:
        identity_obj = RunIdentity(
            dataset=str(identity["dataset"]),
            normalization=str(identity["normalization"]),
            balance_mode=str(identity.get("balance_mode", identity.get("balancing"))),
            seed=int(identity["seed"]),
        )
    now = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id or identity_obj.run_id,
        "identity": identity_obj.as_dict(),
        "config": config_copy,
        "config_fingerprint": config_fingerprint(config_copy),
        "split_fingerprint": split_fingerprint,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "status_updated_at": now,
        "attempt": 0,
    }


def initialize_run(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    identity: RunIdentity | Mapping[str, Any] | None = None,
    split_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create a run layout and manifest, or validate an existing one safely."""

    paths = RunPaths.from_root(run_dir).ensure()
    with append_lock(paths.manifest):
        existing = read_json(paths.manifest)
        if existing is not None:
            validate_manifest_config(existing, config, split_fingerprint=split_fingerprint)
            return existing
        manifest = make_run_manifest(
            config,
            run_id=run_id,
            identity=identity,
            split_fingerprint=split_fingerprint,
        )
        atomic_write_json(paths.manifest, manifest)
        _write_status_snapshot(paths, manifest)
        append_jsonl(
            paths.logs / STATE_EVENTS_FILE,
            {"timestamp": utc_now(), "event": "initialized", "status": manifest["status"]},
        )
        return manifest


def validate_manifest_config(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    split_fingerprint: str | None = None,
) -> None:
    """Reject resume attempts whose configuration/split does not match."""

    expected = str(manifest.get("config_fingerprint", ""))
    actual = config_fingerprint(config)
    if not expected or expected != actual:
        raise ManifestCompatibilityError(
            "A configuração salva não corresponde à configuração solicitada; "
            "crie uma nova saída ou restaure a configuração original."
        )
    saved_split = manifest.get("split_fingerprint")
    if split_fingerprint is not None and saved_split not in {None, split_fingerprint}:
        raise ManifestCompatibilityError("O fingerprint do split salvo não corresponde ao split solicitado.")


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest = read_json(RunPaths.from_root(run_dir).manifest)
    if manifest is None:
        raise FileNotFoundError(f"Manifesto inexistente: {RunPaths.from_root(run_dir).manifest}")
    if not isinstance(manifest, dict):
        raise StateError("Manifesto inválido: esperado objeto JSON")
    return manifest


def update_manifest(
    run_dir: str | Path,
    updates: Mapping[str, Any] | None = None,
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically merge a patch into an existing manifest and return it."""

    paths = RunPaths.from_root(run_dir).ensure()
    with append_lock(paths.manifest):
        manifest = read_manifest(paths.root)
        if expected_config is not None:
            validate_manifest_config(manifest, expected_config)
        if updates:
            protected = {"config_fingerprint", "created_at", "run_id", "identity", "config"}
            attempted = protected.intersection(updates)
            if attempted:
                raise StateError(f"Campos imutáveis do manifesto: {sorted(attempted)}")
            manifest.update(_json_ready(dict(updates)))
        manifest["updated_at"] = utc_now()
        atomic_write_json(paths.manifest, manifest)
        _write_status_snapshot(paths, manifest)
        return manifest


def set_run_status(
    run_dir: str | Path,
    status: str,
    *,
    detail: str | None = None,
    error: str | BaseException | None = None,
    extra: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Transition a run state, persist it atomically, and append an event."""

    if status not in VALID_RUN_STATUSES:
        raise ValueError(f"Status inválido: {status}")
    paths = RunPaths.from_root(run_dir).ensure()
    with append_lock(paths.manifest):
        manifest = read_manifest(paths.root)
        previous = str(manifest.get("status", "pending"))
        if not force and status not in _ALLOWED_TRANSITIONS.get(previous, frozenset()):
            raise InvalidStateTransition(f"Transição não permitida: {previous} -> {status}")
        now = utc_now()
        if status == "running" and previous != "running":
            manifest["attempt"] = int(manifest.get("attempt", 0)) + 1
        manifest["status"] = status
        manifest["updated_at"] = now
        manifest["status_updated_at"] = now
        if detail is not None:
            manifest["status_detail"] = str(detail)
        if error is not None:
            manifest["error"] = str(error)
        elif status in {"running", "completed"}:
            manifest.pop("error", None)
        if extra:
            protected = {"config_fingerprint", "created_at", "run_id", "identity", "config", "status"}
            unsafe = protected.intersection(extra)
            if unsafe:
                raise StateError(f"Campos protegidos no estado extra: {sorted(unsafe)}")
            manifest.update(_json_ready(dict(extra)))
        atomic_write_json(paths.manifest, manifest)
        _write_status_snapshot(paths, manifest)
        append_jsonl(
            paths.logs / STATE_EVENTS_FILE,
            {
                "timestamp": now,
                "event": "status_changed",
                "from_status": previous,
                "status": status,
                "detail": detail,
                "error": None if error is None else str(error),
                "attempt": manifest.get("attempt", 0),
            },
        )
        return manifest


def is_completed(run_dir: str | Path) -> bool:
    """Return true only when the durable manifest marks a run complete."""

    manifest = read_json(RunPaths.from_root(run_dir).manifest, default={})
    return isinstance(manifest, Mapping) and manifest.get("status") == "completed"


__all__ = [
    "MANIFEST_FILE",
    "STATE_EVENTS_FILE",
    "VALID_RUN_STATUSES",
    "InvalidStateTransition",
    "ManifestCompatibilityError",
    "RunIdentity",
    "RunPaths",
    "StateError",
    "append_csv",
    "append_jsonl",
    "append_lock",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json",
    "config_fingerprint",
    "initialize_run",
    "is_completed",
    "make_run_manifest",
    "read_json",
    "read_jsonl",
    "read_manifest",
    "set_run_status",
    "stable_run_id",
    "update_manifest",
    "utc_now",
    "validate_manifest_config",
]
