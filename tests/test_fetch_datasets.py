from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def _load_fetch_module():
    source = Path(__file__).resolve().parents[1] / "scripts" / "fetch_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_datasets_test_module", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_download_finalization_validates_size_and_hashes(tmp_path: Path) -> None:
    fetch = _load_fetch_module()
    content = b"benchmark-cache"
    temporary = tmp_path / "archive.part"
    target = tmp_path / "archive"
    temporary.write_bytes(content)
    remote = fetch.RemoteFile(
        "https://example.invalid/archive",
        "archive",
        md5=hashlib.md5(content).hexdigest(),
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )

    assert fetch._finalize_download(temporary, target, remote) == target
    assert target.read_bytes() == content
    assert not temporary.exists()


def test_extracted_member_checksums_detect_official_payload(tmp_path: Path) -> None:
    fetch = _load_fetch_module()
    payload = tmp_path / "payload"
    payload.mkdir()
    content = b"official-member"
    (payload / "batch").write_bytes(content)

    fetch._verify_member_md5s(payload, {"batch": hashlib.md5(content).hexdigest()})
