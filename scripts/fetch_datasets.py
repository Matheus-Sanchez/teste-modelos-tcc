#!/usr/bin/env python3
"""Download the benchmark's public datasets into ``datasets/<name>/``.

This is intentionally an explicit command rather than an implicit behaviour of
the training CLI. Downloads are resumable, archives are checksum-verified when
an upstream checksum is available, and each installed dataset receives a small
``SOURCE.json`` provenance record.  FER2013 uses a public SourceForge mirror;
review its provenance/license before publishing work derived from it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
)
USER_AGENT = "tcc-dataset-benchmark/0.1 (research downloader)"


class DownloadError(RuntimeError):
    """A source could not be downloaded or did not match its expected layout."""


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    md5: str | None = None
    size: int | None = None
    sha256: str | None = None


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_md5(path: Path) -> str:
    return _hash_file(path, "md5")


def _hash_sha256(path: Path) -> str:
    return _hash_file(path, "sha256")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_record(destination: Path, *, dataset: str, sources: Iterable[RemoteFile], note: str | None = None) -> None:
    payload = {
        "dataset": dataset,
        "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [
            {
                "url": item.url,
                "filename": item.filename,
                "md5": item.md5,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in sources
        ],
    }
    if note:
        payload["note"] = note
    _atomic_json(destination / "SOURCE.json", payload)


def _validate_cached_download(target: Path, remote: RemoteFile) -> Path:
    if target.exists():
        if remote.size is not None and target.stat().st_size != remote.size:
            raise DownloadError(
                f"Tamanho inválido para arquivo já existente: {target}; "
                f"esperado {remote.size}, recebido {target.stat().st_size}."
            )
        if remote.md5 and _hash_md5(target) != remote.md5:
            raise DownloadError(f"Checksum MD5 não confere para arquivo já existente: {target}")
        if remote.sha256 and _hash_sha256(target) != remote.sha256:
            raise DownloadError(f"Checksum SHA-256 não confere para arquivo já existente: {target}")
        print(f"  já baixado: {target.name}")
        return target
    return target


def _finalize_download(temporary: Path, target: Path, remote: RemoteFile) -> Path:
    if remote.size is not None and temporary.stat().st_size != remote.size:
        raise DownloadError(
            f"Download incompleto para {remote.filename}: esperado {remote.size} bytes, "
            f"recebido {temporary.stat().st_size}. O arquivo parcial foi preservado."
        )
    if remote.md5:
        actual = _hash_md5(temporary)
        if actual != remote.md5:
            raise DownloadError(
                f"Checksum MD5 inválido para {target.name}: esperado {remote.md5}, recebido {actual}."
            )
    if remote.sha256:
        actual = _hash_sha256(temporary)
        if actual != remote.sha256:
            raise DownloadError(
                f"Checksum SHA-256 inválido para {target.name}: esperado {remote.sha256}, recebido {actual}."
            )
    os.replace(temporary, target)
    return target


def _download_with_curl(remote: RemoteFile, temporary: Path, target: Path) -> Path:
    """Use curl with bounded connections and safe resume after stalled streams."""

    executable = shutil.which("curl")
    if not executable:
        raise FileNotFoundError("curl não está disponível")
    # Some public mirrors drop long-lived connections.  A new curl invocation
    # can resume from ``temporary`` without risking duplicate bytes, so bound
    # each connection and retry at this level rather than abandoning a useful
    # partial download after one transient timeout.
    attempts = 30
    last_size = temporary.stat().st_size if temporary.exists() else 0
    last_error = ""
    for attempt in range(1, attempts + 1):
        if remote.size is not None and temporary.exists() and temporary.stat().st_size == remote.size:
            return _finalize_download(temporary, target, remote)
        command = [
            executable,
            "--location",
            "--fail",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "--speed-time",
            "45",
            "--speed-limit",
            "1024",
            "--user-agent",
            USER_AGENT,
            "--continue-at",
            "-",
            "--output",
            str(temporary),
            remote.url,
        ]
        try:
            result = subprocess.run(command, check=False)
        except OSError as exc:
            raise DownloadError(f"curl não pôde iniciar para {remote.url}: {exc}") from exc

        current_size = temporary.stat().st_size if temporary.exists() else 0
        if result.returncode == 0:
            try:
                return _finalize_download(temporary, target, remote)
            except DownloadError as exc:
                last_error = str(exc)
        else:
            last_error = f"curl retornou código {result.returncode}"

        if current_size > last_size:
            print(
                f"    retomando {remote.filename}: {current_size / 1024 / 1024:.0f} MiB "
                f"(tentativa {attempt}/{attempts})"
            )
            last_size = current_size
        else:
            print(f"    nova conexão para {remote.filename} (tentativa {attempt}/{attempts})")
        if attempt < attempts:
            time.sleep(min(5, attempt))
    raise DownloadError(f"curl não conseguiu concluir {remote.url}: {last_error}")


def _download_with_urllib(remote: RemoteFile, temporary: Path, target: Path) -> Path:
    """Portable fallback with HTTP Range resume for environments without curl."""

    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        print(f"  retomando {remote.filename} em {offset:,} bytes")
    else:
        print(f"  baixando {remote.filename}")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as exc:
        raise DownloadError(f"Não foi possível baixar {remote.url}: {exc}") from exc
    with response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            # A server that ignored Range must not produce a duplicated file.
            temporary.unlink(missing_ok=True)
            return _download_with_urllib(remote, temporary, target)
        content_length = response.headers.get("Content-Length")
        expected = offset + int(content_length) if content_length and content_length.isdigit() else None
        mode = "ab" if offset else "wb"
        written = offset
        next_report = written + 64 * 1024 * 1024
        with temporary.open(mode) as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                written += len(block)
                if written >= next_report:
                    fraction = "?" if expected is None else f"{written / expected:.0%}"
                    print(f"    {remote.filename}: {written / 1024 / 1024:.0f} MiB ({fraction})")
                    next_report += 64 * 1024 * 1024
            stream.flush()
            os.fsync(stream.fileno())
    return _finalize_download(temporary, target, remote)


def _download(remote: RemoteFile, downloads_dir: Path) -> Path:
    """Download into the downloader-owned cache, preferring resilient curl."""

    downloads_dir.mkdir(parents=True, exist_ok=True)
    target = downloads_dir / remote.filename
    temporary = target.with_suffix(target.suffix + ".part")
    if target.exists():
        return _validate_cached_download(target, remote)
    print(f"  baixando {remote.filename}")
    if shutil.which("curl"):
        return _download_with_curl(remote, temporary, target)
    return _download_with_urllib(remote, temporary, target)


def _safe_tar_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as source:
        base = destination.resolve()
        for member in source.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != base and base not in candidate.parents:
                raise DownloadError(f"Caminho inseguro no tar: {member.name}")
        source.extractall(destination)


def _safe_zip_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        base = destination.resolve()
        for member in source.infolist():
            candidate = (destination / member.filename).resolve()
            if candidate != base and base not in candidate.parents:
                raise DownloadError(f"Caminho inseguro no ZIP: {member.filename}")
        source.extractall(destination)


def _find_parent(root: Path, predicate: Callable[[Path], bool]) -> Path:
    if predicate(root):
        return root
    for candidate in root.rglob("*"):
        if candidate.is_dir() and predicate(candidate):
            return candidate
    raise DownloadError(f"Estrutura esperada não encontrada após extrair {root}")


def _copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _promote_extracted_tree(source: Path, destination: Path) -> None:
    """Promote an extracted dataset without duplicating a large tree on disk.

    Archives are first unpacked under ``.staging``.  On the same filesystem a
    move into an absent destination is a fast rename, which matters for
    CINIC-10's hundreds of thousands of small files.  If a prior interrupted
    install already created the destination, retain the conservative merge.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _copy_tree(source, destination)
    else:
        shutil.move(str(source), str(destination))


def _remove_archive(path: Path, keep_archives: bool) -> None:
    if not keep_archives:
        path.unlink(missing_ok=True)


def _verify_member_md5s(directory: Path, expected: Mapping[str, str]) -> None:
    """Verify the extracted payload when a mirror repacks an official archive."""

    for relative_path, wanted in expected.items():
        candidate = directory / relative_path
        if not candidate.is_file():
            raise DownloadError(f"Arquivo esperado ausente após extração: {candidate}")
        actual = _hash_md5(candidate)
        if actual != wanted:
            raise DownloadError(
                f"Checksum MD5 inválido para conteúdo extraído {relative_path}: esperado {wanted}, recebido {actual}."
            )


def _idx_dataset(root: Path, name: str, files: tuple[RemoteFile, ...], keep_archives: bool) -> None:
    destination = root / name
    required = [destination / item.filename for item in files]
    if not all(path.is_file() for path in required):
        for item in files:
            archive = _download(item, root / ".downloads" / name)
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / item.filename
            if not target.exists():
                shutil.copy2(archive, target)
            _remove_archive(archive, keep_archives)
    _source_record(destination, dataset=name, sources=files)


MNIST_FILES = (
    RemoteFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    RemoteFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    RemoteFile("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    RemoteFile("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
)

FASHION_FILES = (
    RemoteFile("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    RemoteFile("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    RemoteFile("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    RemoteFile("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
)

KMNIST_FILES = (
    RemoteFile("https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz", "kmnist-train-imgs.npz"),
    RemoteFile("https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz", "kmnist-train-labels.npz"),
    RemoteFile("https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz", "kmnist-test-imgs.npz"),
    RemoteFile("https://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz", "kmnist-test-labels.npz"),
)


CIFAR10_MEMBER_MD5S = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}


def _install_emnist(root: Path, keep_archives: bool) -> None:
    destination = root / "emnist_balanced"
    names = (
        "emnist-balanced-train-images-idx3-ubyte.gz",
        "emnist-balanced-train-labels-idx1-ubyte.gz",
        "emnist-balanced-test-images-idx3-ubyte.gz",
        "emnist-balanced-test-labels-idx1-ubyte.gz",
        "emnist-balanced-mapping.txt",
    )
    remote = RemoteFile("https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "emnist-gzip.zip")
    if not all((destination / name).is_file() for name in names[:4]):
        archive = _download(remote, root / ".downloads" / "emnist_balanced")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            members = {Path(item.filename).name: item for item in source.infolist()}
            missing = [name for name in names if name not in members]
            if missing:
                raise DownloadError(f"EMNIST ZIP sem arquivos Balanced esperados: {missing}")
            for name in names:
                target = destination / name
                if not target.exists():
                    with source.open(members[name]) as input_stream, target.open("wb") as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
        _remove_archive(archive, keep_archives)
    _source_record(
        destination,
        dataset="emnist_balanced",
        sources=(remote,),
        note="Somente a variante Balanced foi extraída do ZIP oficial da NIST; a orientação IDX é corrigida no adaptador.",
    )


def _install_cifar(
    root: Path,
    name: str,
    remote: RemoteFile,
    sentinel: str,
    keep_archives: bool,
    *,
    source_note: str | None = None,
    member_md5s: Mapping[str, str] | None = None,
) -> None:
    destination = root / name
    if not (destination / sentinel).is_file():
        archive = _download(remote, root / ".downloads" / name)
        staging_root = root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=staging_root) as temporary:
            staging = Path(temporary)
            _safe_tar_extract(archive, staging)
            source = _find_parent(staging, lambda item: (item / sentinel).is_file())
            if member_md5s:
                _verify_member_md5s(source, member_md5s)
            _promote_extracted_tree(source, destination)
        _remove_archive(archive, keep_archives)
    _source_record(destination, dataset=name, sources=(remote,), note=source_note)


def _install_cinic10(root: Path, keep_archives: bool) -> None:
    destination = root / "cinic10"
    remote = RemoteFile(
        "https://datashare.ed.ac.uk/server/api/core/bitstreams/e8e186cc-2688-48f1-aa5a-3fda1a43f8b6/content",
        "CINIC-10.tar.gz",
        "6ee4d0c996905fe93221de577967a372",
        size=687_544_992,
    )
    if not all((destination / split).is_dir() for split in ("train", "valid", "test")):
        archive = _download(remote, root / ".downloads" / "cinic10")
        staging_root = root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cinic10-", dir=staging_root) as temporary:
            staging = Path(temporary)
            _safe_tar_extract(archive, staging)
            source = _find_parent(
                staging,
                lambda item: all((item / split).is_dir() for split in ("train", "valid", "test")),
            )
            _promote_extracted_tree(source, destination)
        _remove_archive(archive, keep_archives)
    _source_record(destination, dataset="cinic10", sources=(remote,))


def _install_svhn(root: Path, keep_archives: bool) -> None:
    destination = root / "svhn"
    files = (
        RemoteFile("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
        RemoteFile("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"),
    )
    if not all((destination / item.filename).is_file() for item in files):
        for item in files:
            archive = _download(item, root / ".downloads" / "svhn")
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / item.filename
            if not target.exists():
                shutil.copy2(archive, target)
            _remove_archive(archive, keep_archives)
    _source_record(
        destination,
        dataset="svhn",
        sources=files,
        note="A fonte Stanford oferece esses arquivos em HTTP; os hashes MD5 oficiais são verificados antes da instalação.",
    )


def _install_gtsrb(root: Path, keep_archives: bool) -> None:
    destination = root / "gtsrb"
    files = (
        RemoteFile("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip", "GTSRB-Training_fixed.zip", "513f3c79a4c5141765e10e952eaa2478"),
        RemoteFile("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_Images.zip", "c7e4e6327067d32654124b0fe9e82185"),
        RemoteFile("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip", "GTSRB_Final_Test_GT.zip", "fe31e9c9270bbcd7b84b7f21a9d9d9e5"),
    )
    # Public GTSRB archives use either ``Training/<class>`` or
    # ``Final_Training/Images/<class>``.  The adapter accepts both, so the
    # installer must not redownload a valid archive solely due to that wrapper.
    train_dirs = (
        destination / "GTSRB" / "Training",
        destination / "GTSRB" / "Final_Training" / "Images",
    )
    test_dir = destination / "GTSRB" / "Final_Test" / "Images"
    ground_truth_present = any(destination.rglob("GT-final_test.csv")) if destination.exists() else False
    needed_archives = (
        (files[0], not any(path.is_dir() for path in train_dirs)),
        (files[1], not test_dir.is_dir()),
        (files[2], not ground_truth_present),
    )
    if any(required for _, required in needed_archives):
        for item, required in needed_archives:
            if not required:
                continue
            archive = _download(item, root / ".downloads" / "gtsrb")
            staging_root = root / ".staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="gtsrb-", dir=staging_root) as temporary:
                staging = Path(temporary)
                _safe_zip_extract(archive, staging)
                _copy_tree(staging, destination)
            _remove_archive(archive, keep_archives)
    _source_record(destination, dataset="gtsrb", sources=files)


def _validate_fer2013(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"emotion", "pixels"}
        if not required.issubset(fields):
            raise DownloadError(f"FER2013 CSV sem colunas {sorted(required)}: {reader.fieldnames}")
        labels: set[int] = set()
        count = 0
        for count, row in enumerate(reader, start=1):
            labels.add(int(row["emotion"]))
            if len(row["pixels"].split()) != 48 * 48:
                raise DownloadError(f"FER2013 possui pixels inválidos na linha {count + 1}.")
    if count != 35887 or labels != set(range(7)):
        raise DownloadError(f"FER2013 inesperado: {count} linhas e labels {sorted(labels)}; esperado 35887 e 0..6.")


def _install_fer2013(root: Path, keep_archives: bool) -> None:
    destination = root / "fer2013"
    # The direct SourceForge redirector is more reliable for resumable command
    # line downloads than the project HTML ``/download`` page on Windows.
    remote = RemoteFile(
        "https://downloads.sourceforge.net/project/emotion-detector/fer2013.csv",
        "fer2013.csv",
        size=301_072_766,
    )
    target = destination / "fer2013.csv"
    if not target.is_file():
        archive = _download(remote, root / ".downloads" / "fer2013")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, target)
        _remove_archive(archive, keep_archives)
    _validate_fer2013(target)
    _source_record(
        destination,
        dataset="fer2013",
        sources=(remote,),
        note="Fonte é um espelho público SourceForge; confirme a licença/proveniência antes de redistribuir ou publicar resultados.",
    )


INSTALLERS: dict[str, Callable[[Path, bool], None]] = {
    "mnist": lambda root, keep: _idx_dataset(root, "mnist", MNIST_FILES, keep),
    "fashion_mnist": lambda root, keep: _idx_dataset(root, "fashion_mnist", FASHION_FILES, keep),
    "kmnist": lambda root, keep: _idx_dataset(root, "kmnist", KMNIST_FILES, keep),
    "emnist_balanced": _install_emnist,
    # The public mirrors are used for transfer reliability in this environment.
    # The final archives are still checked against the MD5 values published by
    # the original University of Toronto distribution.
    "cifar10": lambda root, keep: _install_cifar(
        root,
        "cifar10",
        RemoteFile(
            "https://huggingface.co/Peyiloo/peyiloo/resolve/main/cifar-10-python.tar.gz?download=true",
            "cifar-10-python.tar.gz",
            size=170_498_071,
            sha256="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        ),
        "data_batch_1",
        keep,
        source_note=(
            "Obtido de espelho público, validado pelo SHA-256 publicado pelo espelho e pelos MD5s "
            "dos arquivos internos da distribuição oficial: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        ),
        member_md5s=CIFAR10_MEMBER_MD5S,
    ),
    "cifar100_coarse": lambda root, keep: _install_cifar(
        root,
        "cifar100_coarse",
        RemoteFile(
            "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz?download=true",
            "cifar-100-python.tar.gz",
            "eb9058c3a382ffc7106e4002c42a8d85",
            size=169_001_437,
        ),
        "train",
        keep,
        source_note=(
            "Obtido de espelho público e validado com o MD5 publicado pela distribuição oficial: "
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        ),
    ),
    "svhn": _install_svhn,
    "gtsrb": _install_gtsrb,
    "fer2013": _install_fer2013,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa os datasets públicos do benchmark para datasets/<nome>/.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Baixa os nove datasets ativos.")
    group.add_argument("--dataset", choices=DATASET_NAMES, action="append", help="Baixa somente um dataset; pode ser repetido.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "datasets", help="Raiz de dados local.")
    parser.add_argument("--keep-archives", action="store_true", help="Mantém arquivos compactados em datasets/.downloads/.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra o plano, sem rede nem escrita.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    selected = list(DATASET_NAMES if args.all else dict.fromkeys(args.dataset))
    if args.dry_run:
        print(f"Raiz: {root}")
        print("Datasets: " + ", ".join(selected))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    (root / ".staging").mkdir(exist_ok=True)
    available = shutil.disk_usage(root).free / 1024**3
    print(f"Raiz: {root} ({available:.1f} GiB livres)")
    failures: list[tuple[str, Exception]] = []
    for name in selected:
        print(f"\n[{name}]")
        try:
            INSTALLERS[name](root, bool(args.keep_archives))
            print("  pronto")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FALHOU: {exc}", file=sys.stderr)
    # Other explicit downloader processes may share the root.  Cleanup is only
    # cosmetic, so a concurrent create/remove must never turn a successful
    # installation into an error.
    staging_root = root / ".staging"
    if staging_root.exists() and not any(staging_root.iterdir()):
        try:
            staging_root.rmdir()
        except OSError:
            pass
    if failures:
        print("\nFalhas: " + ", ".join(f"{name}: {error}" for name, error in failures), file=sys.stderr)
        return 1
    print("\nTodos os datasets solicitados foram instalados. Use: tcc-benchmark audit --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
