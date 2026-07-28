from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from tcc_benchmark.adapters import (
    DatasetSplit,
    LoadedDataset,
    load_cifar10,
    load_cifar100_coarse,
    load_cinic10,
    load_emnist_balanced,
    load_fashion_mnist,
    load_fer2013,
    load_gtsrb,
    load_kmnist,
    load_mnist,
    load_svhn,
)
from tcc_benchmark.audit import audit_dataset


def _write_idx(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array, dtype=np.uint8)
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">HBB", 0, 8, array.ndim))
        stream.write(struct.pack(">" + "I" * array.ndim, *array.shape))
        stream.write(array.tobytes())


def test_mnist_idx_adapter_is_local_and_nhwc(tmp_path: Path) -> None:
    _write_idx(tmp_path / "train-images-idx3-ubyte.gz", np.zeros((3, 2, 2), dtype=np.uint8))
    _write_idx(tmp_path / "train-labels-idx1-ubyte.gz", np.array([0, 1, 0], dtype=np.uint8))
    _write_idx(tmp_path / "t10k-images-idx3-ubyte.gz", np.ones((2, 2, 2), dtype=np.uint8))
    _write_idx(tmp_path / "t10k-labels-idx1-ubyte.gz", np.array([1, 0], dtype=np.uint8))

    loaded = load_mnist(tmp_path)

    assert loaded.splits["train"].images is not None
    assert loaded.splits["train"].images.shape == (3, 2, 2, 1)
    assert loaded.labels_for().tolist() == [0, 1, 0, 1, 0]


def test_cinic_folder_adapter_stays_lazy_and_audit_detects_duplicates_and_corruption(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "airplane").mkdir(parents=True)
        (tmp_path / split / "automobile").mkdir(parents=True)
    pixels = np.full((3, 4, 3), 12, dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "train" / "airplane" / "a.png")
    Image.fromarray(pixels).save(tmp_path / "train" / "automobile" / "same-pixels.png")
    (tmp_path / "valid" / "airplane" / "bad.png").write_bytes(b"not an image")
    Image.fromarray(pixels).save(tmp_path / "test" / "airplane" / "ok.png")

    loaded = load_cinic10(tmp_path)
    report = audit_dataset(loaded)

    assert loaded.splits["train"].images is None
    assert len(loaded.splits["train"].records) == 2
    assert report.duplicate_group_count == 1
    assert len(report.unreadable_images) == 1
    assert report.to_dict()["summary"]["healthy"] is False


def test_fer2013_csv_adapter_parses_native_grayscale(tmp_path: Path) -> None:
    values = " ".join(str(value) for value in range(256))
    pixels = (values + " ") * 9
    pixels = " ".join(pixels.split()[: 48 * 48])
    (tmp_path / "fer2013.csv").write_text(
        "emotion,pixels,Usage\n0,\"" + pixels + "\",Training\n1,\"" + pixels + "\",PrivateTest\n",
        encoding="utf-8",
    )

    loaded = load_fer2013(tmp_path)

    assert loaded.splits["train"].images is not None
    assert loaded.splits["train"].images.shape == (1, 48, 48, 1)
    assert loaded.splits["test"].labels.tolist() == [1]  # type: ignore[union-attr]


def test_audit_array_dataset_is_serializable_and_flags_out_of_range_label(tmp_path: Path) -> None:
    loaded = LoadedDataset(
        "fixture",
        ["zero", "one"],
        {"train": DatasetSplit(images=np.zeros((2, 3, 3, 1), dtype=np.uint8), labels=np.array([0, 3]))},
        tmp_path,
    )

    report = audit_dataset(loaded)

    payload = report.to_dict()
    assert payload["label_counts"] == {"train": {"0": 1, "3": 1}}
    assert payload["invalid_labels"][0]["label"] == 3


def test_all_ten_adapters_accept_a_minimal_local_layout(tmp_path: Path) -> None:
    """Exercise every documented local format without any network access."""

    labels = np.array([0, 1], dtype=np.int64)
    grayscale = np.zeros((2, 4, 4), dtype=np.uint8)
    rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    for name, loader, filename in (
        ("mnist", load_mnist, "mnist.npz"),
        ("fashion", load_fashion_mnist, "fashion-mnist.npz"),
        ("kmnist", load_kmnist, "kmnist.npz"),
        ("emnist", load_emnist_balanced, "emnist-balanced.npz"),
    ):
        root = tmp_path / name
        root.mkdir()
        np.savez(root / filename, x_train=grayscale, y_train=labels, x_test=grayscale, y_test=labels)
        loaded = loader(root)
        assert loaded.sample_count() == 4
        assert audit_dataset(loaded, max_samples=1).checked_samples == 1

    cifar10 = tmp_path / "cifar10.npz"
    np.savez(cifar10, x_train=rgb, y_train=labels, x_test=rgb, y_test=labels)
    assert load_cifar10(cifar10).sample_count() == 4

    cifar100 = tmp_path / "cifar100.npz"
    np.savez(cifar100, x_train=rgb, y_train=labels, x_test=rgb, y_test=labels)
    assert load_cifar100_coarse(cifar100).sample_count() == 4

    cinic = tmp_path / "cinic"
    for split in ("train", "valid", "test"):
        for class_name in ("airplane", "automobile"):
            folder = cinic / split / class_name
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(folder / "one.png")
    assert load_cinic10(cinic).sample_count() == 6

    from scipy.io import savemat

    svhn = tmp_path / "svhn"
    svhn.mkdir()
    # Official SVHN layout is H×W×C×N.
    savemat(svhn / "train_32x32.mat", {"X": np.zeros((32, 32, 3, 2), dtype=np.uint8), "y": np.array([[1], [10]])})
    savemat(svhn / "test_32x32.mat", {"X": np.zeros((32, 32, 3, 2), dtype=np.uint8), "y": np.array([[2], [3]])})
    assert load_svhn(svhn).sample_count() == 4

    gtsrb = tmp_path / "gtsrb" / "Train" / "0"
    gtsrb.mkdir(parents=True)
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(gtsrb / "sign.png")
    assert load_gtsrb(gtsrb.parent.parent).sample_count() == 1

    # The official test-images archive ships an auxiliary CSV without ClassId;
    # the separately supplied ground-truth CSV must be selected instead.
    official_gtsrb = tmp_path / "official_gtsrb" / "GTSRB"
    train_sign = official_gtsrb / "Training" / "00000"
    test_images = official_gtsrb / "Final_Test" / "Images"
    train_sign.mkdir(parents=True)
    test_images.mkdir(parents=True)
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(train_sign / "00000.ppm")
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(test_images / "00000.ppm")
    (test_images / "GT-final_test.test.csv").write_text(
        "Filename;Width;Height\n00000.ppm;5;5\n", encoding="utf-8"
    )
    (test_images / "GT-final_test.csv").write_text(
        "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n00000.ppm;5;5;0;0;4;4;0\n",
        encoding="utf-8",
    )
    official_loaded = load_gtsrb(official_gtsrb.parent)
    assert len(official_loaded.splits["train"]) == 1
    assert len(official_loaded.splits["test"]) == 1

    fer = tmp_path / "fer"
    fer.mkdir()
    pixels = " ".join("0" for _ in range(48 * 48))
    (fer / "fer2013.csv").write_text(
        f"emotion,pixels,Usage\n0,\"{pixels}\",Training\n1,\"{pixels}\",PrivateTest\n",
        encoding="utf-8",
    )
    assert load_fer2013(fer).sample_count() == 2
