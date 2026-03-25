# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multi-class (multi-category) support in datamodules."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD
from tests.helpers.data import DummyImageDatasetGenerator


@pytest.fixture(scope="module")
def multi_category_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a dummy MVTecAD dataset with two categories."""
    root = tmp_path_factory.mktemp("datasets")

    # Generate "dummy" category (standard)
    gen1 = DummyImageDatasetGenerator(data_format="mvtecad", root=root, num_train=5, num_test=5)
    gen1.generate_dataset()

    # Generate a second category "dummy2" by copying the same structure
    src = root / "mvtecad" / "dummy"
    dst = root / "mvtecad" / "dummy2"

    import shutil

    shutil.copytree(src, dst)

    return root


class TestMultiClass:
    """Tests for multi-category support."""

    def test_single_category_backward_compat(self, multi_category_path: Path) -> None:
        """Single string category still works."""
        dm = MVTecAD(
            root=multi_category_path / "mvtecad",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((32, 32)),
        )
        dm.setup()
        assert len(dm.train_data) > 0
        assert "category" in dm.train_data.samples.columns
        assert dm.train_data.samples["category"].unique().tolist() == ["dummy"]

    def test_multi_category_list(self, multi_category_path: Path) -> None:
        """Passing a list of categories loads samples from all of them."""
        dm = MVTecAD(
            root=multi_category_path / "mvtecad",
            category=["dummy", "dummy2"],
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((32, 32)),
        )
        dm.setup()

        # Both categories present
        cats = sorted(dm.train_data.samples["category"].unique().tolist())
        assert cats == ["dummy", "dummy2"]

        cats_test = sorted(dm.test_data.samples["category"].unique().tolist())
        assert cats_test == ["dummy", "dummy2"]

        # More samples than a single category
        single_dm = MVTecAD(
            root=multi_category_path / "mvtecad",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((32, 32)),
        )
        single_dm.setup()
        assert len(dm.train_data) > len(single_dm.train_data)

    def test_category_column_present(self, multi_category_path: Path) -> None:
        """The 'category' column is present even for single-category."""
        dm = MVTecAD(
            root=multi_category_path / "mvtecad",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((32, 32)),
        )
        dm.setup()
        assert "category" in dm.train_data.samples.columns
        assert "category" in dm.test_data.samples.columns

    def test_resolve_categories_from_class_var(self) -> None:
        """_resolve_categories returns CATEGORIES when category=None."""
        dm = MVTecAD.__new__(MVTecAD)
        dm._category = None
        cats = dm._resolve_categories()
        assert isinstance(cats, list)
        assert len(cats) == 15  # MVTecAD has 15 categories
        assert "bottle" in cats

    def test_resolve_categories_single_string(self) -> None:
        """_resolve_categories wraps a single string in a list."""
        dm = MVTecAD.__new__(MVTecAD)
        dm._category = "bottle"
        assert dm._resolve_categories() == ["bottle"]

    def test_resolve_categories_sequence(self) -> None:
        """_resolve_categories converts a sequence to a list."""
        dm = MVTecAD.__new__(MVTecAD)
        dm._category = ("bottle", "cable")
        assert dm._resolve_categories() == ["bottle", "cable"]

    def test_dataloader_works(self, multi_category_path: Path) -> None:
        """Verify that the dataloader produces batches from multi-category data."""
        dm = MVTecAD(
            root=multi_category_path / "mvtecad",
            category=["dummy", "dummy2"],
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((32, 32)),
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch.image.shape[0] <= 4
