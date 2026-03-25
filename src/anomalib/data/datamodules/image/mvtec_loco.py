# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec LOCO Data Module.

This module provides a PyTorch Lightning DataModule for the MVTec LOCO dataset. The
dataset contains 5 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

The dataset distinguishes between structural anomalies (local defects) and
logical anomalies (global defects).

Example:
    Create a MVTec LOCO datamodule::

        >>> from anomalib.data import MVTecLOCO
        >>> datamodule = MVTecLOCO(
        ...     root="./datasets/MVTec_LOCO",
        ...     category="breakfast_box"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── MVTec_LOCO/
            ├── breakfast_box/
            ├── juice_bottle/
            └── ...

License:
    MVTec LOCO dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2022).
    MVTec LOCO - A Dataset for Detecting Logical Anomalies in Images.
    In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mvtec_loco import CATEGORIES, MVTecLOCODataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class MVTecLOCO(AnomalibDataModule):
    """MVTec LOCO Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec_LOCO"``.
        category (str | Sequence[str] | None): Category of the MVTec LOCO dataset (e.g. ``"breakfast_box"`` or
            ``"juice_bottle"``). Pass a list of category names to load
            multiple categories, or ``None`` to load all categories.
            Defaults to ``"breakfast_box"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
            Defaults to ``None``.
        test_split_mode (TestSplitMode | str): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        val_split_mode (ValSplitMode | str): Method to create validation set.
            Defaults to ``ValSplitMode.FROM_DIR``.
        test_split_ratio (float | None): Fraction of data to use for testing.
            Defaults to ``None``.
        val_split_ratio (float | None): Fraction of data to use for validation.
            Defaults to ``None``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create MVTec LOCO datamodule with default settings::

            >>> datamodule = MVTecLOCO()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category::

            >>> datamodule = MVTecLOCO(category="juice_bottle")

        Create validation set from test data::

            >>> datamodule = MVTecLOCO(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set::

            >>> datamodule = MVTecLOCO(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    CATEGORIES = CATEGORIES

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str | Sequence[str] | None = "breakfast_box",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_DIR,
        test_split_ratio: float | None = None,
        val_split_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            val_split_mode=val_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        # MVTec LOCO provides a training set that contains only normal images.
        categories = self._resolve_categories()
        cat0 = categories[0]
        self.train_data = MVTecLOCODataset(split=Split.TRAIN, root=self.root, category=cat0)
        self.val_data = MVTecLOCODataset(split=Split.VAL, root=self.root, category=cat0)
        self.test_data = MVTecLOCODataset(split=Split.TEST, root=self.root, category=cat0)

        self.train_data.samples["category"] = cat0
        self.val_data.samples["category"] = cat0
        self.test_data.samples["category"] = cat0

        for cat in categories[1:]:
            train_ds = MVTecLOCODataset(split=Split.TRAIN, root=self.root, category=cat)
            val_ds = MVTecLOCODataset(split=Split.VAL, root=self.root, category=cat)
            test_ds = MVTecLOCODataset(split=Split.TEST, root=self.root, category=cat)
            train_ds.samples["category"] = cat
            val_ds.samples["category"] = cat
            test_ds.samples["category"] = cat
            self.train_data = self.train_data + train_ds
            self.val_data = self.val_data + val_ds
            self.test_data = self.test_data + test_ds
