# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec 3D-AD Datamodule.

This module provides a PyTorch Lightning DataModule for the MVTec 3D-AD dataset.
The dataset contains RGB and depth image pairs for anomaly detection tasks.

Example:
    Create a MVTec3D datamodule::

        >>> from anomalib.data import MVTec3D
        >>> datamodule = MVTec3D(
        ...     root="./datasets/MVTec3D",
        ...     category="bagel"
        ... )

License:
    MVTec 3D-AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger:
    The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and
    Localization. In: Proceedings of the 17th International Joint Conference
    on Computer Vision, Imaging and Computer Graphics Theory and Applications
    - Volume 5: VISAPP, 202-213, 2022.
    DOI: 10.5220/0010865000003124
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.depth.mvtec_3d import CATEGORIES, MVTec3DDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_3d",
    url="https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b961b/download/428824485-1643285832"
    "/mvtec_3d_anomaly_detection.tar.xz",
    hashsum="d8bb2800fbf3ac88e798da6ae10dc819",
)


class MVTec3D(AnomalibDataModule):
    """MVTec 3D-AD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec3D"``.
        category (str | Sequence[str] | None): Category of the MVTec3D dataset (e.g. ``"bottle"`` or
            ``"cable"``). Pass a list of category names to load multiple
            categories, or ``None`` to load all categories.
            Defaults to ``"bagel"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode | str): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode | str): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.
    """

    CATEGORIES = CATEGORIES

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec3D",
        category: str | Sequence[str] | None = "bagel",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
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
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets.

        Args:
            _stage (str | None, optional): Stage of setup. Not used.
                Defaults to ``None``.
        """
        categories = self._resolve_categories()
        self.train_data = MVTec3DDataset(split=Split.TRAIN, root=self.root, category=categories[0])
        self.test_data = MVTec3DDataset(split=Split.TEST, root=self.root, category=categories[0])
        self.train_data.samples["category"] = categories[0]
        self.test_data.samples["category"] = categories[0]

        for cat in categories[1:]:
            train_ds = MVTec3DDataset(split=Split.TRAIN, root=self.root, category=cat)
            test_ds = MVTec3DDataset(split=Split.TEST, root=self.root, category=cat)
            train_ds.samples["category"] = cat
            test_ds.samples["category"] = cat
            self.train_data = self.train_data + train_ds
            self.test_data = self.test_data + test_ds

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        categories = self._resolve_categories()
        if all((self.root / cat).is_dir() for cat in categories):
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
