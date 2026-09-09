"""Deterministic validation and conservative training transforms for radiographs."""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch

from bone_cancer_ai.data.preprocessing import resize_and_pad

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RadiographTransform:
    """Prepare aligned image/mask tensors with optional radiograph-safe augmentation."""

    def __init__(
        self,
        image_size: int,
        *,
        train: bool,
        seed: int,
        augmentation: dict[str, Any] | None = None,
        pad_value: int = 0,
        normalize: bool = True,
    ) -> None:
        self.image_size = int(image_size)
        self.train = bool(train)
        self.pad_value = int(pad_value)
        self.normalize = bool(normalize)
        settings = augmentation or {}
        transforms: list[A.BasicTransform] = []
        if self.train:
            rotation = float(settings.get("rotation_limit", 7))
            translation = float(settings.get("translate_limit", 0.03))
            scale = float(settings.get("scale_limit", 0.05))
            intensity = float(settings.get("brightness_contrast_limit", 0.1))
            gamma = tuple(int(value) for value in settings.get("gamma_limit", [90, 110]))
            noise = tuple(
                float(value)
                for value in settings.get("gaussian_noise_std_range", [0.01, 0.03])
            )
            transforms = [
                A.Affine(
                    scale=(1.0 - scale, 1.0 + scale),
                    translate_percent=(-translation, translation),
                    rotate=(-rotation, rotation),
                    shear=0.0,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=self.pad_value,
                    fill_mask=0,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=intensity,
                    contrast_limit=intensity,
                    p=0.3,
                ),
                A.RandomGamma(gamma_limit=gamma, p=0.2),
                A.GaussNoise(std_range=noise, mean_range=(0.0, 0.0), p=0.15),
            ]
        self._augmentation = A.Compose(transforms, seed=seed, strict=True)

    @property
    def augmentation_names(self) -> list[str]:
        return [type(transform).__name__ for transform in self._augmentation.transforms]

    def set_seed(self, seed: int) -> None:
        """Reset the isolated Albumentations RNG for a DataLoader worker."""
        self._augmentation.set_random_seed(int(seed))

    def __call__(
        self, *, image: np.ndarray, mask: np.ndarray | None = None
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        prepared_image, prepared_mask, geometry = resize_and_pad(
            image,
            mask,
            size=self.image_size,
            pad_value=self.pad_value,
        )
        mask_for_transform = (
            prepared_mask
            if prepared_mask is not None
            else np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        )
        augmented = self._augmentation(image=prepared_image, mask=mask_for_transform)
        image_values = augmented["image"].astype(np.float32) / 255.0
        if self.normalize:
            mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
            std = np.asarray(IMAGENET_STD, dtype=np.float32)
            image_values = (image_values - mean) / std
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_values.transpose(2, 0, 1)))
        mask_values = (np.asarray(augmented["mask"]) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_values[None, ...]))
        return {"image": image_tensor, "mask": mask_tensor, "geometry": geometry}
