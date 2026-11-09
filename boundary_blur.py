"""Boundary-aware Gaussian blur augmentation (standalone version).

Tumor boundaries get a stronger Gaussian blur than tumor interiors; healthy
tissue is left unchanged. Labels are never modified. See Section 3.5 of the paper.

Usage:
    blurred = boundary_focused_blur(image, mask)
    image: float tensor [C, H, W] (4 MRI modalities scaled to [0, 1])
    mask:  long tensor  [H, W]    (class indices; > 0 means tumor)
"""
import logging

import cv2
import numpy as np
import torch


def boundary_focused_blur(image: torch.Tensor, mask: torch.Tensor,
                          boundary_sigma: float = 0.8,
                          interior_sigma: float = 0.3) -> torch.Tensor:
    try:
        image_np = image.numpy()
        mask_np = mask.numpy()
        blurred_image = np.zeros_like(image_np)

        tumor_mask = (mask_np > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        # Interior = eroded tumor mask; boundary = dilated minus eroded mask
        tumor_eroded = cv2.erode(tumor_mask, kernel, iterations=1)
        tumor_dilated = cv2.dilate(tumor_mask, kernel, iterations=1)
        boundary_mask = tumor_dilated - tumor_eroded

        # 2 = boundary, 1 = interior, 0 = outside
        region_weights = boundary_mask * 2 + tumor_eroded * 1

        for channel in range(image_np.shape[0]):
            channel_img = image_np[channel]

            if np.sum(tumor_mask) == 0:  # no tumor in this slice
                blurred_image[channel] = channel_img
                continue

            result_channel = np.copy(channel_img)

            # Stronger blur on the boundary band
            if np.sum(boundary_mask) > 0:
                boundary_blurred = cv2.GaussianBlur(
                    channel_img, (0, 0), sigmaX=boundary_sigma, sigmaY=boundary_sigma)
                boundary_regions = region_weights == 2
                result_channel[boundary_regions] = boundary_blurred[boundary_regions]

            # Very mild blend in the interior (70% original + 30% blurred)
            if np.sum(tumor_eroded) > 0:
                interior_blurred = cv2.GaussianBlur(
                    channel_img, (0, 0), sigmaX=interior_sigma, sigmaY=interior_sigma)
                interior_regions = region_weights == 1
                result_channel[interior_regions] = (
                    0.7 * result_channel[interior_regions]
                    + 0.3 * interior_blurred[interior_regions])

            blurred_image[channel] = result_channel

        return torch.from_numpy(blurred_image).float()

    except Exception as e:
        logging.warning(f"Boundary-focused blur failed: {e}. Returning original image.")
        return image
