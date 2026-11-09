# improved-MUNet

Code for the paper **"Boundary-Aware Supervision and ET-Focused Training for Brain Tumor Segmentation with a MUNet Backbone"**
Sepideh Mirfendereski, Babak Karasfi. Department of Computer and Information Technology Engineering, Qazvin Branch, Islamic Azad University. Part of my MSc thesis in Artificial Intelligence (2026).

Accurate segmentation of the enhancing tumor (ET) from multi-modal MRI is hard because the region is small and its boundary is ambiguous. We build on a reimplemented **MUNet** baseline (Yang et al., 2025) and study three training-side changes, evaluated under one protocol on BraTS 2020.

## What this work does

The backbone is **not** a contribution of this work. The changes studied are:

1. **Compound loss.** Class-weighted Dice and mIoU, a Sobel-gradient boundary term and a focal term, combined with equal weights chosen from a search over seven weight settings.
2. **ET-focused schedule.** The ET class weight is raised along a smooth-step schedule after a warm-up period and modulated by a conservative performance-feedback rule.
3. **Boundary-aware blur augmentation.** Tumor boundaries get a stronger Gaussian blur than tumor interiors; healthy tissue is unchanged.

![Training pipeline](images/pipeline.png)

## Backbone

MUNet-style U-Net on 4-channel 2D slices (FLAIR, T1, T1ce, T2): a 1x1 embedding to 64 channels, four SD-SSM blocks in the encoder (64 to 512 channels) with strided-convolution downsampling, an SD-SSM bottleneck, and a decoder with transposed convolutions, skip connections and 1x1 channel reduction. The output has four classes: background, non-enhancing core, edema, enhancing tumor.

The SD-SSM block follows the design introduced with MUNet. In this implementation both branches are built from dilated depthwise/spatial-channel convolutions and linear layers (see `sd_ssm_block.py` and the paper, Section 3.2).

![Network](images/network.png)
![SD-SSM block](images/sd_ssm_block.png)

## Results (BraTS 2020, held-out test split, one slice per case)

| Model | WT Dice | TC Dice | ET Dice | WT HD95 | TC HD95 | ET HD95 |
|---|---|---|---|---|---|---|
| Baseline (reimplemented MUNet) | 0.926 | 0.830 | 0.682 | 1.72 | 3.12 | 1.50 |
| + ET-focused changes | 0.907 | 0.774 | 0.715 | 3.09 | 6.24 | 4.50 |
| + boundary-aware blur (final) | 0.914 | 0.801 | **0.741** | 2.81 | 5.30 | 5.11 |

The changes raise ET Dice (0.682 to 0.741) but lower WT and TC Dice and worsen HD95, because Dice, especially for ET, was prioritized over HD95. Five-fold cross-validation (different splits, not directly comparable) gives mean validation Dice of 0.849 (WT), 0.744 (TC) and 0.720 (ET).

HD95 is computed per slice in pixels over all region pixels, so it is not comparable with surface-based HD95 values from other work. Values published in the original MUNet paper were obtained in a different evaluation setting that we did not reproduce, so no superiority or inferiority claim is made with respect to them.

## Limitations

Evaluation is 2D, on one slice per case and one third of the cases, on a single test split with each model trained once. Components were selected in stages rather than jointly, and the baseline and the changed models differ in training budget. The results are preliminary; volumetric evaluation, repeated runs and component-wise ablations are needed.

## Repository layout

- `model.py`, `sd_ssm_block.py`: network (importable and runnable on its own)
- `loss.py`: compound loss, ET-focused schedule and evaluation metrics
- `boundary_blur.py`: boundary-aware blur augmentation (standalone)

The rest of the data pipeline (`dataset.py`) and the training/evaluation scripts (`train.py`, `test.py`) are described in the paper (Sections 3.5 and 4.1) but are not published in this repository. They are available on request for review or reproduction; please contact me.

BraTS 2020 must be obtained separately (e.g. the [Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)).

## References

- L. Yang, Q. Dong, D. Lin, C. Tian, X. Lu, "MUNet: a novel framework for accurate brain tumor segmentation combining UNet and Mamba networks," Front. Comput. Neurosci., vol. 19, 2025.
- B. H. Menze et al., "The multimodal brain tumor image segmentation benchmark (BRATS)," IEEE Trans. Med. Imaging, 2015.

## License

Copyright (c) 2026 Sepideh Mirfendereski. All rights reserved. The code is shared for review and evaluation; reuse or redistribution requires written permission.
