# improved-MUNet

This repository addresses the critical challenge of precise brain tumor segmentation in multi-modal MRI, a task hindered by complex tumor morphology and ambiguous boundaries. To overcome the limitations of conventional convolutional networks like U-Net in modeling long-range spatial dependencies, we propose a novel hybrid architecture that integrates a U-Net backbone with a Mamba module based on selective state space models (SSMs). This synergy enables our model to effectively capture both local textual features and global contextual information, providing a more comprehensive understanding of tumor sub-regions for accurate delineation.

The proposed framework is optimized using a composite loss function combining Dice Loss, IoU Loss, Boundary Loss, and Focal Loss, which collectively address both regional and boundary segmentation errors. To enhance model robustness and generalizability, we incorporate Gaussian blur-based data augmentation into our training pipeline, simulating real-world image quality variations. Our method is rigorously evaluated on the BraTS 2020 dataset following standardized preprocessing and evaluation protocols.



<img width="1280" height="720" alt="improved MUNet" src="https://github.com/user-attachments/assets/80167fcf-97a1-465c-8de1-9525da1e3dcc" />
