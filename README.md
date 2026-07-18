# improved-MUNet

This repository addresses the critical challenge of precise brain tumor segmentation in multi-modal MRI, a task hindered by complex tumor morphology and ambiguous boundaries. To overcome the limitations of conventional convolutional networks like U-Net in modeling long-range spatial dependencies, we propose a novel hybrid architecture that integrates a U-Net backbone with a Mamba module based on selective state space models (SSMs). This synergy enables our model to effectively capture both local textual features and global contextual information, providing a more comprehensive understanding of tumor sub-regions for accurate delineation.

The proposed framework is optimized using a composite loss function combining Dice Loss, IoU Loss, Boundary Loss, and Focal Loss, which collectively address both regional and boundary segmentation errors. To enhance model robustness and generalizability, we incorporate Gaussian blur-based data augmentation into our training pipeline, simulating real-world image quality variations. Our method is rigorously evaluated on the BraTS 2020 dataset following standardized preprocessing and evaluation protocols.



<img width="1280" height="720" alt="improved MUNet0" src="https://github.com/user-attachments/assets/9f344f2b-5a0b-4950-847e-ac54823679a9" />


-----------------------------------------------------------------------------------
# How does it work?


sd_ssm_block.py: defines the core building blocks: SCConv2D (spatial+channel reconstruction), 

DWConv2D (depthwise separable), SDConv2D (combines both with dilation), and SDSSMBlock (splits input,

processes local/global branches, fuses, and adds residual).

model.py: defines the full MUNet architecture: embeds input, stacks SDSSMBlocks with down/up‑sampling 

(PatchMerging/PatchExpanding) in a U‑Net style, and uses skip connections


📦 sd_ssm_block.py
🧠 model.py

🧠📦 model.py imports SDSSMBlock from sd_ssm_block.py as the core.

🔽📉 Encoder stacks blocks + downsamples; 🎯 bottleneck processes deepest map.

🔼📈 Decoder upsamples, 🧵 concatenates skips, 🔧 adjusts channels, runs blocks.

⚡🧩 SDSSMBlock uses 🌐 SCConv2D + 🌀 DWConv2D for local/global features.

🔗✨ Finally fuses everything with a residual shortcut for efficient multi‑scale learning!

-----------------------------------------------------------------------------------
# How to run the code?

python train.py --root_dir ./data --csv_path ./metadata.csv

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

# some of the mask predictions















