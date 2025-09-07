Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation

This repository implements the intelligent framework for robust road segmentation in autonomous mining vehicle systems, as described in the paper: Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation 

Authors: Claudio Urrea\*, Maximiliano VélezAffiliation: Electrical Engineering Department, University of Santiago of ChileCorrespondence: claudio.urrea@usach.cl



Abstract

Intelligent autonomous systems in open-pit mining operations face critical challenges in perception and decision-making due to sensor-based visual degradations, particularly lens soiling and sun glare, which significantly compromise the performance and safety of integrated mining automation systems. We propose a comprehensive intelligent frame-work leveraging single-domain generalization with traditional data augmentation tech-niques, specifically Photometric Distortion (PD) and Contrast Limited Adaptive Histo-gram Equalization (CLAHE), integrated within the BiSeNetV1 architecture. Our systemat-ic approach evaluated four state-of-the-art backbones: ResNet-50, MobileNetV2 (Convolu-tional Neural Networks (CNN)-based), SegFormer-B0, and Twins-PCPVT-S (ViT-based) within an end-to-end autonomous system architecture. The model was trained on clean images from the AutoMine dataset and tested on degraded visual conditions without re-quiring architectural modifications or additional training data from target domains. Res-Net-50 demonstrated superior system robustness with mean Intersection over Union (IoU) of 84.58% for lens soiling and 80.11% for sun glare scenarios, while MobileNetV2 achieved optimal computational efficiency for real-time autonomous systems with 55.0 Frames Per Second (FPS) inference speed while maintaining competitive accuracy (81.54% and 71.65% mIoU respectively). Vision Transformers showed superior stability in system per-formance but lower overall performance under severe degradations. The proposed intelli-gent augmentation-based approach maintains high accuracy while preserving real-time computational efficiency, making it suitable for deployment in autonomous mining vehi-cle systems. Traditional augmentation approaches achieved approximately 30% superior performance compared to advanced GAN-based domain generalization methods, providing a practical solution for robust perception systems without requiring expensive multi-domain training datasets.



Keywords: intelligent autonomous systems; mining automation systems; real-time perception systems; domain generalization; visual degradation robustness; embedded intelligent systems; autonomous vehicle systems; intelligent mining operations; sensor-based decision making



Key Features

Single-domain generalization for handling visual degradations (lens soiling, sun glare) in mining environments.

Implementation based on MMSegmentation framework with custom BiSeNetV1 architecture.

Evaluation of multiple backbones: ResNet-50, MobileNetV2, SegFormer-B0, Twins-PCPVT-S.

Data augmentation techniques: Photometric Distortion (PD) and CLAHE.

Real-time inference capabilities for embedded systems.

Comprehensive documentation, scripts for training/evaluation, and reproducible experiments.

CycleGAN Method for generating stylized images to enhance training data.

Scripts for calculating perceptual consistency, latency, FPS, memory usage, and temporal image separation.



Installation

Follow the detailed instructions in docs/installation.md. Summary:



Clone the repository:

git clone https://github.com/ClaudioUrrea/segmentation.git

cd segmentation





Create conda environment:

conda env create -f environment.yml

conda activate intelligent-mining





Install MMSegmentation:

cd mmsegmentation

pip install -v -e .





Install MMGeneration for CycleGAN configurations:Refer to the documentation at MMGeneration for setup instructions.





Dataset

Download the AutoMine dataset from FigShare: https://doi.org/10.6084/m9.figshare.29897300



Source domain: Clean images for training.

Target domains: Lens soiling and sun glare for testing.



See docs/dataset\_preparation.md for setup instructions.

Usage

Training

Use the Jupyter notebook:

jupyter notebook scripts/For\_training.ipynb



Select a config file from mmsegmentation/configs/ (e.g., ResNet/Resnet\_lr0.001\_Clahe\_photo.py for optimal ResNet-50).

CycleGAN Method

The CycleGAN Method folder contains configuration files and training scripts for CycleGAN to produce stylized images from the original dataset. The script guardar\_imagenes\_estilizadas.py implements the CycleGAN model to generate additional stylized images for training.

To use CycleGAN configs, ensure the MMGeneration environment is installed as per the documentation: https://mmgeneration.readthedocs.io/en/latest/.

Testing

For evaluation on target domains:

jupyter notebook scripts/For\_testing.ipynb



Inference

For single image inference:

jupyter notebook examples/single\_image\_inference.ipynb



Additional Scripts

The scripts folder contains:



A script to calculate the Perceptual Consistency (PC) index.

A script to measure latency, FPS, and maximum memory usage of the models.

A script to separate images from the full sequence, originally captured at 0.1-second intervals, into subsets with temporal thresholds of 1 and 3 seconds.



Image Sequences

The secuencia de imágenes folder contains the complete image sequences with the degraded visual conditions considered in this study. Images with consecutive names are captured at a temporal distance of 0.1 seconds.

Results and Reproduction

Raw experimental results, trained models, and figures are available on FigShare: https://doi.org/10.6084/m9.figshare.29897300.

For reproduction, follow docs/reproduction\_guide.md.

Contributing

See CONTRIBUTING.md for guidelines.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Citation

If you use this code or dataset in your research, please cite the paper:

@article{urrea2025intelligent,

  title={Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation},

  author={Urrea, Claudio and V{\\\\'e}lez, Maximiliano},

  journal={Systems},

  volume={13},

  year={2025},

  publisher={MDPI},

  doi={10.3390/xxxxx}

}



Acknowledgments



Based on MMSegmentation.

Dataset derived from AutoMine with custom annotations.



For questions, contact claudio.urrea@usach.cl.

