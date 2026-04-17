# ICF-Seg: Iterative Correction Learning with Foundation Model Guidance for Boundary-Aware Semi-Supervised Medical Image Segmentation

## Supporting Datasets

为了验证我们 ICF-Seg 在医学图像分割领域的性能和通用性，我们在4个具有挑战性的公共数据集上进行了实验：
[ISIC-2017](https://challenge.isic-archive.com/landing/2017/)、
[ISIC-2018](https://challenge.isic-archive.com/landing/2018/)、
[Kvasir-SEG](https://datasets.simula.no/kvasir/) 和
[CVC-ClinicDB](https://paperswithcode.com/dataset/cvc-clinicdb)。

## Data Preparation

The dataset should be organised as follows, taking ISIC-2018 as an example:

- `ISIC-2018/train/images/`
- `ISIC-2018/train/masks/`
- `ISIC-2018/val/images/`
- `ISIC-2018/val/masks/`
- `ISIC-2018/train.txt`
- `ISIC-2018/val.txt`

Each `images` folder contains the original images (e.g., `ISIC_0000000.jpg`), and each `masks` folder contains the corresponding ground truth masks with the same filename.

## Training

Download the SAM-Med2D model and move the model to the `your_root/weights/sammed2d` directory in your project.

Then run:
python train.py

## Evaluation
To evaluate the model and generate the prediction results, run:
python test.py

## License 📜
The source code is free for research and education use only. Any commercial use should get formal permission first.
