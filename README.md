# Material Recognition

Material recognition with a ResNet-50 classifier trained on the MINC-2500 material dataset. The model predicts one of 23 material categories (for example brick, fabric, glass, wood) and runs live on a webcam feed.

## Features

- Training on the MINC-2500 dataset with ImageNet-pretrained ResNet-50 and a new 23-class head
- AdamW optimizer, cross-entropy loss, 224x224 inputs, per-epoch train/validation metrics
- Real-time webcam prediction with the top material label drawn on the frame
- Press `q` to quit

## Project structure

- `Train.py` - loads the MINC-2500 splits, fine-tunes ResNet-50 and saves `material_classifier.pth`.
- `Test.py` - loads `material_classifier.pth` and predicts the material from the webcam.
- `material_classifier.pth` - trained model weights (state dict) used by `Test.py`.
- `DEMO.mp4` - demo video.
- `dataset/minc-2500/` - expected dataset location (not included in this repo; see Notes).

## Requirements

Python packages imported by the code:

- `torch` (PyTorch)
- `torchvision`
- `opencv-python` (imported as `cv2`)
- `numpy`
- `pillow` (PIL)
- `tqdm` (used by `Train.py`)

Install:

```
pip install torch torchvision opencv-python numpy pillow tqdm
```

## Installation

1. Clone the repository.
2. Install PyTorch and the packages above (see https://pytorch.org for a CUDA build if you have a GPU).
3. For training only, download MINC-2500 and place it as described below.

## Usage

Live material recognition with the included model:

```
python Test.py
```

Expects `material_classifier.pth` in the current directory. Point the webcam at a material; the predicted category is drawn at the top left. Press `q` to exit.

Retrain the model:

```
python Train.py
```

`Train.py` reads `./dataset/minc-2500` (set by `DATASET_PATH`). It expects:

```
dataset/minc-2500/
    categories.txt          one category name per line
    images/<category>/*.jpg images, organized in one folder per category
    labels/train1.txt, train2.txt, validate1.txt, validate2.txt
                            image paths (relative to images/), one per line
```

It trains for 5 epochs with batch size 32 and learning rate 1e-4 and saves `material_classifier.pth`.

## Notes

- The dataset is not in this repository. The source dataset (about 2.3 GB) was excluded, so download MINC-2500 separately and place it at the layout above before running `Train.py`.
- `Test.py` hard-codes the 23 category names; keep them in the same order as `categories.txt` used during training.
- The model is a standard ResNet-50 with the final layer replaced, and `Test.py` runs on CPU if CUDA is not available.
- Training uses 2 of the MINC label splits and randomly samples from them when loading paths; results will vary between runs.
- The demo video is `DEMO.mp4`.
