# OSCAR implementation

This is a modular PyTorch reference implementation of the supplied OSCAR paper's main pipeline: CSV dataset handling, image augmentation, synthetic occlusion generation, DCGAN image synthesis, gated ResNet-18, EMA teacher/student distillation, OCR preprocessing, adaptive fusion, MC Dropout and test-time augmentation.

**Important:** the PDF does not include the claimed 48,200-image dataset, trained weights, exact 91 class names, or all training details. This code does not claim the paper's 94.11% result; it must be trained and evaluated on your own labeled data.

## Setup
Python 3.10/3.11 recommended:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
OCR additionally requires Tesseract installed on your OS and available on PATH.

## CSV
CSV columns: `filepath,label`. Relative paths resolve from the CSV's folder. For `make-csv`, use folders `dataset/<class_name>/<image>`.

## Commands
```bash
python main.py make-csv --input dataset --output data/labels.csv
python main.py occlude --input dataset --output data/occluded --copies 5
python main.py validate-occlusions --input dataset --output data/occluded --copies 5
python main.py train --csv data/labels.csv --output runs/oscar --epochs 30
python main.py evaluate --csv data/test.csv --checkpoint runs/oscar/best.pt
python main.py predict --image sign.jpg --checkpoint runs/oscar/best.pt --mc-samples 10 --tta
python main.py train-dcgan --input dataset --output runs/dcgan --epochs 50
python main.py generate-dcgan --checkpoint runs/dcgan/dcgan.pt --output data/generated --count 100
```

Split original source images into train/validation/test **before** augmentation to avoid leakage. Review GAN images manually. This research prototype must not be the sole safety mechanism in a vehicle.
