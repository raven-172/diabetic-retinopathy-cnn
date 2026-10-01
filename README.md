# Diabetic Retinopathy Severity Classification with CNN

An educational computer-vision project that classifies retinal fundus images into five diabetic-retinopathy (DR) severity levels. The workflow covers data preparation, class balancing, image preprocessing, CNN training, evaluation, and model export with TensorFlow/Keras.

## Verified experiment result

The best documented run used a balanced test set of **525 images** (105 per class):

| Metric | Result |
|---|---:|
| Test accuracy | **38.48%** |
| Macro precision | **42.38%** |
| Macro recall | **38.48%** |
| Macro F1-score | **33.94%** |
| Best validation accuracy | **39.05%** at epoch 20 |

The corresponding checkpoint is [`models/CNNv2.keras`](models/CNNv2.keras). It contains a **306,055-parameter** CNN and was trained with early stopping. These figures are preserved from the original notebook run; they are experimental results, not clinical-performance claims.

## Dataset

The project uses [EyePACKS Diabetic-Retinopathy (pre-processed) V1](https://www.kaggle.com/datasets/rohithgowdax/processed-dr), which contains **35,108 preprocessed JPEG fundus images** and a `labels.csv` file. The Kaggle version stores images at 224×224; this project resizes them to 180×180 during training.

| Label | Severity | Original count | Balanced count |
|---:|---|---:|---:|
| 0 | No DR | 25,802 | 700 |
| 1 | Mild | 2,438 | 700 |
| 2 | Moderate | 5,288 | 700 |
| 3 | Severe | 872 | 700 |
| 4 | Proliferative DR | 708 | 700 |

The retinal images are not mirrored in this repository. The processed Kaggle listing is marked Apache 2.0, but it traces back to the [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), whose data is subject to competition rules. Download the data directly from Kaggle and follow its current terms. See [`data/README.md`](data/README.md) for the expected local layout.

## Pipeline

1. Match each image filename to its numeric label in `labels.csv`.
2. Undersample every class to 700 images with seed 42, producing 3,500 balanced examples.
3. Apply a stratified 70/15/15 train/validation/test split: 2,450 / 525 / 525 images.
4. Resize images to 180×180 RGB and scale pixel values to `[0, 1]`.
5. Train a three-block CNN with 32/64/128 filters, batch normalization, max pooling, global average pooling, dropout, and a five-class softmax output.
6. Optimize with Adam (`learning_rate=5e-4`) and early stopping on validation accuracy.

## Run locally

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract the dataset into `data/raw`:

```bash
kaggle datasets download -d rohithgowdax/processed-dr -p data/raw --unzip
```

Prepare the balanced dataset and train the model:

```bash
python src/prepare_data.py --data-root data/raw --output-dir data/processed
python src/train.py --data-dir data/processed --output-dir artifacts
```

Run inference with the included checkpoint:

```bash
python src/predict.py --image /path/to/fundus.jpeg
```

The cleaned, output-free experiment notebook is available at [`notebooks/diabetic_retinopathy_cnn.ipynb`](notebooks/diabetic_retinopathy_cnn.ipynb).

## Repository structure

```text
.
├── data/                  # Dataset instructions; raw images are excluded
├── models/CNNv2.keras    # Best documented checkpoint
├── notebooks/            # Clean experiment notebook
├── src/
│   ├── model.py          # CNN definition and class names
│   ├── prepare_data.py   # Label matching and deterministic balancing
│   ├── train.py          # Training, evaluation, and artifact export
│   └── predict.py        # Single-image inference CLI
└── requirements.txt
```

## Limitations

- The recorded experiment split images independently, so left and right eyes from one person may appear in different splits.
- The model has no external or prospective clinical validation, and class-level performance is uneven, especially for labels 0 and 1.
- The checkpoint is an educational prototype and must not be used for diagnosis or patient-care decisions.
