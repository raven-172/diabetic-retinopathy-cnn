# Five-Class Diabetic Retinopathy Classification

A Google Colab computer-vision study for grading diabetic retinopathy from retinal fundus images. Starting from a label table with **35,108 records**, the project organizes images into five severity folders, corrects severe class imbalance through deterministic undersampling, compares CNN training strategies, and evaluates the recorded runs on the same balanced holdout set with TensorFlow/Keras.

## Recorded Colab experiment results

All stored runs used a test set of **525 images**, with 105 examples per class.

| Experiment | Training setup | Best validation accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline CNN | 10 epochs; Dense(128); Dropout(0.4) | 33.33% | 26.29% | 15.36% | 26.29% | 18.36% |
| Tuned CNN | Dense(125); Dropout(0.3); early stopping | **39.05%** | **38.48%** | **42.38%** | **38.48%** | **33.94%** |
| Augmentation fine-tuning | Continued the tuned CNN with rotation, zoom, shifts, brightness changes and horizontal flips | 21.71% | 21.52% | 8.82% | 21.52% | 11.80% |

The tuned configuration improved test accuracy by **12.19 percentage points** and macro F1 by **15.58 points** over the baseline. Its matching architecture checkpoint is saved as [`models/CNNv2.keras`](models/CNNv2.keras). These values are transcribed from stored Colab notebook outputs and were not rerun in this repository. The augmentation row is a continuation of the tuned model rather than an independently initialized comparison; a later lighter-augmentation run is excluded because its evaluation cell called the wrong model.

## Dataset and label-based organization

The project uses [EyePACKS Diabetic-Retinopathy (pre-processed) V1](https://www.kaggle.com/datasets/rohithgowdax/processed-dr), which provides 224×224 JPEG images and a `labels.csv` file. Each CSV row maps an image filename to a severity label from 0 to 4.

| Label | Severity | Images before balancing | Images after balancing |
|---:|---|---:|---:|
| 0 | No DR | 25,802 | 700 |
| 1 | Mild | 2,438 | 700 |
| 2 | Moderate | 5,288 | 700 |
| 3 | Severe | 872 | 700 |
| 4 | Proliferative DR | 708 | 700 |

In Colab, the notebook reads `labels.csv`, creates `datasets_split/class_0` through `class_4`, and copies each image into the folder corresponding to its label. Despite its name, `datasets_split` is the label-organized dataset; the train/validation/test split happens later. The notebook then samples 700 images from every class with seed 42 and writes the balanced 3,500-image set to `datasets_undersample`.

The retinal images and labels are not mirrored in this repository. The processed Kaggle listing is marked Apache 2.0, but it traces back to the [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), whose data is subject to competition rules. [`data/README.md`](data/README.md) documents the full processing flow.

## Processing and modeling workflow

1. Use `labels.csv` to place all 35,108 images into five severity folders.
2. Undersample each folder to 700 images with seed 42, producing 3,500 balanced examples.
3. Read images with OpenCV, convert BGR to RGB, resize to 180×180, cast to `float32`, and scale pixels to `[0, 1]`.
4. Apply a stratified 70/15/15 split: 2,450 training, 525 validation, and 525 test images.
5. Train three-block CNN variants with 32/64/128 filters, batch normalization, max pooling, global average pooling, dropout, and a five-class softmax output.
6. Compare accuracy, macro precision, macro recall, macro F1, per-class scores, and confusion matrices.

## Colab notebook and checkpoint

[`notebooks/diabetic_retinopathy_cnn.ipynb`](notebooks/diabetic_retinopathy_cnn.ipynb) is the project implementation and contains the data organization, class balancing, preprocessing, model training, and evaluation steps. Its outputs were removed before publication to avoid embedding dataset images or machine-specific paths.

[`models/CNNv2.keras`](models/CNNv2.keras) contains the 306,055-parameter checkpoint matching the tuned CNN architecture. `requirements.txt` records the main libraries used in the Colab environment.

## Repository structure

```text
.
├── data/                  # Data-processing documentation; no dataset files
├── models/CNNv2.keras    # Tuned-CNN checkpoint
├── notebooks/            # Self-contained Google Colab notebook
└── requirements.txt      # Experiment dependencies
```

## Limitations

- The recorded experiment split images independently, so left and right eyes from one person may appear in different subsets.
- The model has no external or prospective clinical validation, and class-level performance is uneven, especially for labels 0 and 1.
- The augmentation experiment continued training an existing model, so it should not be interpreted as a controlled architecture comparison.
- The checkpoint is an educational prototype and must not be used for diagnosis or patient-care decisions.
