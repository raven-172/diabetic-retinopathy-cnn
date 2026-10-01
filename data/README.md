# Data preparation in Google Colab

The project uses [EyePACKS Diabetic-Retinopathy (pre-processed) V1](https://www.kaggle.com/datasets/rohithgowdax/processed-dr). The Kaggle download contains an `Images` folder and `labels.csv`:

```text
dataset/
├── Images/
│   ├── 10_left.jpeg
│   ├── 10_right.jpeg
│   └── ...
└── labels.csv
```

`labels.csv` contains an `image` column with each filename stem and a `level` column with a value from 0 to 4.

## 1. Divide images into label folders

The workflow reads each CSV row, locates `Images/<image>.jpeg`, creates one directory for every severity level, and copies the image into its matching directory. The public notebook now includes this organization step explicitly:

```text
datasets_split/
├── class_0/    # 25,802 documented images
├── class_1/    #  2,438 documented images
├── class_2/    #  5,288 documented images
├── class_3/    #    872 documented images
└── class_4/    #    708 documented images
```

`datasets_split` therefore means that the source images have been separated by label. It is not the later machine-learning train/validation/test split.

## 2. Balance the five classes

Class 0 contains 25,802 images while class 4 contains only 708. The notebook addresses this imbalance with random undersampling using `TARGET_PER_CLASS = 700` and `SEED = 42`.

For each class folder, it lists the JPEG files, samples 700 without replacement, and copies them into a second directory:

```text
datasets_undersample/
├── class_0/    # 700 images
├── class_1/    # 700 images
├── class_2/    # 700 images
├── class_3/    # 700 images
└── class_4/    # 700 images
```

This produces a balanced dataset of **3,500 images**.

## 3. Load and preprocess images

The notebook loads the balanced JPEG files with OpenCV, converts BGR to RGB, resizes each image to 180×180, casts it to `float32`, and normalizes pixels by dividing by 255. Folder names are encoded as integer labels 0–4.

## 4. Create model subsets

A stratified two-stage split produces:

| Subset | Images | Share | Images per class |
|---|---:|---:|---:|
| Training | 2,450 | 70% | 490 |
| Validation | 525 | 15% | 105 |
| Test | 525 | 15% | 105 |

The public repository intentionally keeps this documentation only. Raw images, `labels.csv`, `datasets_split`, `datasets_undersample`, and notebook image outputs are excluded because the source ultimately derives from the [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), whose data is subject to competition rules.
