# Dataset setup

Download [EyePACKS Diabetic-Retinopathy (pre-processed) V1](https://www.kaggle.com/datasets/rohithgowdax/processed-dr) directly from Kaggle. After extraction, the local files should follow this layout:

```text
data/raw/
├── Images/
│   ├── 10_left.jpeg
│   ├── 10_right.jpeg
│   └── ...
└── labels.csv
```

`labels.csv` contains an `image` column with the filename stem and a `level` column with a value from 0 to 4.

Run the preparation script from the repository root:

```bash
python src/prepare_data.py --data-root data/raw --output-dir data/processed
```

This creates 700 deterministically sampled images per class under `data/processed/class_0` through `class_4`. Raw images, labels, generated class folders, and image previews are excluded from Git because the source ultimately derives from the [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), whose data is subject to competition rules.
