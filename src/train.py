"""Train and evaluate the diabetic-retinopathy CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model import CLASS_NAMES, IMAGE_SIZE, build_model

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def collect_samples(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    paths: list[str] = []
    labels: list[int] = []
    for class_id in CLASS_NAMES:
        class_dir = data_dir / f"class_{class_id}"
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        class_files = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if not class_files:
            raise ValueError(f"No images found in {class_dir}")
        paths.extend(str(path) for path in class_files)
        labels.extend([class_id] * len(class_files))
    return np.asarray(paths), np.asarray(labels, dtype=np.int64)


def split_samples(
    paths: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_paths, other_paths, train_labels, other_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        other_paths,
        other_labels,
        test_size=0.50,
        random_state=seed,
        stratify=other_labels,
    )
    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return (
        dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = collect_samples(args.data_dir)
    train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels = (
        split_samples(paths, labels, args.seed)
    )
    train_data = make_dataset(
        train_paths, train_labels, args.batch_size, shuffle=True, seed=args.seed
    )
    valid_data = make_dataset(
        valid_paths, valid_labels, args.batch_size, shuffle=False, seed=args.seed
    )
    test_data = make_dataset(
        test_paths, test_labels, args.batch_size, shuffle=False, seed=args.seed
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    )
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )

    probabilities = model.predict(test_data, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        test_labels,
        predictions,
        labels=list(CLASS_NAMES),
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=list(CLASS_NAMES))

    model.save(args.output_dir / "CNNv2.keras")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_history(history, args.output_dir / "training_history.png")
    display = ConfusionMatrixDisplay(matrix, display_labels=list(CLASS_NAMES))
    display.plot(cmap="Blues", xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    summary = {
        "train_images": len(train_paths),
        "validation_images": len(valid_paths),
        "test_images": len(test_paths),
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
