"""Run single-image inference with the exported Keras checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import CLASS_NAMES, IMAGE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/CNNv2.keras"))
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(array, axis=0)


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(load_image(args.image), verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    result = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": float(probabilities[class_id]),
        "probabilities": {
            CLASS_NAMES[index]: float(value)
            for index, value in enumerate(probabilities)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
