# Model checkpoint

`CNNv2.keras` is the checkpoint tied to the best documented notebook run.

- Input: 180×180 RGB image scaled to `[0, 1]`
- Output: probabilities for labels 0–4
- Parameters: 306,055
- Optimizer: Adam, learning rate `5e-4`
- Training: batch size 32, up to 50 epochs, early stopping with patience 8
- Recorded test result: 38.48% accuracy and 33.94% macro F1 on 525 balanced images

The checkpoint is provided for reproducibility of an educational experiment and is not validated for clinical use.
