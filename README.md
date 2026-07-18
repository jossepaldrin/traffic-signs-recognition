# Traffic Signs Recognition using CNN and Keras

Trains a convolutional neural network **from scratch** (no transfer learning)
to classify German traffic signs using the GTSRB dataset (43 classes).
Training runs on Google Colab's free GPU; a small local script uses the
trained model for inference on new images.

## Project structure
```
traffic-signs-recognition/
├── notebooks/
│   └── train_traffic_signs.ipynb   # full training pipeline (run in Colab)
├── src/
│   └── predict.py                  # local inference on a single image
├── models/
│   └── traffic_sign_model.keras    # trained model (added after training)
├── requirements.txt                 # for local inference only
└── README.md
```

## Dataset
[GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
on Kaggle — 43 classes, ~50,000 images total. The notebook downloads it
directly via the Kaggle API; the dataset itself is not stored in this repo.

## Training (Google Colab)
1. Open `notebooks/train_traffic_signs.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> select **GPU**.
3. Run all cells. You'll be prompted to upload your `kaggle.json` API token
   (Kaggle account -> Settings -> Create New API Token) to download the dataset.
4. The notebook trains a CNN from scratch, evaluates it on the official
   GTSRB test set, and shows accuracy/loss curves plus a confusion matrix.
5. Download the resulting `traffic_sign_model.keras` file from Colab's file
   browser and place it in a local `models/` folder in this repo.

## Model architecture
A compact CNN: two Conv2D+BatchNorm blocks (32 then 64 filters) each
followed by MaxPooling and Dropout, then a Dense(256) layer before the
43-way softmax output. Built from scratch to demonstrate designing and
training a CNN end-to-end, rather than fine-tuning a pretrained network.

## Local inference
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python src/predict.py path\to\some_sign.jpg
```
Prints the predicted class name and confidence.

## Possible extensions
- Data augmentation (rotation, brightness, zoom) to improve robustness.
- Try transfer learning (e.g. MobileNet) and compare accuracy vs the
  from-scratch model.
- Wrap `predict.py` in a small web app for interactive testing.
