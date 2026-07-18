"""
Run inference with the trained traffic sign CNN on a single image.

Usage:
    python src/predict.py path/to/image.jpg
"""

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 32
MODEL_PATH = "models/traffic_sign_model.keras"

# GTSRB class ID -> human-readable label
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing (vehicles over 3.5 tons)",
}


def predict(image_path):
    model = load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    label = CLASS_NAMES.get(class_id, f"Class {class_id}")
    print(f"Prediction: {label} (class {class_id}), confidence: {confidence:.2%}")
    return class_id, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
