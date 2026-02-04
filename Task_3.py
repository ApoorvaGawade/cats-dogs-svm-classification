import os
import cv2
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

TRAIN_PATH = r"C:\Users\Apoorva\Desktop\cats_vs_dogs_svm\train"
IMG_SIZE = 32         
MAX_IMAGES = 2000     

def load_dataset():
    X, y = [], []

    files = os.listdir(TRAIN_PATH)
    random.shuffle(files)

    for file in files[:MAX_IMAGES]:
        path = os.path.join(TRAIN_PATH, file)
        name = file.lower()

        if name.startswith("cat"):
            label = 0
        elif name.startswith("dog"):
            label = 1
        else:
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img.flatten())
        y.append(label)

    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Training SVM...")
svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Cat", "Dog"]))
