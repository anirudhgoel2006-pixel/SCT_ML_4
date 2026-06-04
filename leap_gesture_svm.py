import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "leapGestRecog"

X = []
y = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):
        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)

            if os.path.isdir(gesture_path):
                for img_name in os.listdir(gesture_path)[:100]:
                    img_path = os.path.join(gesture_path, img_name)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (64, 64))

                    X.append(img.flatten())
                    y.append(gesture)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(svm, "svm_model.pkl")
print("Model saved as svm_model.pkl")
