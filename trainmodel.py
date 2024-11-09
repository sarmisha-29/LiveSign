import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*SymbolDatabase.GetPrototype() is deprecated.*"
)
# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract hand landmarks using MediaPipe
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks)
    return None

# Load and preprocess the dataset
def load_dataset(data_folder):
    X, y = [], []
    for label_folder in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label_folder)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            landmarks = extract_landmarks(image_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_folder)
    return np.array(X), np.array(y)

# Path to your dataset (update this with your own path)
data_folder = 'D:\\DESKTOP\\project\\IP PROJECT\\asl-alphabet-test'
X, y = load_dataset(data_folder)

# Encode the labels into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model (Dense layers only for simplicity)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

# Save the trained model
model.save('cnn_bilstm_sign_language_model.h5')

print("Model training complete and saved as cnn_bilstm_sign_language_model.h5")
