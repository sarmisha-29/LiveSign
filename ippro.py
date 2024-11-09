import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load pre-trained CNN + Bi-LSTM model for sign language recognition
try:
    model = tf.keras.models.load_model('D:\\DESKTOP\\project\\IP PROJECT\\cnn_bilstm_sign_language_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Define labels for sign language gestures (Ensure these match your model's classes)
LABELS = ['Hello', 'Thank you', 'Yes', 'No', 'Please', 'Help']  # Adjust based on your model's training

# Function to predict sign language gesture
def predict_gesture(landmarks):
    input_data = np.array([landmarks])
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction[0])

    # Ensure index is valid for LABELS
    if class_index < len(LABELS):
        gesture = LABELS[class_index]
    else:
        gesture = "Unknown Gesture"
    
    print("Predicted Gesture:", gesture)
    return gesture

# Convert gesture to voice
def gesture_to_voice(text):
    print("Recognized Gesture:", text)
    engine.say(text)
    engine.runAndWait()

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect landmarks for prediction
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Check that landmarks are correctly processed
            if len(landmarks) == 63:  # 21 key points with (x, y, z) coordinates
                gesture_text = predict_gesture(landmarks)
                gesture_to_voice(gesture_text)
                
                # Display the recognized gesture on the screen
                cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                print("Unexpected number of landmarks:", len(landmarks))
                
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
engine.stop()
