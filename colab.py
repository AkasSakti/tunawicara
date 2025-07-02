import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from Model import model
from keras import callbacks
import streamlit as st
import mediapipe as mp

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Load CNN model Bisindo
model = load_model('trained_model.h5')

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.LANCZOS)
    return np.array(img)

def preprocess_frame(frame):
    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Denoising
    img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
    # Resize to model input
    img_resized = cv2.resize(img_denoised, (32, 32))
    # Normalize
    img_norm = img_resized.astype('float32') / 255.0
    # Add batch and channel dims
    return np.expand_dims(img_norm, axis=0)

# function to get the images and label data
def getImagesAndLabels(path):
    
    path = 'dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        
        #if there is an error saving any jpegs
        try:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        except:
            continue    
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces now.")
faces,ids = getImagesAndLabels(path)

K.clear_session()
n_faces = len(set(ids))
model = model((32,32,1),n_faces)
# faces = np.asarray(faces)
faces = np.array([downsample_image(ab) for ab in faces])
ids = np.asarray(ids)
faces = faces[:,:,:,np.newaxis]
print("Shape of Data: " + str(faces.shape))
print("Number of unique faces : " + str(n_faces))


ids = to_categorical(ids)

faces = faces.astype('float32')
faces /= 255.
x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.2, random_state = 0)
checkpoint = callbacks.ModelCheckpoint('./trained_model.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
                                    
model.fit(x_train, y_train,
             batch_size=32,
             epochs=10,
             validation_data=(x_test, y_test),
             shuffle=True,callbacks=[checkpoint])
             

# Print the numer of faces trained and end program
print("\n [INFO] " + str(n_faces) + " faces trained. Exiting Program")

st.title("Bisindo Gesture Recognition for Tunawicara")
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

gesture_labels = ['A', 'B', 'C', ...]  # Isi sesuai label Bisindo

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # MediaPipe Hand Detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Crop/ROI bisa diambil dari bounding box hand_landmarks
        # Untuk contoh, kita pakai seluruh frame
        input_img = preprocess_frame(frame)
        pred = model.predict(input_img)
        gesture = gesture_labels[np.argmax(pred)]
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        gesture = "No hand detected"

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if cap:
    cap.release()