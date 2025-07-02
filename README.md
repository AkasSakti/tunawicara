# Bisindo Gesture Recognition

## Struktur Folder
- app.py
- colab.py
- Model.py
- requirements.txt / environment.yml
- dataset/ (tidak diupload ke GitHub)
- trained_model.h5 (opsional)
- README.md

## Download Dataset
1. Download dataset Bisindo dari [Kaggle](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo)
2. Ekstrak ke folder `dataset/` di root project.

## Instalasi
### Dengan pip
```bash
pip install -r requirements.txt
```
### Dengan conda
```bash
conda env create -f environment.yml
conda activate tunawicara
```

## Menjalankan
```bash
streamlit run app.py
```
```

---

## 3. **Penjelasan Penggunaan .yml (Conda Environment)**

- **Kapan pakai .yml?**  
  Jika Anda ingin environment yang lebih terkontrol (misal di Windows, atau butuh library C++ seperti dlib), gunakan `environment.yml` dengan Conda.
- **Cara pakai:**
  1. Install [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  2. Jalankan:
     ```bash
     conda env create -f environment.yml
     conda activate tunawicara
     ```
  3. Jalankan Streamlit:
     ```bash
     streamlit run app.py
     ```

---

## 4. **Contoh Minimal app.py**

```python
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

st.title("Bisindo Gesture Recognition for Tunawicara")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

model = load_model('trained_model.h5')
gesture_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))
    img_norm = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_norm, axis=0)

cap = None
if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            input_img = preprocess_frame(frame)
            pred = model.predict(input_img)
            gesture = gesture_labels[np.argmax(pred)]
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
```

---

## **Kesimpulan**

- **Struktur folder tetap seperti di atas.**
- **Pilih requirements.txt (pip) atau environment.yml (conda) sesuai kebutuhan.**
- **Dataset besar tidak diupload ke GitHub, cukup instruksi download di README.**
- **.yml diperlukan jika ingin environment yang lebih stabil/portable, terutama untuk dependency berat seperti dlib.**

---

**Jika ingin contoh file environment.yml, requirements.txt, atau README.md siap pakai, silakan minta!**
