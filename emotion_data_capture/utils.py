import os
import cv2
import time

def save_emotion_image(face_img, emotion_label):
    base_dir = "dataset"
    emotion_dir = os.path.join(base_dir, emotion_label)

    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

    filename = f"{int(time.time() * 1000)}.jpg"
    filepath = os.path.join(emotion_dir, filename)
    cv2.imwrite(filepath, face_img)
