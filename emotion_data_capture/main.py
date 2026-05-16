import cv2
from emotion_detector import EmotionDetector
from utils import save_emotion_image

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    detector = EmotionDetector()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame from webcam")
            break

        faces_emotions = detector.detect_emotion(frame)

        # faces_emotions is a list of tuples: (x, y, w, h, dominant_emotion)
        for (x, y, w, h, emotion) in faces_emotions:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            face_img = frame[y:y+h, x:x+w]
            save_emotion_image(face_img, emotion)

        cv2.imshow("Emotion Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
