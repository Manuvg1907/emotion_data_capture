from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        pass

    def detect_emotion(self, frame):
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        faces_emotions = []
        # `results` can be a dict for one face or list for multiple faces
        if isinstance(results, dict):
            # single face
            x, y, w, h = results['region']['x'], results['region']['y'], results['region']['w'], results['region']['h']
            dominant_emotion = results['dominant_emotion']
            faces_emotions.append((x, y, w, h, dominant_emotion))
        else:
            # multiple faces
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                dominant_emotion = res['dominant_emotion']
                faces_emotions.append((x, y, w, h, dominant_emotion))

        return faces_emotions

