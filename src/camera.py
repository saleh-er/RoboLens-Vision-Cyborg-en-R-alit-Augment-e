import cv2

class WebcamStream:
    def __init__(self, source=0):
        # On force l'index en entier (int) pour éviter des bugs
        self.cap = cv2.VideoCapture(int(source))
        if not self.cap.isOpened():
            raise ValueError("Impossible d'ouvrir la caméra")
        
        # Optimisation : fixer une résolution standard
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):
        success, frame = self.cap.read()
        if success:
            # Effet miroir pour que ce soit plus naturel
            return cv2.flip(frame, 1)
        return None

    def release(self):
        self.cap.release()