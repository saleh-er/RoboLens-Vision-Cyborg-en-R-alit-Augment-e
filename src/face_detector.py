import mediapipe as mp
import cv2

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True permet d'avoir les détails des yeux (iris)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, image):
        """
        Prend une image BGR (OpenCV), la convertit en RGB
        et retourne les résultats bruts de MediaPipe.
        """
        # Conversion BGR vers RGB (car MediaPipe déteste le BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.face_mesh.process(image_rgb)
        
        image_rgb.flags.writeable = True
        
        if results.multi_face_landmarks:
            # On retourne uniquement le premier visage détecté
            return results.multi_face_landmarks[0]
        return None