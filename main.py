import cv2
import time
from src.camera import WebcamStream
from src.face_detector import FaceMeshDetector
from src.graphics_engine import RobotRenderer

def main():
    print("Initialisation de RoboLens...")
    
    # 1. Instanciation des modules
    try:
        cam = WebcamStream(source=0) # Changez 0 par 1 si vous avez plusieurs caméras
    except ValueError as e:
        print(f"Erreur: {e}")
        return

    detector = FaceMeshDetector()
    renderer = RobotRenderer()
    
    prev_time = 0

    print("Appuyez sur 'Echap' pour quitter.")

    # 2. Boucle principale
    while True:
        # A. Lecture
        frame = cam.get_frame()
        if frame is None:
            break
        
        # B. Détection
        landmarks = detector.detect(frame)
        
        # C. Rendu Graphique
        if landmarks:
            # On dessine le robot si un visage est trouvé
            renderer.draw_robot_overlay(frame, landmarks)
        
        # On dessine le HUD (Interface) tout le temps
        renderer.draw_hud(frame)

        # D. Calcul FPS (Images par seconde)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # E. Affichage
        cv2.imshow('RoboLens - Python Computer Vision', frame)

        # Quitter
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Nettoyage
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()