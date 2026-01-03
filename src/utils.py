import numpy as np
import math
import cv2

def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int):
    """
    Convertit les coordonnées normalisées (0.0 - 1.0) de MediaPipe 
    en coordonnées pixels (ex: 1280x720) pour OpenCV.
    """
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def get_landmarks_indices(face_landmarks, indices, width, height):
    """
    Récupère une liste de points (x, y) en pixels à partir d'une liste d'indices.
    Utile pour dessiner des formes complexes (mâchoire, yeux).
    """
    points = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        px = normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        points.append(px)
    return points

def rotate_image(image, angle, center=None, scale=1.0):
    """
    Fonction utilitaire pour faire tourner une image (utile si plus tard
    vous voulez coller une texture .png qui suit l'inclinaison de la tête).
    """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # Calcul de la matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated