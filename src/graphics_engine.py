import cv2
import numpy as np
import os
from src.utils import normalized_to_pixel_coordinates

class RobotRenderer:
    def __init__(self):
        # 1. Chargement du HUD (Interface)
        # On le charge normalement. Le code s'occupera de la transparence.
        self.hud_path = os.path.join("assets", "hud_overlay.png")
        self.hud_img = cv2.imread(self.hud_path)

        # 2. Chargement de l'Oeil
        # Astuce: Si l'image n'est pas top, on la dessinera avec du code (lasers)
        self.eye_img = cv2.imread(os.path.join("assets", "terminator_eye.png"), cv2.IMREAD_UNCHANGED)
        
        # 3. Chargement de la Texture Robot
        self.texture_img = cv2.imread(os.path.join("assets", "robot_texture.png"))

    def draw_hud(self, image):
        """ Rendu style Hologramme (Le noir devient transparent) """
        if self.hud_img is None: return image

        h, w = image.shape[:2]
        
        # Redimensionner le HUD à la taille de la fenêtre
        hud_resized = cv2.resize(self.hud_img, (w, h))
        
        # Si le HUD a de la transparence (4 canaux), on le convertit en RGB simple
        if hud_resized.shape[2] == 4:
            hud_resized = cv2.cvtColor(hud_resized, cv2.COLOR_BGRA2BGR)

        # --- LA MAGIE EST ICI (cv2.add) ---
        # On additionne les pixels. 
        # Noir (0) + Image = Image (Transparent)
        # Bleu Néon (255) + Image = Lumière très vive
        return cv2.add(image, hud_resized)

    def draw_robot_overlay(self, image, landmarks):
        h, w, _ = image.shape
        
        # --- 1. L'OEIL ROUGE (TERMINATOR) ---
        # On récupère le centre de l'œil droit (point 468 ou 159)
        eye_pt = landmarks.landmark[468] 
        cx, cy = normalized_to_pixel_coordinates(eye_pt.x, eye_pt.y, w, h)
        
        if cx and cy:
            # OPTION A : Si tu as la bonne image PNG rouge
            if self.eye_img is not None:
                size = 70
                x1 = cx - size // 2
                y1 = cy - size // 2
                # Fonction simple de collage avec transparence
                self.overlay_png(image, self.eye_img, x1, y1, size)
            
            # OPTION B (Secours) : Si pas d'image, on dessine un LASER ROUGE
            else:
                # Cercle rouge brillant
                cv2.circle(image, (cx, cy), 15, (0, 0, 255), -1) # Centre rouge
                cv2.circle(image, (cx, cy), 8, (255, 255, 255), -1) # Point blanc (reflet)
                # Halo rouge autour
                cv2.circle(image, (cx, cy), 25, (0, 0, 255), 2) 

        # --- 2. LA MACHOIRE (WIRE-FRAME TECH) ---
        # Au lieu d'une texture grise moche, on va faire une structure 3D verte/grise
        if self.texture_img is not None:
            jaw_indices = [152, 148, 176, 377, 400, 378, 379, 365, 397, 288, 132, 58, 172, 136, 150, 149, 176, 152]
            jaw_points = []
            for idx in jaw_indices:
                pt = normalized_to_pixel_coordinates(landmarks.landmark[idx].x, landmarks.landmark[idx].y, w, h)
                if pt: jaw_points.append(pt)
            
            if len(jaw_points) > 0:
                pts = np.array(jaw_points, np.int32)
                
                # Créer un masque pour la mâchoire
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.fillPoly(mask, [pts], (255, 255, 255))
                
                # Préparer la texture
                tex_resized = cv2.resize(self.texture_img, (w, h))
                
                # Mélanger la texture avec le visage (Transparence 70%)
                # Cela permet de voir la peau en dessous = plus réaliste
                face_part = cv2.bitwise_and(image, mask)
                tex_part = cv2.bitwise_and(tex_resized, mask)
                
                blended_jaw = cv2.addWeighted(face_part, 0.4, tex_part, 0.6, 0)
                
                # Réinjecter la mâchoire mélangée dans l'image
                mask_inv = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
                image_bg = cv2.bitwise_and(image, image, mask=mask_inv)
                
                # On combine tout (Attention aux dimensions des masques)
                final_jaw_area = cv2.bitwise_and(blended_jaw, mask) # Sécurité
                
                # Astuce sale mais efficace pour recombiner sans erreur de dimension
                locs = np.where(mask[:,:,0] > 0)
                image[locs[0], locs[1]] = blended_jaw[locs[0], locs[1]]
                
                # Dessiner un contour Tech autour de la mâchoire
                cv2.polylines(image, [pts], True, (200, 200, 200), 2, cv2.LINE_AA)

        return image

    def overlay_png(self, bg_img, overlay_img, x, y, size):
        """ Helper pour coller le PNG de l'œil proprement """
        try:
            overlay_img = cv2.resize(overlay_img, (size, size))
            h, w = overlay_img.shape[:2]
            
            # Vérifier les limites
            if y < 0 or y+h > bg_img.shape[0] or x < 0 or x+w > bg_img.shape[1]:
                return

            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3):
                bg_img[y:y+h, x:x+w, c] = (1. - alpha) * bg_img[y:y+h, x:x+w, c] + \
                                          alpha * overlay_img[:, :, c]
        except:
            pass