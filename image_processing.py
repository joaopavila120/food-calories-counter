import cv2
import numpy as np

def analisar_retina(imagem_caminho):
    img_original = cv2.imread(imagem_caminho)
    if img_original is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem {imagem_caminho}!")

    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_equalizada = cv2.equalizeHist(img)
    img_suave = cv2.GaussianBlur(img_equalizada, (5, 5), 0)
    
    return img_original, img, img_equalizada, img_suave
