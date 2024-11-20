import cv2
import numpy as np

def analisar_retina(imagem_caminho):
    img_original = cv2.imread(imagem_caminho)
    if img_original is None:
        raise FileNotFoundError("Imagem inválida ou corrompida!")

    valida = verificar_imagem_retina(img_original)
    if not valida:
        raise ValueError("A imagem fornecida não parece ser de retina. Verifique o arquivo selecionado.")

    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.GaussianBlur(cv2.equalizeHist(img), (5, 5), 0)
    
    return img_original, img, img_suave

def verificar_imagem_retina(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(img_gray, 50, 150)
    proporcao_red = np.sum(img[:, :, 2] > 100) / (img.shape[0] * img.shape[1])

    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas_circulares = sum(
        1 for contorno in contornos if cv2.arcLength(contorno, True) > 0
    )

    return areas_circulares >= 1 and proporcao_red > 0.01
