import cv2
import numpy as np

def analisar_retina(imagem_caminho):
    img_original = cv2.imread(imagem_caminho)
    if img_original is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem {imagem_caminho}!")

    valida, motivo = verificar_imagem_retina(img_original)
    if not valida:
        raise ValueError(f"A imagem fornecida não parece ser de retina. {motivo}")

    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_equalizada = cv2.equalizeHist(img)
    img_suave = cv2.GaussianBlur(img_equalizada, (5, 5), 0)
    detectar_bordas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    destaca_estruturas = cv2.morphologyEx(img_suave, cv2.MORPH_TOPHAT, detectar_bordas)
    bordas = cv2.Canny(destaca_estruturas, 50, 150)
    
    vasos_dilatados = np.sum(bordas) / 255
    if vasos_dilatados > 10000:
        status = "Risco de Diabetes"
    elif vasos_dilatados > 5000:
        status = "Risco de Hipertensão"
    else:
        status = "Olho Saudável"

    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    return img_original_rgb, img, destaca_estruturas, bordas, status

def verificar_imagem_retina(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(img_gray, 50, 150)
    quantidade_bordas = np.sum(bordas > 0)
    proporcao_red = np.sum(img[:, :, 2] > 100) / (img.shape[0] * img.shape[1])
    
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas_circulares = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro ** 2))
        if 0.7 < circularidade < 1.2:
            areas_circulares += 1

    if proporcao_red < 0.01:
        return False, "Proporção de vermelho muito baixa, não parece uma retina."
    if quantidade_bordas < 1000 or quantidade_bordas > 30000:  # Aumentamos o limite superior
        return False, "Quantidade de bordas fora do intervalo típico de uma retina."
    if areas_circulares < 1:
        return False, "Não foram detectadas áreas circulares típicas de uma retina."

    return True, "Imagem validada como retina."
