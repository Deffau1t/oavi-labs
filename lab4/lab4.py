import cv2
import numpy as np

# Загрузка изображения PNG (ваше исходное цветное)
image = cv2.imread('image.png')
me = cv2.imread('klinika.png')
# Приводим в полутоновое изображение
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png', gray) # сохраняем полутоновое изображение


gray_me = cv2.cvtColor(me, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_klinika.png', gray_me)

# Оператор Прюитта
kernelx = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]], dtype=int)

kernely = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]], dtype=int)

# Градиенты по X и Y
Gx = cv2.filter2D(gray, cv2.CV_32F, kernelx)
Gy = cv2.filter2D(gray, cv2.CV_32F, kernely)


Gx_ = cv2.filter2D(gray_me, cv2.CV_32F, kernelx)
Gy_ = cv2.filter2D(gray_me, cv2.CV_32F, kernely)

# Итоговый градиент G
G = np.sqrt(Gx ** 2 + Gy ** 2)


G_ = np.sqrt(Gx_ ** 2 + Gy_ ** 2)

# Нормализуем Gx, Gy, и G в диапазон [0,255]
Gx_norm = cv2.normalize(Gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Gy_norm = cv2.normalize(Gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
G_norm = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


Gx_norm_me = cv2.normalize(Gx_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Gy_norm_me = cv2.normalize(Gy_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
G_norm_me = cv2.normalize(G_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Сохраняем нормализованные фото
cv2.imwrite('Gradient_X.png', Gx_norm)
cv2.imwrite('Gradient_Y.png', Gy_norm)
cv2.imwrite('Gradient_Result.png', G_norm)


cv2.imwrite('Gradient_X_klinika.png', Gx_norm_me)
cv2.imwrite('Gradient_Y_klinika.png', Gy_norm_me)
cv2.imwrite('Gradient_Result_klinika.png', G_norm_me)


# Подбираем порог бинаризации (примерное значение, подбирается опытным путем)
# Например 100 (можешь менять, подбирая самостоятельно!)
_, binary_G = cv2.threshold(G_norm, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('Gradient_Binary.png', binary_G)


_, binary_G_me = cv2.threshold(G_norm_me, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('Gradient_Binary_klinika.png', binary_G_me)