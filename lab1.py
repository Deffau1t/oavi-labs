import numpy as np
import cv2


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def save_image(image, filename):
    cv2.imwrite(filename, image)

# Разбиение изображения на каналы R, G, B
def split_rgb(image):
    B, G, R = cv2.split(image)  #в градациях серого
    
    R_colored = np.zeros_like(image)
    R_colored[:, :, 2] = R  # Устанавливаем красный канал
    G_colored = np.zeros_like(image)
    G_colored[:, :, 1] = G  # Устанавливаем зелёный канал
    B_colored = np.zeros_like(image)
    B_colored[:, :, 0] = B  # Устанавливаем синий канал
    
    save_image(R_colored, "R_colored.png")  
    save_image(G_colored, "G_colored.png")  
    save_image(B_colored, "B_colored.png")  
    
    return R, G, B


# Преобразование RGB в HSI и сохранение яркостной компоненты
def rgb_to_hsi(image):
    B, G, R = image[:,:,0] / 255.0, image[:,:,1] / 255.0, image[:,:,2] / 255.0
    intensity = (R + G + B) / 3
    save_image((intensity * 255).astype(np.uint8), "brightness_component.png")
    return intensity


# Инверсия яркости изображения
def invert_intensity(image):
    rgb_to_hsi(image)
    inverted_image = np.clip(255 - image, 0, 255).astype(np.uint8)
    save_image(inverted_image, "Inverted_intensity.png")


# Увеличение изображения методом 
def nearest_neighbor_resize(image, M):
    h, w, c = image.shape
    new_h, new_w = int(h * M), int(w * M)
    resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x, y = int(i / M), int(j / M)
            resized[i, j] = image[x, y]
    return(resized)


# Уменьшение изображения методом пропуска строк и столбцов
def downsample(image, N):
    return image[::N, ::N]


# Передискретизация в два прохода: сначала увеличение, затем уменьшение
def resample_two_pass(image, M, N):
    upsampled = nearest_neighbor_resize(image, M)
    downsampled = downsample(upsampled, N)
    return downsampled


# Передискретизация за один проход методом ближайшего соседа
def resample_one_pass(image, K):
    return nearest_neighbor_resize(image, K)


image_path = "adv_time.png"
image = load_image(image_path)

split_rgb(image)
rgb_to_hsi(image)
invert_intensity(image)

save_image(nearest_neighbor_resize(image, 2), "Interpolation.png")
save_image(downsample(image, 3), "Decimation.png")

resized_MN = resample_two_pass(image, 2, 3)
save_image(resized_MN, "Resized_MN.png")

resized_K = resample_one_pass(image, 0.1)
save_image(resized_K, "Resized_K.png")
