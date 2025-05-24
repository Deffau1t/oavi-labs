import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from datetime import datetime

# === Параметры ===
radius = 1
n_points = 8 * radius
gamma = 1.5  # Параметр для степенного преобразования
output_image = f"lbp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


# === Функция степенного преобразования яркости ===
def power_law_transform(image, gamma=1.5):
    norm_image = image / 255.0
    transformed = np.power(norm_image, gamma)
    return (transformed * 255).astype(np.uint8)


# === Функция расчёта LBP и гистограммы H(LBP) ===
def compute_lbp(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-7)  # Нормализация
    return lbp, hist


# === Основная функция обработки изображения ===
def process_image(image_path):
    # Загрузка и подготовка изображения
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Преобразование яркости
    enhanced_image = power_law_transform(gray_image, gamma=gamma)

    # Расчёт LBP и гистограммы
    lbp, hist = compute_lbp(enhanced_image, radius, n_points)

    # Визуализация
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(gray_image, cmap='gray')
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(enhanced_image, cmap='gray')
    axs[0, 1].set_title("Степенное преобразование")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(lbp, cmap='gray')
    axs[1, 0].set_title("Матрица LBP")
    axs[1, 0].axis('off')

    axs[1, 1].bar(np.arange(len(hist)), hist, color='gray')
    axs[1, 1].set_title("Гистограмма H(LBP)")

    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()
    print(f"✅ Анализ завершён. Изображение сохранено в файл: {output_image}")


# === Пример запуска ===
process_image("texture.png")