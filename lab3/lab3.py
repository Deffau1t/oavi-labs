import cv2
import numpy as np


def get_cross_neighbors(img, x, y):
    """
    Функция возвращает отсортированный список значений пикселей в маске "косой крест".
    В маске "косой крест" включены пиксели, расположенные по диагоналям вокруг текущего пикселя (x, y).
    """
    neighbors = []
    height, width = img.shape 
    for dx, dy in [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]:  
        nx, ny = x + dx, y + dy  
        if 0 <= nx < width and 0 <= ny < height:  
            neighbors.append(img[ny, nx])  
    return sorted(neighbors)


def rank_filter(image, rank=7):
    """
    Фильтрует изображение, применяя ранговый фильтр с маской "косой крест".
    Для каждого пикселя выбирается значение, соответствующее заданному рангу в отсортированном списке соседей.
    """
    height, width = image.shape  
    filtered_image = np.zeros((height, width), dtype=np.uint8)  

    for y in range(height):
        for x in range(width):
            neighbors = get_cross_neighbors(image, x, y)  
            
            filtered_image[y, x] = neighbors[min(rank - 1, len(neighbors) - 1)]  
    
    return filtered_image


def difference_image(img1, img2):
    """
    Вычисляет разностное изображение как модуль разности между исходным и фильтрованным изображением.
    """
    return cv2.absdiff(img1, img2)


input_image = cv2.imread("image_gray.png", cv2.IMREAD_GRAYSCALE)  # Загрузка изображения в формате grayscale (оттенки серого)
filtered_image = rank_filter(input_image)  # Применяем ранговый фильтр
diff_image = difference_image(input_image, filtered_image)  # Вычисляем разностное изображение

# Сохранение результатов
cv2.imwrite("filtered_image.png", filtered_image)  # Сохраняем отфильтрованное изображение
cv2.imwrite("difference_image.png", diff_image)  # Сохраняем разностное изображение


# Загрузка изображения в оттенках серого
input_image = cv2.imread("RR.png", cv2.IMREAD_GRAYSCALE)  # Загрузка изображения в формате grayscale (оттенки серого)
filtered_image = rank_filter(input_image)  # Применяем ранговый фильтр
diff_image = difference_image(input_image, filtered_image)  # Вычисляем разностное изображение

# Сохранение результатов
cv2.imwrite("filtered_RR.png", filtered_image)  # Сохраняем отфильтрованное изображение
cv2.imwrite("difference_RR.png", diff_image)  # Сохраняем разностное изображение