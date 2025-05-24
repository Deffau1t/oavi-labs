from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

# Пути
image_path = "lab6/phrase.bmp"
output_dir = "lab6/profiles_output"
symbols_dir = os.path.join(output_dir, "symbol_profiles")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(symbols_dir, exist_ok=True)

# Загрузка и бинаризация
image = Image.open(image_path).convert("L")
binary = np.array(image) < 128  # 1 — чёрный, 0 — белый

# Профили
horizontal_profile = np.sum(binary, axis=1)
vertical_profile = np.sum(binary, axis=0)

# Горизонтальный профиль
plt.figure(figsize=(10, 3))
plt.bar(np.arange(len(horizontal_profile)), horizontal_profile, color="black")
plt.title("Горизонтальный профиль")
plt.xlabel("Y (строки)")
plt.ylabel("Чёрные пиксели")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "horizontal_profile.png"))
plt.close()

# Вертикальный профиль
plt.figure(figsize=(10, 3))
plt.bar(np.arange(len(vertical_profile)), vertical_profile, color="black")
plt.title("Вертикальный профиль")
plt.xlabel("X (столбцы)")
plt.ylabel("Чёрные пиксели")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "vertical_profile.png"))
plt.close()


# Функция поиска сегментов
def find_segments(profile, min_val=1):
    segments = []
    in_segment = False
    for i, val in enumerate(profile):
        if val > min_val and not in_segment:
            start = i
            in_segment = True
        elif val <= min_val and in_segment:
            end = i
            in_segment = False
            if end - start > 1:
                segments.append((start, end))
    if in_segment:
        segments.append((start, len(profile)))
    return segments


# Поиск строк
lines = find_segments(horizontal_profile)

# Обработка символов в каждой строке
draw_image = image.convert("RGB")
draw = ImageDraw.Draw(draw_image)
char_count = 0

for line_index, (top, bottom) in enumerate(lines):
    line_slice = binary[top:bottom, :]
    vertical_profile = np.sum(line_slice, axis=0)
    symbols = find_segments(vertical_profile)

    for symbol_index, (left, right) in enumerate(symbols):
        symbol_region = binary[top:bottom, left:right]
        rows = np.any(symbol_region, axis=1)
        cols = np.any(symbol_region, axis=0)
        if not rows.any() or not cols.any():
            continue  # Пустая область

        dy = np.where(rows)[0]
        dx = np.where(cols)[0]
        refined_top = top + dy[0]
        refined_bottom = top + dy[-1]
        refined_left = left + dx[0]
        refined_right = left + dx[-1]

        # Рисуем прямоугольник
        draw.rectangle(
            [(refined_left, refined_top), (refined_right, refined_bottom)],
            outline="red", width=1
        )

        # Вырезаем символ для анализа
        symbol_img = binary[refined_top:refined_bottom + 1, refined_left:refined_right + 1]

        # Построение и сохранение горизонтального профиля
        profile_y = np.sum(symbol_img, axis=1)
        plt.figure(figsize=(6, 2))
        plt.bar(np.arange(len(profile_y)), profile_y, color="black")
        plt.title(f"H-профиль символа {char_count+1}")
        plt.xlabel("Y")
        plt.ylabel("Чёрные пиксели")
        plt.tight_layout()
        plt.savefig(os.path.join(symbols_dir, f"profile_horizontal_{char_count+1}.png"))
        plt.close()

        # Построение и сохранение вертикального профиля
        profile_x = np.sum(symbol_img, axis=0)
        plt.figure(figsize=(6, 2))
        plt.bar(np.arange(len(profile_x)), profile_x, color="black")
        plt.title(f"V-профиль символа {char_count+1}")
        plt.xlabel("X")
        plt.ylabel("Чёрные пиксели")
        plt.tight_layout()
        plt.savefig(os.path.join(symbols_dir, f"profile_vertical_{char_count+1}.png"))
        plt.close()

        char_count += 1

# Сохранение изображения с обводкой
output_path = os.path.join(output_dir, "segmented_clean.png")
draw_image.save(output_path)
print(f"Готово! Найдено символов: {char_count}")
print(f"Профили сохранены в: {symbols_dir}")