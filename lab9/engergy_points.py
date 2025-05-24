import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# === Параметры ===
filename = r"music.wav"
top_n_peaks = 10

# === Загрузка аудиофайла ===
rate, data = wav.read(filename)

# Переводим в моно, если стерео
if len(data.shape) > 1:
    data = data[:, 0]

# Нормализация данных
data = data.astype(np.float32)
data = data / np.max(np.abs(data))

# === Параметры спектрограммы ===
window = signal.windows.hann(1024)
frequencies, times, Sxx = signal.spectrogram(
    data, fs=rate, window=window, nperseg=1024, noverlap=512,
    scaling='density', mode='magnitude'
)

# === Вычисление энергии ===
energy_matrix = np.sum(Sxx, axis=0)

# === Поиск пиков энергии ===
peaks, _ = signal.find_peaks(energy_matrix, height=0.01)

# === Извлечение частоты и времени для пиков ===
peak_data = []
for peak in peaks:
    time_stamp = times[peak]
    peak_freq_idx = np.argmax(Sxx[:, peak])
    peak_freq = frequencies[peak_freq_idx]
    peak_energy = energy_matrix[peak]
    peak_data.append((time_stamp, peak_freq, peak_energy))

# === Сортировка по значению энергии ===
peak_data.sort(key=lambda x: x[2], reverse=True)

# === Форматированный вывод ===
report_lines = []
for i, (time_stamp, peak_freq, peak_energy) in enumerate(peak_data[:top_n_peaks]):
    report_lines.append(f"{time_stamp:.3f}s @ {peak_freq:.1f}Hz — energy={peak_energy:.2f}")

# === Вывод на экран ===
print("\n".join(report_lines))
