from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from helpers import *

def spectrogram_plot(samples, sample_rate,t = 11000):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling = 'spectrum', window = ('hann'))
    spec = np.log10(my_spectrogram)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())


    plt.ylim(top=t)
    plt.yticks(np.arange(min(frequencies), max(frequencies), 500))
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    return my_spectrogram, frequencies

if __name__ == '__main__':
   change_sample_rate("voice_a.wav")
   sample_rate_a , samples_a = wavfile.read("results/wavs/voice_a.wav")

   dpi = 500
   spectogram_a, frequencies_a = spectrogram_plot(samples_a, sample_rate_a, 11000)
   plt.axhline(y = 344,  color = 'r', linestyle = '-', lw= 0.5, label = "Форманты")
   plt.axhline(y = 602,  color = 'r', linestyle = '-', lw= 0.5)
   plt.axhline(y = 86,  color = 'r', linestyle = '-', lw= 0.5)
   plt.axhline(y = 1119,  color = 'r', linestyle = '-', lw= 0.5)
   plt.legend()
   plt.savefig('results/spectrogram_a.png', dpi = dpi)
   plt.clf()

   change_sample_rate("voice_i.wav")
   sample_rate_i , samples_i = wavfile.read("results/wavs/voice_i.wav")
   spectogram_i, frequencies_i = spectrogram_plot(samples_i, sample_rate_i, 11000)
   plt.axhline(y = 344,  color = 'r', linestyle = '-', lw= 0.5, label = "Форманты")
   plt.axhline(y = 86,  color = 'r', linestyle = '-', lw= 0.5)
   plt.axhline(y = 602,  color = 'r', linestyle = '-', lw= 0.5)
   plt.axhline(y = 1119,  color = 'r', linestyle = '-', lw= 0.5)
   plt.legend()
   plt.savefig('results/spectrogram_i.png', dpi = dpi)
   plt.clf()

   change_sample_rate("voice_gav.wav")
   sample_rate_gav , samples_gav = wavfile.read("results/wavs/voice_gav.wav")
   spectogram_gav, frequencies_gav = spectrogram_plot(samples_gav, sample_rate_gav, 11000)
   plt.savefig('results/spectrogram_gav.png', dpi = dpi)
   plt.clf()

   spec_a = integral_image(spectogram_a)

   formants_a = list(find_all_formants(frequencies_a, spec_a, 3))
   formants_a.sort()

   print("Минимальная частота для звука А: " + str(formants_a[0]))
   print("Максимальная частота для звука А: " + str(formants_a[-1]))

   print("Тембрально окрашенный тон для звука А: " + str(formants_a[0]))

   power_a = power(frequencies_a, spec_a, 3, formants_a)
   print(sorted(power_a.items(), key = lambda item: item[1], reverse=True))
   print("Четыре самые сильные форманты: " + str(sorted(power_a, key=lambda i: power_a[i])[-4:]))

   spec_i = integral_image(spectogram_i)

   formants_i = list(find_all_formants(frequencies_i, spec_i, 3))
   formants_i.sort()

   print("\n\nМинимальная частота для звука И: " + str(formants_i[0]))
   print("Максимальная частота для звука И: " + str(formants_i[-1]))

   print("Тембрально окрашенный тон для звука И: " + str(formants_i[0]))

   power_i = power(frequencies_i, spec_i, 3, formants_i)
   print(sorted(power_i.items(), key = lambda item: item[1], reverse=True))
   print("Четыре самые сильные форманты: " + str(sorted(power_i, key=lambda i: power_i[i])[-4:]))

   spec_gav = integral_image(spectogram_gav)

   formants_gav = list(find_all_formants(frequencies_gav, spec_gav, 5))
   formants_gav.sort()

   print("\n\nМинимальная частота для звука ГАВ: " + str(formants_gav[0]))
   print("Максимальная частота для звука ГАВ: " + str(formants_gav[-2]))

   print("Тембрально окрашенный тон для звука ГАВ: " + str(formants_gav[0]))

  # power_gav = power(frequencies_gav, spec_gav, 5, formants_gav)
   #power_gav.pop(8268)
   #print(sorted(power_gav.items(), key = lambda item: item[1], reverse=True))
   #print("Четыре самые сильные форманты: " + str(sorted(power_gav, key=lambda i: power_gav[i])[-4:]))