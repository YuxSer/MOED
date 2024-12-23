import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile

class Model:
    def trend(self, trend_type, a, N, delta):
        t = np.arange(0, N) * delta  # Время t = k * delta
        if trend_type == 'linear_up':
            data = a * t  # Восходящий линейный тренд
        elif trend_type == 'linear_down':
            data = -a * t  # Нисходящий линейный тренд
        elif trend_type == 'exp_up':
            data = np.exp(a * t)  # Восходящий экспоненциальный тренд
        elif trend_type == 'exp_down':
            data = np.exp(-a * t)  # Нисходящий экспоненциальный тренд
        else:
            raise ValueError("Неизвестный тип тренда")
        return data

    def shift(self, data, C):
        return data + C  # Сдвиг на константу C

    def mult(self, data, C):
        return C * data  # Умножение на константу C

    def noise(self, N, R, delta):
        noise_data = np.random.rand(N)
        xmin, xmax = np.min(noise_data), np.max(noise_data)
        data = ((noise_data - xmin) / (xmax - xmin) - 0.5) * 2 * R
        return data

    def myNoise(self, N, R, delta):
        a, c, m, seed = 1664525, 1013904223, 2 ** 32, 12345
        noise_data = np.zeros(N)
        for i in range(N):
            seed = (a * seed + c) % m
            noise_data[i] = seed / m
        xmin, xmax = np.min(noise_data), np.max(noise_data)
        normalized_data = ((noise_data - xmin) / (xmax - xmin) - 0.5) * 2 * R
        return normalized_data

    def spikes(self, N, M, R, Rs):
        data = np.zeros(N)
        num_spikes = np.random.randint(int(M * N * 0.001), int(M * N * 0.01))
        spike_positions = np.random.choice(N, num_spikes, replace=False)
        spike_amplitudes = np.random.uniform(R - Rs, R + Rs, num_spikes)
        spike_amplitudes *= np.random.choice([-1, 1], size=num_spikes)
        data[spike_positions] = spike_amplitudes
        return data

    def randVector(self, M, N, R, generator_type='builtin'):
        data = np.zeros((M, N))
        for i in range(M):
            if generator_type == 'builtin':
                data[i] = self.noise(N, R, 1)
            elif generator_type == 'custom':
                data[i] = self.myNoise(N, R, 1)
            else:
                raise ValueError("Неизвестный тип генератора: используйте 'builtin' или 'custom'")
        return data

    def harm(self, N, A0, f0, delta_t):
        harm_data = A0 * np.sin(2 * np.pi * f0 * delta_t * np.arange(N))
        return harm_data

    def polyHarm(self, N, A, f, M, delta_t):
        polyharm_data = np.zeros(N)
        for j in range(M):
            polyharm_data += A[j] * np.sin(2 * np.pi * f[j] * delta_t * np.arange(N))
        return polyharm_data

    def addModel(self, data1, data2,N):
        data3 = []
        for i in range(N):
            data3.append(data1[i]+data2[i])
        return data3

    def multModel(self, data1, data2,N):
        data3 = []
        for i in range(N):
            data3.append(data1[i] * data2[i])
        return data3
    @staticmethod
    def convModel(x, h, N, M):
        y = []
        for k in range(N + M - 1):
            value = 0
            for m in range(M):
                if 0 <= k - m < N:
                    value += x[k - m] * h[m]
            y.append(value)
        return y[M:]



class Analysis:
    def stationarity(self, data, N, M, P):
        # Стационарность
        mean_values_m = np.mean(data, axis=1)  # Средние по строкам (для каждого m)
        stationary = True
        for m in range(M):
            for n in range(m + 1, M):
                delta_mn = abs(mean_values_m[m] - mean_values_m[n])  # Разница средних
                if delta_mn > P:  # Если разница превышает порог, процесс не стационарен
                    stationary = False
                    break
        # Эргодичность
        mean_values_k = np.mean(data, axis=0)  # Средние по столбцам (для каждого k)
        ergodic = True
        for j in range(N):
            for l in range(j + 1, N):
                delta_jl = abs(mean_values_k[j] - mean_values_k[l])  # Разница средних
                if delta_jl > P:  # Если разница превышает порог, процесс не эргодичен
                    ergodic = False
                    break

        return stationary, ergodic

    def hist(self,data,N,M):
        counts, bin_edges = np.histogram(data, bins=M, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts

    def acf(self, data, N, type='AKF'):
        # Вычисляем среднее значение данных
        mean_data = np.mean(data)
        # Инициализация массивов для хранения значений Rxx(L) и Rx(L)
        Rxx = []
        Rx = []
        # Рассчитываем Rxx(L) и Rx(L) для каждого L
        for L in range(N):
            # Автокорреляционная функция Rxx(L)
            numerator_Rxx = sum((data[k] - mean_data) * (data[k + L] - mean_data) for k in range(N - L))
            denominator_Rxx = sum((data[k] - mean_data) ** 2 for k in range(N))
            Rxx_value = numerator_Rxx / denominator_Rxx if denominator_Rxx != 0 else 0
            Rxx.append(Rxx_value)

            # Ковариационная функция Rx(L)
            numerator_Rx = sum((data[k] - mean_data) * (data[k + L] - mean_data) for k in range(N - L))
            Rx_value = numerator_Rx / N
            Rx.append(Rx_value)

        # Выбор типа отображаемой функции
        if type == 'AKF':
            result = Rxx
        if type == 'KF':
            result = Rx

        return result

    def ccf(self, dataX, dataY, N):
        mean_x = np.mean(dataX)
        mean_y = np.mean(dataY)
        R_xy = np.zeros(N)
        for L in range(N):
            sum_xy = 0
            for k in range(N - L):
                sum_xy += (dataX[k] - mean_x) * (dataY[k + L] - mean_y)
            R_xy[L] = sum_xy / N
        return R_xy

    @staticmethod
    def Fourier(data,N):
        fft_result = np.fft.fft(data, N)  # Преобразование Фурье
        Re = np.real(fft_result) / N  # Действительная часть (нормализация)
        Im = np.imag(fft_result) / N  # Мнимая часть (нормализация)
        return Re, Im



    @staticmethod
    def spectrFourier(Re, Im, halfN, delta_t):
        Spectr = np.sqrt(Re ** 2 + Im ** 2)  # Амплитудный спектр
        freq = np.fft.fftfreq(halfN * 2, delta_t)[:halfN]  # Частоты
        return Spectr, freq

    @staticmethod
    def window(data,L):
        windowed_data = data.copy()
        windowed_data[-L:] = 0
        return windowed_data

    @staticmethod
    def statistics(data, N):
        # 1. Минимальное и максимальное значение
        min_val = np.min(data)
        max_val = np.max(data)
        print(f"Минимум: {min_val}, Максимум: {max_val}")
        # 2. Среднее значение (СЗ)
        mean = np.mean(data)
        print(f"Среднее значение (СЗ): {mean}")
        # 3. Дисперсия (D)
        variance = np.var(data)
        print(f"Дисперсия (D): {variance}")
        # 4. Стандартное отклонение (СО)
        std_dev = np.sqrt(variance)
        print(f"Стандартное отклонение (СО): {std_dev}")
        # 5. Асимметрия (A)
        mu3 = np.mean((data - mean) ** 3)
        print(f"Асимметрия (A): {mu3}")
        # 6. Коэффициент асимметрии (КА)
        skewness = mu3 / std_dev**3
        print(f"Коэффициент асимметрии (КА): {skewness}")
        # 7. Эксцесс (Э)
        mu4 = np.mean((data - mean) ** 4)
        print(f"Эксцесс (Э): {mu4}")
        # 8. Куртозис (K)
        kurtosis = mu4 / std_dev**4 - 3
        print(f"Куртозис (K): {kurtosis}")
        # 9. Средний квадрат (СК)
        psi2 = np.mean(data ** 2)
        print(f"Средний квадрат (СК): {psi2}")
        # 10. Среднеквадратическая ошибка (СКО)
        rms_error = np.sqrt(psi2)
        print(f"Среднеквадратическая ошибка (СКО): {rms_error}")

    @staticmethod
    def transferFunction(signal, delta_t, m):
        # Рассчёт спектра Фурье
        N = len(signal)
        Re = np.fft.fft(signal).real
        Im = np.fft.fft(signal).imag
        halfN = N // 2

        # Амплитудный спектр
        Spectr = Analysis.spectrFourier(Re, Im, halfN, delta_t)
        # Частотная характеристика: умножение на (2 * m + 1)
        transfer_func = Spectr * (2 * m + 1)

        # Построение графика частотной характеристики
        freq = np.fft.fftfreq(N, delta_t)[:halfN]

        return freq,transfer_func,halfN

    @staticmethod
    def SNR(signal, noise, N):
        sigma_S = np.std(signal)
        sigma_N = np.std(noise)
        snr_value = 20 * np.log10(sigma_S / sigma_N)
        return snr_value


class Processing:
    @staticmethod
    def antiSpike(data, N, R):
        procData = data.copy()
        for k in range(1, N - 1):
            if abs(procData[k]) > R:
                procData[k] = (procData[k - 1] + procData[k + 1]) / 2
        return procData

    @staticmethod
    def antiTrendLinear(data, N,):
        derivative = np.gradient(data)
        plt.plot(derivative)
        plt.title('Первая производная данных для удаления линейного тренда')
        plt.grid(True)
        plt.show()
        return derivative

    @staticmethod
    def antiTrendNonLinear(data, N, W):
        # Вычисление скользящего среднего
        trend = np.zeros(N)
        for i in range(N):
            if i < W // 2:
                # Если недостаточно данных в начале, берем среднее первых i+1 точек
                trend[i] = np.mean(data[:i + W // 2 + 1])
            elif i > N - W // 2 - 1:
                # Если недостаточно данных в конце, берем среднее последних точек
                trend[i] = np.mean(data[i - W // 2:])
            else:
                # Основная часть данных, берем среднее в окне W
                trend[i] = np.mean(data[i - W // 2:i + W // 2 + 1])

        # Убираем тренд из данных
        detrended_data = data - trend
        # Построение графиков
        plt.figure(figsize=(12, 6))

        # Исходные данные
        plt.subplot(2, 1, 1)
        plt.plot(data, label='Исходные данные')
        plt.plot(trend, label='Вычисленный тренд', linestyle='--')
        plt.legend()
        plt.xlabel('Индекс')
        plt.ylabel('Амплитуда')
        plt.title('Исходные данные и тренд')

        # Данные без тренда
        plt.subplot(2, 1, 2)
        plt.plot(detrended_data, label='Данные без тренда')
        plt.legend()
        plt.xlabel('Индекс')
        plt.ylabel('Амплитуда')
        plt.title('Данные после удаления тренда')

        plt.tight_layout()
        plt.show()

        return detrended_data

    @staticmethod
    def antiShift(data, N):
        """Удаление смещения в данных"""
        mean_value = np.mean(data)  # Находим среднее значение
        procData = data - mean_value  # Вычитаем его из всех данных
        return procData

    def antiNoise(data, M_values, N):

        procData = {}
        std_devs = []

        for M in M_values:
            # Генерация M реализаций случайного шума
            random_vectors = [data(N) for _ in range(M)]

            # Усреднение по M реализациям
            averaged_vector = np.mean(random_vectors, axis=0)
            procData[M] = averaged_vector

            # Вычисление стандартного отклонения
            std_dev = np.std(averaged_vector)
            std_devs.append(std_dev)

            # Построение графика для текущего M
            plt.figure(figsize=(8, 4))
            plt.plot(averaged_vector, label=f"Усреднение для M={M}")
            plt.title(f"Осреднённая реализация шума (M={M})")
            plt.xlabel("t")
            plt.ylabel("Амплитуда")
            plt.show()

        return procData, std_devs

    @staticmethod
    def empirical_sigma(N, max_M, increment=10):
        original_std = 1  # Стандартное отклонение одной реализации
        sigma_values = []
        M_values = list(range(1, max_M + 1, increment))

        for M in M_values:
            # Генерация M случайных реализаций
            data = np.random.normal(0, original_std, (M, N))

            # Усреднение M реализаций
            averaged_data = np.mean(data, axis=0)

            # Вычисление σ_M (эмпирическое стандартное отклонение)
            sigma_M = np.std(averaged_data)
            sigma_values.append(sigma_M)

        return M_values, sigma_values, original_std

    @staticmethod
    def lpf(fc, m, dt):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        fact = 2 * fc * dt
        lpw = np.zeros(m + 1)
        lpw[0] = fact

        # Расчет первых m весов
        for i in range(1, m + 1):
            lpw[i] = np.sin(np.pi * fact * i) / (np.pi * i)
        lpw[m] /= 2  # Коррекция для m-го веса
        # Сглаживание
        sumg = lpw[0]
        for i in range(1, m + 1):
            arg = np.pi * i / m
            sum = d[0]
            for k in range(1, 4):
                sum += 2 * d[k] * np.cos(arg * k)
            lpw[i] *= sum
            sumg += 2 * lpw[i]
        lpw /= sumg  # Нормировка
        # Формирование симметричного массива весов
        weights = np.concatenate((lpw[::-1], lpw[1:]))
        return weights

    @staticmethod
    def hpf(fc, m, dt):

        lpw = processing.lpf(fc, m, dt)  # Получаем веса для ФНЧ
        loper = 2 * m + 1
        hpw = np.zeros(loper)
        for k in range(loper):
            if k == m:
                hpw[k] = 1 - lpw[k]
            else:
                hpw[k] = -lpw[k]
        return hpw

    @staticmethod
    def bpf(fc1, fc2, m, dt):
        if fc1 < fc2:
            lpw1 = processing.lpf(fc1, m, dt)  # Веса ФНЧ для fc1
            lpw2 = processing.lpf(fc2, m, dt)  # Веса ФНЧ для fc2
            loper = 2 * m + 1
            bpf = np.zeros(loper)
            for k in range(loper):
                bpf[k] = lpw2[k] - lpw1[k]
            return bpf
        else:
            print("fc1>=fc2")

    @staticmethod
    def bsf(fc1, fc2, m, dt):
        if fc1 < fc2:
            lpw1 = processing.lpf(fc1, m, dt)  # Веса ФНЧ для fc1
            lpw2 = processing.lpf(fc2, m, dt)  # Веса ФНЧ для fc2
            loper = 2 * m + 1
            bsf = np.zeros(loper)
            for k in range(loper):
                if k == m:
                    bsf[k] = 1 + lpw1[k] - lpw2[k]
                else:
                    bsf[k] = lpw1[k] - lpw2[k]
            return bsf
        else:
            print("fc1>=fc2")


class IN_OUT:
    @staticmethod
    def readWAV(file_path):
        rate, data = wavfile.read(file_path)  # Чтение файла
        N = len(data)  # Длина записи
        print(f"Частота дискретизации: {rate} Гц")
        print(f"Длина записи: {N} отсчётов")

        return data, rate, N

    @staticmethod
    def writeWAV(file_path, data, rate):
        # Увеличение громкости
        amplified_data = np.clip(data, -32768, 32767).astype(np.int16)

        # Запись в файл
        wavfile.write(file_path, rate, amplified_data)
        print(f"Аудиофайл записан в {file_path}")

    @staticmethod
    def rw(c1,c2,n1,n2,n3,n4,N):
        data = [1] * N
        for i in range(n1,n2):
            data[i] = c1
        for i in range(n3,n4):
            data[i] = c2
        return data

model = Model()
analysis = Analysis()
processing = Processing()
in_out = IN_OUT()

'''
    # Входные параметры
N = 1000  # Длина данных
a = 0.02  # Коэффициент для трендов
delta = 1  # Интервал времени



linear_up = model.trend('linear_up', a, N, delta)
linear_down = model.trend('linear_down', a, N, delta)
exp_up = model.trend('exp_up', a, N, delta)
exp_down = model.trend('exp_down', a, N, delta)

# Отображение графиков
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(linear_up)


plt.subplot(2, 2, 2)
plt.plot(linear_down)

plt.subplot(2, 2, 3)
plt.plot(exp_up)

plt.subplot(2, 2, 4)
plt.plot(exp_down)

plt.tight_layout()
plt.show()

# Пример сдвига и умножения данных
C_shift = 100  # Константа для сдвига
C_mult = 3  # Константа для умножения

shifted_data = model.shift(linear_up, C_shift)
multiplied_data = model.mult(exp_up, C_mult)

# Отображение сдвинутых и умноженных данных
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(shifted_data)
plt.title('Сдвинутые данные')

plt.subplot(1, 2, 2)
plt.plot(multiplied_data)
plt.title('Умноженные данные')

plt.tight_layout()
plt.show()

# Данные для шумов
N = 100
R = 10
delta = 1
data1 = model.noise(N, R, delta)
data2 = model.myNoise(N, R, delta)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(data1)
plt.title('Шум с использованием встроенного генератора')

plt.subplot(1, 2, 2)
plt.plot(data2)
plt.title('Шум с использованием разработанного генератора')

plt.tight_layout()
plt.show()

# Задание 3
N = 1000  # Длина данных
M = 8  # Коэффициент количества выбросов
R = 10000  # Опорное значение амплитуды выбросов
Rs = R * 0.1  # Варьирование амплитуд
spike_data = model.spikes(N, M, R, Rs)

plt.plot(spike_data)
plt.title('Импульсный шум с выбросами')
plt.xlabel('Индекс')
plt.ylabel('Амплитуда')
plt.show()

# Задание 4
N = 1000  # Длина каждой реализации
M = 100  # Количество реализаций
R = 100  # Диапазон шума
P = R * 0.01  # Порог для оценки стационарности и эргодичности (1% от R)

model = Model()
analysis = Analysis()


# 1. Генерация случайного вектора с использованием встроенного генератора
data_builtin = model.randVector(M, N, R, generator_type='builtin')

# 2. Оценка стационарности и эргодичности для встроенного генератора
stationary_builtin, ergodic_builtin = analysis.stationarity(data_builtin, N, M, P)
print(f"Встроенный генератор - Стационарность: {'да' if stationary_builtin else 'нет'}, Эргодичность: {'да' if ergodic_builtin else 'нет'}")

# 3. Генерация случайного вектора с использованием собственного генератора
data_custom = model.randVector(M, N, R, generator_type='custom')

# 4. Оценка стационарности и эргодичности для собственного генератора
stationary_custom, ergodic_custom = analysis.stationarity(data_custom, N, M, P)
print(f"Собственный генератор - Стационарность: {'да' if stationary_custom else 'нет'}, Эргодичность: {'да' if ergodic_custom else 'нет'}")


# Задание 5
N = 1000
A0 = 100
delta_t = 0.001
f0 = 515
harm_data = model.harm(N,A0,f0,delta_t)

plt.plot(harm_data)
plt.title(f'Гармонический процесс (f0 = {f0} Hz)')
plt.show()

N = 1000          # Длина данных
M = 3             # Количество гармоник
A = [100, 15, 20] # Амплитуды гармоник
f = [33, 5, 170]  # Частоты гармоник в Гц
delta_t = 0.002   # Временной шаг
data = model.polyHarm(N, A, f, M, delta_t)
plt.plot(data)
plt.title('Полигармонический процесс')
plt.show()

# Расчёт частоты главного повторения
f_gp = max(f)
print(f"Частота главного повторения f_гп = {f_gp} Гц")


N = 1000         # Длина данных
delta_t = 0.002  # Временной шаг
# Вариант (а): Линейный тренд и гармонический процесс
a_trend = 0.3
b_trend = 20
A_harm = 5
f_harm = 50

# Генерация данных
linear_trend = model.trend('linear_up', a_trend, N, delta_t) + b_trend
harm_signal = model.harm(N, A_harm, f_harm, delta_t)

# Аддитивная модель для (а)
additive_data_a = model.addModel(linear_trend, harm_signal,N)
plt.plot(harm_signal)
plt.tight_layout()
plt.show()


'''
# Задание 6
# Параметры

'''
N = 1000
A0 = 100
delta_t = 0.001
f0 = 15
harm_data1 = model.harm(N,A0,f0,delta_t)
harm_data2 = model.harm(N,A0,215,delta_t)

acf_values = analysis.acf(harm_data, N, type='AKF')
plt.figure(figsize=(10, 5))
plt.plot(range(N), acf_values)
plt.xlabel("L")
plt.ylabel("Значение функции")
plt.show()

ccf_values = analysis.ccf(data2,data2,N)
plt.figure(figsize=(10, 5))
plt.plot(range(N), ccf_values)
plt.xlabel("L")
plt.ylabel("Значение функции")
plt.show()



a,b =Analysis.Fourier(data,N)
Analysis.spectrFourier(a,b,N,delta)



N = 1000  # Длина данных
a = 0.02  # Коэффициент для трендов
delta = 1  # Интервал времени



exp_up = model.trend('exp_up', a, N, delta)

a,b =Analysis.Fourier(exp_up,N)
Analysis.spectrFourier(a,b,500,delta)


N = 1024
L_values = [24, 124, 224]
delta_t = 0.001  # Интервал между отсчетами

A0 = 100
f0 = 15
harm_data1 = model.harm(N,A0,f0,delta_t)


for L in L_values:
    windowed_harmonic = Analysis.window(harm_data1,L)
    Re, Im = Analysis.Fourier(windowed_harmonic, N)
    print(f"Амплитудный спектр Фурье для гармонического процесса с обнулением {L} последних значений:")
    print (Analysis.spectrFourier(Re, Im,512, delta_t))

M = 3             # Количество гармоник
A = [100, 15, 20] # Амплитуды гармоник
f = [33, 5, 170]  # Частоты гармоник в Гц
data = model.polyHarm(N, A, f, M, delta_t)

for L in L_values:
    windowed_harmonic = Analysis.window(data,L)
    Re, Im = Analysis.Fourier(windowed_harmonic, N)
    print(f"Амплитудный спектр Фурье для полигармонического процесса с обнулением {L} последних значений:")
    print (Analysis.spectrFourier(Re, Im,512, delta_t))
    


dt = 0.0005
with open('pgp_dt0005.dat') as file:
    data = np.fromfile(file,dtype=np.float32)

plt.figure(figsize=(10,5))
plt.plot(data)
plt.show()

a,b =Analysis.Fourier(data,1000)
Analysis.spectrFourier(a,b,500,0.0005)


N = 1000
dt = 0.002

a = 0.05
b = 10
R = 10

data1 = model.trend('exp_up',a,N,dt) * 20
data2 = model.noise(N,R,dt)
data3 = model.addModel(data1, data2, N)
plt.figure(figsize=(10,5))
plt.title('Аддитивная модель')
plt.plot(data3)
plt.show()


# Задание 8
N = 1000
A0 = R = 100
data1 = model.noise(N,R,1)
harm_data = model.harm(N,A0,15,0.001)
data2 = model.spikes(N,1,100*R,10*R)
data3 = model.addModel(data1,data2,N)
data4 = model.addModel(harm_data,data2,N)

proc_data = processing.antiSpike(data4,N,R)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(data4)
plt.title('Необработанные данные')

plt.subplot(1,2,2)
plt.plot(proc_data)
plt.title('Обработанные данные')

plt.show()


N = 1000         # Длина данных
delta_t = 0.002  # Временной шаг
# Вариант (а): Линейный тренд и гармонический процесс
a_trend = 0.3
b_trend = 20
A_harm = 5
f_harm = 50

# Генерация данных
exp_trend = model.trend('exp_up', a_trend, N, delta_t) + b_trend
harm_signal = model.harm(N, A_harm, f_harm, delta_t)

# Аддитивная модель для (а)
additive_data_a = model.addModel(exp_trend, harm_signal,N)


plt.figure(figsize=(10,5))
plt.plot(additive_data_a)
plt.title('Аддитивная модель')
plt.show()

N = 1000
W_values = [10, 50, 100]  # Разные длины окна

plt.figure(figsize=(12, 8))

for i, W in enumerate(W_values, 1):
    detrended_data = Processing.antiTrendNonLinear(additive_data_a, N, W)

    # Графики для сравнения исходных данных, тренда и очищенных данных
    plt.subplot(len(W_values), 1, i)
    plt.plot(detrended_data, label='Данные без тренда')
    plt.grid(True)
    plt.title(f'Удаление тренда с использованием окна W = {W}')

plt.tight_layout()
plt.show()


N = 1000         # Длина данных
delta_t = 1 # Временной шаг
# Вариант (а): Линейный тренд и гармонический процесс
a_trend = 0.3
b_trend = 20
A_harm = 5
f_harm = 50

# Генерация данных
linear_trend = model.trend('linear_up', a_trend, N, delta_t) + b_trend
harm_signal = model.harm(N, A_harm, f_harm, 0.002)

# Аддитивная модель для (а)
additive_data_a = model.addModel(linear_trend, harm_signal,N)
processing.antiTrendLinear(additive_data_a,N)

exp_trend = model.trend('exp_up',0.05,1000,10)
noise_data = model.noise(1000,100,1)

add_data2 = model.multModel(exp_trend,noise_data,N)

proc_data = processing.antiTrendNonLinear(add_data2,N,20)

plt.figure(figsize=(10,5))
plt.plot(proc_data)
plt.title('Аддитивная модель')
plt.show()

N = 1000  # Длина данных
a = 0.3  # Коэффициент для трендов
delta = 1  # Интервал времени

linear_up = model.trend('linear_up', a, N, delta)+20
Analysis.statistics(linear_up, N)
a1,b1 = Analysis.Fourier(linear_up,N)
plt.figure(figsize=(10,7))


plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(linear_up)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(linear_up,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(linear_up,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,500,delta))
plt.tight_layout()
plt.show()

N = 1000
A0 = 100
delta_t = 0.001
f0 = 15
harm_data = model.harm(N,A0,f0,delta_t)

Analysis.statistics(harm_data, N)
a1,b1 = Analysis.Fourier(harm_data,N)
plt.figure(figsize=(10,7))


plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(harm_data)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(harm_data,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(harm_data,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,500,delta_t))
plt.tight_layout()
plt.show()


N=1000
delta_t = 0.002
M = 3             # Количество гармоник
A = [100, 15, 20] # Амплитуды гармоник
f = [33, 5, 170]  # Частоты гармоник в Гц
data = model.polyHarm(N, A, f, M, delta_t)

Analysis.statistics(data, N)
a1,b1 = Analysis.Fourier(data,N)
plt.figure(figsize=(10,7))


plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(data)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(data,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(data,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,500,delta_t))
plt.tight_layout()
plt.show()

N = 100
R = 10
delta = 1
data1 = model.noise(N, R, delta)
data = model.myNoise(N, R, delta)
Analysis.statistics(data, N)
a1,b1 = Analysis.Fourier(data,N)
plt.figure(figsize=(10,7))


plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(data)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(data,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(data,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,50,delta))
plt.tight_layout()
plt.show()


N = 1000  # Длина данных
M = 8  # Коэффициент количества выбросов
R = 100  # Опорное значение амплитуды выбросов
Rs = R * 0.1  # Варьирование амплитуд
data = model.spikes(N, M, R, Rs)
Analysis.statistics(data, N)
a1,b1 = Analysis.Fourier(data,N)
plt.figure(figsize=(10,7))

plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(data)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(data,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(data,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,500,1))
plt.tight_layout()
plt.show()


N = 1000
dt = 0.1

a = 0.05
b = 10
R = 10
data = model.trend('exp_up',a,N,dt) * 20

Analysis.statistics(data, N)
a1,b1 = Analysis.Fourier(data,N)
plt.figure(figsize=(10,7))

plt.subplot(2,2,1)
plt.title('Исходные данные')
plt.plot(data)
plt.subplot(2,2,2)
plt.title('Гистограмма')
plt.hist(data,bins=100,density=True)
plt.subplot(2,2,3)
plt.title('Автокорреляция')
plt.plot(analysis.acf(data,N,type='AKF'))
plt.subplot(2,2,4)
plt.title('Амплитудный спектр Фурье')
plt.plot(analysis.spectrFourier(a1,b1,500,1))
plt.tight_layout()
plt.show()


#Задание 10
N = 1000  # Длина данных
a = 0.02  # Коэффициент для трендов
delta = 1  # Интервал времени

linear_up = model.trend('linear_up', a, N, delta)
C_shift = 100  # Константа для сдвига

shifted_data = model.shift(linear_up, C_shift)
antishifted_data = processing.antiShift(shifted_data,N)

# Отображение сдвинутых и умноженных данных
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(shifted_data)
plt.title('Сдвинутые данные')

plt.subplot(1, 2, 2)
plt.plot(antishifted_data)
plt.title('Удаление смещения')
plt.show()

exp_trend = model.trend('exp_up',0.05,1000,10)
noise_data = model.noise(1000,100,1)

add_data2 = model.multModel(exp_trend,noise_data,1000)
antinoise_data = processing.antiNoise(add_data2,100,1000)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(add_data2)
plt.title('Сдвинутые данные')

plt.subplot(1, 2, 2)
plt.plot(antinoise_data)
plt.title('Удаление смещения')
plt.show()

# Задание 11

A = 1
f = 7
dt = 0.005
M = 200
N=1000
a=30
b=1
h1 = model.harm(N,A,f,dt)
h2 = model.trend('exp_down',a,N,dt)
h = model.multModel(h1,h2,N)
hh = h/ np.max(h) * 120


t = np.arange(0, N * dt, dt)
x = np.zeros(N)
impulse_positions = [200, 400, 600, 800]
impulse_amplitudes = [0.9, 1.0, 1.1, 0.95]
for pos, amp in zip(impulse_positions, impulse_amplitudes):
    x[pos] = amp

M2 = 8  # Коэффициент количества выбросов
R = 10000  # Опорное значение амплитуды выбросов
Rs = R * 0.1  # Варьирование амплитуд
x2 = model.spikes(N, M2, R, Rs)



y = model.convModel(x,hh,N,M)
y2 = model.convModel(x2,hh,N,M)
# Визуализация

plt.plot(hh)
plt.title('Импульсивная реакция')
plt.show()


plt.plot(x)
plt.title('Управляющая функция')
plt.show()

plt.plot(y)
plt.title('Кардиограмма')
plt.show()

plt.plot(y2)
plt.title('Патологическая кардиограмма')
plt.show()

# ЗАДАНИЕ 12
fc1 = 100  # Частота среза
fc2 = 150
m = 150   # Количество весов (m)
dt = 0.002  # Шаг по времени

# Расчет симметричных весов
x1 = processing.lpf(fc1,m,dt)
x2 = processing.hpf(fc1,m,dt)
x3 = processing.bpf(fc1,fc2,m,dt)
x4 = processing.bsf(fc1,fc2,m,dt)



freq1,transfer_func1,halfN1 = analysis.transferFunction(x1,dt,m)
freq2,transfer_func2,halfN2 = analysis.transferFunction(x2,dt,m)
freq3,transfer_func3,halfN3 = analysis.transferFunction(x4,dt,m)


with open('pgp_dt0005.dat') as file:
    data = np.fromfile(file,dtype=np.float32)



filt_data2 = model.convModel(data,x4,1000,m)
freq4,transfer_func4,halfN4 = analysis.transferFunction(filt_data2,dt,m)

plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.plot(filt_data2)
plt.xlabel('Время(с)')
plt.title('Фильтрованные данные')

plt.subplot(1,2,2)
plt.plot(freq4, transfer_func4[:halfN4])
plt.xlabel('Частота (Гц)')
plt.title('Частотная характеристика')
plt.tight_layout()
plt.show()


input_path = "doma.wav"  # Замените на путь к вашему файлу
data, rate, N = IN_OUT.readWAV(input_path)

# Пример записи
output_path = "output.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, data, rate)

data, rate, N = IN_OUT.readWAV(output_path)


N = 1000  # Длина одной реализации
M_values = [1, 10, 100, 1000, 10000]  # Значения M
random_generator = lambda N: np.random.randn(N)  # Функция генерации шума

# Запуск функции
procData, std_devs = Processing.antiNoise(random_generator, M_values, N)

# Вывод стандартных отклонений
for M, std_dev in zip(M_values, std_devs):
    print(f"M = {M}, стандартное отклонение σ_M = {std_dev:.5f}")


A0 = 10
R = 30
N = 1000
M_values = [1, 10, 100, 1000, 10000]
data1 = model.noise(N,R,1)
harm_data = model.harm(N,A0,5,0.001)
data3 = model.addModel(data1,harm_data,N)

procData, std_devs = Processing.antiNoise(data3,M_values , N)

'''

input_path = "doma.wav"
data1, rate, N = IN_OUT.readWAV(input_path)
c1 = 0.5
c2 = 2
n1 = 1000
n2 = 12000
n3 = 18000
n4 = 28000

data2 = IN_OUT.rw(c1,c2,n1,n2,n3,n4,N)
data3 = model.multModel(data1,data2,N)

'''
# Отображение короткого фрагмента (0.5-1 сек)
duration = min(rate, N)  # 1 секунда или меньше
time = np.linspace(0, duration / rate, duration)
plt.plot(time, data1[:duration])
plt.title("Исходный фрагмент")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# Пример записи
output_path = "output.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, data3, rate)
new_data, new_rate, N = IN_OUT.readWAV(output_path)
'''
# Отображение короткого фрагмента (0.5-1 сек)
duration = min(rate, N)  # 1 секунда или меньше
time = np.linspace(0, duration / rate, duration)
plt.plot(time, new_data[:duration])
plt.title("Фрагмент с изменённым ударением")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.tight_layout()
plt.show()
'''



'''
# Построение графиков для каждого слога
plt.figure(figsize=(12, 6))

# Первый слог
plt.subplot(2, 1, 1)
plt.plot(time_syllable1, syllable1, label="Первый слог")
plt.title("Первый слог")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()

# Второй слог
plt.subplot(2, 1, 2)
plt.plot(time_syllable2, syllable2, label="Второй слог", color="orange")
plt.title("Второй слог")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
'''


'''
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(time_syllable2, syllable2)
plt.title("Второй слог")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(freq2, spectr2[:halfN2])
plt.grid(True)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Амплитудный спектр Фурье')
plt.tight_layout()
plt.show()
'''

syllable1 = data1[n1:n2]
syllable2 = data1[n3:n4]
N1 = len(syllable1)
N2 = len(syllable2)
print(f"Частота дискретизации для 1 слога: {rate} Гц")
print(f"Длина записи 1 слога: {N1} отсчётов")
print(f"Частота дискретизации для 2 слога: {rate} Гц")
print(f"Длина записи 2 слога: {N2} отсчётов")

output_path = "syllable1.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, syllable1, rate)

output_path = "syllable2.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, syllable2, rate)

# Создаём временную шкалу для каждого слога
time_syllable1 = np.linspace(n1 / rate, n2 / rate, len(syllable1))
time_syllable2 = np.linspace(n3 / rate, n4 / rate, len(syllable2))

fc1= 657
fc2= 1290
m = 64
dt=1/rate

x1 = processing.lpf(fc1,m,dt)
x2 = processing.hpf(fc1,m,dt)
x3 = processing.bpf(fc1,fc2,m,dt)


filt_data1 = model.convModel(syllable2,x1,N2,m)
filt_data2 = model.convModel(syllable2,x2,N2,m)
filt_data3 = model.convModel(syllable2,x3,N2,m)
Re1,Im1 = analysis.Fourier(filt_data1,N2)
Re2,Im2 = analysis.Fourier(filt_data2,N2)
Re3,Im3 = analysis.Fourier(filt_data3,N2)
delta_t = 1 / rate
halfN1 = N1 // 2
halfN2 = N2 // 2

spectr1,freq1 = analysis.spectrFourier(Re1,Im1,halfN2,delta_t)
spectr2,freq2 = analysis.spectrFourier(Re2,Im2,halfN2,delta_t)
spectr3,freq3 = analysis.spectrFourier(Re3,Im3,halfN2,delta_t)

# Обновление временной шкалы для фильтрованных данных
delay = m / rate  # Задержка, вызванная фильтром
filtered_length1 = len(filt_data1)  # Длина после свёртки (уже укорочена)
filtered_length2 = len(filt_data2)  # Длина после свёртки (уже укорочена)
filtered_length3 = len(filt_data3)  # Длина после свёртки (уже укорочена)
time_filt_data1 = np.linspace(delay, delay + filtered_length1 * dt, filtered_length1)
time_filt_data2 = np.linspace(delay, delay + filtered_length2 * dt, filtered_length2)
time_filt_data3 = np.linspace(delay, delay + filtered_length3 * dt, filtered_length3)

# Построение графиков
plt.figure(figsize=(8, 4))
# Фильтрованные данные
plt.subplot(1, 2, 1)
plt.plot(time_filt_data1, filt_data1, )
plt.xlabel('Время (с)')
plt.title('2 слог после фильтра lpf')
# Спектр данных
plt.subplot(1, 2, 2)
plt.plot(freq1, spectr1[:halfN2])
plt.xlabel('Частота[Гц]')
plt.title('Спектр данных')
plt.tight_layout()
plt.show()

output_path = "ma2_lpf.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, filt_data1, rate)

# Построение графиков
plt.figure(figsize=(8, 4))
# Фильтрованные данные
plt.subplot(1, 2, 1)
plt.plot(time_filt_data2, filt_data2, )
plt.xlabel('Время (с)')
plt.title('2 слог после фильтра hpf')
# Спектр данных
plt.subplot(1, 2, 2)
plt.plot(freq2, spectr2[:halfN2])
plt.xlabel('Частота[Гц]')
plt.title('Спектр данных')
plt.tight_layout()
plt.show()

output_path = "ma2_hpf.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, filt_data2, rate)


# Построение графиков
plt.figure(figsize=(8, 4))
# Фильтрованные данные
plt.subplot(1, 2, 1)
plt.plot(time_filt_data3, filt_data3, )
plt.xlabel('Время (с)')
plt.title('2 слог после фильтра bpf')
# Спектр данных
plt.subplot(1, 2, 2)
plt.plot(freq3, spectr3[:halfN2])
plt.xlabel('Частота[Гц]')
plt.title('Спектр данных')
plt.tight_layout()
plt.show()

output_path = "ma2_bpf.wav"  # Имя выходного файла
IN_OUT.writeWAV(output_path, filt_data3, rate)


'''
data_Fourier = data1.astype(np.float32)
Re,Im = analysis.Fourier(data_Fourier,N)
Re1,Im1 = analysis.Fourier(syllable1,N1)
Re2,Im2 = analysis.Fourier(syllable2,N2)
delta_t = 1 / rate
halfN = N // 2
halfN1 = N1 // 2
halfN2 = N2 // 2
spectr,freq = analysis.spectrFourier(Re,Im,halfN,delta_t)
spectr1,freq1 = analysis.spectrFourier(Re1,Im1,halfN1,delta_t)
spectr2,freq2 = analysis.spectrFourier(Re2,Im2,halfN2,delta_t)
'''