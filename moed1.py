import numpy as np
import matplotlib.pyplot as plt

class Model:
    def trend(self, trend_type, a, N, delta):
        t = np.arange(0, N) * delta  # Время t = k * delta
        if trend_type == 'linear_up':
            return a * t  # Восходящий линейный тренд
        elif trend_type == 'linear_down':
            return -a * t  # Нисходящий линейный тренд
        elif trend_type == 'exp_up':
            return np.exp(a * t)  # Восходящий экспоненциальный тренд
        elif trend_type == 'exp_down':
            return np.exp(-a * t)  # Нисходящий экспоненциальный тренд
        else:
            raise ValueError("Неизвестный тип тренда")

    def shift(self, data, C):
        return data + C  # Сдвиг на константу C

    def mult(self, data, C):
        return C * data  # Умножение на константу C

    def noise(self,N,R,delta):
        t = np.arange(0, N) * delta
        noise_data = np.random.rand(N)
        xmin = np.min(noise_data)
        xmax = np.max(noise_data)
        data = ((noise_data - xmin) / (xmax - xmin) - 0.5) * 2 * R
        return t,data

    def myNoise(self,N,R,delta):
        t = np.arange(0, N) * delta
        a = 1664525
        c = 1013904223
        m = 2 ** 32
        seed = 12345  # Начальное значение
        noise_data = np.zeros(N)
        for i in range(N):
            seed = (a * seed + c) % m
            noise_data[i] = seed / m
        xmin, xmax = np.min(noise_data), np.max(noise_data)
        normalized_data = ((noise_data - xmin) / (xmax - xmin) - 0.5) * 2 * R

        return t, normalized_data

    def spikes(self,N,M,R,Rs):
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
                _, data[i] = self.noise(N, R, 1)  # Использование встроенного генератора
            elif generator_type == 'custom':
                _, data[i] = self.myNoise(N, R, 1)  # Использование собственного генератора
            else:
                raise ValueError("Неизвестный тип генератора: используйте 'builtin' или 'custom'")
        return data

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



    # Входные параметры
N = 1000  # Длина данных
a = 0.02  # Коэффициент для трендов
delta = 1  # Интервал времени

model = Model()

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
t_builtin, builtin_noise = model.noise(N, R, delta)
t_custom, custom_noise = model.myNoise(N, R, delta)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t_builtin, builtin_noise)
plt.title('Шум с использованием встроенного генератора')

plt.subplot(1, 2, 2)
plt.plot(t_custom, custom_noise)
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
