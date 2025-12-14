# -*- coding: utf-8 -*-
import numpy as np                   # библиотека для векторов/матриц
from math import *                   # базовые математические функции
import datetime                      # для вывода времени старта
import pandas as pd                  # для удобного хранения результатов
from atmosphere import atmo          # модель атмосферы (должна лежать рядом)
from numba import njit               # ускорение некоторых функций

# -------- Приветствие и стартовое время --------
# создаем объект атмосферы
atm = atmo()                         # инициализация вашей атмосферы
# фиксируем и выводим время старта
start = datetime.datetime.now()      # берем текущее время
print('Время старта: ' + str(start)) # печатаем время старта

# -------- Параметры расчёта --------
calcParams = np.array([0.1, 15])     # массив [dt, r_por]: шаг по времени и радиус поражения
calcParams = calcParams.astype('float64')  # приводим к типу float64

# -------- Гравитация --------
g = 9.80665                          # ускорение свободного падения, м/с^2

# -------- Словарь результатов --------
result = dict({                      # здесь будем копить всё, что посчитали
    't': [],                         # время
    'Theta_r': [],                   # угол ракеты
    'v_r': [],                       # скорость ракеты
    'x_r': [],                       # координата X ракеты
    'y_r': [],                       # координата Y ракеты
    'n_xa_r': [],                    # продольная перегрузка ракеты
    'n_ya_r': [],                    # поперечная перегрузка ракеты
    'P_r': [],                       # тяга двигателя
    'mass_r': [],                    # масса ракеты
    'Theta_c': [],                   # угол цели
    'v_c': [],                       # скорость цели
    'x_c': [],                       # координата X цели
    'y_c': [],                       # координата Y цели
    'n_xa_c': [],                    # продольная перегрузка цели (задаётся)
    'n_ya_c': [],                    # поперечная перегрузка цели (задаётся)
    'Theta_n': [],                   # угол носителя (нос)
    'v_n': [],                       # скорость носителя
    'x_n': [],                       # координата X носителя
    'y_n': [],                       # координата Y носителя
    'n_xa_n': [],                    # продольная перегрузка носителя (задаётся)
    'n_ya_n': [],                    # поперечная перегрузка носителя (задаётся)
    'switch_time': 0                 # момент переключения режима двигателя
})

# -------- Начальные условия (вектор inits) --------
# inits: [x_r0, y_r0, x_c0, y_c0, Theta_c0, v_c0, Theta_r0, v_r0, Theta_n0, v_n0,
#         mu_0, eta_0, k_m, k_p, m_start, m_bch, diameter, I_ud, length, (2 свободных)]
inits = np.array([
    0,       # x_r0: начальная X ракеты [м]
    10000,    # y_r0: начальная Y ракеты [м]
    30000,  # x_c0: начальная X цели [м]
    5000,   # y_c0: начальная Y цели [м]
    0,       # Theta_c0: начальный угол цели [рад]
    600,    # v_c0: скорость цели [м/с] (отрицательная — летит "влево")
    -999,    # Theta_r0: начальный угол ракеты [рад] (-999 = рассчитать по геометрии)
    700,     # v_r0: скорость ракеты [м/с]
    0,       # Theta_n0: начальный угол носителя [рад]
    340,     # v_n0: скорость носителя [м/с]
    0.825,       # mu_0: относительный запас топлива (0 = без тяги для простоты, поставьте >0 для тяги)
    15.0,       # eta_0: тяговооружённость (0 = без тяги)
    0.55,       # k_m: распределение массы топлива по режимам
    0.2,       # k_p: отношение тяг режимов
    161.5,     # m_start: стартовая масса [кг]
    5,      # m_bch: масса БЧ [кг]
    0.178,    # diameter: диаметр [м]
    2500,    # I_ud: удельный импульс [м/с]
    3.906,     # length: длина [м]
    0, 0     # свободные
], dtype='float64')

# -------- Ограничения --------
limits = np.array([
    300,  # t_max: макс. время [с]
    40,   # n_ya_r_max: макс. поперечная перегрузка ракеты
    0     # y_r_min: минимальная высота [м]
], dtype='float64')

# -------- Параметры метода наведения --------
MethodData = np.array([
    8,              # метод наведения (8 = пропорциональное сближение)
    np.deg2rad(10), # phi_upr (не используется здесь)
    np.deg2rad(10), # Theta_r (стартовый угол, если не -999)
    4,              # k: коэффициент для PN
    3,              # lambda (не используется)
    0.00035         # C (не используется)
], dtype='float64')

# -------- Задаваемые перегрузки для цели и носителя --------
NRCData = np.array([
    0,  # n_xa_r_pot (не используется — продольная ракеты считается)
    0,  # n_xa_c_pot: продольная цели
    1,  # n_ya_c_pot: поперечная цели
    0,  # n_xa_n_pot: продольная носителя
    1   # n_ya_n_pot: поперечная носителя
], dtype='float64')

# -------- Для “линий видимости” (не обязательно) --------
VisionLines = dict({
    't': [],
    'x_r': [], 'y_r': [],
    'x_c': [], 'y_c': [],
    'x_n': [], 'y_n': []
})

# ------------------- Ускоряемые функции движения -------------------

@njit(fastmath=True)
def r(x_r, y_r, x_c, y_c):
    return np.sqrt((y_c - y_r) ** 2 + (x_c - x_r) ** 2)  # расстояние между ракетой и целью

@njit(fastmath=True)
def dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c):
    r_rc = r(x_r, y_r, x_c, y_c)                         # текущее расстояние
    if r_rc == 0.0:                                      # защита от деления на 0
        return 0.0
    Delta_x = x_c - x_r                                  # разность по x
    Delta_y = y_c - y_r                                  # разность по y
    Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)  # относительная скорость по x
    Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)  # относительная скорость по y
    return (Delta_y * Delta_dot_y + Delta_x * Delta_dot_x) / r_rc # производная расстояния

@njit(fastmath=True)
def epsilon(x_r, y_r, x_c, y_c):
    # угол линии визирования (упрощенно, через atan)
    if (x_c - x_r > 0):
        return atan((y_c - y_r) / (x_c - x_r))
    elif ((x_c - x_r < 0) and (y_c - y_r >= 0)):
        return atan((y_c - y_r) / (x_c - x_r)) + np.pi
    elif ((x_c - x_r < 0) and (y_c - y_r < 0)):
        return atan((y_c - y_r) / (x_c - x_r)) - np.pi
    elif ((x_c == x_r) and (y_c >= y_r)):
        return np.pi / 2
    else:
        return -np.pi / 2

@njit(fastmath=True)
def dot_epsilon(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c):
    r_rc = r(x_r, y_r, x_c, y_c)                         # расстояние
    if r_rc == 0.0:
        return 0.0
    Delta_x = x_c - x_r                                  # разности координат
    Delta_y = y_c - y_r
    Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)  # относ. скорость
    Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)
    # производная угла линии визирования
    return (Delta_x * Delta_dot_y - Delta_y * Delta_dot_x) / (r_rc ** 2)

@njit(fastmath=True)
def dpar_i(n_xa_i, v_i, Theta_i):
    # дифференциалы: производная скорости и координат
    dv_i = (n_xa_i - np.sin(Theta_i)) * g                # изменение скорости
    dx_i = v_i * np.cos(Theta_i)                         # изменение x
    dy_i = v_i * np.sin(Theta_i)                         # изменение y
    return dv_i, dx_i, dy_i

@njit(fastmath=True)
def par_i1(Theta_i, dTheta_i, dt, v_i, dv_i, x_i, dx_i, y_i, dy_i):
    # шаг вперед методом Эйлера
    Theta_i1 = Theta_i + dTheta_i * dt                   # новый угол
    v_i1 = v_i + dv_i * dt                               # новая скорость
    x_i1 = x_i + dx_i * dt                               # новый x
    y_i1 = y_i + dy_i * dt                               # новый y
    return Theta_i1, v_i1, x_i1, y_i1

@njit(fastmath=True)
def value_cx(mah):
    # аппроксимация Cx по числу Маха (как в dz_pro)
    if mah < 0.61:
        return 0.308
    if mah >= 0.61 and mah <= 1:
        return 0.505 * (mah - 0.61) ** 2.31 + 0.308
    if mah >= 1 and mah < 1.4:
        return 0.4485 * ((mah - 1) ** 0.505) * (exp(-5.68 * (mah - 1))) + 0.551
    if mah >= 1.4 and mah < 4:
        return mah / (0.356 * mah ** 2 + 2.237 * mah - 1.4)
    return 0.302

@njit(fastmath=True)
def n_xa_tiaga(X_a, P, alf, m_r):
    # продольная перегрузка (тяга - лобовое) / вес
    return (-X_a + P) / (m_r * g)

@njit(fastmath=True)
def n_ya_pot_ProportionalNavConst(v_r, dot_epsilon_rc, Theta_r, k):
    # потребная поперечная перегрузка по PN
    return (v_r / g * k * dot_epsilon_rc + np.cos(Theta_r))

# -------- Перевод кода возврата в текст --------
def RC(ResCode):
    if ResCode == 0: return 'Успешное поражение цели'
    if ResCode == 1: return 'Столкновение ракеты с землёй'
    if ResCode == 2: return 'Превышение максимальной перегрузки'
    if ResCode == 4: return 'Превышено максимальное время полёта ракеты'
    if ResCode == 5: return 'Закончилось топливо'
    return ''

# ==================== Основная функция расчёта ====================

def CalcTrajectory(calcParams, inits, limits, MethodData, NRCData, result):
    dt = calcParams[0]                  # шаг по времени [с]
    N_max = ceil(limits[0] / dt)        # макс. число шагов (t_max / dt)

    # создаём массивы для истории по времени (размер N_max)
    t = np.zeros(N_max)                 # время
    Theta_r = np.zeros(N_max)           # угол ракеты
    v_r = np.zeros(N_max)               # скорость ракеты
    x_r = np.zeros(N_max)               # X ракеты
    y_r = np.zeros(N_max)               # Y ракеты
    n_xa_r = np.zeros(N_max)            # продольная перегрузка ракеты
    n_ya_r = np.zeros(N_max)            # поперечная перегрузка ракеты
    P_r = np.zeros(N_max)               # тяга двигателя
    mass_r = np.zeros(N_max)            # масса ракеты

    Theta_c = np.zeros(N_max)           # угол цели
    v_c = np.zeros(N_max)               # скорость цели
    x_c = np.zeros(N_max)               # X цели
    y_c = np.zeros(N_max)               # Y цели
    n_xa_c = np.zeros(N_max)            # продольная перегрузка цели (задана)
    n_ya_c = np.zeros(N_max)            # поперечная перегрузка цели (задана)

    Theta_n = np.zeros(N_max)           # угол носителя
    v_n = np.zeros(N_max)               # скорость носителя
    x_n = np.zeros(N_max)               # X носителя
    y_n = np.zeros(N_max)               # Y носителя
    n_xa_n = np.zeros(N_max)            # продольная перегрузка носителя (задана)
    n_ya_n = np.zeros(N_max)            # поперечная перегрузка носителя (задана)

    # списки для точек режимов двигателя (для картинки)
    flight_mode_raz_x = []              # точки первого режима по X
    flight_mode_dva_x = []              # точки второго режима по X
    flight_mode_raz_y = []              # точки первого режима по Y
    flight_mode_dva_y = []              # точки второго режима по Y

    # начальные значения из inits
    t[0] = 0
    Theta_r[0] = inits[6]               # начальный угол ракеты
    v_r[0] = inits[7]                   # скорость ракеты
    x_r[0] = inits[0]                   # X ракеты
    y_r[0] = inits[1]                   # Y ракеты

    Theta_c[0] = inits[4]               # угол цели
    v_c[0] = inits[5]                   # скорость цели
    x_c[0] = inits[2]                   # X цели
    y_c[0] = inits[3]                   # Y цели

    Theta_n[0] = inits[8]               # угол носителя
    v_n[0] = inits[9]                   # скорость носителя
    x_n[0] = inits[0]                   # X носителя (старт совпадает с ракетой)
    y_n[0] = inits[1] - 1               # Y носителя (чуть ниже)

    # параметры двигателя из inits
    mu_0 = inits[10]                    # относительный запас топлива
    eta_0 = inits[11]                   # тяговооружённость
    k_m = inits[12]                     # распределение массы
    k_p = inits[13]                     # отношение тяг режимов
    m_start = inits[14]                 # стартовая масса
    m_bch = inits[15]                   # масса БЧ
    diametr = inits[16]                 # диаметр
    I_ud = inits[17]                    # удельный импульс
    dlina = inits[18]                   # длина

    # топливо и тяга по режимам
    m_fuel = (m_start - m_bch) * mu_0   # масса топлива (общая)
    m_fuel_raz = m_fuel / (1 + k_m)     # топливо 1-й режим
    m_fuel_dva = m_fuel_raz * k_m       # топливо 2-й режим
    m_0 = m_bch + m_fuel                # начальная масса ракеты
    S_a = 0.8 * diametr ** 2 * pi / 4   # эффективная площадь сопла/донного сечения
    tiaga_raz = eta_0 * m_0 * g         # тяга 1-го режима
    tiaga_dva = tiaga_raz * k_p         # тяга 2-го режима
    G_raz = tiaga_raz / I_ud            # расход топлива на 1-м режиме
    G_dva = tiaga_dva / I_ud            # расход на 2-м режиме
    switch_time = 0                     # время переключения режима
    m_00 = m_0                          # запас (необязательно)

    # если угол ракеты -999, выровнять по ЛВ (линии визирования)
    if (Theta_r[0] == -999):
        epsilon_rc = epsilon(x_r[0], y_r[0], x_c[0], y_c[0])             # угол линии визирования
        Theta_r[0] = epsilon_rc - asin((v_c[0] / v_r[0]) * np.sin(epsilon_rc - Theta_c[0])) # угол наведения

    # главный цикл интегрирования по времени
    i = 0                               # текущий шаг
    while (t[i] <= limits[0]):          # пока t <= t_max
        # 1) аэродинамика: лобовое и тяга
        M = v_r[i] / 340                # число Маха (скорость/звук)
        X_a = value_cx(M) * atm.rho(y_r[i]) * v_r[i] ** 2 * S_a / 2  # лобовое сопротивление

        # выбираем текущую тягу P в зависимости от режима (по остатку топлива)
        if m_fuel >= m_fuel_dva and m_fuel_raz != 0:
            P = tiaga_raz + (atm.p(0) - atm.p(y_r[i])) * S_a  # тяга 1-го режима + перепад давлений
        else:
            P = tiaga_dva + (atm.p(0) - atm.p(y_r[i])) * S_a  # тяга 2-го режима + перепад давлений

        n_xa_r[i] = n_xa_tiaga(X_a, P, 0.0, m_0)  # продольная перегрузка ракеты

        # 2) задаем перегрузки цели и носителя (из NRCData)
        n_xa_c[i] = NRCData[1]         # продольная цели
        n_ya_c[i] = NRCData[2]         # поперечная цели
        n_xa_n[i] = NRCData[3]         # продольная носителя
        n_ya_n[i] = NRCData[4]         # поперечная носителя

        # 3) относительное движение: r и производные
        r_rc = r(x_r[i], y_r[i], x_c[i], y_c[i])  # расстояние до цели
        if (r_rc == 0):                           # защита от деления на 0
            dot_r_rc = 0
            dot_epsilon_rc = 0
        else:
            dot_r_rc = dot_r(x_r[i], y_r[i], x_c[i], y_c[i], v_r[i], v_c[i], Theta_r[i], Theta_c[i]) # производная расстояния
            dot_epsilon_rc = dot_epsilon(x_r[i], y_r[i], x_c[i], y_c[i], v_r[i], v_c[i], Theta_r[i], Theta_c[i]) # производная угла ЛВ

        # 4) потребная перегрузка по PN (если метод = 8)
        if i >= 1:
            if MethodData[0] == 8:
                k = MethodData[3]                             # коэффициент PN
                n_ya_r[i] = n_ya_pot_ProportionalNavConst(v_r[i], dot_epsilon_rc, Theta_r[i], k)  # требуемая n_ya

        # 5) проверка условий завершения
        if (r_rc < calcParams[1]):            # если ближе радиуса поражения
            returnCode = 0                    # успех
            break
        else:
            if (abs(n_ya_r[i]) > limits[1]):  # превышение поперечной перегрузки
                returnCode = 2
                break
            elif (y_r[i] <= limits[2] and i >= 1):  # упали на землю
                returnCode = 1
                break
            if m_fuel <= 0:                   # топливо закончилось
                returnCode = 5
                break

        # 6) дифференциалы (Эйлер)
        dTheta_r = 0
        if (v_r[i] != 0):                     # если есть скорость, угловая скорость PN
            k = 4                             # упрощенно: dTheta_r = k * dot_epsilon_rc
            dTheta_r = k * dot_epsilon_rc

        dv_r, dx_r, dy_r = dpar_i(n_xa_r[i], v_r[i], Theta_r[i])   # продольное уравнение ракеты

        dTheta_c = 0
        if v_c[i] != 0:                                                 # поворот цели (если надо)
            dTheta_c = (n_ya_c[i] - np.cos(Theta_c[i])) * g / v_c[i]
        dv_c, dx_c, dy_c = dpar_i(n_xa_c[i], v_c[i], Theta_c[i])        # движение цели

        dTheta_n = 0
        if v_n[i] != 0:                                                 # поворот носителя (если надо)
            dTheta_n = (n_ya_n[i] - np.cos(Theta_n[i])) * g / v_n[i]
        dv_n, dx_n, dy_n = dpar_i(n_xa_n[i], v_n[i], Theta_n[i])        # движение носителя

        # 7) шаг вперед (Эйлер)
        Theta_r[i + 1], v_r[i + 1], x_r[i + 1], y_r[i + 1] = par_i1(Theta_r[i], dTheta_r, dt, v_r[i], dv_r, x_r[i], dx_r, y_r[i], dy_r)
        Theta_c[i + 1], v_c[i + 1], x_c[i + 1], y_c[i + 1] = par_i1(Theta_c[i], dTheta_c, dt, v_c[i], dv_c, x_c[i], dx_c, y_c[i], dy_c)
        Theta_n[i + 1], v_n[i + 1], x_n[i + 1], y_n[i + 1] = par_i1(Theta_n[i], dTheta_n, dt, v_n[i], dv_n, x_n[i], dx_n, y_n[i], dy_n)

        # 8) расход топлива по режимам (простая модель)
        if MethodData[0] == 8:
            if m_fuel >= m_fuel_dva:          # режим 1
                m_fuel -= G_raz * dt          # вычитаем расход
                flight_mode_raz_x.append(x_r[i])  # точка для графика
                flight_mode_raz_y.append(y_r[i])
                result['switch_time'] = t[i]      # время возможного переключения
            else:                              # режим 2
                flight_mode_dva_x.append(x_r[i])
                flight_mode_dva_y.append(y_r[i])
                m_fuel -= G_dva * dt

            m_0 = m_bch + m_fuel              # масса ракеты обновляется

        # 9) запас по границе массива
        if (i + 1 >= N_max):
            returnCode = 4                    # превысили время
            break

        # 10) обновляем время
        t[i + 1] = t[i] + dt
        i += 1                                # следующий шаг

    # 11) финальная подготовка результатов
    mass_r[0] = mass_r[1]                    # сглаживаем стартовую точку
    P_r[0] = P                               # последняя тяга

    N = i + 1                                # фактическое число шагов

    # записываем в словарь result
    result['t'] = t[0:N]
    result['Theta_r'] = Theta_r[0:N]; result['v_r'] = v_r[0:N]; result['x_r'] = x_r[0:N]; result['y_r'] = y_r[0:N]
    result['n_xa_r'] = n_xa_r[0:N]; result['n_ya_r'] = n_ya_r[0:N]; result['P_r'] = P_r[0:N]; result['mass_r'] = mass_r[0:N]

    result['Theta_c'] = Theta_c[0:N]; result['v_c'] = v_c[0:N]; result['x_c'] = x_c[0:N]; result['y_c'] = y_c[0:N]
    result['n_xa_c'] = n_xa_c[0:N]; result['n_ya_c'] = n_ya_c[0:N]

    result['Theta_n'] = Theta_n[0:N]; result['v_n'] = v_n[0:N]; result['x_n'] = x_n[0:N]; result['y_n'] = y_n[0:N]
    result['n_xa_n'] = n_xa_n[0:N]; result['n_ya_n'] = n_ya_n[0:N]

    # печать диагностик
    print('Расстояние до цели:', r_rc, ' Загрузка топлива:', m_fuel_raz + m_fuel_dva)
    if returnCode == 0:
        print('Остаток топлива:', m_fuel, ' Нормальный mu_0', (m_start - m_bch) * mu_0 / m_start)

    # возвращаем код и точки режимов для графика
    return returnCode, flight_mode_raz_x, flight_mode_raz_y, flight_mode_dva_x, flight_mode_dva_y

# ==================== Запуск и простая визуализация ====================

if __name__ == "__main__":
    # запускаем расчёт
    ResCode, fx1, fy1, fx2, fy2 = CalcTrajectory(calcParams, inits, limits, MethodData, NRCData, result)
    print(RC(ResCode))                     # печатаем текстовый код результата

    # делаем DataFrame для удобства
    CalcRes = pd.DataFrame(result)

    # === Простая визуализация ===
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = "Times New Roman"
    plt.rcParams['mathtext.it'] = "Times New Roman:italic"

    # Траектории (цель и режимы ракеты)
    fig, ax = plt.subplots()
    ax.plot(CalcRes['x_c'], CalcRes['y_c'], color='blue', label='Цель')         # траектория цели
    ax.plot(fx1, fy1, color='red', label='ракета 1 (режим)')                   # точки режима 1
    ax.plot(fx2, fy2, color='orange', label='ракета 2 (режим)')                # точки режима 2
    ax.legend(loc=2); ax.grid(True)
    ax.set_xlabel('$X$, м'); ax.set_ylabel('$Y$, м'); ax.set_title('Траектории')
    plt.tight_layout(); plt.show()

    # Скорость ракеты
    fig, ax = plt.subplots()
    ax.plot(CalcRes['t'], CalcRes['v_r'], color='blue')
    ax.grid(True); ax.set_xlabel('$t$, c'); ax.set_ylabel('$v_{р}$, м/с'); ax.set_title('Скорость ракеты')
    plt.tight_layout(); plt.show()

    # Перегрузки ракеты
    fig, ax = plt.subplots()
    ax.plot(CalcRes['t'], CalcRes['n_ya_r'], color='blue', label='n_ya_r')
    ax.plot(CalcRes['t'], CalcRes['n_xa_r'], color='red', label='n_xa_r')
    ax.legend(loc=1); ax.set_xlabel('$t$, c'); ax.set_ylabel('$n$'); ax.grid(True); ax.set_title('Перегрузки ракеты')
    plt.tight_layout(); plt.show()

    # Масса ракеты
    fig, ax = plt.subplots()
    ax.plot(CalcRes['t'], CalcRes['mass_r'], color='blue')
    ax.set_xlabel('$t$, c'); ax.set_ylabel('$m$, кг'); ax.grid(True); ax.set_title('Масса ракеты')
    plt.tight_layout(); plt.show()

    # Тяга
    fig, ax = plt.subplots()
    ax.plot(CalcRes['t'], CalcRes['P_r'], color='blue')
    ax.set_xlabel('$t$, c'); ax.set_ylabel('$P$, Н'); ax.grid(True); ax.set_title('$P_{r}$')
    plt.tight_layout(); plt.show()
