import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import AeroBDSM

# =============================================================================
# ПАРАМЕТРЫ И КОНСТАНТЫ
# =============================================================================

# Основные параметры
N = 1000
l_f = 3.700
b_kr = 0.8
b_op = 0.260
S_Kons = 4 * 0.075
l_raszmah = 0.64
d = 0.180
S_op = 4 * 0.0598
l_raszmah_op = 0.64
#S_op = (4 * 0.260 * (l_raszmah_op - d))/2
print('S op = ', S_op)
# Расчетные параметры
l_raszmah_kons = l_raszmah - d
lamb = (l_raszmah_kons**2) / S_Kons
lamb_op = (l_raszmah_op**2) / S_op
bar_c = 0.01
chi_05 = 0
zeta = 1
lambda_Nos = 1.38
lambda_Cil = 17.305
D_bar = d / l_raszmah

# Параметры для интерференции
xb = 1.980
L1 = xb + b_kr / 2 
xb_op = 3.440
L1_op = xb_op + b_op / 2
nu = 15.1 * 1e-6
eta_k = 1
kappa_M = 0.96

# Параметры для скоса потока
alpha_p, phi_alpha, psi_I, psi_II = 0, 0, 0, 0
l_1c_II = 0.036
zeta_II = 0
b_b_II = b_kr
chi_0_II = 0
L_vI_bII = 0.882

# Параметры ориентации
orientation = "XX"
psi = 45

# Параметры для расчета сопротивления
delta_I = 10  # угол поворота первой несущей поверхности - руля
delta_II = 0  # угол поворота второй несущей поверхности - крыла
c_otn = bar_c
lambda_r = lamb_op
lambda_kr = lamb
eta_c = 1.13

# Коэффициенты нелинейности
A_kr = 0.8
A_op = 0.6

# Новые параметры для коэффициента щели
chi_p = math.radians(45)  # угол стреловидности (пример)
# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def safe_divide(a, b):
    """Безопасное деление с обработкой нуля"""
    return a / b if b != 0 else 0

def calculate_wing_normal_coefficient(c_ya, alpha_deg, A):
    """Расчет коэффициента нормальной силы крыла с правильной обработкой отрицательных углов"""
    alpha_rad = math.radians(alpha_deg)
    
    # Линейная часть
    term1 = 57.3 * c_ya * math.sin(alpha_rad) * math.cos(alpha_rad)
    
    # Нелинейная часть - квадратичная зависимость от sin(alpha)
    # Для отрицательных углов sin(alpha_rad) будет отрицательным, но квадрат - всегда положительный
    # Поэтому умножаем на sign(alpha) чтобы сохранить знак
    term2 = A * (math.sin(alpha_rad))**2 * (1 if alpha_deg >= 0 else -1)
    
    return term1 + term2

def calculate_wing_lift_coefficient(c_ya, alpha_deg, A):
    """Расчет коэффициента подъемной силы крыла с правильной обработкой отрицательных углов"""
    alpha_rad = math.radians(alpha_deg)
    
    # Для подъемной силы используем косинусное преобразование от нормальной силы
    c_n = calculate_wing_normal_coefficient(c_ya, alpha_deg, A)
    
    # Подъемная сила = нормальная сила * cos(alpha) - осевая сила * sin(alpha)
    # Упрощенно: c_y ≈ c_n * cos(alpha) для малых углов
    return c_n * math.cos(alpha_rad)

def calculate_fuselage_lift_coefficient(c_ya, alpha_deg):
    """Расчет коэффициента подъемной силы корпуса"""
    return c_ya * alpha_deg

# =============================================================================
# ОСНОВНОЙ РАСЧЕТ
# =============================================================================

print("Запуск полного аэродинамического анализа...")

# Создаем массивы для чисел Маха
Mach = np.linspace(0.1, 4, N)

# Массивы для результатов
c_ya_kr = np.zeros_like(Mach)
c_ya_op = np.zeros_like(Mach)
c_ya_izkor = np.zeros_like(Mach)
c_ya_total = np.zeros_like(Mach)

c_y_kr_5deg = np.zeros_like(Mach)
c_y_op_5deg = np.zeros_like(Mach)
c_y_izkor_5deg = np.zeros_like(Mach)
c_y_total_5deg = np.zeros_like(Mach)

K_aa_kr = np.zeros_like(Mach)
K_aa_op = np.zeros_like(Mach)
k_aa_kr = np.zeros_like(Mach)
k_aa_op = np.zeros_like(Mach)

eps_cp = np.zeros_like(Mach)
alpha_eff_op_5deg = np.zeros_like(Mach)
kappa_t1 = np.ones_like(Mach)
kappa_t2 = np.ones_like(Mach)

# Новые массивы для коэффициента щели и связанных параметров
k_sch = np.zeros_like(Mach)
n_I = np.zeros_like(Mach)
n_II = np.zeros_like(Mach)
eps_delta_sr = np.zeros_like(Mach)
c_y_delta_I = np.zeros_like(Mach)
c_y_delta_II = np.zeros_like(Mach)

# =============================================================================
# РАСЧЕТ ИЗОЛИРОВАННЫХ ХАРАКТЕРИСТИК
# =============================================================================

print("Расчет изолированных характеристик...")

for i in range(N):
    try:
        # Для крыла
        result_kr = AeroBDSM.get_c_y_alpha_IsP(Mach[i], lamb, bar_c, chi_05, zeta)
        c_ya_kr[i] = result_kr.Value / 57.3 if hasattr(result_kr, 'Value') else 0
        
        # Для оперения
        result_op = AeroBDSM.get_c_y_alpha_IsP(Mach[i], lamb_op, bar_c, chi_05, zeta)
        c_ya_op[i] = result_op.Value / 57.3 if hasattr(result_op, 'Value') else 0
        
        # Для корпуса
        result_izkor = AeroBDSM.get_c_y_alpha_NosCil_Par(Mach[i], lambda_Nos, lambda_Cil)
        c_ya_izkor[i] = result_izkor.Value / 57.3 if hasattr(result_izkor, 'Value') else 0
        
    except Exception as e:
        print(f"Ошибка при M={Mach[i]}: {e}")
        c_ya_kr[i] = c_ya_op[i] = c_ya_izkor[i] = 0

# =============================================================================
# РАСЧЕТ КОЭФФИЦИЕНТОВ ИНТЕРФЕРЕНЦИИ
# =============================================================================

print("Расчет коэффициентов интерференции и щели")
# Дополнительные параметры для расчета коэффициента щели (примерные значения)
l_razmah_r = 0.35  # размах рулей
l_razmah_kr = 0.64  # размах крыльев
S_r = S_op  # площадь рулей
S_kr = S_Kons  # площадь крыльев
S = S_op + S_Kons + (2 * math.pi * d * (l_f - d / 2) + 2 * math.pi * (d / 2)**2)  # полная площадь

# Предполагаемые коэффициенты (замените на реальные расчеты)
i_v = np.ones_like(Mach) * 0.1
c_y_alpha_is_kr = c_ya_kr * 57.3  # производная коэффициента подъемной силы крыла
k_delta_0_kr = np.ones_like(Mach) * 0.8
psi_eps = np.ones_like(Mach) * 1.0
z_v_otn = np.ones_like(Mach) * 0.5
K_delta_0_r = np.ones_like(Mach) * 0.9
k_t_I = np.ones_like(Mach) * 1.0
k_t_II = np.ones_like(Mach) * 1.0


for i in range(N):
    # Упрощенный расчет коэффициентов интерференции
    if Mach[i] < 1:
        # Для дозвуковых режимов
        K_aa_kr[i] = 1.1
        K_aa_op[i] = 1.05
        k_aa_kr[i] = 1.0
        k_aa_op[i] = 1.0
    else:
        # Для сверхзвуковых режимов
        K_aa_kr[i] = 1.2
        K_aa_op[i] = 1.1
        k_aa_kr[i] = 1.05
        k_aa_op[i] = 1.05
    
    # Расчет коэффициента щели
    if Mach[i] > 1.4:
        k_sch[i] = 0.85 + (Mach[i] - 1.4) * (1 - 0.85) / (2 - 1.4)
    else:
        k_sch[i] = 0.85

    n_I[i] = k_sch[i] * math.cos(chi_p)
    n_II[i] = k_sch[i] * math.cos(chi_p)
    
    # Расчет эффективности рулей с учетом коэффициента щели
    eps_delta_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_kr[i] * k_delta_0_kr[i] * n_I[i] * psi_eps[i]) / (2 * math.pi * z_v_otn[i] * l_razmah_kr * lambda_kr * K_aa_kr[i])
    
    # sqrt(2) - учет крестокрылого ЛА (вариант ++) стр. 195
    c_y_delta_I[i] = (c_ya_op[i] * 57.3 * K_delta_0_r[i] * n_I[i] * S_r / S * k_t_I[i] + 
                     c_ya_kr[i] * 57.3 * K_aa_kr[i] * eps_delta_sr[i] * S_kr / S * k_t_II[i])
    
    c_y_delta_II[i] = (c_ya_kr[i] * 57.3 * k_delta_0_kr[i] * S_kr / S * k_t_II[i])
# =============================================================================
# РАСЧЕТ СКОСА ПОТОКА
# =============================================================================

print("Расчет скоса потока...")

for i in range(N):
    try:
        # Упрощенный расчет скоса потока
        eps_cp[i] = 0.5 * (c_ya_op[i] / c_ya_kr[i]) if c_ya_kr[i] != 0 else 0
    except Exception as e:
        print(f"Ошибка расчета ε_cp при M={Mach[i]}: {e}")
        eps_cp[i] = 0

# =============================================================================
# РАСЧЕТ КОЭФФИЦИЕНТОВ ПОДЪЕМНОЙ СИЛЫ ПРИ α=5°
# =============================================================================

print("Расчет коэффициентов подъемной силы при α=5°...")

for i in range(N):
    if orientation == "XX":
        # Для XX-схемы - ИСПРАВЛЕНО: убрана распаковка
        c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[i], 5, A_kr)
        c_y_kr_5deg[i] = (K_aa_kr[i] / k_aa_kr[i]) * c_n_kr * math.cos(math.radians(5)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(5))
        
        c_n_op = calculate_wing_normal_coefficient(c_ya_op[i], 5, A_op)
        c_y_op_5deg[i] = (K_aa_op[i] / k_aa_op[i]) * c_n_op * math.cos(math.radians(5)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(5))
        alpha_eff_op_5deg[i] = 5  # Упрощенное значение
    else:
        # Для других схем
        c_y_kr_5deg[i] = calculate_wing_lift_coefficient(c_ya_kr[i], 5, A_kr)
        
        alpha_eff_rad = 1.0 * (math.radians(5) - math.radians(eps_cp[i]))
        alpha_eff_deg = math.degrees(alpha_eff_rad)
        c_n_op = (57.3 * c_ya_op[i] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                 A_op * (math.sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
        c_y_op_5deg[i] = (K_aa_op[i] / k_aa_op[i]) * c_n_op
        alpha_eff_op_5deg[i] = alpha_eff_deg
    
    # Коэффициент подъемной силы корпуса
    c_y_izkor_5deg[i] = calculate_fuselage_lift_coefficient(c_ya_izkor[i], 5)

# =============================================================================
# РАСЧЕТ СУММАРНЫХ ХАРАКТЕРИСТИК
# =============================================================================

print("Расчет суммарных характеристик...")

S_f = 2 * math.pi * d * (l_f - d / 2) + 2 * math.pi * (d / 2)**2
S = S_op + S_Kons + S_f
s_f_bar = safe_divide(S_f, S)
s_1_bar = safe_divide(S_op, S)
s_2_bar = safe_divide(S_Kons, S)

# Итоговые коэффициенты
c_ya_f = c_ya_izkor
c_ya_1_with_interference = c_ya_op * K_aa_op
c_ya_2_with_interference = c_ya_kr 

# Суммарный коэффициент производной
c_ya_total = (c_ya_f * s_f_bar + 
             c_ya_1_with_interference * s_1_bar + 
             c_ya_2_with_interference * s_2_bar)

# Суммарный коэффициент подъемной силы для угла 5°
c_y_total_5deg = (c_y_izkor_5deg * s_f_bar + 
                 c_y_op_5deg * s_1_bar + 
                 c_y_kr_5deg * s_2_bar)

# =============================================================================
# РАСЧЕТ КОЭФФИЦИЕНТОВ СОПРОТИВЛЕНИЯ
# =============================================================================

print("Расчет коэффициентов лобового сопротивления...")

angles = [0, 5, 10, 15, 20, 25]
c_x_data = {}

for angle_ataki in angles:
    c_x_total = np.zeros_like(Mach)
    c_x0_total = np.zeros_like(Mach)
    c_x_i_total = np.zeros_like(Mach)
    
    c_x_fuselage = np.zeros_like(Mach)
    c_x_wing = np.zeros_like(Mach)
    c_x_tail = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        # Расчет коэффициентов сопротивления при нулевой подъемной силе
        # Сопротивление корпуса
        Re = Mach[i] * 340 * l_f / nu
        c_f = AeroBDSM.get_c_f0(Re, 0).Value
        c_x_tr = 2 * c_f / 2 * S_f / S
        c_x_Nos = AeroBDSM.get_c_x0_p_Nos_Con(Mach[i], lambda_Nos).Value
        
        if Mach[i] <= 0.8:
            c_x_dn = (0.0155 / math.sqrt(lambda_Cil * c_f)) * eta_k * (S/S_f)
        else:
            c_x_dn = (0.0155 / math.sqrt(lambda_Cil * c_f)) * eta_k * (S/S_f)
        
        c_x0_f = c_x_tr + c_x_Nos + c_x_dn
        
        # Сопротивление несущих поверхностей
        c_x_p_I = 2 * c_f * eta_c
        
        # Волновое сопротивление
        if Mach[i] < 1:
            c_x_v_I = 0
            c_x_v_II = 0
        else:
            try:
                c_x_v_I = AeroBDSM.get_c_x0_w_IsP_Rmb(Mach[i], bar_c, zeta, chi_05, lambda_r).Value
            except Exception as e:
                print(f"Ошибка при расчете c_x_v_I для M={Mach[i]}: {e}")
                c_x_v_I = 0
            
            try:
                c_x_v_II = AeroBDSM.get_c_x0_w_IsP_Rmb(Mach[i], bar_c, zeta, chi_05, lambda_kr).Value
            except Exception as e:
                print(f"Ошибка при расчете c_x_v_II для M={Mach[i]}: {e}")
                c_x_v_II = 0
        
        c_x_0I = c_x_p_I + c_x_v_I
        c_x_0II = c_x_p_I + c_x_v_II
        
        # Расчет индуктивного сопротивления
        alpha_rad = math.radians(angle_ataki)
        delta_I_rad = math.radians(delta_I)
        delta_II_rad = math.radians(delta_II)
        
        # Сопротивление корпуса
        if Mach[i] >= 1:
            sigma = AeroBDSM.get_sigma_cp_Nos_Par(Mach[i], lambda_Nos).Value
        else:
            sigma = 0
            
        delta_c_x1 = 2 * sigma * (math.sin(alpha_rad))**2
        
        # Подъемная сила корпуса
        c_y_f = calculate_fuselage_lift_coefficient(c_ya_izkor[i], angle_ataki)
        
        c_x_i_f = (c_y_f + c_x0_f * math.sin(alpha_rad)) * math.tan(alpha_rad) + delta_c_x1 * math.cos(alpha_rad)
        
        # Сопротивление несущих поверхностей
        # Нормальные силы
        if orientation == "XX":
            c_nI = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op)
            c_nII = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr)
        else:
            c_nI = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op)
            c_nII = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr)
        
        # Индуктивное сопротивление несущих поверхностей
        term_I = (K_aa_op[i] - k_aa_op[i]) / k_aa_op[i] if k_aa_op[i] != 0 else 0
        c_x_i_I = c_nI * (math.sin(alpha_rad + delta_I_rad) + term_I * math.sin(alpha_rad) * math.cos(delta_I_rad))
        
        term_II = (K_aa_kr[i] - k_aa_kr[i]) / k_aa_kr[i] if k_aa_kr[i] != 0 else 0
        c_x_i_II = c_nII * (math.sin(alpha_rad + delta_II_rad) + term_II * math.sin(alpha_rad) * math.cos(delta_II_rad))
        
        # Суммарное индуктивное сопротивление
        c_x_i_total[i] = (c_x_i_f * s_f_bar + 
                         c_x_i_I * s_1_bar + 
                         c_x_i_II * s_2_bar)
        
        # Суммарный коэффициент сопротивления
        c_x0_total[i] = 1.05 * (c_x0_f * s_f_bar + c_x_0I * 2 * s_1_bar + c_x_0II * 2 * s_2_bar)
        c_x_total[i] = c_x0_total[i] + c_x_i_total[i]
        
        # Сохраняем компоненты для анализа
        c_x_fuselage[i] = c_x0_f * s_f_bar + c_x_i_f * s_f_bar
        c_x_wing[i] = c_x_0II * 2 * s_2_bar + c_x_i_II * s_2_bar
        c_x_tail[i] = c_x_0I * 2 * s_1_bar + c_x_i_I * s_1_bar
    
    c_x_data[angle_ataki] = {
        'total': c_x_total.copy(),
        'c_x0': c_x0_total.copy(),
        'c_x_i': c_x_i_total.copy(),
        'fuselage': c_x_fuselage.copy(),
        'wing': c_x_wing.copy(),
        'tail': c_x_tail.copy()
    }


# =============================================================================
# РАСЧЕТ MZ
# =============================================================================

# Работаем в связанной системе координат, начало координат совпадает с центром масс

# =============================================================================
# РАСЧЕТ КОЭФФИЦИЕНТА МОМЕНТА ТАНГАЖА MZ
# =============================================================================

print("Расчет коэффициента момента тангажа...")

# Определение selected_mach_numbers для использования в разделе MZ
selected_mach_numbers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
angles_for_analysis = list(range(-20, 21, 1))

# Параметры центров давления и фокусов (примерные значения, нужно уточнить)
# Координаты отсчитываются от носика аппарата
x_d_fuselage = 1.85  # координата центра давления корпуса
x_d_wing = 2.38      # координата центра давления крыльев  
x_d_tail = 3.57      # координата центра давления оперения

# Фокусы аппарата (примерные значения)
x_Fa = 2.1          # фокус по углу атаки
x_Fdelta_I = 3.1    # фокус по углу отклонения передних рулей
x_Fdelta_II = 2.8   # фокус по углу отклонения задних рулей

# Центр масс аппарата (центровка)
x_T = 2.023  # координата центра масс от носика

# Характерный линейный размер
L = 3.7  # используем длину корпуса как характерный размер

# Параметры отклонения рулей
delta_I_deg = 0  # угол отклонения передних рулей [град]
delta_II_deg = 10  # угол отклонения задних рулей [град]

# Момент при нулевой подъемной силе (примерное значение)
m_z0 = 0.0

# Массивы для результатов
m_z_data = {}
m_z_fuselage_data = {}
m_z_wing_data = {} 
m_z_tail_data = {}

# Расчет по формуле (5.17) через центры давления
print("Расчет m_z через центры давления...")

for angle_ataki in angles:
    m_z_total = np.zeros_like(Mach)
    m_z_fuselage = np.zeros_like(Mach)
    m_z_wing = np.zeros_like(Mach)
    m_z_tail = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        # Расчет коэффициентов нормальной силы для компонентов
        alpha_rad = math.radians(angle_ataki)
        
        # Корпус
        c_y1_fuselage = calculate_fuselage_lift_coefficient(c_ya_izkor[i], angle_ataki)
        
        # Крылья
        if orientation == "XX":
            c_n_wing = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr)
            c_y1_wing = (K_aa_kr[i] / k_aa_kr[i]) * c_n_wing * math.cos(alpha_rad) * math.sqrt(2)
        else:
            c_y1_wing = calculate_wing_lift_coefficient(c_ya_kr[i], angle_ataki, A_kr)
        
        # Оперение (рули)
        if orientation == "XX":
            c_n_tail = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op)
            c_y1_tail = (K_aa_op[i] / k_aa_op[i]) * c_n_tail * math.cos(alpha_rad) * math.sqrt(2)
        else:
            # Учет скоса потока для оперения
            alpha_eff_rad = 1.0 * (alpha_rad - math.radians(eps_cp[i]))
            alpha_eff_deg = math.degrees(alpha_eff_rad)
            c_n_tail = (57.3 * c_ya_op[i] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                       A_op * (math.sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y1_tail = (K_aa_op[i] / k_aa_op[i]) * c_n_tail
        
        # Моменты от каждой компоненты по формуле (5.17)
        m_z_fuselage[i] = c_y1_fuselage * (x_T - x_d_fuselage) / L
        m_z_wing[i] = -c_y1_wing * (x_T - x_d_wing) / L
        m_z_tail[i] = -c_y1_tail * (x_T - x_d_tail) / L
        
        # Суммарный момент
        m_z_total[i] = m_z_fuselage[i] + m_z_wing[i] + m_z_tail[i]
    
    m_z_data[angle_ataki] = {
        'total': m_z_total.copy(),
        'fuselage': m_z_fuselage.copy(),
        'wing': m_z_wing.copy(),
        'tail': m_z_tail.copy()
    }

# Расчет по формуле (5.23) через аэродинамические фокусы
print("Расчет m_z через аэродинамические фокусы...")

# Производные коэффициента подъемной силы (примерные значения)
c_y1_alpha = c_ya_total * 57.3  # производная по углу атаки [1/рад]
c_y1_delta_I = c_y_delta_I / 57.3  # производная по углу отклонения передних рулей
c_y1_delta_II = c_y_delta_II / 57.3  # производная по углу отклонения задних рулей

m_z_focus_data = {}

for angle_ataki in angles:
    m_z_focus = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        alpha_rad = math.radians(angle_ataki)
        delta_I_rad = math.radians(delta_I_deg)
        delta_II_rad = math.radians(delta_II_deg)
        
        # Расчет по формуле (5.23)
        m_z_focus[i] = (m_z0 + 
                       c_y1_alpha[i] * alpha_rad * (x_T - x_Fa) / L +
                       c_y1_delta_I[i] * delta_I_rad * (x_T - x_Fdelta_I) / L +
                       c_y1_delta_II[i] * delta_II_rad * (x_T - x_Fdelta_II) / L)
    
    m_z_focus_data[angle_ataki] = m_z_focus.copy()
    
# Расчет по формуле (5.20) через центры давления частей аппарата
print("Расчет m_z через центры давления частей аппарата...")

# Приведенные площади (уже рассчитаны ранее)
S_f_bar = s_f_bar  # относительная площадь корпуса
S_1_bar = s_1_bar  # относительная площадь оперения  
S_2_bar = s_2_bar  # относительная площадь крыльев

# Коэффициенты k_r (примерные значения)
k_r_I = 1.0  # для передних рулей
k_r_II = 1.0  # для задних рулей

m_z_parts_data = {}

for angle_ataki in angles:
    m_z_parts = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        # Расчет коэффициентов нормальной силы для частей
        alpha_rad = math.radians(angle_ataki)
        
        # Корпус
        c_y1_fuselage = calculate_fuselage_lift_coefficient(c_ya_izkor[i], angle_ataki)
        
        # Крылья
        c_n_wing = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr)
        c_y1_wing = c_n_wing * math.cos(alpha_rad)
        
        # Оперение
        if orientation == "XX":
            c_n_tail = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op)
            c_y1_tail = c_n_tail * math.cos(alpha_rad) * math.sqrt(2)
        else:
            alpha_eff_rad = 1.0 * (alpha_rad - math.radians(eps_cp[i]))
            c_n_tail = (57.3 * c_ya_op[i] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                       A_op * (math.sin(alpha_eff_rad))**2)
            c_y1_tail = c_n_tail
        
        # Расчет по формуле (5.20)
        term_fuselage = (c_y1_fuselage * S_f_bar) * (x_T - x_d_fuselage) / L
        term_wing = -(c_y1_wing * S_2_bar * k_r_II) * (x_T - x_d_wing) / L  
        term_tail = -(c_y1_tail * S_1_bar * k_r_I) * (x_T - x_d_tail) / L
        
        m_z_parts[i] = term_fuselage + term_wing + term_tail
    
    m_z_parts_data[angle_ataki] = m_z_parts.copy()


# =============================================================================
# РАСЧЕТ КОЭФФИЦИЕНТА МОМЕНТА ТАНГАЖА ЧЕРЕЗ АЭРОДИНАМИЧЕСКИЕ ФОКУСЫ ПО КОМПОНЕНТАМ
# =============================================================================

print("Расчет m_z через аэродинамические фокусы по компонентам...")

# Определим фокусы для каждой компоненты отдельно
x_Fa_wing = 1.7      # фокус крыльев по углу атаки
x_Fa_tail = 1.8      # фокус оперения по углу атаки  
x_Fa_fuselage = 1.6  # фокус корпуса по углу атаки

# Фокусы по отклонению рулей
x_Fdelta_I = 1.75    # фокус по углу отклонения передних рулей
x_Fdelta_II = 1.7    # фокус по углу отклонения задних рулей

# Производные коэффициента подъемной силы для каждой компоненты
c_y1_alpha_wing = c_ya_kr * 57.3 * S_2_bar      # производная крыльев [1/рад]
c_y1_alpha_tail = c_ya_op * 57.3 * S_1_bar      # производная оперения [1/рад] 
c_y1_alpha_fuselage = c_ya_izkor * 57.3 * S_f_bar  # производная корпуса [1/рад]

# Производные по отклонению рулей
c_y1_delta_I = c_y_delta_I / 57.3  # производная по углу отклонения передних рулей
c_y1_delta_II = c_y_delta_II / 57.3  # производная по углу отклонения задних рулей

# Массивы для результатов по компонентам
m_z_focus_data = {}  # Используем один словарь для всех данных

for angle_ataki in angles:
    m_z_focus_total = np.zeros_like(Mach)
    m_z_focus_wing = np.zeros_like(Mach)
    m_z_focus_tail = np.zeros_like(Mach)
    m_z_focus_fuselage = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        alpha_rad = math.radians(angle_ataki)
        delta_I_rad = math.radians(delta_I_deg)
        delta_II_rad = math.radians(delta_II_deg)
        
        # Моменты от каждой компоненты через фокусы
        # Крылья
        m_z_focus_wing[i] = -(c_y1_alpha_wing[i] * alpha_rad * (x_Fa_wing - x_T) / L +
                            c_y1_delta_II[i] * delta_II_rad * (x_Fdelta_II - x_T) / L)
        
        # Оперение
        m_z_focus_tail[i] = -(c_y1_alpha_tail[i] * alpha_rad * (x_Fa_tail - x_T) / L +
                            c_y1_delta_I[i] * delta_I_rad * (x_Fdelta_I - x_T) / L)
        
        # Корпус
        m_z_focus_fuselage[i] = -c_y1_alpha_fuselage[i] * alpha_rad * (x_Fa_fuselage - x_T) / L
        
        # Суммарный момент
        m_z_focus_total[i] = (m_z0 + 
                            m_z_focus_wing[i] + 
                            m_z_focus_tail[i] + 
                            m_z_focus_fuselage[i])
    

    m_z_focus_data[angle_ataki] = {
        'total': m_z_focus_total.copy(),
        'fuselage': m_z_focus_fuselage.copy(),
        'wing': m_z_focus_wing.copy(),
        'tail': m_z_focus_tail.copy()
    }

# =============================================================================
# ГРАФИКИ ДЛЯ MZ
# =============================================================================

print("Построение графиков для m_z...")

# График 1: Mz от числа Маха для разных углов атаки
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))

for angle in angles:
    if angle in m_z_data:
        data = m_z_data[angle]
        
        axes1[0, 0].plot(Mach, data['total'], linewidth=2, label=f'α={angle}°')
        axes1[0, 1].plot(Mach, data['fuselage'], linewidth=2, label=f'α={angle}°')
        axes1[1, 0].plot(Mach, data['wing'], linewidth=2, label=f'α={angle}°')
        axes1[1, 1].plot(Mach, data['tail'], linewidth=2, label=f'α={angle}°')

axes1[0, 0].set_xlabel('Число Маха')
axes1[0, 0].set_ylabel('m_z')
axes1[0, 0].set_title('Суммарный коэффициент момента тангажа\n(через центры давления)')
axes1[0, 0].grid(True, alpha=0.3)
axes1[0, 0].legend()

axes1[0, 1].set_xlabel('Число Маха')
axes1[0, 1].set_ylabel('m_z')
axes1[0, 1].set_title('Момент тангажа корпуса')
axes1[0, 1].grid(True, alpha=0.3)
axes1[0, 1].legend()

axes1[1, 0].set_xlabel('Число Маха')
axes1[1, 0].set_ylabel('m_z')
axes1[1, 0].set_title('Момент тангажа крыльев')
axes1[1, 0].grid(True, alpha=0.3)
axes1[1, 0].legend()

axes1[1, 1].set_xlabel('Число Маха')
axes1[1, 1].set_ylabel('m_z')
axes1[1, 1].set_title('Момент тангажа оперения')
axes1[1, 1].grid(True, alpha=0.3)
axes1[1, 1].legend()

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\mz_vs_mach_centers.png', dpi=300, bbox_inches='tight')
plt.show()

print("Построение графиков m_z от числа Маха через фокусы по компонентам...")

# График 1: моменты через фокусы
fig2, axes1 = plt.subplots(2, 2, figsize=(15, 12))

for angle in angles:
    if angle in m_z_focus_data:  # Используем правильный словарь
        data = m_z_focus_data[angle]
        axes1[0, 0].plot(Mach, data['total'], linewidth=2, label=f'α={angle}°')
        axes1[0, 1].plot(Mach, data['fuselage'], linewidth=2, label=f'α={angle}°')
        axes1[1, 0].plot(Mach, data['wing'], linewidth=2, label=f'α={angle}°')
        axes1[1, 1].plot(Mach, data['tail'], linewidth=2, label=f'α={angle}°')

axes1[0, 0].set_xlabel('Число Маха')
axes1[0, 0].set_ylabel('m_z')
axes1[0, 0].set_title('Суммарный коэффициент момента тангажа\n(через фокусы)')  # Исправлено название
axes1[0, 0].grid(True, alpha=0.3)
axes1[0, 0].legend()

axes1[0, 1].set_xlabel('Число Маха')
axes1[0, 1].set_ylabel('m_z')
axes1[0, 1].set_title('Момент тангажа корпуса\n(через фокусы)')  # Исправлено название
axes1[0, 1].grid(True, alpha=0.3)
axes1[0, 1].legend()

axes1[1, 0].set_xlabel('Число Маха')
axes1[1, 0].set_ylabel('m_z')
axes1[1, 0].set_title('Момент тангажа крыльев\n(через фокусы)')  # Исправлено название
axes1[1, 0].grid(True, alpha=0.3)
axes1[1, 0].legend()

axes1[1, 1].set_xlabel('Число Маха')
axes1[1, 1].set_ylabel('m_z')
axes1[1, 1].set_title('Момент тангажа оперения\n(через фокусы)')  # Исправлено название
axes1[1, 1].grid(True, alpha=0.3)
axes1[1, 1].legend()

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\mz_focus_vs_mach_components.png', dpi=300, bbox_inches='tight')  # Исправлено имя файла
plt.show()



# График 3: Mz от угла атаки для разных чисел Маха
print("Построение графиков m_z от угла атаки...")

# Создаем массивы для m_z от угла атаки
mz_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}
mz_wing_vs_alpha = {mach: [] for mach in selected_mach_numbers}
mz_tail_vs_alpha = {mach: [] for mach in selected_mach_numbers}
mz_fuselage_vs_alpha = {mach: [] for mach in selected_mach_numbers}

for mach in selected_mach_numbers:
    mach_idx = np.argmin(np.abs(Mach - mach))
    
    for alpha in angles_for_analysis:
        # Используем метод через центры давления
        alpha_rad = math.radians(alpha)
        
        # Корпус
        c_y1_fuselage = calculate_fuselage_lift_coefficient(c_ya_izkor[mach_idx], alpha)
        
        # Крылья
        if orientation == "XX":
            c_n_wing = calculate_wing_normal_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
            c_y1_wing = (K_aa_kr[mach_idx] / k_aa_kr[mach_idx]) * c_n_wing * math.cos(alpha_rad) * math.sqrt(2)
        else:
            c_y1_wing = calculate_wing_lift_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
        
        # Оперение
        if orientation == "XX":
            c_n_tail = calculate_wing_normal_coefficient(c_ya_op[mach_idx], alpha, A_op)
            c_y1_tail = (K_aa_op[mach_idx] / k_aa_op[mach_idx]) * c_n_tail * math.cos(alpha_rad) * math.sqrt(2)
        else:
            alpha_eff_rad = 1.0 * (alpha_rad - math.radians(eps_cp[mach_idx]))
            alpha_eff_deg = math.degrees(alpha_eff_rad)
            c_n_tail = (57.3 * c_ya_op[mach_idx] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                       A_op * (math.sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y1_tail = (K_aa_op[mach_idx] / k_aa_op[mach_idx]) * c_n_tail
        
        # Моменты
        mz_fuselage = c_y1_fuselage * (x_T - x_d_fuselage) / L
        mz_wing = -c_y1_wing * (x_T - x_d_wing) / L
        mz_tail = -c_y1_tail * (x_T - x_d_tail) / L
        
        mz_total = mz_fuselage + mz_wing + mz_tail
        
        mz_vs_alpha_data[mach].append(mz_total)
        mz_wing_vs_alpha[mach].append(mz_wing)
        mz_tail_vs_alpha[mach].append(mz_tail)
        mz_fuselage_vs_alpha[mach].append(mz_fuselage)

# Построение графиков
fig4, axes3 = plt.subplots(2, 2, figsize=(15, 12))

colors = plt.cm.viridis(np.linspace(0, 1, len(selected_mach_numbers)))

for i, mach in enumerate(selected_mach_numbers):
    axes3[0, 0].plot(angles_for_analysis, mz_vs_alpha_data[mach], 
                    color=colors[i], linewidth=2, label=f'M={mach}')
    axes3[0, 1].plot(angles_for_analysis, mz_wing_vs_alpha[mach],
                    color=colors[i], linewidth=2, label=f'M={mach}')
    axes3[1, 0].plot(angles_for_analysis, mz_tail_vs_alpha[mach],
                    color=colors[i], linewidth=2, label=f'M={mach}')
    axes3[1, 1].plot(angles_for_analysis, mz_fuselage_vs_alpha[mach],
                    color=colors[i], linewidth=2, label=f'M={mach}')

axes3[0, 0].set_xlabel('Угол атаки, град')
axes3[0, 0].set_ylabel('m_z')
axes3[0, 0].set_title('Суммарный коэффициент момента тангажа')
axes3[0, 0].grid(True, alpha=0.3)
axes3[0, 0].legend()
axes3[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axes3[0, 1].set_xlabel('Угол атаки, град')
axes3[0, 1].set_ylabel('m_z')
axes3[0, 1].set_title('Момент тангажа крыльев')
axes3[0, 1].grid(True, alpha=0.3)
axes3[0, 1].legend()
axes3[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axes3[1, 0].set_xlabel('Угол атаки, град')
axes3[1, 0].set_ylabel('m_z')
axes3[1, 0].set_title('Момент тангажа оперения')
axes3[1, 0].grid(True, alpha=0.3)
axes3[1, 0].legend()
axes3[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axes3[1, 1].set_xlabel('Угол атаки, град')
axes3[1, 1].set_ylabel('m_z')
axes3[1, 1].set_title('Момент тангажа корпуса')
axes3[1, 1].grid(True, alpha=0.3)
axes3[1, 1].legend()
axes3[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\mz_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.show()


print("Расчет коэффициента момента тангажа завершен!")

# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

print("Сохранение результатов...")

all_data = {
    'Mach': Mach,
    'c_ya_kr': c_ya_kr,
    'c_ya_op': c_ya_op,
    'c_ya_izkor': c_ya_izkor,
    'c_ya_total': c_ya_total,
    'c_y_total_5deg': c_y_total_5deg,
    'c_y_kr_5deg': c_y_kr_5deg,
    'c_y_op_5deg': c_y_op_5deg,
    'c_y_izkor_5deg': c_y_izkor_5deg,
    'K_aa_kr': K_aa_kr,
    'K_aa_op': K_aa_op,
    'eps_cp': eps_cp,
    'alpha_eff_op_5deg': alpha_eff_op_5deg,
    'kappa_t1': kappa_t1,
    'kappa_t2': kappa_t2,
    'c_x_data': c_x_data,
    # Новые данные
    'k_sch': k_sch,
    'n_I': n_I,
    'n_II': n_II,
    'eps_delta_sr': eps_delta_sr,
    'c_y_delta_I': c_y_delta_I,
    'c_y_delta_II': c_y_delta_II
}

# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================

print("Построение графиков...")

# График 1: Подъемная сила крыла и оперения
test_angles = [0, 5, 10, 15, 20, 25]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(test_angles)))

for i, angle in enumerate(test_angles):
    c_y_kr = np.zeros_like(Mach)
    c_y_op = np.zeros_like(Mach)
    
    for j, mach in enumerate(Mach):
        if orientation == "XX":
            # ИСПРАВЛЕНО: убрана распаковка
            c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[j], angle, A_kr)
            c_y_kr[j] = (K_aa_kr[j] / k_aa_kr[j]) * c_n_kr * math.cos(math.radians(angle)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(angle))
            
            c_n_op = calculate_wing_normal_coefficient(c_ya_op[j], angle, A_op)
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op * math.cos(math.radians(angle)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(angle))
        else:
            c_y_kr[j] = calculate_wing_lift_coefficient(c_ya_kr[j], angle, A_kr)
            
            alpha_eff_rad = 1.0 * (math.radians(angle) - math.radians(eps_cp[j]))
            alpha_eff_deg = math.degrees(alpha_eff_rad)
            c_n_op = (57.3 * c_ya_op[j] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                     A_op * (math.sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op
    
    ax1.plot(Mach, c_y_kr, color=colors[i], linewidth=2, label=fr'$\alpha = {angle}^\circ$')
    ax2.plot(Mach, c_y_op, color=colors[i], linewidth=2, label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('Число Маха (M)')
ax1.set_ylabel('Коэффициент подъемной силы (C_y)')
ax1.set_title(f'Коэффициент подъемной силы крыла\n({orientation}-образная ориентация)')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_xlabel('Число Маха (M)')
ax2.set_ylabel('Коэффициент подъемной силы (C_y)')
ax2.set_title(f'Коэффициент подъемной силы оперения\n({orientation}-образная ориентация)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(f'C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\wing_tail_lift_coefficients_{orientation}.png', dpi=300, bbox_inches='tight')
plt.show()

# График 2: Нормальная сила
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for i, alpha in enumerate(test_angles):
    c_n_kr = np.zeros_like(Mach)
    c_n_op = np.zeros_like(Mach)
    c_n_izkor = np.zeros_like(Mach)
    
    for j, mach in enumerate(Mach):
        c_n_kr[j] = calculate_wing_normal_coefficient(c_ya_kr[j], alpha, A_kr)
        c_n_op[j] = calculate_wing_normal_coefficient(c_ya_op[j], alpha, A_op)
        c_n_izkor[j] = calculate_fuselage_lift_coefficient(c_ya_izkor[j], alpha)
    
    axes2[0].plot(Mach, c_n_kr, color=colors[i], linewidth=2, label=f'α={alpha}°')
    axes2[1].plot(Mach, c_n_op, color=colors[i], linewidth=2, label=f'α={alpha}°')
    axes2[2].plot(Mach, c_n_izkor, color=colors[i], linewidth=2, label=f'α={alpha}°')

axes2[0].set_xlabel('Число Маха')
axes2[0].set_ylabel('C_n')
axes2[0].set_title('Коэффициент нормальной силы крыла')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].set_xlabel('Число Маха')
axes2[1].set_ylabel('C_n')
axes2[1].set_title('Коэффициент нормальной силы оперения')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

axes2[2].set_xlabel('Число Маха')
axes2[2].set_ylabel('C_n')
axes2[2].set_title('Коэффициент нормальной силы корпуса')
axes2[2].legend()
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\normal_force_vs_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Подъемная сила всех компонентов
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

for i, alpha in enumerate(test_angles):
    c_y_kr = np.zeros_like(Mach)
    c_y_op = np.zeros_like(Mach)
    c_y_izkor = np.zeros_like(Mach)
    
    for j, mach in enumerate(Mach):
        if orientation == "XX":
            # ИСПРАВЛЕНО: убрана распаковка
            c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[j], alpha, A_kr)
            c_y_kr[j] = (K_aa_kr[j] / k_aa_kr[j]) * c_n_kr * math.cos(math.radians(alpha)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(alpha))
            
            c_n_op = calculate_wing_normal_coefficient(c_ya_op[j], alpha, A_op)
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op * math.cos(math.radians(alpha)) * math.sqrt(2) - 2 * 0.01 * math.sin(math.radians(alpha))
        else:
            c_y_kr[j] = calculate_wing_lift_coefficient(c_ya_kr[j], alpha, A_kr)
            
            alpha_eff_rad = 1.0 * (math.radians(alpha) - math.radians(eps_cp[j]))
            alpha_eff_deg = math.degrees(alpha_eff_rad)
            c_n_op = (57.3 * c_ya_op[j] * math.sin(alpha_eff_rad) * math.cos(alpha_eff_rad) + 
                     A_op * (math.sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op
        
        c_y_izkor[j] = calculate_fuselage_lift_coefficient(c_ya_izkor[j], alpha)
    
    axes3[0].plot(Mach, c_y_kr, color=colors[i], linewidth=2, label=f'α={alpha}°')
    axes3[1].plot(Mach, c_y_op, color=colors[i], linewidth=2, label=f'α={alpha}°')
    axes3[2].plot(Mach, c_y_izkor, color=colors[i], linewidth=2, label=f'α={alpha}°')

axes3[0].set_xlabel('Число Маха')
axes3[0].set_ylabel('C_y')
axes3[0].set_title('Коэффициент подъемной силы крыла')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

axes3[1].set_xlabel('Число Маха')
axes3[1].set_ylabel('C_y')
axes3[1].set_title('Коэффициент подъемной силы оперения')
axes3[1].legend()
axes3[1].grid(True, alpha=0.3)

axes3[2].set_xlabel('Число Маха')
axes3[2].set_ylabel('C_y')
axes3[2].set_title('Коэффициент подъемной силы корпуса')
axes3[2].legend()
axes3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\lift_force_vs_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 4: Сопротивление
fig4, axes4 = plt.subplots(2, 2, figsize=(15, 12))

for i, alpha in enumerate(angles):
    if alpha in c_x_data:
        data = c_x_data[alpha]
        
        axes4[0, 0].plot(Mach, data['total'], color=colors[i], linewidth=2, label=f'α={alpha}°')
        axes4[0, 1].plot(Mach, data['fuselage'], color=colors[i], linewidth=2, label=f'α={alpha}°')
        axes4[1, 0].plot(Mach, data['wing'], color=colors[i], linewidth=2, label=f'α={alpha}°')
        axes4[1, 1].plot(Mach, data['tail'], color=colors[i], linewidth=2, label=f'α={alpha}°')

axes4[0, 0].set_xlabel('Число Маха')
axes4[0, 0].set_ylabel('C_x')
axes4[0, 0].set_title('Суммарный коэффициент лобового сопротивления')
axes4[0, 0].legend()
axes4[0, 0].grid(True, alpha=0.3)

axes4[0, 1].set_xlabel('Число Маха')
axes4[0, 1].set_ylabel('C_x')
axes4[0, 1].set_title('Коэффициент сопротивления корпуса')
axes4[0, 1].legend()
axes4[0, 1].grid(True, alpha=0.3)

axes4[1, 0].set_xlabel('Число Маха')
axes4[1, 0].set_ylabel('C_x')
axes4[1, 0].set_title('Коэффициент сопротивления крыла')
axes4[1, 0].legend()
axes4[1, 0].grid(True, alpha=0.3)

axes4[1, 1].set_xlabel('Число Маха')
axes4[1, 1].set_ylabel('C_x')
axes4[1, 1].set_title('Коэффициент сопротивления оперения')
axes4[1, 1].legend()
axes4[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\drag_coefficients_vs_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ: cn, cy, cx ОТ УГЛА АТАКИ ПРИ РАЗНЫХ ЧИСЛАХ МАХА
# =============================================================================

print("Построение графиков коэффициентов от угла атаки...")

# Выбранные числа Маха для анализа
selected_mach_numbers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# Углы атаки от -20 до +20 градусов с шагом 1 градус
angles_for_analysis = list(range(-20, 21, 1))

# Создаем массивы для хранения данных
cn_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}
cy_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}
cx_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}

# Функция для интерполяции коэффициента сопротивления с учетом симметрии
def interpolate_cx(alpha, mach_idx, c_x_data):
    """Интерполяция коэффициента сопротивления для произвольного угла атаки с учетом симметрии"""
    # Используем симметрию: сопротивление одинаково для положительных и отрицательных углов
    alpha_positive = abs(alpha)

    if alpha_positive in c_x_data:
        return c_x_data[alpha_positive]['total'][mach_idx]

    available_angles = sorted(c_x_data.keys())
    
    # Если alpha_positive больше максимального из доступных, экстраполируем от последних двух точек
    if alpha_positive > available_angles[-1]:
        alpha1, alpha2 = available_angles[-2], available_angles[-1]
        cx1 = c_x_data[alpha1]['total'][mach_idx]
        cx2 = c_x_data[alpha2]['total'][mach_idx]
        slope = (cx2 - cx1) / (alpha2 - alpha1)
        return cx2 + slope * (alpha_positive - alpha2)
    
    # Если alpha_positive меньше минимального, но больше нуля, то используем минимальный угол (это 0)
    if alpha_positive < available_angles[0]:
        return c_x_data[available_angles[0]]['total'][mach_idx]
    
    # Иначе интерполируем между двумя ближайшими углами
    for i in range(len(available_angles) - 1):
        if available_angles[i] < alpha_positive < available_angles[i + 1]:
            alpha_low = available_angles[i]
            alpha_high = available_angles[i + 1]
            cx_low = c_x_data[alpha_low]['total'][mach_idx]
            cx_high = c_x_data[alpha_high]['total'][mach_idx]
            fraction = (alpha_positive - alpha_low) / (alpha_high - alpha_low)
            return cx_low + fraction * (cx_high - cx_low)
    
    return 0

# Заполняем данные для каждого числа Маха
for mach in selected_mach_numbers:
    # Находим ближайший индекс в массиве Mach
    mach_idx = np.argmin(np.abs(Mach - mach))
    actual_mach = Mach[mach_idx]
    
    print(f"Расчет для M = {actual_mach:.2f}")
    
    for alpha in angles_for_analysis:
        # Расчет коэффициента нормальной силы с использованием исправленных функций
        c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
        c_n_op = calculate_wing_normal_coefficient(c_ya_op[mach_idx], alpha, A_op)
        
        # Для корпуса - линейная зависимость
        c_n_izkor = c_ya_izkor[mach_idx] * alpha
        
        # Суммарный коэффициент нормальной силы с учетом интерференции
        if orientation == "XX":
            # Для XX-схемы учитываем интерференцию
            cn_total = (c_n_izkor * s_f_bar + 
                       c_n_op * s_1_bar * (K_aa_op[mach_idx] / k_aa_op[mach_idx]) + 
                       c_n_kr * s_2_bar * (K_aa_kr[mach_idx] / k_aa_kr[mach_idx]))
        else:
            cn_total = (c_n_izkor * s_f_bar + 
                       c_n_op * s_1_bar + 
                       c_n_kr * s_2_bar)
        
        # Расчет коэффициента подъемной силы
        c_y_kr = calculate_wing_lift_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
        c_y_op = calculate_wing_lift_coefficient(c_ya_op[mach_idx], alpha, A_op)
        c_y_izkor = c_ya_izkor[mach_idx] * alpha
        
        # Суммарный коэффициент подъемной силы с учетом интерференции
        if orientation == "XX":
            cy_total = (c_y_izkor * s_f_bar + 
                       c_y_op * s_1_bar * (K_aa_op[mach_idx] / k_aa_op[mach_idx]) + 
                       c_y_kr * s_2_bar * (K_aa_kr[mach_idx] / k_aa_kr[mach_idx]))
        else:
            cy_total = (c_y_izkor * s_f_bar + 
                       c_y_op * s_1_bar + 
                       c_y_kr * s_2_bar)
        
        # Расчет коэффициента сопротивления с интерполяцией и симметрией
        cx_total = interpolate_cx(alpha, mach_idx, c_x_data)
        
        # Сохраняем результаты
        cn_vs_alpha_data[mach].append(cn_total)
        cy_vs_alpha_data[mach].append(cy_total)
        cx_vs_alpha_data[mach].append(cx_total)

# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ ОТ УГЛА АТАКИ
# =============================================================================

print("Построение графиков от угла атаки...")

# График 1: Коэффициент нормальной силы от угла атаки
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_mach_numbers)))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cn_vs_alpha_data[mach], 
             color=colors[i], linewidth=2, 
             label=f'M = {mach}')

plt.xlabel('Угол атаки, град', fontsize=12)
plt.ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
plt.title('Зависимость коэффициента нормальной силы от угла атаки\nпри различных числах Маха', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(min(angles_for_analysis), max(angles_for_analysis))
plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\cn_vs_alpha_different_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 2: Коэффициент подъемной силы от угла атаки
plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cy_vs_alpha_data[mach], 
             color=colors[i], linewidth=2, 
             label=f'M = {mach}')

plt.xlabel('Угол атаки, град', fontsize=12)
plt.ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
plt.title('Зависимость коэффициента подъемной силы от угла атаки\nпри различных числах Маха', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(min(angles_for_analysis), max(angles_for_analysis))
plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\cy_vs_alpha_different_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Коэффициент лобового сопротивления от угла атаки
plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cx_vs_alpha_data[mach], 
             color=colors[i], linewidth=2, 
             label=f'M = {mach}')

plt.xlabel('Угол атаки, град', fontsize=12)
plt.ylabel('Коэффициент лобового сопротивления (C_x)', fontsize=12)
plt.title('Зависимость коэффициента лобового сопротивления от угла атаки\nпри различных числах Маха', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(min(angles_for_analysis), max(angles_for_analysis))
plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\cx_vs_alpha_different_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 4: Аэродинамическое качество K = Cy/Cx от угла атаки
plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    k_values = []
    for j in range(len(angles_for_analysis)):
        if abs(cx_vs_alpha_data[mach][j]) > 1e-6:  # Избегаем деления на ноль
            k_values.append(cy_vs_alpha_data[mach][j] / cx_vs_alpha_data[mach][j])
        else:
            k_values.append(0)
    
    plt.plot(angles_for_analysis, k_values, 
             color=colors[i], linewidth=2, 
             label=f'M = {mach}')

plt.xlabel('Угол атаки, град', fontsize=12)
plt.ylabel('Аэродинамическое качество (K = C_y/C_x)', fontsize=12)
plt.title('Зависимость аэродинамического качества от угла атаки\nпри различных числах Маха', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(min(angles_for_analysis), max(angles_for_analysis))
plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\k_vs_alpha_different_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 5: Поляра летательного аппарата (Cy от Cx) для разных чисел Маха
plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(cx_vs_alpha_data[mach], cy_vs_alpha_data[mach], 
             color=colors[i], linewidth=2, marker='o', markersize=3,
             label=f'M = {mach}')
    
    # Добавляем маркеры для некоторых характерных углов
    characteristic_angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    for angle in characteristic_angles:
        if angle in angles_for_analysis:
            idx = angles_for_analysis.index(angle)
            plt.plot(cx_vs_alpha_data[mach][idx], cy_vs_alpha_data[mach][idx], 
                    'ko', markersize=5)
            plt.annotate(f'{angle}°', 
                        (cx_vs_alpha_data[mach][idx], cy_vs_alpha_data[mach][idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Коэффициент лобового сопротивления (C_x)', fontsize=12)
plt.ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
plt.title('Поляра летательного аппарата при различных числах Маха', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\polar_diagram_different_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# График 6: Сводный график всех коэффициентов для M=2.0 (пример)
mach_example = 2.0
if mach_example in selected_mach_numbers:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Верхний график - Cy и Cn
    ax1.plot(angles_for_analysis, cy_vs_alpha_data[mach_example], 
             'b-', linewidth=2, label='C_y')
    ax1.plot(angles_for_analysis, cn_vs_alpha_data[mach_example], 
             'r-', linewidth=2, label='C_n')
    ax1.set_xlabel('Угол атаки, град', fontsize=12)
    ax1.set_ylabel('Коэффициенты C_y, C_n', fontsize=12)
    ax1.set_title(f'Коэффициенты подъемной и нормальной силы (M = {mach_example})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    # Нижний график - Cx и K
    ax2.plot(angles_for_analysis, cx_vs_alpha_data[mach_example], 
             'g-', linewidth=2, label='C_x')
    
    # Расчет аэродинамического качества
    k_values = []
    for j in range(len(angles_for_analysis)):
        if abs(cx_vs_alpha_data[mach_example][j]) > 1e-6:
            k_values.append(cy_vs_alpha_data[mach_example][j] / cx_vs_alpha_data[mach_example][j])
        else:
            k_values.append(0)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(angles_for_analysis, k_values, 
                  'm-', linewidth=2, label='K = C_y/C_x')
    
    ax2.set_xlabel('Угол атаки, град', fontsize=12)
    ax2.set_ylabel('Коэффициент сопротивления C_x', fontsize=12, color='g')
    ax2_twin.set_ylabel('Аэродинамическое качество K', fontsize=12, color='m')
    ax2.set_title(f'Коэффициент сопротивления и аэродинамическое качество (M = {mach_example})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\summary_coefficients_M2.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# ГРАФИКИ cn, cy, cx ОТ УГЛА АТАКИ ДЛЯ КОМПОНЕНТОВ
# =============================================================================

print("Построение графиков коэффициентов от угла атаки для компонентов...")

# Выбранные числа Маха для анализа
selected_mach_numbers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# Углы атаки от -20 до +20 градусов с шагом 1 градус
angles_for_analysis = list(range(-20, 21, 1))

# Создаем массивы для хранения данных по компонентам
cn_wing_data = {mach: [] for mach in selected_mach_numbers}
cy_wing_data = {mach: [] for mach in selected_mach_numbers}
cx_wing_data = {mach: [] for mach in selected_mach_numbers}

cn_tail_data = {mach: [] for mach in selected_mach_numbers}
cy_tail_data = {mach: [] for mach in selected_mach_numbers}
cx_tail_data = {mach: [] for mach in selected_mach_numbers}

cn_fuselage_data = {mach: [] for mach in selected_mach_numbers}
cy_fuselage_data = {mach: [] for mach in selected_mach_numbers}
cx_fuselage_data = {mach: [] for mach in selected_mach_numbers}

# Функция для интерполяции коэффициента сопротивления компонента
def interpolate_cx_component(alpha, mach_idx, c_x_data, component):
    """Интерполяция коэффициента сопротивления для компонента"""
    # Используем симметрию: сопротивление одинаково для положительных и отрицательных углов
    alpha_positive = abs(alpha)

    if alpha_positive in c_x_data:
        return c_x_data[alpha_positive][component][mach_idx]

    available_angles = sorted(c_x_data.keys())
    
    # Если alpha_positive больше максимального из доступных, экстраполируем от последних двух точек
    if alpha_positive > available_angles[-1]:
        alpha1, alpha2 = available_angles[-2], available_angles[-1]
        cx1 = c_x_data[alpha1][component][mach_idx]
        cx2 = c_x_data[alpha2][component][mach_idx]
        slope = (cx2 - cx1) / (alpha2 - alpha1)
        return cx2 + slope * (alpha_positive - alpha2)
    
    # Если alpha_positive меньше минимального, но больше нуля, то используем минимальный угол
    if alpha_positive < available_angles[0]:
        return c_x_data[available_angles[0]][component][mach_idx]
    
    # Иначе интерполируем между двумя ближайшими углами
    for i in range(len(available_angles) - 1):
        if available_angles[i] < alpha_positive < available_angles[i + 1]:
            alpha_low = available_angles[i]
            alpha_high = available_angles[i + 1]
            cx_low = c_x_data[alpha_low][component][mach_idx]
            cx_high = c_x_data[alpha_high][component][mach_idx]
            fraction = (alpha_positive - alpha_low) / (alpha_high - alpha_low)
            return cx_low + fraction * (cx_high - cx_low)
    
    return 0

# Заполняем данные для каждого числа Маха
for mach in selected_mach_numbers:
    # Находим ближайший индекс в массиве Mach
    mach_idx = np.argmin(np.abs(Mach - mach))
    actual_mach = Mach[mach_idx]
    
    print(f"Расчет компонентов для M = {actual_mach:.2f}")
    
    for alpha in angles_for_analysis:
        # Расчет коэффициентов для крыла
        cn_wing = calculate_wing_normal_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
        cy_wing = calculate_wing_lift_coefficient(c_ya_kr[mach_idx], alpha, A_kr)
        cx_wing = interpolate_cx_component(alpha, mach_idx, c_x_data, 'wing')
        
        # Расчет коэффициентов для оперения (рулей)
        cn_tail = calculate_wing_normal_coefficient(c_ya_op[mach_idx], alpha, A_op)
        cy_tail = calculate_wing_lift_coefficient(c_ya_op[mach_idx], alpha, A_op)
        cx_tail = interpolate_cx_component(alpha, mach_idx, c_x_data, 'tail')
        
        # Расчет коэффициентов для корпуса
        cn_fuselage = calculate_fuselage_lift_coefficient(c_ya_izkor[mach_idx], alpha)
        cy_fuselage = cn_fuselage  # Для корпуса подъемная сила равна нормальной
        cx_fuselage = interpolate_cx_component(alpha, mach_idx, c_x_data, 'fuselage')
        
        # Сохраняем результаты
        cn_wing_data[mach].append(cn_wing)
        cy_wing_data[mach].append(cy_wing)
        cx_wing_data[mach].append(cx_wing)
        
        cn_tail_data[mach].append(cn_tail)
        cy_tail_data[mach].append(cy_tail)
        cx_tail_data[mach].append(cx_tail)
        
        cn_fuselage_data[mach].append(cn_fuselage)
        cy_fuselage_data[mach].append(cy_fuselage)
        cx_fuselage_data[mach].append(cx_fuselage)

# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ КОМПОНЕНТОВ
# =============================================================================

print("Построение графиков для компонентов...")

colors = plt.cm.viridis(np.linspace(0, 1, len(selected_mach_numbers)))

# График 1: Коэффициенты для крыла
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

for i, mach in enumerate(selected_mach_numbers):
    axes1[0].plot(angles_for_analysis, cn_wing_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes1[1].plot(angles_for_analysis, cy_wing_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes1[2].plot(angles_for_analysis, cx_wing_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')

axes1[0].set_xlabel('Угол атаки, град', fontsize=12)
axes1[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes1[0].set_title('Коэффициент нормальной силы крыла', fontsize=14)
axes1[0].grid(True, alpha=0.3)
axes1[0].legend(fontsize=10)
axes1[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes1[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes1[0].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes1[1].set_xlabel('Угол атаки, град', fontsize=12)
axes1[1].set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
axes1[1].set_title('Коэффициент подъемной силы крыла', fontsize=14)
axes1[1].grid(True, alpha=0.3)
axes1[1].legend(fontsize=10)
axes1[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes1[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes1[1].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes1[2].set_xlabel('Угол атаки, град', fontsize=12)
axes1[2].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes1[2].set_title('Коэффициент сопротивления крыла', fontsize=14)
axes1[2].grid(True, alpha=0.3)
axes1[2].legend(fontsize=10)
axes1[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes1[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes1[2].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\wing_coefficients_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.show()

# График 2: Коэффициенты для оперения (рулей)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for i, mach in enumerate(selected_mach_numbers):
    axes2[0].plot(angles_for_analysis, cn_tail_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes2[1].plot(angles_for_analysis, cy_tail_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes2[2].plot(angles_for_analysis, cx_tail_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')

axes2[0].set_xlabel('Угол атаки, град', fontsize=12)
axes2[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes2[0].set_title('Коэффициент нормальной силы оперения', fontsize=14)
axes2[0].grid(True, alpha=0.3)
axes2[0].legend(fontsize=10)
axes2[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes2[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes2[0].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes2[1].set_xlabel('Угол атаки, град', fontsize=12)
axes2[1].set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
axes2[1].set_title('Коэффициент подъемной силы оперения', fontsize=14)
axes2[1].grid(True, alpha=0.3)
axes2[1].legend(fontsize=10)
axes2[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes2[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes2[1].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes2[2].set_xlabel('Угол атаки, град', fontsize=12)
axes2[2].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes2[2].set_title('Коэффициент сопротивления оперения', fontsize=14)
axes2[2].grid(True, alpha=0.3)
axes2[2].legend(fontsize=10)
axes2[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes2[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes2[2].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\tail_coefficients_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Коэффициенты для корпуса
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

for i, mach in enumerate(selected_mach_numbers):
    axes3[0].plot(angles_for_analysis, cn_fuselage_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes3[1].plot(angles_for_analysis, cy_fuselage_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')
    axes3[2].plot(angles_for_analysis, cx_fuselage_data[mach], 
                  color=colors[i], linewidth=2, label=f'M = {mach}')

axes3[0].set_xlabel('Угол атаки, град', fontsize=12)
axes3[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes3[0].set_title('Коэффициент нормальной силы корпуса', fontsize=14)
axes3[0].grid(True, alpha=0.3)
axes3[0].legend(fontsize=10)
axes3[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes3[0].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes3[1].set_xlabel('Угол атаки, град', fontsize=12)
axes3[1].set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
axes3[1].set_title('Коэффициент подъемной силы корпуса', fontsize=14)
axes3[1].grid(True, alpha=0.3)
axes3[1].legend(fontsize=10)
axes3[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes3[1].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

axes3[2].set_xlabel('Угол атаки, град', fontsize=12)
axes3[2].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes3[2].set_title('Коэффициент сопротивления корпуса', fontsize=14)
axes3[2].grid(True, alpha=0.3)
axes3[2].legend(fontsize=10)
axes3[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes3[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes3[2].set_xlim(min(angles_for_analysis), max(angles_for_analysis))

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\fuselage_coefficients_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.show()

# График 4: Сравнительные графики для всех компонентов при M=2.0
mach_example = 2.0
if mach_example in selected_mach_numbers:
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Находим индекс для M=2.0
    mach_idx = selected_mach_numbers.index(mach_example)
    
    # График нормальной силы
    axes4[0].plot(angles_for_analysis, cn_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes4[0].plot(angles_for_analysis, cn_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes4[0].plot(angles_for_analysis, cn_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes4[0].set_xlabel('Угол атаки, град', fontsize=12)
    axes4[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
    axes4[0].set_title(f'Сравнение нормальной силы компонентов (M = {mach_example})', fontsize=14)
    axes4[0].grid(True, alpha=0.3)
    axes4[0].legend(fontsize=10)
    axes4[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes4[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes4[0].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    # График подъемной силы
    axes4[1].plot(angles_for_analysis, cy_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes4[1].plot(angles_for_analysis, cy_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes4[1].plot(angles_for_analysis, cy_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes4[1].set_xlabel('Угол атаки, град', fontsize=12)
    axes4[1].set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
    axes4[1].set_title(f'Сравнение подъемной силы компонентов (M = {mach_example})', fontsize=14)
    axes4[1].grid(True, alpha=0.3)
    axes4[1].legend(fontsize=10)
    axes4[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes4[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes4[1].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    # График сопротивления
    axes4[2].plot(angles_for_analysis, cx_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes4[2].plot(angles_for_analysis, cx_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes4[2].plot(angles_for_analysis, cx_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes4[2].set_xlabel('Угол атаки, град', fontsize=12)
    axes4[2].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
    axes4[2].set_title(f'Сравнение сопротивления компонентов (M = {mach_example})', fontsize=14)
    axes4[2].grid(True, alpha=0.3)
    axes4[2].legend(fontsize=10)
    axes4[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes4[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes4[2].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\components_comparison_M2.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Графики для компонентов успешно построены и сохранены!")