import AeroBDSM
from math import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"

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

print('S_op = ', S_op)

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
A_kr_const = 0.8
A_op_const = 0.6

# Новые параметры для коэффициента щели
chi_p = radians(45)  # угол стреловидности

# Параметры для расчета фюзеляжа как во второй программе
lambda_Nos_cx = lambda_Nos  # Удлинение для расчета cx носовой части

# Площадь поверхности фюзеляжа (как во второй программе)
S_f = pi * d * (l_f - d / 2) + 2 * pi * (d / 2)**2

# Характерная площадь (площадь миделя) - как во второй программе
S_char = pi * d**2 / 4

# Суммарная площадь для приведения коэффициентов сопротивления
S_cx_total = S_op + S_Kons + S_f

print(f"S_f = {S_f:.4f}")
print(f"S_char = {S_char:.4f}")
print(f"S_cx_total = {S_cx_total:.4f}")

# Параметры положения
D_otn = d / l_raszmah
L_1_otn_kr = L1 / l_f
L_1_otn_r = L1_op / l_f
L_xv_kr = xb
L_xv_r = xb_op

# Центр масс и фокусы
x_T = 2.023
x_d_fuselage = 1.85
x_d_wing = 2.38
x_d_tail = 3.57
x_Fa = 2.1
x_Fdelta_I = 3.1
x_Fdelta_II = 2.8
L = 3.7

# Углы отклонения рулей
delta_I_deg = 0
delta_II_deg = 10

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def safe_divide(a, b):
    return a / b if b != 0 else 0

def calculate_wing_normal_coefficient(c_ya, alpha_deg, A):
    alpha_rad = radians(alpha_deg)
    term1 = 57.3 * c_ya * sin(alpha_rad) * cos(alpha_rad)
    term2 = A * (sin(alpha_rad))**2 * (1 if alpha_deg >= 0 else -1)
    return term1 + term2

def calculate_wing_lift_coefficient(c_ya, alpha_deg, A):
    alpha_rad = radians(alpha_deg)
    c_n = calculate_wing_normal_coefficient(c_ya, alpha_deg, A)
    return c_n * cos(alpha_rad)

def calculate_fuselage_lift_coefficient_simple(c_ya, alpha_deg):
    """Упрощенная версия для расчета сопротивления, как во второй программе"""
    return c_ya * alpha_deg

def calculate_fuselage_lift_coefficient(M, alpha_deg, c_y_alpha_is_f_val):
    """Полная версия для расчета подъемной силы"""
    try:
        alpha_rad = radians(alpha_deg)
        alpha_abs_rad = radians(abs(alpha_deg))
        
        kappa_alpha_val = 1 - 0.45 * (1 - exp(-0.06 * M**2)) * (1 - exp(-0.12 * alpha_abs_rad))
        
        M_y_val = M * sin(alpha_abs_rad)
        c_y_perp_Cil_val = AeroBDSM.get_c_yPerp_Cil(M_y_val)
        
        S_bok = pi * d**2 / 8 + pi * d * (l_f - d/2)
        term1 = 57.3 * c_y_alpha_is_f_val * kappa_alpha_val * sin(alpha_rad) * cos(alpha_rad)
        term2 = 4 * S_bok * c_y_perp_Cil_val * (sin(alpha_abs_rad))**2 * (1 if alpha_deg >= 0 else -1) / (pi * d**2)
        
        return term1 + term2
    except Exception as e:
        print(f"Ошибка в calculate_fuselage_lift_coefficient: M={M}, alpha={alpha_deg}, error={e}")
        return 0

def interpolate_cx(alpha, mach_idx, c_x_data):
    alpha_positive = abs(alpha)
    if alpha_positive in c_x_data:
        return c_x_data[alpha_positive]['total'][mach_idx]
    
    available_angles = sorted(c_x_data.keys())
    if alpha_positive > available_angles[-1]:
        alpha1, alpha2 = available_angles[-2], available_angles[-1]
        cx1 = c_x_data[alpha1]['total'][mach_idx]
        cx2 = c_x_data[alpha2]['total'][mach_idx]
        slope = (cx2 - cx1) / (alpha2 - alpha1)
        return cx2 + slope * (alpha_positive - alpha2)
    
    if alpha_positive < available_angles[0]:
        return c_x_data[available_angles[0]]['total'][mach_idx]
    
    for i in range(len(available_angles) - 1):
        if available_angles[i] < alpha_positive < available_angles[i + 1]:
            alpha_low = available_angles[i]
            alpha_high = available_angles[i + 1]
            cx_low = c_x_data[alpha_low]['total'][mach_idx]
            cx_high = c_x_data[alpha_high]['total'][mach_idx]
            fraction = (alpha_positive - alpha_low) / (alpha_high - alpha_low)
            return cx_low + fraction * (cx_high - cx_low)
    
    return 0

# =============================================================================
# ОСНОВНОЙ РАСЧЕТ
# =============================================================================

print("Запуск полного аэродинамического анализа...")

# Создаем массивы для чисел Маха
Mach = np.linspace(0.1, 4, N)
angles = [0, 5, 10, 15, 20, 25]
angles_for_analysis = list(range(-20, 21, 1))
selected_mach_numbers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Инициализация массивов для фюзеляжа
c_y_alpha_is_f = np.zeros_like(Mach)

# Для сопротивления фюзеляжа (как во второй программе)
c_f = np.zeros_like(Mach)
Re = np.zeros_like(Mach)
c_x_tr = np.zeros_like(Mach)
c_x_Nos = np.zeros_like(Mach)
c_x_dn = np.zeros_like(Mach)
c_x0_f = np.zeros_like(Mach)

# Для сопротивления несущих поверхностей
c_x_0I = np.zeros_like(Mach)
c_x_0II = np.zeros_like(Mach)
c_x_p_I = np.zeros_like(Mach)
c_x_p_II = np.zeros_like(Mach)
c_x_v_I = np.zeros_like(Mach)
c_x_v_II = np.zeros_like(Mach)

# Остальные массивы
c_ya_kr = np.zeros_like(Mach)
c_ya_op = np.zeros_like(Mach)
c_ya_izkor = np.zeros_like(Mach)  # Старый массив для обратной совместимости
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

k_sch = np.zeros_like(Mach)
n_I = np.zeros_like(Mach)
n_II = np.zeros_like(Mach)
eps_delta_sr = np.zeros_like(Mach)
c_y_delta_I = np.zeros_like(Mach)
c_y_delta_II = np.zeros_like(Mach)

# Добавляем массивы для k_t_I и k_t_II как во второй программе
k_t_I = np.ones_like(Mach)
k_t_II = np.ones_like(Mach)

# Расчетные параметры для интерференции
K_aa_teor = (1 + D_otn)**2
k_aa_teor = (1 + 0.41 * D_otn)**2
K_aa_zv = 1 + 3 * D_otn - D_otn / eta_k * (1 - D_otn)
k_aa_zv = (1 + 0.41 * D_otn)**2 * (1 + 3 * D_otn - D_otn / eta_k * (1 - D_otn)) / (1 + D_otn)**2

kappa_nos_kr = 0.6 + 0.4 * (1 - exp(-0.5 * L_1_otn_kr))
kappa_nos_r = 0.6 + 0.4 * (1 - exp(-0.5 * L_1_otn_r))

# Коэффициенты для функций Лапласа
c_val = (4 + 1 / eta_k) * (1 + 8 * D_otn**2)
FI1_kr = 0.5407
FI2_kr = 0.1507
FI1_r = 1 - 10e-8
FI2_r = 1 - 10e-7

print(f"S_f = {S_f:.4f}")
print(f"S_char = {S_char:.4f}")
print(f"S_cx_total = {S_cx_total:.4f}")

# =============================================================================
# ЦИКЛ РАСЧЕТА ПО ЧИСЛАМ МАХА - ХАРАКТЕРИСТИКИ ПРИ НУЛЕВОЙ ПОДЪЕМНОЙ СИЛЕ
# =============================================================================

print("Расчет характеристик при нулевой подъемной силе...")

for i in range(N):
    # Изолированные характеристики крыльев и оперения
    try:
        result_kr = AeroBDSM.get_c_y_alpha_IsP(Mach[i], lamb, bar_c, chi_05, zeta)
        c_ya_kr[i] = result_kr.Value / 57.3 if hasattr(result_kr, 'Value') else 0
        
        result_op = AeroBDSM.get_c_y_alpha_IsP(Mach[i], lamb_op, bar_c, chi_05, zeta)
        c_ya_op[i] = result_op.Value / 57.3 if hasattr(result_op, 'Value') else 0
        
        # ФЮЗЕЛЯЖ: используем метод из второй программы
        c_y_alpha_is_f[i] = AeroBDSM.get_c_y_alpha_NosCil_Ell(Mach[i], lambda_Nos, lambda_Cil) / 57.3
        
        # Для обратной совместимости
        c_ya_izkor[i] = c_y_alpha_is_f[i]
        
    except Exception as e:
        print(f"Ошибка при M={Mach[i]}: {e}")
        c_ya_kr[i] = c_ya_op[i] = c_y_alpha_is_f[i] = c_ya_izkor[i] = 0
    
    # ========== СОПРОТИВЛЕНИЕ КОРПУСА (как во второй программе) ==========
    Re[i] = Mach[i] * 340 * l_f / nu
    c_f[i] = AeroBDSM.get_c_f0(Re[i], 0).Value
    
    # Трение корпуса (отнесенное к S_char)
    c_x_tr[i] = c_f[i] * S_f / S_char
    
    # Сопротивление носовой части
    try:
        # Пробуем функцию из второй программы
        c_x_Nos[i] = AeroBDSM.get_c_x0_p_Nos_Con(Mach[i], lambda_Nos_cx).Value
    except:
        # Если не работает, используем альтернативную
        c_x_Nos[i] = AeroBDSM.get_c_x0_p_Nos_Par(Mach[i], lambda_Nos_cx).Value
    
    # Донное сопротивление (формула из второй программы)
    if Mach[i] <= 0.8:
        c_x_dn[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2
    else:
        c_x_dn[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2
    
    # Итого сопротивление корпуса при нулевой подъемной силе
    c_x0_f[i] = c_x_tr[i] + c_x_Nos[i] + c_x_dn[i]
    
    # ========== СОПРОТИВЛЕНИЕ НЕСУЩИХ ПОВЕРХНОСТЕЙ (как во второй программе) ==========
    c_x_p_I[i] = 2 * c_f[i] * eta_c
    c_x_p_II[i] = 2 * c_f[i] * eta_c
    
    # Волновое сопротивление
    if Mach[i] < 1:
        c_x_v_I[i] = 0
        c_x_v_II[i] = 0
    else:
        try:
            c_x_v_I[i] = AeroBDSM.get_c_x0_w_IsP_Rmb(Mach[i], c_otn, zeta, chi_05, lambda_r).Value
        except:
            c_x_v_I[i] = 0
        
        try:
            c_x_v_II[i] = AeroBDSM.get_c_x0_w_IsP_Rmb(Mach[i], c_otn, zeta, chi_05, lambda_kr).Value
        except:
            c_x_v_II[i] = 0
    
    c_x_0I[i] = c_x_p_I[i] + c_x_v_I[i]
    c_x_0II[i] = c_x_p_II[i] + c_x_v_II[i]
    
    # ========== КОЭФФИЦИЕНТЫ ИНТЕРФЕРЕНЦИИ ==========
    if Mach[i] > 1:
        L_xv_otn_kr = L_xv_kr / (pi / 2 * d * sqrt(Mach[i]**2 - 1))
        b_otn_kr = b_kr / (pi / 2 * d * sqrt(Mach[i]**2 - 1))
        FL_xv_kr = 1 - (sqrt(pi)) / (2 * b_otn_kr * sqrt(c_val)) * (FI1_kr - FI2_kr)

        L_xv_otn_r = L_xv_r / (pi / 2 * d * sqrt(Mach[i]**2 - 1))
        b_otn_r = b_op / (pi / 2 * d * sqrt(Mach[i]**2 - 1))
        FL_xv_r = 1 - (sqrt(pi)) / (2 * b_otn_r * sqrt(c_val)) * (FI1_r - FI2_r)
        
        if L_xv_kr > (pi / 2 * d * sqrt(Mach[i]**2 - 1)):
            K_aa_kr[i] = K_aa_teor * kappa_M * kappa_nos_kr
        else:
            K_aa_kr[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_kr) * kappa_M * kappa_nos_kr

        if L_xv_r > (pi / 2 * d * sqrt(Mach[i]**2 - 1)):
            K_aa_op[i] = K_aa_teor * kappa_M * kappa_nos_r
        else:
            K_aa_op[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_r) * kappa_M * kappa_nos_r

        k_aa_kr[i] = k_aa_zv * kappa_M * kappa_nos_kr
        k_aa_op[i] = k_aa_kr[i]
    else:
        k_aa_kr[i] = k_aa_zv * kappa_M * kappa_nos_kr
        k_aa_op[i] = k_aa_kr[i]
        K_aa_kr[i] = K_aa_zv * kappa_M * kappa_nos_kr
        K_aa_op[i] = K_aa_zv * kappa_M * kappa_nos_r
    
    # Коэффициент щели
    if Mach[i] > 1.4:
        k_sch[i] = 0.85 + (Mach[i] - 1.4) * (1 - 0.85) / (2 - 1.4)
    else:
        k_sch[i] = 0.85

    n_I[i] = k_sch[i] * cos(chi_p)
    n_II[i] = k_sch[i] * cos(chi_p)
    
    # Производные для эффективности рулей
    c_y_alpha_is_kr = c_ya_kr[i] * 57.3
    i_v = 0.1
    k_delta_0_kr = 0.8
    psi_eps = 1.0
    z_v_otn = 0.5
    K_delta_0_r = 0.9
    
    l_razmah_r = 0.35
    l_razmah_kr = 0.64
    S_r = S_op
    S_kr = S_Kons
    
    eps_delta_sr[i] = (57.3 * i_v * l_razmah_r * c_y_alpha_is_kr * k_delta_0_kr * n_I[i] * psi_eps) / (2 * pi * z_v_otn * l_razmah_kr * lamb * K_aa_kr[i])
    
    c_y_delta_I[i] = (c_ya_op[i] * 57.3 * K_delta_0_r * n_I[i] * S_r / S_char * k_t_I[i] + 
                     c_ya_kr[i] * 57.3 * K_aa_kr[i] * eps_delta_sr[i] * S_kr / S_char * k_t_II[i])
    
    c_y_delta_II[i] = (c_ya_kr[i] * 57.3 * k_delta_0_kr * S_kr / S_char * k_t_II[i])
    
    # Скос потока
    eps_cp[i] = 0.5 * (c_ya_op[i] / c_ya_kr[i]) if c_ya_kr[i] != 0 else 0
    
    # Коэффициенты подъемной силы при α=5° для крыльев и оперения
    if orientation == "XX":
        c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[i], 5, A_kr_const)
        c_y_kr_5deg[i] = (K_aa_kr[i] / k_aa_kr[i]) * c_n_kr * cos(radians(5)) * sqrt(2) - 2 * 0.01 * sin(radians(5))
        
        c_n_op = calculate_wing_normal_coefficient(c_ya_op[i], 5, A_op_const)
        c_y_op_5deg[i] = (K_aa_op[i] / k_aa_op[i]) * c_n_op * cos(radians(5)) * sqrt(2) - 2 * 0.01 * sin(radians(5))
        alpha_eff_op_5deg[i] = 5
    else:
        c_y_kr_5deg[i] = calculate_wing_lift_coefficient(c_ya_kr[i], 5, A_kr_const)
        
        alpha_eff_rad = 1.0 * (radians(5) - radians(eps_cp[i]))
        alpha_eff_deg = degrees(alpha_eff_rad)
        c_n_op = (57.3 * c_ya_op[i] * sin(alpha_eff_rad) * cos(alpha_eff_rad) + 
                 A_op_const * (sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
        c_y_op_5deg[i] = (K_aa_op[i] / k_aa_op[i]) * c_n_op
        alpha_eff_op_5deg[i] = alpha_eff_deg
    
    # ФЮЗЕЛЯЖ при α=5° (полная версия)
    c_y_izkor_5deg[i] = calculate_fuselage_lift_coefficient(Mach[i], 5, c_y_alpha_is_f[i]) * (S_char / S_cx_total)

# Суммарные характеристики
c_ya_1_with_interference = c_ya_op * K_aa_op
c_ya_2_with_interference = c_ya_kr 

c_ya_total = (c_y_alpha_is_f * (S_char / S_cx_total) * (S_f / S_cx_total) + 
             c_ya_1_with_interference * (S_op / S_cx_total) + 
             c_ya_2_with_interference * (S_Kons / S_cx_total))

c_y_total_5deg = (c_y_izkor_5deg * (S_f / S_cx_total) + 
                 c_y_op_5deg * (S_op / S_cx_total) + 
                 c_y_kr_5deg * (S_Kons / S_cx_total))

# =============================================================================
# РАСЧЕТ СОПРОТИВЛЕНИЯ ДЛЯ РАЗНЫХ УГЛОВ АТАКИ (как во второй программе)
# =============================================================================

print("Расчет коэффициентов лобового сопротивления для разных углов атаки...")

c_x_data = {}
c_x_fuselage_data = {}
c_x_wing_data = {}
c_x_tail_data = {}

# Массивы для хранения данных сопротивления при разных углах
c_x0_f_data = np.zeros((len(angles), len(Mach)))
c_x_i_f_data = np.zeros((len(angles), len(Mach)))
c_x_fuselage_data_storage = np.zeros((len(angles), len(Mach)))

for idx, angle_ataki in enumerate(angles):
    c_x_total = np.zeros_like(Mach)
    c_x0_total = np.zeros_like(Mach)
    c_x_i_total = np.zeros_like(Mach)
    
    c_x_fuselage = np.zeros_like(Mach)
    c_x_wing = np.zeros_like(Mach)
    c_x_tail = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        # ========== ИНДУКТИВНОЕ СОПРОТИВЛЕНИЕ ФЮЗЕЛЯЖА ==========
        if Mach[i] > 1:
            sigma_val = AeroBDSM.get_sigma_cp_Nos_Con(Mach[i], lambda_Nos).Value
        else:
            sigma_val = 0
        
        delta_c_x1_val = 2 * sigma_val * sin(radians(angle_ataki))**2
        
        # Упрощенный расчет подъемной силы фюзеляжа для индуктивного сопротивления
        c_y_f_val = calculate_fuselage_lift_coefficient_simple(c_y_alpha_is_f[i], angle_ataki)
        
        # Формула индуктивного сопротивления фюзеляжа из второй программы
        c_x_i_f_val = (c_y_f_val + c_x0_f[i] * sin(radians(angle_ataki))) * \
                     tan(radians(angle_ataki)) + delta_c_x1_val * cos(radians(angle_ataki))
        
        # ========== ИНДУКТИВНОЕ СОПРОТИВЛЕНИЕ НЕСУЩИХ ПОВЕРХНОСТЕЙ ==========
        alpha_rad = radians(angle_ataki)
        delta_I_rad = radians(delta_I)
        delta_II_rad = radians(delta_II)
        
        # Коэффициенты нелинейности
        try:
            A_kr_val = AeroBDSM.get_A_IsP(Mach[i], zeta, c_ya_kr[i])
        except:
            A_kr_val = A_kr_const
        
        try:
            A_op_val = AeroBDSM.get_A_IsP(Mach[i], zeta, c_ya_op[i])
        except:
            A_op_val = A_op_const
        
        # Нормальные силы
        c_nI = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op_val)
        c_nII = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr_val)
        
        # Индуктивное сопротивление несущих поверхностей (формула из второй программы)
        term_I = (K_aa_op[i] - k_aa_op[i]) / k_aa_op[i] if k_aa_op[i] != 0 else 0
        c_x_i_I = c_nI * (sin(alpha_rad + delta_I_rad) + term_I * sin(alpha_rad) * cos(delta_I_rad))
        
        term_II = (K_aa_kr[i] - k_aa_kr[i]) / k_aa_kr[i] if k_aa_kr[i] != 0 else 0
        c_x_i_II = c_nII * (sin(alpha_rad + delta_II_rad) + term_II * sin(alpha_rad) * cos(delta_II_rad))
        
        # ========== СУММИРОВАНИЕ КОМПОНЕНТОВ (как во второй программе) ==========
        # Общее индуктивное сопротивление
        c_x_i_total[i] = (c_x_i_f_val * S_f / S_cx_total + 
                         c_x_i_I * S_op / S_cx_total + 
                         c_x_i_II * S_Kons / S_cx_total)
        
        # Общее сопротивление при нулевой подъемной силе
        c_x0_total[i] = 1.05 * (c_x0_f[i] * S_f / S_cx_total + 
                               c_x_0I[i] * k_t_I[i] * S_op / S_cx_total + 
                               c_x_0II[i] * k_t_II[i] * S_Kons / S_cx_total)
        
        # Суммарное сопротивление
        c_x_total[i] = c_x0_total[i] + c_x_i_total[i]
        
        # Сохраняем компоненты для анализа
        c_x_fuselage[i] = c_x0_f[i] * S_f / S_cx_total + c_x_i_f_val * S_f / S_cx_total
        c_x_wing[i] = c_x_0II[i] * S_Kons / S_cx_total + c_x_i_II * S_Kons / S_cx_total
        c_x_tail[i] = c_x_0I[i] * S_op / S_cx_total + c_x_i_I * S_op / S_cx_total
        
        # Сохраняем данные фюзеляжа
        c_x0_f_data[idx, i] = c_x0_f[i]
        c_x_i_f_data[idx, i] = c_x_i_f_val
        c_x_fuselage_data_storage[idx, i] = c_x_fuselage[i]
    
    c_x_data[angle_ataki] = {
        'total': c_x_total.copy(),
        'c_x0': c_x0_total.copy(),
        'c_x_i': c_x_i_total.copy(),
        'fuselage': c_x_fuselage.copy(),
        'wing': c_x_wing.copy(),
        'tail': c_x_tail.copy()
    }

# =============================================================================
# РАСЧЕТ МОМЕНТА ТАНГАЖА
# =============================================================================

print("Расчет коэффициента момента тангажа...")

m_z_data = {}
m_z_focus_data = {}

for angle_ataki in angles:
    m_z_total = np.zeros_like(Mach)
    m_z_fuselage = np.zeros_like(Mach)
    m_z_wing = np.zeros_like(Mach)
    m_z_tail = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        # Расчет через центры давления
        alpha_rad = radians(angle_ataki)
        
        # ФЮЗЕЛЯЖ: коэффициент подъемной силы
        c_y1_fuselage = calculate_fuselage_lift_coefficient(Mach[i], angle_ataki, c_y_alpha_is_f[i]) * (S_char / S_cx_total)
        
        # Крылья
        if orientation == "XX":
            c_n_wing = calculate_wing_normal_coefficient(c_ya_kr[i], angle_ataki, A_kr_const)
            c_y1_wing = (K_aa_kr[i] / k_aa_kr[i]) * c_n_wing * cos(alpha_rad) * sqrt(2)
        else:
            c_y1_wing = calculate_wing_lift_coefficient(c_ya_kr[i], angle_ataki, A_kr_const)
        
        # Оперение
        if orientation == "XX":
            c_n_tail = calculate_wing_normal_coefficient(c_ya_op[i], angle_ataki, A_op_const)
            c_y1_tail = (K_aa_op[i] / k_aa_op[i]) * c_n_tail * cos(alpha_rad) * sqrt(2)
        else:
            alpha_eff_rad = 1.0 * (alpha_rad - radians(eps_cp[i]))
            alpha_eff_deg = degrees(alpha_eff_rad)
            c_n_tail = (57.3 * c_ya_op[i] * sin(alpha_eff_rad) * cos(alpha_eff_rad) + 
                       A_op_const * (sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y1_tail = (K_aa_op[i] / k_aa_op[i]) * c_n_tail
        
        # Приведение к общей площади
        c_y1_wing = c_y1_wing * (S_char / S_cx_total)
        c_y1_tail = c_y1_tail * (S_char / S_cx_total)
        
        # Моменты
        m_z_fuselage[i] = c_y1_fuselage * (x_T - x_d_fuselage) / L
        m_z_wing[i] = -c_y1_wing * (x_T - x_d_wing) / L
        m_z_tail[i] = -c_y1_tail * (x_T - x_d_tail) / L
        
        m_z_total[i] = m_z_fuselage[i] + m_z_wing[i] + m_z_tail[i]
    
    m_z_data[angle_ataki] = {
        'total': m_z_total.copy(),
        'fuselage': m_z_fuselage.copy(),
        'wing': m_z_wing.copy(),
        'tail': m_z_tail.copy()
    }
    
    # Расчет через фокусы
    m_z_focus_total = np.zeros_like(Mach)
    m_z_focus_wing = np.zeros_like(Mach)
    m_z_focus_tail = np.zeros_like(Mach)
    m_z_focus_fuselage = np.zeros_like(Mach)
    
    for i in range(len(Mach)):
        alpha_rad = radians(angle_ataki)
        delta_I_rad = radians(delta_I_deg)
        delta_II_rad = radians(delta_II_deg)
        
        # Производные, приведенные к общей площади
        c_y1_alpha_wing = c_ya_kr[i] * 57.3 * (S_Kons / S_cx_total)
        c_y1_alpha_tail = c_ya_op[i] * 57.3 * (S_op / S_cx_total)
        c_y1_alpha_fuselage = c_y_alpha_is_f[i] * 57.3 * (S_f / S_cx_total)
        
        # Фокусы по компонентам (примерные значения)
        x_Fa_wing = 1.7
        x_Fa_tail = 1.8
        x_Fa_fuselage = 1.6
        x_Fdelta_I = 1.75
        x_Fdelta_II = 1.7
        m_z0 = 0.0
        
        m_z_focus_wing[i] = -(c_y1_alpha_wing * alpha_rad * (x_Fa_wing - x_T) / L +
                            c_y_delta_II[i] * delta_II_rad * (x_Fdelta_II - x_T) / L)
        
        m_z_focus_tail[i] = -(c_y1_alpha_tail * alpha_rad * (x_Fa_tail - x_T) / L +
                            c_y_delta_I[i] * delta_I_rad * (x_Fdelta_I - x_T) / L)
        
        m_z_focus_fuselage[i] = -c_y1_alpha_fuselage * alpha_rad * (x_Fa_fuselage - x_T) / L
        
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
# РАСЧЕТ КОЭФФИЦИЕНТОВ ОТ УГЛА АТАКИ ДЛЯ РАЗНЫХ ЧИСЕЛ МАХА
# =============================================================================

print("Расчет коэффициентов от угла атаки для разных чисел Маха...")

cn_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}
cy_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}
cx_vs_alpha_data = {mach: [] for mach in selected_mach_numbers}

cn_wing_data = {mach: [] for mach in selected_mach_numbers}
cy_wing_data = {mach: [] for mach in selected_mach_numbers}
cx_wing_data = {mach: [] for mach in selected_mach_numbers}

cn_tail_data = {mach: [] for mach in selected_mach_numbers}
cy_tail_data = {mach: [] for mach in selected_mach_numbers}
cx_tail_data = {mach: [] for mach in selected_mach_numbers}

cn_fuselage_data = {mach: [] for mach in selected_mach_numbers}
cy_fuselage_data = {mach: [] for mach in selected_mach_numbers}
cx_fuselage_data = {mach: [] for mach in selected_mach_numbers}

# Функция для интерполяции сопротивления компонента
def interpolate_cx_component(alpha, mach_idx, c_x_data, component):
    alpha_positive = abs(alpha)
    if alpha_positive in c_x_data:
        return c_x_data[alpha_positive][component][mach_idx]
    
    available_angles = sorted(c_x_data.keys())
    if alpha_positive > available_angles[-1]:
        alpha1, alpha2 = available_angles[-2], available_angles[-1]
        cx1 = c_x_data[alpha1][component][mach_idx]
        cx2 = c_x_data[alpha2][component][mach_idx]
        slope = (cx2 - cx1) / (alpha2 - alpha1)
        return cx2 + slope * (alpha_positive - alpha2)
    
    if alpha_positive < available_angles[0]:
        return c_x_data[available_angles[0]][component][mach_idx]
    
    for i in range(len(available_angles) - 1):
        if available_angles[i] < alpha_positive < available_angles[i + 1]:
            alpha_low = available_angles[i]
            alpha_high = available_angles[i + 1]
            cx_low = c_x_data[alpha_low][component][mach_idx]
            cx_high = c_x_data[alpha_high][component][mach_idx]
            fraction = (alpha_positive - alpha_low) / (alpha_high - alpha_low)
            return cx_low + fraction * (cx_high - cx_low)
    
    return 0

for mach in selected_mach_numbers:
    mach_idx = np.argmin(np.abs(Mach - mach))
    actual_mach = Mach[mach_idx]
    
    print(f"Расчет для M = {actual_mach:.2f}")
    
    for alpha in angles_for_analysis:
        # Общие коэффициенты
        c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[mach_idx], alpha, A_kr_const)
        c_n_op = calculate_wing_normal_coefficient(c_ya_op[mach_idx], alpha, A_op_const)
        
        # ФЮЗЕЛЯЖ: коэффициент нормальной силы
        c_n_fuselage = calculate_fuselage_lift_coefficient(Mach[mach_idx], alpha, c_y_alpha_is_f[mach_idx]) * (S_char / S_cx_total)
        
        if orientation == "XX":
            cn_total = (c_n_fuselage + 
                       c_n_op * (S_op / S_cx_total) * (K_aa_op[mach_idx] / k_aa_op[mach_idx]) + 
                       c_n_kr * (S_Kons / S_cx_total) * (K_aa_kr[mach_idx] / k_aa_kr[mach_idx]))
        else:
            cn_total = (c_n_fuselage + 
                       c_n_op * (S_op / S_cx_total) + 
                       c_n_kr * (S_Kons / S_cx_total))
        
        c_y_kr = calculate_wing_lift_coefficient(c_ya_kr[mach_idx], alpha, A_kr_const)
        c_y_op = calculate_wing_lift_coefficient(c_ya_op[mach_idx], alpha, A_op_const)
        
        # ФЮЗЕЛЯЖ: коэффициент подъемной силы
        c_y_fuselage = calculate_fuselage_lift_coefficient(Mach[mach_idx], alpha, c_y_alpha_is_f[mach_idx]) * (S_char / S_cx_total)
        
        if orientation == "XX":
            cy_total = (c_y_fuselage + 
                       c_y_op * (S_op / S_cx_total) * (K_aa_op[mach_idx] / k_aa_op[mach_idx]) + 
                       c_y_kr * (S_Kons / S_cx_total) * (K_aa_kr[mach_idx] / k_aa_kr[mach_idx]))
        else:
            cy_total = (c_y_fuselage + 
                       c_y_op * (S_op / S_cx_total) + 
                       c_y_kr * (S_Kons / S_cx_total))
        
        cx_total = interpolate_cx(alpha, mach_idx, c_x_data)
        
        cn_vs_alpha_data[mach].append(cn_total)
        cy_vs_alpha_data[mach].append(cy_total)
        cx_vs_alpha_data[mach].append(cx_total)
        
        # Компоненты
        cn_wing_data[mach].append(c_n_kr)
        cy_wing_data[mach].append(c_y_kr)
        cx_wing = interpolate_cx_component(alpha, mach_idx, c_x_data, 'wing')
        cx_wing_data[mach].append(cx_wing)
        
        cn_tail_data[mach].append(c_n_op)
        cy_tail_data[mach].append(c_y_op)
        cx_tail = interpolate_cx_component(alpha, mach_idx, c_x_data, 'tail')
        cx_tail_data[mach].append(cx_tail)
        
        cn_fuselage_data[mach].append(c_n_fuselage)
        cy_fuselage_data[mach].append(c_y_fuselage)
        cx_fuselage = interpolate_cx_component(alpha, mach_idx, c_x_data, 'fuselage')
        cx_fuselage_data[mach].append(cx_fuselage)

print("Расчеты завершены!")
print("Программа готова к построению графиков...")




# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================

# =============================================================================
# НАСТРОЙКА ЦВЕТОВЫХ СХЕМ
# =============================================================================

# Выберите одну из предложенных схем:
COLOR_SCHEME = "plasma"  # или "viridis", "plasma", "coolwarm", "Set2"

if COLOR_SCHEME == "viridis":
    colormap = plt.cm.viridis
elif COLOR_SCHEME == "plasma":
    colormap = plt.cm.plasma
elif COLOR_SCHEME == "coolwarm":
    colormap = plt.cm.coolwarm
elif COLOR_SCHEME == "Set2":
    colormap = plt.cm.Set2
elif COLOR_SCHEME == "tab10":
    colormap = plt.cm.tab10
else:
    colormap = plt.cm.viridis  # по умолчанию

# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================

print("Построение графиков...")

# Цвета для графиков с разными числами Маха
colors_mach = colormap(np.linspace(0, 1, len(selected_mach_numbers)))

# Цвета для графиков с разными углами атаки
test_angles = [0, 5, 10, 15, 20, 25]
colors_angles = colormap(np.linspace(0, 1, len(test_angles)))

# =============================================================================
# ГРАФИКИ КОМПОНЕНТОВ
# =============================================================================

print("Построение графиков для компонентов...")

# График 1: Коэффициенты для крыла
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

for i, mach in enumerate(selected_mach_numbers):
    axes1[0].plot(angles_for_analysis, cn_wing_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes1[1].plot(angles_for_analysis, cy_wing_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes1[2].plot(angles_for_analysis, cx_wing_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')

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

# График 2: Коэффициенты для оперения
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for i, mach in enumerate(selected_mach_numbers):
    axes2[0].plot(angles_for_analysis, cn_tail_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes2[1].plot(angles_for_analysis, cy_tail_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes2[2].plot(angles_for_analysis, cx_tail_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')

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
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes3[1].plot(angles_for_analysis, cy_fuselage_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')
    axes3[2].plot(angles_for_analysis, cx_fuselage_data[mach], 
                  color=colors_mach[i], linewidth=2, label=f'M = {mach}')

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

# =============================================================================
# ГРАФИКИ ДЛЯ РАЗНЫХ УГЛОВ АТАКИ
# =============================================================================

print("Построение графиков для разных углов атаки...")

# График 4: Подъемная сила крыла и оперения от числа Маха
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, angle in enumerate(test_angles):
    c_y_kr = np.zeros_like(Mach)
    c_y_op = np.zeros_like(Mach)
    
    for j, mach_val in enumerate(Mach):
        if orientation == "XX":
            c_n_kr = calculate_wing_normal_coefficient(c_ya_kr[j], angle, A_kr_const)
            c_y_kr[j] = (K_aa_kr[j] / k_aa_kr[j]) * c_n_kr * cos(radians(angle)) * sqrt(2) - 2 * 0.01 * sin(radians(angle))
            
            c_n_op = calculate_wing_normal_coefficient(c_ya_op[j], angle, A_op_const)
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op * cos(radians(angle)) * sqrt(2) - 2 * 0.01 * sin(radians(angle))
        else:
            c_y_kr[j] = calculate_wing_lift_coefficient(c_ya_kr[j], angle, A_kr_const)
            
            alpha_eff_rad = 1.0 * (radians(angle) - radians(eps_cp[j]))
            alpha_eff_deg = degrees(alpha_eff_rad)
            c_n_op = (57.3 * c_ya_op[j] * sin(alpha_eff_rad) * cos(alpha_eff_rad) + 
                     A_op_const * (sin(alpha_eff_rad))**2 * (1 if alpha_eff_deg >= 0 else -1))
            c_y_op[j] = (K_aa_op[j] / k_aa_op[j]) * c_n_op
    
    ax1.plot(Mach, c_y_kr, color=colors_angles[i], linewidth=2, label=fr'$\alpha = {angle}^\circ$')
    ax2.plot(Mach, c_y_op, color=colors_angles[i], linewidth=2, label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('Число Маха (M)', fontsize=12)
ax1.set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
ax1.set_title(f'Коэффициент подъемной силы крыла\n({orientation}-образная ориентация)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

ax2.set_xlabel('Число Маха (M)', fontsize=12)
ax2.set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
ax2.set_title(f'Коэффициент подъемной силы оперения\n({orientation}-образная ориентация)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\wing_tail_lift_coefficients_{orientation}.png', dpi=300, bbox_inches='tight')
plt.show()

# График 5: Нормальная сила от числа Маха
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))

for i, alpha in enumerate(test_angles):
    c_n_kr = np.zeros_like(Mach)
    c_n_op = np.zeros_like(Mach)
    c_n_fuselage = np.zeros_like(Mach)
    
    for j, mach_val in enumerate(Mach):
        c_n_kr[j] = calculate_wing_normal_coefficient(c_ya_kr[j], alpha, A_kr_const)
        c_n_op[j] = calculate_wing_normal_coefficient(c_ya_op[j], alpha, A_op_const)
        c_n_fuselage[j] = calculate_fuselage_lift_coefficient(Mach[j], alpha, c_y_alpha_is_f[j])
    
    axes5[0].plot(Mach, c_n_kr, color=colors_angles[i], linewidth=2, label=f'α={alpha}°')
    axes5[1].plot(Mach, c_n_op, color=colors_angles[i], linewidth=2, label=f'α={alpha}°')
    axes5[2].plot(Mach, c_n_fuselage, color=colors_angles[i], linewidth=2, label=f'α={alpha}°')

axes5[0].set_xlabel('Число Маха', fontsize=12)
axes5[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes5[0].set_title('Коэффициент нормальной силы крыла', fontsize=14)
axes5[0].grid(True, alpha=0.3)
axes5[0].legend(fontsize=10)

axes5[1].set_xlabel('Число Маха', fontsize=12)
axes5[1].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes5[1].set_title('Коэффициент нормальной силы оперения', fontsize=14)
axes5[1].grid(True, alpha=0.3)
axes5[1].legend(fontsize=10)

axes5[2].set_xlabel('Число Маха', fontsize=12)
axes5[2].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
axes5[2].set_title('Коэффициент нормальной силы корпуса', fontsize=14)
axes5[2].grid(True, alpha=0.3)
axes5[2].legend(fontsize=10)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\normal_force_vs_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ГРАФИКИ СОПРОТИВЛЕНИЯ
# =============================================================================

print("Построение графиков сопротивления...")

# График 6: Сопротивление от числа Маха
fig6, axes6 = plt.subplots(2, 2, figsize=(15, 12))

for i, alpha in enumerate(angles):
    if alpha in c_x_data:
        data = c_x_data[alpha]
        
        axes6[0, 0].plot(Mach, data['total'], color=colors_angles[i % len(colors_angles)], linewidth=2, label=f'α={alpha}°')
        axes6[0, 1].plot(Mach, data['fuselage'], color=colors_angles[i % len(colors_angles)], linewidth=2, label=f'α={alpha}°')
        axes6[1, 0].plot(Mach, data['wing'], color=colors_angles[i % len(colors_angles)], linewidth=2, label=f'α={alpha}°')
        axes6[1, 1].plot(Mach, data['tail'], color=colors_angles[i % len(colors_angles)], linewidth=2, label=f'α={alpha}°')

axes6[0, 0].set_xlabel('Число Маха', fontsize=12)
axes6[0, 0].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes6[0, 0].set_title('Суммарный коэффициент лобового сопротивления', fontsize=14)
axes6[0, 0].grid(True, alpha=0.3)
axes6[0, 0].legend(fontsize=10)

axes6[0, 1].set_xlabel('Число Маха', fontsize=12)
axes6[0, 1].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes6[0, 1].set_title('Коэффициент сопротивления корпуса', fontsize=14)
axes6[0, 1].grid(True, alpha=0.3)
axes6[0, 1].legend(fontsize=10)

axes6[1, 0].set_xlabel('Число Маха', fontsize=12)
axes6[1, 0].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes6[1, 0].set_title('Коэффициент сопротивления крыла', fontsize=14)
axes6[1, 0].grid(True, alpha=0.3)
axes6[1, 0].legend(fontsize=10)

axes6[1, 1].set_xlabel('Число Маха', fontsize=12)
axes6[1, 1].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
axes6[1, 1].set_title('Коэффициент сопротивления оперения', fontsize=14)
axes6[1, 1].grid(True, alpha=0.3)
axes6[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\drag_coefficients_vs_mach.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ГРАФИКИ МОМЕНТА ТАНГАЖА
# =============================================================================

print("Построение графиков момента тангажа...")

# График 7: Mz от числа Маха (через центры давления)
fig7, axes7 = plt.subplots(2, 2, figsize=(15, 12))

for idx, angle in enumerate(angles):
    if angle in m_z_data:
        data = m_z_data[angle]
        
        axes7[0, 0].plot(Mach, data['total'], color=colors_angles[idx % len(colors_angles)], linewidth=2, label=f'α={angle}°')
        axes7[0, 1].plot(Mach, data['fuselage'], color=colors_angles[idx % len(colors_angles)], linewidth=2, label=f'α={angle}°')
        axes7[1, 0].plot(Mach, data['wing'], color=colors_angles[idx % len(colors_angles)], linewidth=2, label=f'α={angle}°')
        axes7[1, 1].plot(Mach, data['tail'], color=colors_angles[idx % len(colors_angles)], linewidth=2, label=f'α={angle}°')

axes7[0, 0].set_xlabel('Число Маха', fontsize=12)
axes7[0, 0].set_ylabel('Коэффициент момента тангажа (m_z)', fontsize=12)
axes7[0, 0].set_title('Суммарный коэффициент момента тангажа\n(через центры давления)', fontsize=14)
axes7[0, 0].grid(True, alpha=0.3)
axes7[0, 0].legend(fontsize=10)

axes7[0, 1].set_xlabel('Число Маха', fontsize=12)
axes7[0, 1].set_ylabel('Коэффициент момента тангажа (m_z)', fontsize=12)
axes7[0, 1].set_title('Момент тангажа корпуса', fontsize=14)
axes7[0, 1].grid(True, alpha=0.3)
axes7[0, 1].legend(fontsize=10)

axes7[1, 0].set_xlabel('Число Маха', fontsize=12)
axes7[1, 0].set_ylabel('Коэффициент момента тангажа (m_z)', fontsize=12)
axes7[1, 0].set_title('Момент тангажа крыльев', fontsize=14)
axes7[1, 0].grid(True, alpha=0.3)
axes7[1, 0].legend(fontsize=10)

axes7[1, 1].set_xlabel('Число Маха', fontsize=12)
axes7[1, 1].set_ylabel('Коэффициент момента тангажа (m_z)', fontsize=12)
axes7[1, 1].set_title('Момент тангажа оперения', fontsize=14)
axes7[1, 1].grid(True, alpha=0.3)
axes7[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\mz_vs_mach_centers.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# СРАВНИТЕЛЬНЫЕ ГРАФИКИ
# =============================================================================

print("Построение сравнительных графиков...")

# График 8: Сравнительные графики для всех компонентов при M=2.0
mach_example = 2.0
if mach_example in selected_mach_numbers:
    fig8, axes8 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Находим индекс для M=2.0
    mach_idx = selected_mach_numbers.index(mach_example)
    
    # График нормальной силы
    axes8[0].plot(angles_for_analysis, cn_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes8[0].plot(angles_for_analysis, cn_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes8[0].plot(angles_for_analysis, cn_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes8[0].set_xlabel('Угол атаки, град', fontsize=12)
    axes8[0].set_ylabel('Коэффициент нормальной силы (C_n)', fontsize=12)
    axes8[0].set_title(f'Сравнение нормальной силы компонентов (M = {mach_example})', fontsize=14)
    axes8[0].grid(True, alpha=0.3)
    axes8[0].legend(fontsize=10)
    axes8[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes8[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes8[0].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    # График подъемной силы
    axes8[1].plot(angles_for_analysis, cy_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes8[1].plot(angles_for_analysis, cy_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes8[1].plot(angles_for_analysis, cy_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes8[1].set_xlabel('Угол атаки, град', fontsize=12)
    axes8[1].set_ylabel('Коэффициент подъемной силы (C_y)', fontsize=12)
    axes8[1].set_title(f'Сравнение подъемной силы компонентов (M = {mach_example})', fontsize=14)
    axes8[1].grid(True, alpha=0.3)
    axes8[1].legend(fontsize=10)
    axes8[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes8[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes8[1].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    # График сопротивления
    axes8[2].plot(angles_for_analysis, cx_wing_data[mach_example], 
                  'b-', linewidth=2, label='Крыло')
    axes8[2].plot(angles_for_analysis, cx_tail_data[mach_example], 
                  'r-', linewidth=2, label='Оперение')
    axes8[2].plot(angles_for_analysis, cx_fuselage_data[mach_example], 
                  'g-', linewidth=2, label='Корпус')
    axes8[2].set_xlabel('Угол атаки, град', fontsize=12)
    axes8[2].set_ylabel('Коэффициент сопротивления (C_x)', fontsize=12)
    axes8[2].set_title(f'Сравнение сопротивления компонентов (M = {mach_example})', fontsize=14)
    axes8[2].grid(True, alpha=0.3)
    axes8[2].legend(fontsize=10)
    axes8[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes8[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes8[2].set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\components_comparison_M2.png', dpi=300, bbox_inches='tight')
    plt.show()

# График 9: Сводный график всех коэффициентов для M=2.0
if mach_example in selected_mach_numbers:
    fig9, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
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
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlim(min(angles_for_analysis), max(angles_for_analysis))
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\fanto\\OneDrive\\Рабочий стол\\PRO\\7 Semestr\\Kursach\\Python\\AIM-120_DZ-main\\AIM-120_DZ-main\\Meteor\\Image\\summary_coefficients_M2.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# ПОЛЯРНЫЕ ДИАГРАММЫ И АЭРОДИНАМИЧЕСКОЕ КАЧЕСТВО
# =============================================================================

print("Построение полярных диаграмм...")

# График 10: Поляра летательного аппарата для разных чисел Маха
fig10 = plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(cx_vs_alpha_data[mach], cy_vs_alpha_data[mach], 
             color=colors_mach[i], linewidth=2, marker='o', markersize=3,
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

# График 11: Аэродинамическое качество от угла атаки
fig11 = plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    k_values = []
    for j in range(len(angles_for_analysis)):
        if abs(cx_vs_alpha_data[mach][j]) > 1e-6:
            k_values.append(cy_vs_alpha_data[mach][j] / cx_vs_alpha_data[mach][j])
        else:
            k_values.append(0)
    
    plt.plot(angles_for_analysis, k_values, 
             color=colors_mach[i], linewidth=2, 
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

# =============================================================================
# ГРАФИКИ ОТ УГЛА АТАКИ ДЛЯ РАЗНЫХ ЧИСЕЛ МАХА
# =============================================================================

print("Построение графиков от угла атаки для разных чисел Маха...")

# График 12: Коэффициент нормальной силы от угла атаки
fig12 = plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cn_vs_alpha_data[mach], 
             color=colors_mach[i], linewidth=2, 
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

# График 13: Коэффициент подъемной силы от угла атаки
fig13 = plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cy_vs_alpha_data[mach], 
             color=colors_mach[i], linewidth=2, 
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

# График 14: Коэффициент сопротивления от угла атаки
fig14 = plt.figure(figsize=(12, 8))

for i, mach in enumerate(selected_mach_numbers):
    plt.plot(angles_for_analysis, cx_vs_alpha_data[mach], 
             color=colors_mach[i], linewidth=2, 
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

print("Все графики построены!")