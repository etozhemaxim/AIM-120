import AeroBDSM
from math import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"

d = 0.0722                                          # Калибр
N = 500                                            # Число шагов
M = np.linspace(0.01, 2, N + 1)                    # Скорость: от 0 до 2 Махов
ataka = np.linspace(-10, 10, N + 1)  
l_f = 1.635                                         # Длина фюзеляжа
lambda_Nos = (d / 2) / d                            # Удлинение носовой части
lambda_Nos_cx = d / 2 + 63.4 / 72.2                 # Величина для расчета с_x с учетом иглы на носике ракеты   
lambda_Cil = (l_f - d / 2) / d                      # Удлинение цилиндрической части
S_f = pi * d * (l_f - d / 2) + 2 * pi * (d / 2)**2  # Площадь поверхности фюзеляжа - цилиндр + половина сферы
c_otn = 0.05                                        # Относительная площадь крыла 
chi_05 = 0
zeta = 1                                            # Обратное сужение несущей поверхности
eta_k = 1                                           # Сужение кормовой части (сужения нет)
b_kr = 0.1                                          # Ширина крыла по бортовой хорде
b_r = 0.025                                         # Ширина руля по бортовой хорде
nu = 15.1e-6
L_xv_kr = 0.085                                     # Расстояние от середины бортовой хорды крыла до кормового среза
L_xv_r = l_f - 0.368                                # Расстояние от середины бортовой хорды руля до кормового среза
L_A = 0.882+0.05
l_razmah = 0.144                                    # Полный размах
l_razmah_r = 0.066                                  # Размах рулей без фюзеляжа
S_r = 2 * 0.033 * 0.025                             # Площадь рулей
lambda_r = (l_razmah_r**2) / (S_r + b_kr**2)
l_razmah_kr = l_razmah - d                          # Размах крыльев без фюзеляжа
S_kr = 2 * 0.036 * 0.1                              # Площадь крыльев
lambda_kr = l_razmah_kr**2 / S_kr
D_otn = d / l_razmah                                # Относительный диаметр
x_bort_kr = l_f - L_xv_kr                           # Расстояние от носика корпуса до середины бортовой хорды крыла
x_bort_r = l_f - L_xv_r                             # Расстояние от носика корпуса до середины бортовой хорды руля
L1_kr = x_bort_kr + b_kr / 2
L1_r = x_bort_r + b_r / 2
L_1_otn_kr = L1_kr / d
L_1_otn_r = L1_r / d
alpha_p, phi_alpha, psi_I, psi_II = 0, 0 , 0, 0
l_1c_II = 0.036
zeta_II = 0
b_b_II = b_kr
chi_0_II = 0
L_vI_bII = 0.882
chi_p = 0                                           # Угол между осью вращения рулей и перпендикуляром к фюзеляжу
S_b_Nos = pi * d**2 / 8                             # Площадь боковой поверхности носовой части
S_bok = S_b_Nos + pi * d * (l_f - d / 2)

# angles = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
angles = [10, 8, 6, 4, 2, 0]
deltas = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]

c_y_alpha_is_f = np.zeros_like(M)
c_y_alpha_is_kr = np.zeros_like(M)
c_y_alpha_is_r = np.zeros_like(M)

K_aa_teor = (1 + D_otn)**2
k_aa_teor = (1 + 0.41 * D_otn)**2

K_aa_zv = 1 + 3 * D_otn - D_otn / eta_k * (1 - D_otn)
k_aa_zv = (1 + 0.41 * D_otn)**2 * (1 + 3 * D_otn - D_otn / eta_k * (1 - D_otn)) / (1 + D_otn)**2

delta_otn_zv_kr = np.zeros_like(M)
kappa_ps_kr = np.zeros_like(M)
delta_otn_zv_r = np.zeros_like(M)
kappa_ps_r = np.zeros_like(M)

# Учет числа Маха
kappa_M = 0.96 # по рис. 3.13 стр. 162

# Учет влияния длины передней части корпуса
kappa_nos_kr = 0.6 + 0.4 * (1 - exp(-0.5 * L_1_otn_kr))
kappa_nos_r = 0.6 + 0.4 * (1 - exp(-0.5 * L_1_otn_r))

pol_step_lin_M = np.zeros_like(M)

L_xv_otn_kr = np.zeros_like(M)
b_otn_kr = np.zeros_like(M)
FL_xv_kr = np.zeros_like(M)

L_xv_otn_r = np.zeros_like(M)
b_otn_r = np.zeros_like(M)
FL_xv_r = np.zeros_like(M)

K_aa_kr = np.zeros_like(M)
K_aa_r = np.zeros_like(M)
k_aa_kr = np.zeros_like(M)
k_aa_r = np.zeros_like(M)

z_v_otn = np.zeros_like(M)
z_v = np.zeros_like(M)
y_v = 0
i_v = np.zeros_like(M)
psi_eps = np.zeros_like(M)
eps_alpha_sr = np.zeros_like(M)

kappa_q_Nos_Con = np.zeros_like(M)
kappa_q_IsP = np.zeros_like(M)
k_t_I = np.zeros_like(M)
k_t_II = np.zeros_like(M)

c = (4 + 1 / eta_k) * (1 + 8 * D_otn**2)
FI1_kr = 0.5407         # значение функции Лапласа в точке (b_kr + L_xv_kr) * sqrt(2 * c) по таблице
FI2_kr = 0.1507         # значение функции Лапласа в точке L_xv_kr * sqrt(2 * c) по таблице
FI1_r = 1 - 10e-8       # значение функции Лапласа в точке (b_r + L_xv_r) * sqrt(2 * c) по таблице
FI2_r = 1 - 10e-7       # значение функции Лапласа в точке L_xv_r * sqrt(2 * c) по таблице
# print((b_kr + L_xv_kr) * sqrt(2 * c), '0.5407')
# print((L_xv_kr) * sqrt(2 * c), '0.1507')
# print((b_r + L_xv_r) * sqrt(2 * c), '1')
# print((L_xv_r) * sqrt(2 * c), '1')

S = pi * d**2 / 4 # характерная площадь (площадь миделя)
S_cx = S_r + S_kr + S_f

c_y_alpha = np.zeros_like(M)
c_y1 = np.zeros_like(M)

k_delta_0_kr = np.zeros_like(M) 
K_delta_0_kr = np.zeros_like(M) 
k_delta_0_r = np.zeros_like(M) 
K_delta_0_r = np.zeros_like(M) 
kappa_ps_otn_kr = np.zeros_like(M)
kappa_ps_otn_r = np.zeros_like(M)
eps_delta_sr = np.zeros_like(M)
k_sch = np.zeros_like(M)
n_I = np.zeros_like(M)
n_II = np.zeros_like(M)

c_y_delta_I = np.zeros_like(M)
c_y_delta_II = np.zeros_like(M)

c_y = np.zeros_like(M)

c_y_f = np.zeros_like(M)
kappa_alpha = np.zeros_like(M)
c_y_perp_Cil = np.zeros_like(M)
M_y = np.zeros_like(M)
c_y_is_kr = np.zeros_like(M)
c_y_is_r = np.zeros_like(M)
A_kr = np.zeros_like(M)
A_r = np.zeros_like(M)
alpha_eff_I = np.zeros_like(M)
alpha_eff_II = np.zeros_like(M)
c_nI = np.zeros_like(M)
c_nII = np.zeros_like(M)
c_y_I = np.zeros_like(M)
c_y_II = np.zeros_like(M)
eps_sr = np.zeros_like(M)
c_y_b1 = np.zeros_like(M)
c_y_b = np.zeros_like(M)


c_f = np.zeros_like(M)
c_x_Nos = np.zeros_like(M)
Re = np.zeros_like(M)
c_x_tr = np.zeros_like(M)
c_x_dn = np.zeros_like(M)
c_x0_f = np.zeros_like(M)

c_x_0I = np.zeros_like(M)
c_x_p_I = np.zeros_like(M)
c_x_v_I = np.zeros_like(M)

c_x_0II = np.zeros_like(M)
c_x_p_II = np.zeros_like(M)
c_x_v_II = np.zeros_like(M)

c_x_i = np.zeros_like(M)
c_x_i_f = np.zeros_like(M)
c_x_i_I = np.zeros_like(M)
c_x_i_II = np.zeros_like(M)

sigma = np.zeros_like(M)
delta_c_x1 = np.zeros_like(M)

c_x0 = np.zeros_like(M)
c_x_data = np.zeros((len(angles), len(M)))

c_x_fuselage = np.zeros_like(M)
c_x_wing = np.zeros_like(M)
c_x_fins = np.zeros_like(M)

c_nI_cx = np.zeros_like(M)
c_nII_cx = np.zeros_like(M)

# Инициализация массивов для хранения данных при разных углах атаки
c_x0_f_data = np.zeros((len(angles), len(M)))
c_x_0I_data = np.zeros((len(angles), len(M)))
c_x_0II_data = np.zeros((len(angles), len(M)))
c_x_i_f_data = np.zeros((len(angles), len(M)))
c_x_i_I_data = np.zeros((len(angles), len(M)))
c_x_i_II_data = np.zeros((len(angles), len(M)))
c_x_i_data = np.zeros((len(angles), len(M)))
c_x0_data = np.zeros((len(angles), len(M)))
c_x_data = np.zeros((len(angles), len(M)))
c_x_fuselage_data = np.zeros((len(angles), len(M)))
c_x_wing_data = np.zeros((len(angles), len(M)))
c_x_fins_data = np.zeros((len(angles), len(M)))

def calculate_wing_normal_coefficient(c_ya, alpha_deg, A):
    alpha_rad = radians(alpha_deg)
    term1 = 57.3 * c_ya * sin(alpha_rad) * cos(alpha_rad)
    term2 = A * (sin(alpha_rad))**2 * (1 if alpha_deg >= 0 else -1)

    return term1 + term2

def calculate_wing_lift_coefficient(c_ya, alpha_deg, A):
    alpha_rad = radians(alpha_deg)
    c_n = calculate_wing_normal_coefficient(c_ya, alpha_deg, A)
    return c_n * cos(alpha_rad)

def calculate_fuselage_lift_coefficient(c_ya, alpha_deg):
    return c_ya * alpha_deg

for idx, angle_ataki in enumerate(angles):
    for i in range(len(M)):
        delta_I = 10 # угол поворота первой несущей поверхности - руля
        delta_II = 0 # угол поворота второй несущей поверхности - крыла
## Коэффициент лобового сопротивления

        # Коэффициент лобового сопротивления корпуса при alpha = 0
        Re[i] = M[i] * 340 * l_f / nu
        c_f[i] = AeroBDSM.get_c_f0(Re[i], 0).Value
        c_x_tr[i] = c_f[i] * S_f / S
        c_x_Nos[i] = AeroBDSM.get_c_x0_p_Nos_Con(M[i], lambda_Nos_cx).Value

        if M[i] <= 0.8:
            c_x_dn[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2
        else:
            c_x_dn[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2

        c_x0_f[i] = c_x_tr[i] + c_x_Nos[i] + c_x_dn[i]

        # Коэффициент лобового сопротивления несущих поверхностей при alpha = delta = 0
        eta_c = 1.13    # стр. 232 ЛиЧ рис. 4.28
        c_x_p_I[i] = 2 * c_f[i] * eta_c
        c_x_p_II[i] = 2 * c_f[i] * eta_c

        # Волновое сопротивление
        if M[i] < 1:
            c_x_v_I[i] = 0
            c_x_v_II[i] = 0
        else:
            c_x_v_I[i] = AeroBDSM.get_c_x0_w_IsP_Rmb(M[i], c_otn, zeta, chi_05, lambda_r).Value
            c_x_v_II[i] = AeroBDSM.get_c_x0_w_IsP_Rmb(M[i], c_otn, zeta, chi_05, lambda_kr).Value

        c_x_0I[i] = c_x_p_I[i] + c_x_v_I[i]
        c_x_0II[i] = c_x_p_II[i] + c_x_v_II[i]


## Подъёмная сила
for idx, angle_ataki in enumerate(angles):
    for i in range(len(M)):
        delta_I = 10 # угол поворота первой несущей поверхности - руля
        delta_II = 0 # угол поворота второй несущей поверхности - крыла

## Производная коэф-та подъёмной силы ЛА по углу атаки c_y_alpha
        c_y_alpha_is_f[i] = AeroBDSM.get_c_y_alpha_NosCil_Ell(M[i], lambda_Nos, lambda_Cil) / 57.3
        c_y_alpha_is_kr[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_kr, c_otn, chi_05, zeta) / 57.3
        c_y_alpha_is_r[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_r, c_otn, chi_05, zeta) / 57.3
        delta_otn_zv_kr[i] = 0.093 * L1_kr * (1 + 0.4 * M[i] + 0.147 * M[i]**2 - 0.006 * M[i]**3) / (d * (M[i] * L1_kr / nu)**(1/5))
        kappa_ps_kr[i] = (1 - (2 * D_otn**2) / (1 - D_otn**2) * delta_otn_zv_kr[i]) * (1 - (D_otn * (eta_k - 1)) / ((1 - D_otn) * (eta_k + 1)) * delta_otn_zv_kr[i])
        delta_otn_zv_r[i] = 0.093 * L1_r * (1 + 0.4 * M[i] + 0.147 * M[i]**2 - 0.006 * M[i]**3) / (d * (M[i] * L1_r / nu)**(1/5))
        kappa_ps_r[i] = (1 - (2 * D_otn**2) / (1 - D_otn**2) * delta_otn_zv_r[i]) * (1 - (D_otn * (eta_k - 1)) / ((1 - D_otn) * (eta_k + 1)) * delta_otn_zv_r[i])

        if M[i] > 1:  
            # Учет задней части крыльев
            L_xv_otn_kr[i] = L_xv_kr / (pi / 2 * d * sqrt(M[i]**2 - 1))
            b_otn_kr[i] = b_kr / (pi / 2 * d * sqrt(M[i]**2 - 1))
            FL_xv_kr[i] = 1 - (sqrt(pi)) / (2 * b_otn_kr[i] * sqrt(c)) * (FI1_kr - FI2_kr)

            # Учет задней части рулей
            L_xv_otn_r[i] = L_xv_r / (pi / 2 * d * sqrt(M[i]**2 - 1))
            b_otn_r[i] = b_r / (pi / 2 * d * sqrt(M[i]**2 - 1))
            FL_xv_r[i] = 1 - (sqrt(pi)) / (2 * b_otn_r[i] * sqrt(c)) * (FI1_r - FI2_r)
            
            # Коэф-ты интерференции:
            if L_xv_kr > (pi / 2 * d * sqrt(M[i]**2 - 1)):
                K_aa_kr[i] = K_aa_teor * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            else:
                K_aa_kr[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_kr[i]) * kappa_ps_kr[i] * kappa_M * kappa_nos_kr

            if L_xv_r > (pi / 2 * d * sqrt(M[i]**2 - 1)):
                K_aa_r[i] = K_aa_teor * kappa_ps_r[i] * kappa_M * kappa_nos_r
            else:
                K_aa_r[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_r[i]) * kappa_ps_r[i] * kappa_M * kappa_nos_r

            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]   

        else:
            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]
            K_aa_kr[i] = K_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            K_aa_r[i] = K_aa_zv * kappa_ps_r[i] * kappa_M * kappa_nos_r

        # Скос потока
        z_v_otn[i] = AeroBDSM.get_bar_z_v(M[i], lambda_kr, chi_05, zeta)
        z_v[i] = (d + z_v_otn[i] * (l_razmah - d)) / 2
        i_v[i] = AeroBDSM.get_i_v(zeta, d, l_razmah, y_v, z_v[i])
        psi_eps[i] = AeroBDSM.get_psi_eps(M[i], np.radians(angle_ataki), phi_alpha, psi_I, psi_II, z_v[i], y_v, L_vI_bII, d, l_1c_II, zeta_II, b_b_II, chi_0_II)
        psi_eps[i] = 1 # аэробдсм выдает такое значение, но оно почему то ломает формулу ниже, если писать туда через аэробдсм
        eps_alpha_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_r[i] * k_aa_r[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_r * K_aa_kr[i])
        #print(psi_eps[i], eps_alpha_sr[i], i)

        # Коэффициент торможения потока
        kappa_q_Nos_Con[i] = AeroBDSM.get_kappa_q_Nos_Con(M[i], lambda_Nos) # коэф торможения потока, вызванный обтеканием коническиой носовой части
        kappa_q_IsP[i] = AeroBDSM.get_kappa_q_IsP(M[i], L_A, b_r)
        k_t_I[i] = kappa_q_Nos_Con[i]
        k_t_II[i] = kappa_q_IsP[i]

        c_y_alpha[i] = c_y_alpha_is_f[i] * S_f / S + c_y_alpha_is_r[i] * K_aa_r[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * (1 - eps_alpha_sr[i]/57.3) * S_kr / S * k_t_II[i]

## Производные коэф-та подьёмной силы ЛА по углам отклонения ОУ c_y_delta_I и c_y_delta_II
        
        k_delta_0_zv = k_aa_zv**2 / K_aa_zv
        K_delta_0_zv = k_aa_zv

        kappa_ps_otn_kr[i] = ((1 - D_otn * (1 + delta_otn_zv_kr[i])) * (1 - (eta_k - 1) * D_otn * (1 + delta_otn_zv_kr[i]) / (eta_k + 1 - 2 * D_otn))) / \
        ((1 - D_otn) * (1 - (eta_k - 1) * D_otn / (eta_k + 1 - 2 * D_otn)))
        kappa_ps_otn_r[i] = ((1 - D_otn * (1 + delta_otn_zv_r[i])) * (1 - (eta_k - 1) * D_otn * (1 + delta_otn_zv_r[i]) / (eta_k + 1 - 2 * D_otn))) / \
        ((1 - D_otn) * (1 - (eta_k - 1) * D_otn / (eta_k + 1 - 2 * D_otn)))

        k_delta_0_kr[i] = k_delta_0_zv * kappa_ps_otn_kr[i] * kappa_M
        K_delta_0_kr[i] = (k_delta_0_zv + (K_delta_0_zv - k_delta_0_zv) * FL_xv_kr[i]) * kappa_ps_otn_kr[i] * kappa_M
        k_delta_0_r[i] = k_delta_0_zv * kappa_ps_otn_r[i] * kappa_M
        K_delta_0_r[i] = (k_delta_0_zv + (K_delta_0_zv - k_delta_0_zv) * FL_xv_r[i]) * kappa_ps_otn_r[i] * kappa_M

        # if M[i] > 1.4:
        #     k_sch[i] = 1
        # else:
        #     k_sch[i] = 0.85

        if M[i] > 1.4:
            k_sch[i] = 0.85 + (M[i] - 1.4) * (1 - 0.85) / (M[N] - 1.4)
        else:
            k_sch[i] = 0.85

        n_I[i] = k_sch[i] * cos(chi_p)
        n_II[i] = k_sch[i] * cos(chi_p)

        eps_delta_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_r[i] * k_delta_0_r[i] * n_I[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_r * K_aa_kr[i])
        # sqrt(2) - учет крестокрылого ЛА (вариант ++) стр. 195
        c_y_delta_I[i] = (c_y_alpha_is_r[i] * K_delta_0_r[i] * n_I[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * eps_delta_sr[i] * S_kr / S * k_t_II[i])
        c_y_delta_II[i] = (c_y_alpha_is_kr[i] * K_delta_0_kr[i] * S_kr / S * k_t_II[i])
        #print(f"Угол атаки: {angle_ataki}°, M = {M[i]:.2f}, c_y_delta_I = {c_y_delta_I[i]:.6f}, c_y_delta_II = {c_y_delta_II[i]:.6f}")

## Коэффициент подъёмной силы при больших углах alpha и delta

        # Коэффициент подъёмной силы корпуса
        kappa_alpha[i] = 1 - 0.45 * (1 - exp(-0.06 * M[i]**2)) * (1 - exp(-0.12 * np.radians(abs(angle_ataki))))
        M_y[i] = M[i] * sin(np.radians(angle_ataki))
        c_y_perp_Cil[i] = AeroBDSM.get_c_yPerp_Cil(M_y[i])
        # S_bok[i] = AeroBDSM.get_S_b_F(S_b_Nos, ds[i], Ls[i], Ms[i], b_bs[i], L_hvs[i])
        c_y_f[i] = 57.3 * c_y_alpha_is_f[i] * kappa_alpha[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) \
        + 4 * S_bok * c_y_perp_Cil[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki) / (pi * d**2)
        print(f"c_y_alpha_is_f[i]={c_y_alpha_is_f[i]}")
        print(f"kappa_alpha[i]={kappa_alpha[i]}")
        print(f"sin(np.radians(angle_ataki))={sin(np.radians(angle_ataki))}")
        print(f"cos(np.radians(angle_ataki))={cos(np.radians(angle_ataki))}")
        print(f"c_y_perp_Cil[i]={c_y_perp_Cil[i]}")
        print(f"(sin(np.radians(angle_ataki)))**2={(sin(np.radians(angle_ataki)))**2}")
        print(f"sign(angle_ataki)={sign(angle_ataki)}")
        print((4 * S_bok) /(pi * d**2) )
        print(60 * '=' )
        A_kr[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_kr[i])
        c_y_is_kr[i] = 57.3 * c_y_alpha_is_kr[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) + A_kr[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki)
        A_r[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_r[i])
        c_y_is_r[i] = 57.3 * c_y_alpha_is_r[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) + A_r[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki)

        eps_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_nI[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_r * K_aa_kr[i])

        alpha_eff_I[i] = k_aa_r[i] * np.radians(angle_ataki) + k_delta_0_r[i] * n_I[i] * np.radians(delta_I)
        alpha_eff_II[i] = k_aa_kr[i] * (np.radians(angle_ataki) - eps_sr[i]) + k_delta_0_kr[i] * np.radians(delta_II)
        c_nI[i] = 57.3 * c_y_alpha_is_r[i] * sin(alpha_eff_I[i]) * cos(alpha_eff_I[i]) + A_r[i] * (sin(alpha_eff_I[i]))**2 * sign(alpha_eff_I[i])
        c_nII[i] = 57.3 * c_y_alpha_is_kr[i] * sin(alpha_eff_II[i]) * cos(alpha_eff_II[i]) + A_kr[i] * (sin(alpha_eff_II[i]))**2 * sign(alpha_eff_II[i])
        c_y_I[i] = c_nI[i] * (K_aa_r[i] / k_aa_r[i] * cos(np.radians(angle_ataki)) * cos(np.radians(delta_I))) - c_x_0I[i] * sin(np.radians(angle_ataki + delta_I))
        c_y_II[i] = c_nII[i] * (K_aa_kr[i] / k_aa_kr[i] * cos(np.radians(angle_ataki)) * cos(np.radians(delta_II))) - c_x_0II[i] * sin(np.radians(angle_ataki + delta_II))

## Итоговый коэффициент подъёмной силы ЛА
        # c_y при малых углах:
        c_y[i] = c_y_alpha[i] * np.radians(angle_ataki) + c_y_delta_I[i] * np.radians(delta_I) + c_y_delta_II[i] * np.radians(delta_II)

        # c_y при больших углах:
        c_y_b1[i] = c_y_f[i] * S_f / S + c_y_I[i] * S_r / S * k_t_I[i] + c_y_II[i] * S_kr / S * k_t_II[i]
        c_y_b[i] = c_y_b1[i] * np.radians(angle_ataki) + c_y_I[i] * np.radians(delta_I) + c_y_II[i] * np.radians(delta_II)


# График коэффициента подъёмной силы c_y при различных малых углах атаки и углах отклонения ОУ от числа Маха
for angle_ataki in angles:
    for i in range(len(M)):
        c_y[i] = c_y_alpha[i] * np.radians(angle_ataki) + c_y_delta_I[i] * np.radians(delta_I) + c_y_delta_II[i] * np.radians(delta_II)
    plt.plot(M, c_y, label=f'α = {angle_ataki}°', linewidth=2)
plt.xlabel('M', fontsize = 12)
plt.ylabel('$c_{y}$', fontsize = 12)
plt.title('Зависимость коэффициента подъёмной силы $c_{y}$\nпри различных малых углах атаки от числа Маха', fontsize = 12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

# График коэф-та подъёмной силы ЛА по углу атаки c_y_alpha от числа Маха при разных alpha
for angle_ataki in angles:
    for i in range(len(M)):
        c_y1[i] = c_y_alpha[i] * np.radians(angle_ataki)
    plt.plot(M, c_y1, label=f'α = {angle_ataki}°', linewidth=2)
plt.xlabel('M', fontsize = 12)
plt.ylabel('$c_{y}^{\\mathrm{\\alpha}}$', fontsize = 12)
plt.title('Зависимость коэффициента подъёмной силы по углу атаки $c_{y}^{\\mathrm{\\alpha}}$\nпри различных углах атаки от числа Маха', fontsize=12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

c_y_delta_1 = np.zeros_like(M)
c_y_delta_2 = np.zeros_like(M)

# График коэф-та подъёмной силы ЛА по углам отклонения ОУ c_y_delta_I от числа Маха при разных alpha
for angle_ataki in angles:
    for i in range(len(M)):
        c_y_delta_1[i] = c_y_delta_I[i] * np.radians(angle_ataki)
    plt.plot(M, c_y_delta_1, label=f'α = {angle_ataki}°', linewidth=2)
plt.xlabel('M', fontsize=12)
plt.ylabel('$c_{y}^{\\mathrm{\\delta}_{I}}$', fontsize=12)
plt.title('Зависимость коэф-та подъёмной силы по углам отклонения ОУ $c_{y}^{\\mathrm{\\delta}_{I}}$\nпри различных углах атаки от числа Маха', fontsize = 12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

# График коэф-та подъёмной силы ЛА по углам отклонения ОУ c_y_delta_II от числа Маха при разных alpha
for angle_ataki in angles:
    for i in range(len(M)):
        c_y_delta_2[i] = c_y_delta_II[i] * np.radians(angle_ataki)
    plt.plot(M, c_y_delta_2, label = f'α = {angle_ataki}°', linewidth = 2)
plt.xlabel('M', fontsize=12)
plt.ylabel('$c_{y}^{\\mathrm{\\delta}_{II}}$', fontsize=12)
plt.title('Зависимость коэф-та подъёмной силы по углам отклонения ОУ $c_{y}^{\\mathrm{\\delta}_{II}}$\nпри различных углах атаки от числа Маха', fontsize = 12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

# График c_y_alpha изолированного руля
a = np.zeros_like(M)
for angle_ataki in angles:
    for i in range(len(M)):
        c_y_alpha_is_r[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_r, c_otn, chi_05, zeta)
        a[i] = c_y_alpha_is_r[i] * np.radians(angle_ataki)
    plt.plot(M, a, label = f'α = {angle_ataki}°', linewidth = 2)
plt.xlabel('M', fontsize=14)
plt.ylabel('$c_{yиз.р}^{\\mathrm{\\alpha}}$', fontsize=14)
plt.title('График $c_{y}^{\\mathrm{\\alpha}}$ изолированного руля', fontsize = 14)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

# График c_y_alpha изолированного крыла
b = np.zeros_like(M)
for angle_ataki in angles:
    for i in range(len(M)):
        c_y_alpha_is_kr[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_kr, c_otn, chi_05, zeta)
        b[i] = c_y_alpha_is_kr[i] * np.radians(angle_ataki)
    plt.plot(M, b, label = f'α = {angle_ataki}°', linewidth = 2)
plt.xlabel('M', fontsize=14)
plt.ylabel('$c_{yиз.кр}^{\\mathrm{\\alpha}}$', fontsize=14)
plt.title('График $c_{y}^{\\mathrm{\\alpha}}$ изолированного крыла', fontsize = 14)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()

# График c_y_alpha изолированного фюзеляжа
c = np.zeros_like(M)
for angle_ataki in angles:
    for i in range(len(M)):
        c_y_alpha_is_f[i] = AeroBDSM.get_c_y_alpha_NosCil_Ell(M[i], lambda_Nos, lambda_Cil)
        c[i] = c_y_alpha_is_f[i] * np.radians(angle_ataki)
    plt.plot(M, c, label = f'α = {angle_ataki}°', linewidth = 2)
plt.xlabel('M', fontsize=14)
plt.ylabel('$c_{yиз.ф}^{\\mathrm{\\alpha}}$', fontsize=14)
plt.title('График $c_{y}^{\\mathrm{\\alpha}}$ изолированного фюзеляжа', fontsize = 14)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()


for idx, angle_ataki in enumerate(angles):
    for i in range(len(M)):
        delta_I = 10 # угол поворота первой несущей поверхности - руля
        delta_II = 0 # угол поворота второй несущей поверхности - крыла

## Коэффициент лобового сопротивления (продолжение)

        # Коэффициент индуктивного сопротивления
        if M[i] > 1:
            sigma[i] = AeroBDSM.get_sigma_cp_Nos_Con(M[i], lambda_Nos).Value
        else:
            sigma[i] = 0

        delta_c_x1[i] = 2 * sigma[i] * sin(np.radians(angle_ataki)) * sin(np.radians(angle_ataki))

        c_y_f[i] = calculate_fuselage_lift_coefficient(c_y_alpha_is_f[i], angle_ataki)

        c_x_i_f[i] = (c_y_f[i] + c_x0_f[i] * sin(np.radians(angle_ataki))) * tan(np.radians(angle_ataki)) + delta_c_x1[i] * cos(np.radians(angle_ataki))

        A_kr[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_kr[i])
        A_r[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_r[i])

        c_nI_cx[i] = calculate_wing_normal_coefficient(c_y_alpha_is_r[i], angle_ataki, A_r[i])
        c_nII_cx[i] = calculate_wing_normal_coefficient(c_y_alpha_is_kr[i], angle_ataki, A_kr[i])

        # вторая часть формул ниже не написана, так как кси = 0 изза ромбовидного профиля (нет подсасывающей силы)
        c_x_i_I[i] = c_nI_cx[i] * (sin(np.radians(angle_ataki + delta_I)) + (K_aa_r[i] - k_aa_r[i]) * sin(np.radians(angle_ataki)) * cos(np.radians(delta_I)) / k_aa_r[i])
        c_x_i_II[i] = c_nII_cx[i] * (sin(np.radians(angle_ataki + delta_II)) + (K_aa_kr[i] - k_aa_kr[i]) * sin(np.radians(angle_ataki)) * cos(np.radians(delta_II)) / k_aa_kr[i])

        c_x_i[i] = c_x_i_f[i] * S_f / S_cx + c_x_i_I[i] * S_r / S_cx + c_x_i_II[i] * S_kr / S_cx

        c_x0[i] = 1.05 * (c_x0_f[i] * S_f / S_cx + c_x_0I[i] * k_t_I[i] * S_r / S_cx + c_x_0II[i] * k_t_II[i] * S_kr / S_cx)

        c_x_fuselage[i] = c_x0_f[i] * S_f / S_cx + c_x_i_f[i] * S_f / S_cx
        c_x_wing[i] = c_x_0II[i] * S_kr / S_cx + c_x_i_II[i] * S_kr / S_cx
        c_x_fins[i] = c_x_0I[i] * S_r / S_cx + c_x_i_I[i] * S_r / S_cx

        # Сохранение данных для каждого угла атаки
        c_x0_f_data[idx, i] = c_x0_f[i]
        c_x_0I_data[idx, i] = c_x_0I[i]
        c_x_0II_data[idx, i] = c_x_0II[i]
        c_x_i_f_data[idx, i] = c_x_i_f[i]
        c_x_i_I_data[idx, i] = c_x_i_I[i]
        c_x_i_II_data[idx, i] = c_x_i_II[i]
        c_x_i_data[idx, i] = c_x_i[i]
        c_x0_data[idx, i] = c_x0[i]
        c_x_data[idx, i] = c_x0[i] + c_x_i[i]

        c_x_fuselage_data[idx, i] = c_x_fuselage[i]
        c_x_wing_data[idx, i] = c_x_wing[i]
        c_x_fins_data[idx, i] = c_x_fins[i]

plt.figure(figsize=(10, 6))
for idx, angle_ataki in enumerate(angles):
    plt.plot(M, c_x_fuselage_data[idx, :], linewidth=2, label=f'α={angle_ataki}°')
plt.xlabel('M', fontsize = 12)
plt.ylabel('$c_{x}$', fontsize = 12)
plt.title('Зависимость $c_{x}$\nпри различных малых углах атаки от числа Маха', fontsize = 12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()


Delta_x_Fa_f = np.zeros_like(M)
x_Fa_f = np.zeros_like(M)
x_Fa_I = np.zeros_like(M)
x_Fa_II = np.zeros_like(M)
x_Fa = np.zeros_like(M)

x_F_is_r_otn = np.zeros_like(M)
x_F_is_r = np.zeros_like(M)
x_F_Delta_r = np.zeros_like(M)
x_Fi_f_r = np.zeros_like(M)
f_1 = np.zeros_like(M)

x_Fb_r_otn = np.zeros_like(M)
F_r = np.zeros_like(M)
F_1_r = np.zeros_like(M)

x_F_is_kr_otn = np.zeros_like(M)
x_F_is_kr = np.zeros_like(M)
x_F_Delta_kr = np.zeros_like(M)
x_Fi_f_kr = np.zeros_like(M)

x_Fb_kr_otn = np.zeros_like(M)
F_kr = np.zeros_like(M)
F_1_kr = np.zeros_like(M)

x_F_delta_I = np.zeros_like(M)
x_F_delta_II = np.zeros_like(M)
x_FdI = np.zeros_like(M)
x_FdII = np.zeros_like(M)

mz = np.zeros_like(M)
mz_a = np.zeros_like(M)
mz_dI = np.zeros_like(M)
mz_dII = np.zeros_like(M)

# Исходя из геометрии моего крыла и рис. 5.1 стр. 253
x_Ak = 0
b_Ak_r = 0.025
b_Ak_kr = 0.1

mz_omega_z_I = np.zeros_like(M)
mz_omega_z_II = np.zeros_like(M)
mz_omega_z = np.zeros_like(M)
delta_mz = np.zeros_like(M)
mz_alpha_dot = np.zeros_like(M)
mz_delta_dot = np.zeros_like(M)
mz_10 = np.zeros_like(M)
## Момент тангажа
for delta_I in deltas:
    for i in range(len(ataka)):
        delta_II = 0
        Mach = 2
        # Координата фокусу по углу атаки фюзеляжа
        Delta_x_Fa_f[i] = AeroBDSM.get_Delta_bar_x_Falpha_NosCil(Mach, lambda_Nos, lambda_Cil)
        x_Fa_f[i] = d / 2 - (2 * pi * (d / 2)**3 / 3) / S + Delta_x_Fa_f[i]
        
        # Координата фокусу по углу атаки первого пояса (рулей)
        x_F_is_r_otn[i] = AeroBDSM.get_bar_x_Falpha_IsP(Mach, lambda_r, chi_05, zeta)
        x_F_is_r[i] = x_Ak + b_Ak_r * x_F_is_r_otn[i]

        f_1[i] = AeroBDSM.get_Delta_bar_z_Falpha_iC(d)
        x_F_Delta_r[i] = x_F_is_r[i] - f_1[i] * tan(chi_05)

        x_b_r = 0.355 # координата начала бортовой хорды руля (от носа)
        x_Fb_r_otn[i] = x_F_is_r_otn[i] + 0.02 * lambda_r * tan(chi_05)

        if Mach <= 1:
            x_Fi_f_r[i] = x_b_r + b_r * x_Fb_r_otn[i]

            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]
            K_aa_kr[i] = K_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            K_aa_r[i] = K_aa_zv * kappa_ps_r[i] * kappa_M * kappa_nos_r
        else:
            L_xv_otn_r[i] = L_xv_r / (pi / 2 * d * sqrt(Mach**2 - 1))
            b_otn_r[i] = b_r / (pi / 2 * d * sqrt(Mach**2 - 1))
            FL_xv_r[i] = 1 - (sqrt(pi)) / (2 * b_otn_r[i] * sqrt(c)) * (FI1_r - FI2_r)
            
            # Коэф-ты интерференции:
            if L_xv_r > (pi / 2 * d * sqrt(Mach**2 - 1)):
                K_aa_r[i] = K_aa_teor * kappa_ps_r[i] * kappa_M * kappa_nos_r
            else:
                K_aa_r[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_r[i]) * kappa_ps_r[i] * kappa_M * kappa_nos_r

            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]    

            F_r[i] = FL_xv_r[i]
            F_1_r[i] = 1 - 1 / (c * b_otn_r[i]**2) * (exp(0 - c * L_xv_otn_r[i]**2) - exp(0 - c * (b_otn_r[i] + L_xv_otn_r[i])**2)) + sqrt(pi) / (b_otn_r[i] * sqrt(c)) * FI2_r
            x_Fi_f_r[i] = x_b_r + b_r * x_Fb_r_otn[i] * F_r[i] * F_1_r[i]
        
        x_Fa_I[i] = 1 / K_aa_r[i] * (x_F_is_r[i] + (k_aa_r[i] - 1) * x_F_Delta_r[i] + (K_aa_r[i] - k_aa_r[i]) * x_Fi_f_r[i])

        # Координата фокусу по углу атаки второго пояса (крыльев)
        x_F_is_kr_otn[i] = AeroBDSM.get_bar_x_Falpha_IsP(Mach, lambda_kr, chi_05, zeta)
        x_F_is_kr[i] = x_Ak + b_Ak_kr * x_F_is_kr_otn[i]

        f_1[i] = AeroBDSM.get_Delta_bar_z_Falpha_iC(d)
        x_F_Delta_kr[i] = x_F_is_kr[i] - f_1[i] * tan(chi_05)

        x_b_kr = 1.26 # координата начала бортовой хорды руля (от носа)
        x_Fb_kr_otn[i] = x_F_is_kr_otn[i] + 0.02 * lambda_kr * tan(chi_05)

        if Mach <= 1:
            x_Fi_f_kr[i] = x_b_kr + b_kr * x_Fb_kr_otn[i]

            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]
            K_aa_kr[i] = K_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            K_aa_r[i] = K_aa_zv * kappa_ps_r[i] * kappa_M * kappa_nos_r
        else:
            L_xv_otn_kr[i] = L_xv_kr / (pi / 2 * d * sqrt(Mach**2 - 1))
            b_otn_kr[i] = b_kr / (pi / 2 * d * sqrt(Mach**2 - 1))
            FL_xv_kr[i] = 1 - (sqrt(pi)) / (2 * b_otn_kr[i] * sqrt(c)) * (FI1_kr - FI2_kr)
            
            if L_xv_r > (pi / 2 * d * sqrt(Mach**2 - 1)):
                K_aa_r[i] = K_aa_teor * kappa_ps_r[i] * kappa_M * kappa_nos_r
            else:
                K_aa_r[i] = (k_aa_zv + (K_aa_zv - k_aa_zv) * FL_xv_r[i]) * kappa_ps_r[i] * kappa_M * kappa_nos_r

            k_aa_kr[i] = k_aa_zv * kappa_ps_kr[i] * kappa_M * kappa_nos_kr
            k_aa_r[i] = k_aa_kr[i]    

            F_kr[i] = FL_xv_kr[i]
            F_1_kr[i] = 1 - 1 / (c * b_otn_kr[i]**2) * (exp(0 - c * L_xv_otn_kr[i]**2) - exp(0 - c * (b_otn_kr[i] + L_xv_otn_kr[i])**2)) + sqrt(pi) / (b_otn_kr[i] * sqrt(c)) * FI2_kr
            x_Fi_f_kr[i] = x_b_kr + b_kr * x_Fb_kr_otn[i] * F_kr[i] * F_1_kr[i]
        
        x_Fa_II[i] = 1 / K_aa_kr[i] * (x_F_is_kr[i] + (k_aa_kr[i] - 1) * x_F_Delta_kr[i] + (K_aa_kr[i] - k_aa_kr[i]) * x_Fi_f_kr[i])

        c_y_alpha_is_f[i] = AeroBDSM.get_c_y_alpha_NosCil_Ell(Mach, lambda_Nos, lambda_Cil) / 57.3
        c_y_alpha_is_kr[i] = AeroBDSM.get_c_y_alpha_IsP(Mach, lambda_kr, c_otn, chi_05, zeta) / 57.3
        c_y_alpha_is_r[i] = AeroBDSM.get_c_y_alpha_IsP(Mach, lambda_r, c_otn, chi_05, zeta) / 57.3
        delta_otn_zv_kr[i] = 0.093 * L1_kr * (1 + 0.4 * Mach + 0.147 * Mach**2 - 0.006 * Mach**3) / (d * (Mach * L1_kr / nu)**(1/5))
        kappa_ps_kr[i] = (1 - (2 * D_otn**2) / (1 - D_otn**2) * delta_otn_zv_kr[i]) * (1 - (D_otn * (eta_k - 1)) / ((1 - D_otn) * (eta_k + 1)) * delta_otn_zv_kr[i])
        delta_otn_zv_r[i] = 0.093 * L1_r * (1 + 0.4 * Mach + 0.147 * Mach**2 - 0.006 * Mach**3) / (d * (Mach * L1_r / nu)**(1/5))
        kappa_ps_r[i] = (1 - (2 * D_otn**2) / (1 - D_otn**2) * delta_otn_zv_r[i]) * (1 - (D_otn * (eta_k - 1)) / ((1 - D_otn) * (eta_k + 1)) * delta_otn_zv_r[i])

        kappa_q_Nos_Con[i] = AeroBDSM.get_kappa_q_Nos_Con(Mach, lambda_Nos) # коэф торможения потока, вызванный обтеканием коническиой носовой части
        kappa_q_IsP[i] = AeroBDSM.get_kappa_q_IsP(Mach, L_A, b_r)
        k_t_I[i] = kappa_q_Nos_Con[i]
        k_t_II[i] = kappa_q_IsP[i]
        z_v_otn[i] = AeroBDSM.get_bar_z_v(Mach, lambda_kr, chi_05, zeta)
        z_v[i] = (d + z_v_otn[i] * (l_razmah - d)) / 2
        i_v[i] = AeroBDSM.get_i_v(zeta, d, l_razmah, y_v, z_v[i])
        psi_eps[i] = 1
        eps_alpha_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_r[i] * k_aa_r[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_r * K_aa_kr[i])

        c_y_alpha[i] = c_y_alpha_is_f[i] * S_f / S + c_y_alpha_is_r[i] * K_aa_r[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * (1 - eps_alpha_sr[i]/57.3) * S_kr / S * k_t_II[i]

        # Суммарная координата фокуса по углу атаки
        x_Fa[i] = 1 / c_y_alpha[i] * (c_y_alpha_is_f[i] * S_f / S * x_Fa_f[i] + c_y_alpha_is_r[i] * S_r / S * x_Fa_I[i] * k_t_I[i] + c_y_alpha_is_kr[i] * S_kr / S * x_Fa_II[i] * k_t_II[i])

        k_delta_0_zv = k_aa_zv**2 / K_aa_zv
        K_delta_0_zv = k_aa_zv

        kappa_ps_otn_kr[i] = ((1 - D_otn * (1 + delta_otn_zv_kr[i])) * (1 - (eta_k - 1) * D_otn * (1 + delta_otn_zv_kr[i]) / (eta_k + 1 - 2 * D_otn))) / \
        ((1 - D_otn) * (1 - (eta_k - 1) * D_otn / (eta_k + 1 - 2 * D_otn)))
        kappa_ps_otn_r[i] = ((1 - D_otn * (1 + delta_otn_zv_r[i])) * (1 - (eta_k - 1) * D_otn * (1 + delta_otn_zv_r[i]) / (eta_k + 1 - 2 * D_otn))) / \
        ((1 - D_otn) * (1 - (eta_k - 1) * D_otn / (eta_k + 1 - 2 * D_otn)))

        k_delta_0_kr[i] = k_delta_0_zv * kappa_ps_otn_kr[i] * kappa_M
        K_delta_0_kr[i] = (k_delta_0_zv + (K_delta_0_zv - k_delta_0_zv) * FL_xv_kr[i]) * kappa_ps_otn_kr[i] * kappa_M
        k_delta_0_r[i] = k_delta_0_zv * kappa_ps_otn_r[i] * kappa_M
        K_delta_0_r[i] = (k_delta_0_zv + (K_delta_0_zv - k_delta_0_zv) * FL_xv_r[i]) * kappa_ps_otn_r[i] * kappa_M

        # if Mach[i] > 1.4:
        #     k_sch[i] = 1
        # else:
        #     k_sch[i] = 0.85

        if Mach > 1.4:
            k_sch[i] = 0.85 + (Mach - 1.4) * (1 - 0.85) / (Mach - 1.4)
        else:
            k_sch[i] = 0.85

        n_I[i] = k_sch[i] * cos(chi_p)
        n_II[i] = k_sch[i] * cos(chi_p)
        eps_delta_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_r[i] * k_delta_0_r[i] * n_I[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_r * K_aa_kr[i])
        c_y_delta_I[i] = (c_y_alpha_is_r[i] * K_delta_0_r[i] * n_I[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * eps_delta_sr[i] * S_kr / S * k_t_II[i])
        c_y_delta_II[i] = (c_y_alpha_is_kr[i] * K_delta_0_kr[i] * S_kr / S * k_t_II[i])

        # Координата фокуса по углу отклонения несущих поверхностей
        x_F_delta_I[i] = 1 / K_delta_0_r[i] * (k_delta_0_r[i] * x_F_is_r[i] + x_Fi_f_r[i] * (K_delta_0_r[i] - k_delta_0_r[i]))
        x_FdI[i] = 1 / c_y_delta_I[i] * (c_y_alpha_is_r[i] * K_delta_0_r[i] * n_I[i] * S_r / S * k_t_I[i] * x_F_delta_I[i])

        x_F_delta_II[i] = 1 / K_delta_0_kr[i] * (k_delta_0_kr[i] * x_F_is_kr[i] + x_Fi_f_kr[i] * (K_delta_0_kr[i] - k_delta_0_kr[i]))
        x_FdII[i] = 1 / c_y_delta_II[i] * (c_y_alpha_is_kr[i] * K_delta_0_kr[i] * n_II[i] * S_kr / S * k_t_II[i] * x_F_delta_II[i])

        x_t = 0.6938
        mz_a[i] = c_y_alpha[i] * ((x_t - x_Fa[i]) / l_f)
        mz_dI[i] = c_y_delta_I[i] * ((x_t - x_FdI[i]) / l_f)
        mz_dII[i] = c_y_delta_II[i] * ((x_t - x_FdII[i]) / l_f)
        mz_10 = 0.01

        omega = 120 #рад/с, в Интернете нашел, что скорость вращения 15-20 об/с
        V = Mach * 340
        omega_z = omega * l_f / V
        alpha_dot = 0.1
        alpha_dot_otn = alpha_dot * l_f / V
        delta_dot = 0.87
        delta_dot_otn = delta_dot * l_f / V

        x_c_ob = 0.8
        mz_omega_z_f = -2 * (1 - x_t / l_f + (x_t / l_f)**2 - x_c_ob / l_f)
        x_t_otn = (x_t - x_Ak) / b_Ak_r
        mz_omega_z_I[i] = -57.3 * (c_y_alpha_is_r[i] * K_aa_r[i] * (x_t_otn - 1 / 2)**2)
        delta_mz[i] = -57.3 * c_y_alpha_is_kr[i] * K_aa_kr[i] * eps_alpha_sr[i] * (x_t - x_b_r + b_r / 2) / b_kr * (x_t - x_Fa_II[i]) / b_kr
        mz_omega_z_II[i] = -57.3 * (c_y_alpha_is_kr[i] * K_aa_kr[i] * (x_t_otn - 1 / 2)**2) + delta_mz[i]
        mz_omega_z[i] = mz_omega_z_f * S_f / S * 1**2 + mz_omega_z_I[i] * S_r / S * (b_r / l_f)**2 * sqrt(k_t_I[i]) + mz_omega_z_II[i] * S_kr / S * (b_kr / l_f)**2 * sqrt(k_t_II[i])

        mz_alpha_dot[i] = -57.3 * c_y_alpha_is_kr[i] * K_aa_kr[i] * S_kr / S * sqrt(k_t_II[i]) * eps_alpha_sr[i] * ((x_Fa_II[i] - x_t) / l_f) * ((x_b_kr + b_kr / 2 - x_b_r + b_r / 2) / l_f)
        mz_delta_dot[i] = mz_alpha_dot[i] * eps_delta_sr[i] / eps_alpha_sr[i]

for delta_I in deltas:
    for i in range(len(ataka)):
        mz[i] = mz_10 + mz_a[i] * np.radians(ataka[i]) + mz_dI[i] * np.radians(delta_I) + mz_dII[i] * np.radians(delta_II) \
        #+ mz_omega_z[i] * omega_z + mz_alpha_dot[i] * alpha_dot_otn + mz_delta_dot[i] * delta_dot_otn
    plt.plot(ataka, mz, label=f'delta = {delta_I}°', linewidth=2)
plt.xlabel('alpha', fontsize = 12)
plt.ylabel('$mz$', fontsize = 12)
plt.title('', fontsize = 12)
plt.legend(fontsize = 10, loc = 'best')
plt.grid(True, alpha = 0.3)
plt.minorticks_on()
plt.grid(True, which = 'major', linestyle = '-', alpha = 0.5)
plt.grid(True, which = 'minor', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()