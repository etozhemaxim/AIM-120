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

d = 0.18                                          # Калибр
N = 4000                                            # Число шагов
M = np.linspace(0.001, 4, N + 1)                    # Скорость: от 0 до 2 Махов
ataka = np.linspace(-10, 10, N + 1)
d_ruli = np.linspace(-25, 25, N + 1)    
H = np.linspace(0, 30000, N + 1)
l_f = 3.700                                         # Длина фюзеляжа
lambda_Nos = 3.25                            # Удлинение носовой части
lambda_Cil = 17.305                      # Удлинение цилиндрической части
S_f = pi * d * (l_f - d / 2) + 2 * pi * (d / 2)**2  # Площадь поверхности фюзеляжа - цилиндр + половина сферы
c_otn = 0.05                                        # Относительная площадь крыла 
chi_05 = 0
zeta = 1                                            # Обратное сужение несущей поверхности
eta_k = 1
b_kr = 0.8                                          # Ширина крыла по бортовой хорде
b_r = 0.025                                         # Ширина руля по бортовой хорде
nu = 15.1e-6
L_xv_kr = 1.98                                     # Расстояние от середины бортовой хорды крыла до кормового среза
L_xv_r = l_f - 0.882                                # Расстояние от середины бортовой хорды руля до кормового среза
L_A = 0.882+0.05
l_razmah = 0.64                                   # Полный размах
l_razmah_r = 0.35                                  # Размах рулей без фюзеляжа
S_r = 2 * 0.033 * 0.025                             # Площадь рулей
lambda_r = (l_razmah_r**2) / (S_r + b_kr**2)
l_razmah_kr = l_razmah - d                          # Размах крыльев без фюзеляжа
S_kr = 2 * 0.036 * 0.1                              # Площадь крыльев
lambda_kr = (l_razmah_kr**2) / S_kr
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

# angles = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
angles = [10, 8, 6, 4, 2, 0]
deltas = [25, 20, 15, 10, 5, 0, 5, 10, 15, 20, 25]

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

S = S_r + S_kr + S_f # характерная площадь
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

c_y_delta_1 = np.zeros_like(M)
c_y_delta_2 = np.zeros_like(M)

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
c_y_b = np.zeros_like(M)

for angle_ataki in angles:
    for i in range(len(M)):
        delta_I = 10 # угол поворота первой несущей поверхности - руля
        delta_II = 0 # угол поворота второй несущей поверхности - крыла
## Производная коэф-та подъёмной силы ЛА по углу атаки c_y_alpha
        c_y_alpha_is_f[i] = AeroBDSM.get_c_y_alpha_NosCil_Ell(M[i], lambda_Nos, lambda_Cil)
        c_y_alpha_is_kr[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_kr, c_otn, chi_05, zeta)
        c_y_alpha_is_r[i] = AeroBDSM.get_c_y_alpha_IsP(M[i], lambda_r, c_otn, chi_05, zeta)
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
        eps_alpha_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_kr[i] * k_aa_kr[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_kr * K_aa_kr[i])
        #print(psi_eps[i], eps_alpha_sr[i], i)

        # Коэффициент торможения потока
        kappa_q_Nos_Con[i] = AeroBDSM.get_kappa_q_Nos_Con(M[i], lambda_Nos) # коэф торможения потока, вызванный обтеканием коническиой носовой части
        kappa_q_IsP[i] = AeroBDSM.get_kappa_q_IsP(M[i], L_A, b_r)
        k_t_I[i] = kappa_q_Nos_Con[i]
        k_t_II[i] = kappa_q_IsP[i]

        c_y_alpha[i] = c_y_alpha_is_f[i] * S_f / S + c_y_alpha_is_r[i] * K_aa_r[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * (1 - eps_alpha_sr[i]/57.3) * S_kr / S * k_t_II[i]
        #print(f"Угол атаки: {angle_ataki}°, M = {M[i]:.2f}, c_y_alpha = {c_y_alpha[i]:.6f}")

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

        if M[i] > 1.4:
            k_sch[i] = 1
        else:
            k_sch[i] = 0.85

        n_I[i] = k_sch[i] * cos(chi_p)
        n_II[i] = k_sch[i] * cos(chi_p)

        eps_delta_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_y_alpha_is_kr[i] * k_delta_0_kr[i] * n_I[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_kr * K_aa_kr[i])
        # sqrt(2) - учет крестокрылого ЛА (вариант ХХ) стр. 195
        c_y_delta_I[i] = sqrt(2) * (c_y_alpha_is_r[i] * K_delta_0_r[i] * n_I[i] * S_r / S * k_t_I[i] + c_y_alpha_is_kr[i] * K_aa_kr[i] * eps_delta_sr[i] * S_kr / S * k_t_II[i])
        c_y_delta_II[i] = sqrt(2) * (c_y_alpha_is_kr[i] * K_delta_0_kr[i] * n_II[i] * S_kr / S * k_t_II[i])
        #print(f"Угол атаки: {angle_ataki}°, M = {M[i]:.2f}, c_y_delta_I = {c_y_delta_I[i]:.6f}, c_y_delta_II = {c_y_delta_II[i]:.6f}")

## Коэффициент подъёмной силы при больших углах alpha и delta

        # Коэффициент подъёмной силы корпуса
        kappa_alpha[i] = 1 - 0.45 * (1 - exp(-0.06 * M[i]**2)) * (1 - exp(-0.12 * np.radians(abs(angle_ataki))))
        M_y[i] = M[i] * sin(np.radians(angle_ataki))
        c_y_perp_Cil[i] = AeroBDSM.get_c_yPerp_Cil(M_y[i])
        # c_y_f[i] = 57.3 * c_y_alpha_is_f[i] * kappa_alpha[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) \
        # + 4 * S_bok * c_y_perp_Cil[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki) / (pi * d**2)

        A_kr[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_kr[i])
        c_y_is_kr[i] = 57.3 * c_y_alpha_is_kr[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) + A_kr[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki)
        A_r[i] = AeroBDSM.get_A_IsP(M[i], zeta, c_y_alpha_is_r[i])
        c_y_is_r[i] = 57.3 * c_y_alpha_is_r[i] * sin(np.radians(angle_ataki)) * cos(np.radians(angle_ataki)) + A_r[i] * (sin(np.radians(angle_ataki)))**2 * sign(angle_ataki)

        eps_sr[i] = (57.3 * i_v[i] * l_razmah_r * c_nI[i] * psi_eps[i]) / (2 * pi * z_v_otn[i] * l_razmah_kr * lambda_kr * K_aa_kr[i])

        alpha_eff_I[i] = k_aa_r[i] * np.radians(angle_ataki) + k_delta_0_r[i] * n_I[i] * np.radians(delta_I)
        alpha_eff_II[i] = k_aa_kr[i] * (np.radians(angle_ataki) - eps_sr[i]) + k_delta_0_kr[i] * n_II[i] * np.radians(delta_II)
        c_nI[i] = 57.3 * c_y_alpha_is_r[i] * sin(alpha_eff_I[i]) * cos(alpha_eff_I[i]) + A_r[i] * (sin(alpha_eff_I[i]))**2 * sign(alpha_eff_I[i])
        c_nII[i] = 57.3 * c_y_alpha_is_kr[i] * sin(alpha_eff_II[i]) * cos(alpha_eff_II[i]) + A_kr[i] * (sin(alpha_eff_II[i]))**2 * sign(alpha_eff_II[i])
        #c_y_I[i] = c_nI[i] * (K_aa_r[i] / k_aa_r[i] * cos(np.radians(angle_ataki)) * cos(np.radians(delta_I))) - c_x_0I * sin(np.radians(angle_ataki + delta_I))
        #c_y_II[i] = c_nII[i] * (K_aa_kr[i] / k_aa_kr[i] * cos(np.radians(angle_ataki)) * cos(np.radians(delta_II))) - c_x_0II * sin(np.radians(angle_ataki + delta_II))

## Итоговый коэффициент подъёмной силы ЛА
        # c_y при малых углах:
        c_y[i] = c_y_alpha[i] * np.radians(angle_ataki) + c_y_delta_I[i] * np.radians(delta_I) + c_y_delta_II[i] * np.radians(delta_II)

        # c_y при больших углах:
        c_y_b[i] = c_y_f[i] * S_f / S + c_y_I[i] * S_r / S * k_t_I[i] + c_y_II[i] * S_kr / S * k_t_II[i]


## Построение графиков
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