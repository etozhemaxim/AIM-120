import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, tan, sqrt, fabs, radians

# Настройки графиков
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"

# =============================================================================
# ФУНКЦИЯ РАСЧЕТА ДОННОГО СОПРОТИВЛЕНИЯ ПО НОВОЙ ЗАВИСИМОСТИ
# =============================================================================

def sigma(x):
    """Сигмоид-функция"""
    return 1 / (1 + exp(-x))

def A_func(x):
    """Функция A(x) = tan(x² - 1) / |x² - 1|"""
    x2_1 = x**2 - 1
    if fabs(x2_1) < 1e-10:  # избегаем деления на 0
        return 0
    return tan(x2_1) / fabs(x2_1)

def calculate_bottom_drag_coefficient(M, lambda_Cil, c_f, eta_k):
    """
    Расчет коэффициента донного сопротивления по новой зависимости
    """
    # Для M <= 0.8 используем старую формулу
    if M <= 0.8:
        return 0.0155 / sqrt(lambda_Cil * c_f) * eta_k**2
    
    # Для M > 0.8 используем новую кусочно-заданную функцию
    if M <= 0.723672:
        # F1(M) = ln(1.410839 + 1.0432458*A(M) + 1.167756*A(M)^2 + 0.43818533*A(M)^3)
        A_val = A_func(M)
        F = log(1.410839 + 1.0432458 * A_val + 1.167756 * A_val**2 + 0.43818533 * A_val**3)
        
    elif M <= 0.949985:
        # F2(M) = exp(-1.1994635 + 4.1244161*A(M) + 5.4425535*A(M)^2 + 2.6464556*A(M)^3)
        A_val = A_func(M)
        F = exp(-1.1994635 + 4.1244161 * A_val + 5.4425535 * A_val**2 + 2.6464556 * A_val**3)
        
    elif M <= 1.045254:
        # F3(M) = ln(1508.25 - 6119.1935*(1/M) + 9304.258*(1/M^2) - 6277.8414*(1/M^3) + 1585.749*(1/M^4))
        inv_M = 1 / M
        inv_M2 = inv_M**2
        inv_M3 = inv_M**3
        inv_M4 = inv_M**4
        F = log(1508.25 - 6119.1935 * inv_M + 9304.258 * inv_M2 - 6277.8414 * inv_M3 + 1585.749 * inv_M4)
        
    elif M <= 1.335822:
        # F4(M) = exp(-78.455781 + 249.63042*M - 302.51885*M^2 + 162.4847*M^3 - 32.690523*M^4)
        M2 = M**2
        M3 = M**3
        M4 = M**4
        F = exp(-78.455781 + 249.63042 * M - 302.51885 * M2 + 162.4847 * M3 - 32.690523 * M4)
        
    elif M <= 3.74289:
        # F5(M) = tan(4.6685604 - 15.483104*σ(M) + 18.393009*σ(M)^2 - 7.5350128*σ(M)^3)
        sigma_val = sigma(M)
        sigma2 = sigma_val**2
        sigma3 = sigma_val**3
        F = tan(4.6685604 - 15.483104 * sigma_val + 18.393009 * sigma2 - 7.5350128 * sigma3)
        
    else:
        # F6(M) = 1.43/M^2 - 0.772/M^2 * (1 - 0.011*M^2)^3.5
        M2 = M**2
        term1 = 1.43 / M2
        term2 = 0.772 / M2 * (1 - 0.011 * M2)**3.5
        F = term1 - term2
    
    return F

# =============================================================================
# ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ ИЗ СУЩЕСТВУЮЩЕГО КОДА
# =============================================================================

# Импортируем необходимые функции и данные из основного кода
import AeroBDSM
from math import *

# Основные параметры (из исходного кода)
N = 1000
l_f = 3.906
b_kr = 0.2
b_op = 0.316
l_raszmah = 0.498
S_Kons = 4 * (b_kr * l_raszmah) * 0.1
d = 0.178
S_op = 0.0996
l_raszmah_op = 0.651

# Расчетные параметры
l_raszmah_kons = l_raszmah - d
lamb = (l_raszmah_kons**2) / S_Kons
lamb_op = (l_raszmah_op**2) / S_op
bar_c = 0.04
chi_05 = 0
zeta = 1
lambda_Nos = 2.64
lambda_Cil = 21.943

# Параметры для расчета сопротивления
nu = 15.1 * 1e-6
eta_k = 1
eta_c = 1.13
c_otn = bar_c
lambda_kr = lamb
lambda_r = lamb_op

# Параметры положения
xb = 1.632
L1 = xb + b_kr / 2 
xb_op = 3.632
L1_op = xb_op + b_op / 2

# Площади (из исходного кода)
S_f = pi * d * (l_f - d / 2) + 2 * pi * (d / 2)**2
S_char = pi * d**2 / 4
S_cx_total = S_op + S_Kons + S_f

# =============================================================================
# РАСЧЕТ СОПРОТИВЛЕНИЯ ПРИ α=0
# =============================================================================

print("Расчет лобового сопротивления при α=0...")

# Создаем массив чисел Маха от 0.1 до 4
Mach = np.linspace(0.1, 4, N)

# Инициализация массивов
c_f = np.zeros_like(Mach)
c_x_tr = np.zeros_like(Mach)
c_x_Nos = np.zeros_like(Mach)
c_x_dn_old = np.zeros_like(Mach)  # старое донное сопротивление
c_x_dn_new = np.zeros_like(Mach)  # новое донное сопротивление
c_x0_f_active = np.zeros_like(Mach)   # для активного участка (без донного)
c_x0_f_passive = np.zeros_like(Mach)  # для пассивного участка (с новым донным)

c_x_0I = np.zeros_like(Mach)
c_x_0II = np.zeros_like(Mach)
c_x_p_I = np.zeros_like(Mach)
c_x_p_II = np.zeros_like(Mach)
c_x_v_I = np.zeros_like(Mach)
c_x_v_II = np.zeros_like(Mach)

# Коэффициенты для несущих поверхностей (из исходного кода)
k_t_I = np.ones_like(Mach)
k_t_II = np.ones_like(Mach)

# Расчет для всех чисел Маха
for i in range(N):
    # Коэффициент трения
    Re = Mach[i] * 340 * l_f / nu
    c_f[i] = AeroBDSM.get_c_f0(Re, 0).Value
    
    # Трение корпуса
    c_x_tr[i] = c_f[i] * S_f / S_char
    
    # Сопротивление носовой части
    try:
        c_x_Nos[i] = AeroBDSM.get_c_x0_p_Nos_Con(Mach[i], lambda_Nos).Value
    except:
        c_x_Nos[i] = AeroBDSM.get_c_x0_p_Nos_Par(Mach[i], lambda_Nos).Value
    
    # Старое донное сопротивление (для M <= 0.8)
    if Mach[i] <= 0.8:
        c_x_dn_old[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2
    else:
        c_x_dn_old[i] = 0.0155 / sqrt(lambda_Cil * c_f[i]) * eta_k**2  # из исходного кода
    
    # Новое донное сопротивление по зависимости из картинки
    c_x_dn_new[i] = calculate_bottom_drag_coefficient(Mach[i], lambda_Cil, c_f[i], eta_k)
    
    # Сопротивление фюзеляжа для активного участка (без донного сопротивления)
    c_x0_f_active[i] = c_x_tr[i] + c_x_Nos[i]
    
    # Сопротивление фюзеляжа для пассивного участка (с новым донным сопротивлением)
    c_x0_f_passive[i] = c_x_tr[i] + c_x_Nos[i] + c_x_dn_new[i]
    
    # Сопротивление несущих поверхностей (как в исходном коде)
    c_x_p_I[i] = 2 * c_f[i] * eta_c
    c_x_p_II[i] = 2 * c_f[i] * eta_c
    
    # Волновое сопротивление несущих поверхностей
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

# =============================================================================
# РАСЧЕТ ОБЩЕГО СОПРОТИВЛЕНИЯ
# =============================================================================

print("Расчет общего коэффициента лобового сопротивления...")

# Инициализация массивов для общего сопротивления
c_x_total_active = np.zeros_like(Mach)   # активный участок
c_x_total_passive = np.zeros_like(Mach)  # пассивный участок

# Расчет общего сопротивления для активного и пассивного участков
for i in range(N):
    # Общее сопротивление при нулевой подъемной силе (α=0)
    # Активный участок (без донного сопротивления)
    c_x0_total_active = (c_x0_f_active[i] * (S_f / S_cx_total) + 
                        c_x_0I[i] * k_t_I[i] * (S_op / S_cx_total) + 
                        c_x_0II[i] * k_t_II[i] * (S_Kons / S_cx_total))
    
    # Пассивный участок (с донным сопротивлением)
    c_x0_total_passive = (c_x0_f_passive[i] * (S_f / S_cx_total) + 
                         c_x_0I[i] * k_t_I[i] * (S_op / S_cx_total) + 
                         c_x_0II[i] * k_t_II[i] * (S_Kons / S_cx_total))
    
    # Увеличиваем на 5% для учета дополнительной интерференции (как в исходном коде)
    c_x0_total_active *= 1.05
    c_x0_total_passive *= 1.05
    
    # При α=0 индуктивное сопротивление отсутствует
    c_x_total_active[i] = c_x0_total_active
    c_x_total_passive[i] = c_x0_total_passive

# =============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================

print("Построение графиков...")

# Создаем фигуру с тремя графиками
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# График 1: Общее сопротивление на активном участке (без донного)
axes[0].plot(Mach, c_x_total_active, 'b-', linewidth=2.5, label='Активный участок')
axes[0].set_xlabel('Число Маха (M)', fontsize=12)
axes[0].set_ylabel('Коэффициент лобового сопротивления $(c_{x_{a}})$', fontsize=12)
axes[0].set_title('Общее сопротивление на активном участке\n(без донного сопротивления, α=0)', fontsize=14)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(fontsize=10, loc='best')
axes[0].set_xlim(0, 4)
axes[0].set_ylim(bottom=0)

# График 2: Общее сопротивление на пассивном участке (с донным)
axes[1].plot(Mach, c_x_total_passive, 'r-', linewidth=2.5, label='Пассивный участок')
axes[1].set_xlabel('Число Маха (M)', fontsize=12)
axes[1].set_ylabel('Коэффициент лобового сопротивления $(c_{x_{a}})$', fontsize=12)
axes[1].set_title('Общее сопротивление на пассивном участке\n(с донным сопротивлением, α=0)', fontsize=14)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(fontsize=10, loc='best')
axes[1].set_xlim(0, 4)
axes[1].set_ylim(bottom=0)

# График 3: Донное сопротивление
axes[2].plot(Mach, c_x_dn_new, 'g-', linewidth=2.5, label='Донное сопротивление')
axes[2].set_xlabel('Число Маха (M)', fontsize=12)
axes[2].set_ylabel('Коэффициент донного сопротивления', fontsize=12)
axes[2].set_title('Донное сопротивление от числа Маха\n(по новой зависимости)', fontsize=14)
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].legend(fontsize=10, loc='best')
axes[2].set_xlim(0, 4)
axes[2].set_ylim(bottom=0)

# Добавляем вертикальные линии для границ кусочной функции
boundaries = [0.723672, 0.949985, 1.045254, 1.335822, 3.74289]
boundary_labels = ['0.724', '0.950', '1.045', '1.336', '3.743']

for ax_idx in [2]:  # Только на графике донного сопротивления
    for i, boundary in enumerate(boundaries):
        axes[ax_idx].axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        if i < len(boundary_labels):
            axes[ax_idx].text(boundary, axes[ax_idx].get_ylim()[1]*0.95, 
                            boundary_labels[i], rotation=90, fontsize=8,
                            verticalalignment='top', horizontalalignment='right')

# Общие настройки
plt.suptitle('Зависимости коэффициентов сопротивления от числа Маха при α=0°', fontsize=16, y=1.02)
plt.tight_layout()

# Сохранение графика

plt.show()

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: СРАВНЕНИЕ ВСЕХ ТРЕХ КРИВЫХ НА ОДНОМ ГРАФИКЕ
# =============================================================================

print("Построение сводного графика...")

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Строим все три кривые на одном графике
ax2.plot(Mach, c_x_total_active, 'b-', linewidth=2.5, label='Активный участок (без донного)')
ax2.plot(Mach, c_x_total_passive, 'r-', linewidth=2.5, label='Пассивный участок (с донным)')
ax2.plot(Mach, c_x_dn_new, 'g-', linewidth=2.5, label='Донное сопротивление')

# Настройки графика
ax2.set_xlabel('Число Маха (M)', fontsize=14)
ax2.set_ylabel('Коэффициент сопротивления', fontsize=14)
ax2.set_title('Сравнение коэффициентов сопротивления от числа Маха при α=0°', fontsize=16)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=12, loc='best')
ax2.set_xlim(0, 4)
ax2.set_ylim(bottom=0)

# Добавляем пояснения
ax2.text(0.5, 0.95, 'Разница между кривыми:\nКрасная - Синяя = Зеленая', 
         transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

plt.show()

# =============================================================================
# ВЫВОД ИНФОРМАЦИИ О РАСЧЕТАХ
# =============================================================================

print("\n" + "="*70)
print("ИНФОРМАЦИЯ О РАСЧЕТАХ:")
print("="*70)

# Выбираем несколько характерных чисел Маха для отображения
sample_mach_numbers = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

print(f"\n{'M':>6} | {'c_x_active':>12} | {'c_x_passive':>12} | {'c_x_dn':>12} | {'Разница':>10}")
print("-" * 70)

for M in sample_mach_numbers:
    # Находим ближайший индекс
    idx = np.argmin(np.abs(Mach - M))
    
    c_x_act = c_x_total_active[idx]
    c_x_pass = c_x_total_passive[idx]
    c_x_dn_val = c_x_dn_new[idx]
    diff = c_x_pass - c_x_act
    
    print(f"{M:6.2f} | {c_x_act:12.6f} | {c_x_pass:12.6f} | {c_x_dn_val:12.6f} | {diff:10.6f}")

print("\nГрафики успешно построены и сохранены!")
print("1. График 1: Общее сопротивление на активном участке (без донного)")
print("2. График 2: Общее сопротивление на пассивном участке (с донным)")
print("3. График 3: Донное сопротивление от числа Маха")
print("4. График 4: Сводный график со всеми тремя кривыми")