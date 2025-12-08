import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

angles_big_deg = [-25, -20, -15, -10, 10, 15,20 , 25]
deltas = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
M_values = np.arange(0.5, 4.1 , 0.1)
# # # Чтение данных
data = pd.read_csv('all_angles_data.csv')

# Функция для расчета коэффициента торможения (аналогичная C++ версии)
# def calculate_kappa_q_nos_con(M, lambda_nos):
#     # Проверка граничных условий
#     if lambda_nos < 0.7 or lambda_nos > 5.0:
#         return 1.0
    
#     # Случай дозвуковых скоростей
#     if M <= 1.0:
#         return 1.0
    
#     # Данные для M=2.0
#     lambda_values_M2 = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
#     kappa_values_M2 = [0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.75, 0.74]
    
#     # Интерполяция для M=2
#     kappa_at_M2 = np.interp(lambda_nos, lambda_values_M2, kappa_values_M2)
    
#     # Интерполяция между M=1 и M=2
#     if M <= 2.0:
#         return 1.0 + (M - 1.0) * (kappa_at_M2 - 1.0)
#     else:
#         # Для M > 2 используем значение при M=2
#         return kappa_at_M2

# # Параметры фюзеляжа (должны совпадать с C++ кодом)
# l_nos = 0.47  # длина носовой части
# D = 0.178     # диаметр
# lambda_nos = l_nos / D

# # Применяем торможение потока к данным
# data_with_kappa = data.copy()

# # Добавляем столбец с коэффициентом торможения
# data_with_kappa['kappa_q'] = data_with_kappa['Mach'].apply(
#     lambda M: calculate_kappa_q_nos_con(M, lambda_nos)
# )

# # Применяем торможение ко всем коэффициентам подъемной силы
# angles = [-10, -5, 0, 5, 10]
# for angle in angles:
#     col = f'alpha_{angle}'
#     data_with_kappa[col + '_with_kappa'] = data_with_kappa[col] * data_with_kappa['kappa_q']

# # Создание фигуры с двумя графиками бок о бок
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α (С ТОРМОЖЕНИЕМ И БЕЗ)
# for i, angle in enumerate(angles):
#     col = f'alpha_{angle}'
    
#     # Без торможения (исходные данные) - пунктирная линия
#     ax1.plot(data['Mach'], data[col], 
#              color=colors[i % len(colors)], 
#              linewidth=2,
#              linestyle='--',
#              alpha=0.7,
#              label=fr'$\alpha = {angle}^\circ$ (без торможения)')
    
#     # С торможением - сплошная линия
#     ax1.plot(data_with_kappa['Mach'], data_with_kappa[col + '_with_kappa'], 
#              color=colors[i % len(colors)], 
#              linewidth=3,
#              linestyle='-',
#              label=fr'$\alpha = {angle}^\circ$ (с торможением)')

# ax1.set_xlabel('M',fontsize=15 )
# ax1.set_ylabel(r'$C_y\text{из.ф}$', fontsize=16)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right', fontsize=10)

# # ПРАВЫЙ ГРАФИК: C_y vs α для разных M (С ТОРМОЖЕНИЕМ И БЕЗ)
# selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# print(f"Выбранные числа Маха: {selected_mach}")

# # Строим правый график
# for j, mach in enumerate(selected_mach):
#     # Находим ближайшее число Маха в данных
#     idx = (data['Mach'] - mach).abs().idxmin()
#     closest_mach = data['Mach'].iloc[idx]
    
#     # Данные без торможения
#     mach_data = data.iloc[idx]
#     cy_values_no_kappa = []
#     for angle in angles:
#         col = f'alpha_{angle}'
#         cy_values_no_kappa.append(mach_data[col])
    
#     # Данные с торможением
#     mach_data_with_kappa = data_with_kappa.iloc[idx]
#     cy_values_with_kappa = []
#     for angle in angles:
#         col = f'alpha_{angle}_with_kappa'
#         cy_values_with_kappa.append(mach_data_with_kappa[col])
    
#     # Без торможения - пунктир
#     ax2.plot(angles, cy_values_no_kappa, 
#              color=mach_colors[j % len(mach_colors)], 
#              linewidth=2, 
#              linestyle='--',
#              alpha=0.7,
#              marker='s',
#              markersize=4,
#              label=fr'$M = {mach}$ (без торможения)')
    
#     # С торможением - сплошная
#     ax2.plot(angles, cy_values_with_kappa, 
#              color=mach_colors[j % len(mach_colors)], 
#              linewidth=3, 
#              linestyle='-',
#              marker='o',
#              markersize=5,
#              label=fr'$M = {mach}$ (с торможением)')

# ax2.set_xlabel('$\\alpha$' ,fontsize=16)
# ax2.set_ylabel(r'$C_y\text{из.ф}$', fontsize=16)
# ax2.grid(True, alpha=0.3)
# ax2.legend(loc='upper left', fontsize=10) 

# # Добавляем линию при α = 0 для наглядности
# ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()


# # # Чтение данных для изолированного крыла
# # data = pd.read_csv('krylo_isP.csv')

# # # Создание фигуры с двумя графиками бок о бок
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # # Цвета для графиков
# # colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# # mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# # angles = [-10, -5, 0, 5, 10]

# # # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α
# # for i, angle in enumerate(angles):
# #     col = f'alpha_{angle}'
    
# #     ax1.plot(data['Mach'], data[col], 
# #              color=colors[i % len(colors)], 
# #              linewidth=2,
# #              linestyle='-',
# #              label=fr'$\alpha = {angle}^\circ$')

# # ax1.set_xlabel('M', fontsize=15)
# # ax1.set_ylabel(r'$C_y\text{из.кр}$', fontsize=16)
# # ax1.grid(True, alpha=0.3)
# # ax1.legend(loc='upper right', fontsize=10)

# # # ПРАВЫЙ ГРАФИК: C_y vs α для разных M
# # selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# # print(f"Выбранные числа Маха: {selected_mach}")

# # # Строим правый график
# # for j, mach in enumerate(selected_mach):
# #     # Находим ближайшее число Маха в данных
# #     idx = (data['Mach'] - mach).abs().idxmin()
# #     closest_mach = data['Mach'].iloc[idx]
    
# #     # Данные
# #     mach_data = data.iloc[idx]
# #     cy_values = []
# #     for angle in angles:
# #         col = f'alpha_{angle}'
# #         cy_values.append(mach_data[col])
    
# #     ax2.plot(angles, cy_values, 
# #              color=mach_colors[j % len(mach_colors)], 
# #              linewidth=2, 
# #              linestyle='-',
# #              marker='o',
# #              markersize=5,
# #              label=fr'$M = {mach}$')

# # ax2.set_xlabel('$\\alpha$', fontsize=16)
# # ax2.set_ylabel(r'$C_y\text{из.кр}$', fontsize=16)
# # ax2.grid(True, alpha=0.3)
# # ax2.legend(loc='upper left', fontsize=10) 

# # # Добавляем линию при α = 0 для наглядности
# # ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# # plt.tight_layout()
# # plt.show()

# # # Чтение данных для изолированного крыла
# # data = pd.read_csv('krylo_isP_Intrf.csv')

# # # Создание фигуры с двумя графиками бок о бок
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # # Цвета для графиков
# # colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# # mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# # angles = [-10, -5, 0, 5, 10]

# # # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α
# # for i, angle in enumerate(angles):
# #     col = f'alpha_{angle}'
    
# #     ax1.plot(data['Mach'], data[col], 
# #              color=colors[i % len(colors)], 
# #              linewidth=2,
# #              linestyle='-',
# #              label=fr'$\alpha = {angle}^\circ$')

# # ax1.set_xlabel('M', fontsize=15)
# # ax1.set_ylabel(r'$C_y\text{из.кр}$', fontsize=16)
# # ax1.grid(True, alpha=0.3)
# # ax1.legend(loc='upper right', fontsize=10)

# # # ПРАВЫЙ ГРАФИК: C_y vs α для разных M
# # selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# # print(f"Выбранные числа Маха: {selected_mach}")

# # # Строим правый график
# # for j, mach in enumerate(selected_mach):
# #     # Находим ближайшее число Маха в данных
# #     idx = (data['Mach'] - mach).abs().idxmin()
# #     closest_mach = data['Mach'].iloc[idx]
    
# #     # Данные
# #     mach_data = data.iloc[idx]
# #     cy_values = []
# #     for angle in angles:
# #         col = f'alpha_{angle}'
# #         cy_values.append(mach_data[col])
    
# #     ax2.plot(angles, cy_values, 
# #              color=mach_colors[j % len(mach_colors)], 
# #              linewidth=2, 
# #              linestyle='-',
# #              marker='o',
# #              markersize=5,
# #              label=fr'$M = {mach}$')

# # ax2.set_xlabel('$\\alpha$', fontsize=16)
# # ax2.set_ylabel(r'$C_y\text{из.кр}$', fontsize=16)
# # ax2.grid(True, alpha=0.3)
# # ax2.legend(loc='upper left', fontsize=10) 

# # # Добавляем линию при α = 0 для наглядности
# # ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# # plt.tight_layout()
# # plt.show()




# # Чтение данных для изолированного крыла
# data = pd.read_csv('Kappa_aa_kr.csv')

# # Создание фигуры с двумя графиками бок о бок
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# angles = [-10, -5, 0, 5, 10]

# # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α
# for i, angle in enumerate(angles):
#     col = f'alpha_{angle}'
    
#     ax1.plot(data['Mach'], data[col], 
#              color=colors[i % len(colors)], 
#              linewidth=2,
#              linestyle='-',
#              label=fr'$\alpha = {angle}^\circ$')

# ax1.set_xlabel('M', fontsize=15)
# ax1.set_ylabel(r'$K_{\alpha \alpha \text{кр}}$', fontsize=16)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right', fontsize=10)

# # ПРАВЫЙ ГРАФИК: C_y vs α для разных M
# selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# print(f"Выбранные числа Маха: {selected_mach}")

# # Строим правый график
# for j, mach in enumerate(selected_mach):
#     # Находим ближайшее число Маха в данных
#     idx = (data['Mach'] - mach).abs().idxmin()
#     closest_mach = data['Mach'].iloc[idx]
    
#     # Данные
#     mach_data = data.iloc[idx]
#     cy_values = []
#     for angle in angles:
#         col = f'alpha_{angle}'
#         cy_values.append(mach_data[col])
    
#     ax2.plot(angles, cy_values, 
#              color=mach_colors[j % len(mach_colors)], 
#              linewidth=2, 
#              linestyle='-',
#              marker='o',
#              markersize=5,
#              label=fr'$M = {mach}$')

# ax2.set_xlabel('$\\alpha$', fontsize=16)
# ax2.set_ylabel(r'$K_{\alpha \alpha text{кр}}$', fontsize=16)
# ax2.grid(True, alpha=0.3)
# ax2.legend(loc='upper left', fontsize=10) 

# # Добавляем линию при α = 0 для наглядности
# ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()



# data = pd.read_csv('rul_isP.csv')

# # Создание фигуры с двумя графиками бок о бок
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# angles = [-10, -5, 0, 5, 10]

# # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α
# for i, angle in enumerate(angles):
#     col = f'alpha_{angle}'
    
#     ax1.plot(data['Mach'], data[col], 
#              color=colors[i % len(colors)], 
#              linewidth=2,
#              linestyle='-',
#              label=fr'$\alpha = {angle}^\circ$')

# ax1.set_xlabel('M', fontsize=15)
# ax1.set_ylabel(r'$C^\alpha_y\text{из.рл}$', fontsize=16)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right', fontsize=10)

# # ПРАВЫЙ ГРАФИК: C_y vs α для разных M
# selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# print(f"Выбранные числа Маха: {selected_mach}")

# # Строим правый график
# for j, mach in enumerate(selected_mach):
#     # Находим ближайшее число Маха в данных
#     idx = (data['Mach'] - mach).abs().idxmin()
#     closest_mach = data['Mach'].iloc[idx]
    
#     # Данные
#     mach_data = data.iloc[idx]
#     cy_values = []
#     for angle in angles:
#         col = f'alpha_{angle}'
#         cy_values.append(mach_data[col])
    
#     ax2.plot(angles, cy_values, 
#              color=mach_colors[j % len(mach_colors)], 
#              linewidth=2, 
#              linestyle='-',
#              marker='o',
#              markersize=5,
#              label=fr'$M = {mach}$')

# ax2.set_xlabel('$\\alpha$', fontsize=16)
# ax2.set_ylabel(r'$C^\alpha_y\text{из.рл}$', fontsize=16)
# ax2.grid(True, alpha=0.3)
# ax2.legend(loc='upper left', fontsize=10) 

# # Добавляем линию при α = 0 для наглядности
# ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()




# # Чтение данных для изолированного крыла
# data = pd.read_csv('Kappa_aa_rl.csv')

# # Создание фигуры с двумя графиками бок о бок
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
# mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# angles = [-10, -5, 0, 5, 10]

# # ЛЕВЫЙ ГРАФИК: C_y vs M для разных α
# for i, angle in enumerate(angles):
#     col = f'alpha_{angle}'
    
#     ax1.plot(data['Mach'], data[col], 
#              color=colors[i % len(colors)], 
#              linewidth=2,
#              linestyle='-',
#              label=fr'$\alpha = {angle}^\circ$')

# ax1.set_xlabel('M', fontsize=15)
# ax1.set_ylabel(r'$K_{\alpha \alpha \text{рл}}$', fontsize=16)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right', fontsize=10)

# # ПРАВЫЙ ГРАФИК: C_y vs α для разных M
# selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# print(f"Выбранные числа Маха: {selected_mach}")

# # Строим правый график
# for j, mach in enumerate(selected_mach):
#     # Находим ближайшее число Маха в данных
#     idx = (data['Mach'] - mach).abs().idxmin()
#     closest_mach = data['Mach'].iloc[idx]
    
#     # Данные
#     mach_data = data.iloc[idx]
#     cy_values = []
#     for angle in angles:
#         col = f'alpha_{angle}'
#         cy_values.append(mach_data[col])
    
#     ax2.plot(angles, cy_values, 
#              color=mach_colors[j % len(mach_colors)], 
#              linewidth=2, 
#              linestyle='-',
#              marker='o',
#              markersize=5,
#              label=fr'$M = {mach}$')

# ax2.set_xlabel('$\\alpha$', fontsize=16)
# ax2.set_ylabel(r'$K_{\alpha \alpha \text{рл}}$', fontsize=16)
# ax2.grid(True, alpha=0.3)
# ax2.legend(loc='upper left', fontsize=10) 

# # Добавляем линию при α = 0 для наглядности
# ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()


# data = pd.read_csv('z_b_v.csv')

# # Создание фигуры с одним графиком
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']

# # ГРАФИК: z_b vs M
# ax.plot(data['Mach'], data['z_b'], 
#         color='blue', 
#         linewidth=2,
#         linestyle='-',
#         marker='s',
#         markersize=4
# )

# ax.set_xlabel('M', fontsize=15)
# ax.set_ylabel(r'$\bar{z}_\text{В}$', fontsize=16)
# ax.grid(True, alpha=0.3)
# ax.legend(loc='best', fontsize=12)

# plt.tight_layout()
# plt.show()



# data = pd.read_csv('i_v.csv')

# # Создание фигуры с одним графиком
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# # Цвета для графиков
# colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']

# # ГРАФИК: z_b vs M
# ax.plot(data['Mach'], data['i_v'], 
#         color='red', 
#         linewidth=2,
#         linestyle='-',
#         marker='s',
#         markersize=4
# )

# ax.set_xlabel('M', fontsize=15)
# ax.set_ylabel(r'$i_\text{В}$', fontsize=20)
# ax.grid(True, alpha=0.3)
# ax.legend(loc='best', fontsize=12)

# plt.tight_layout()
# plt.show()





# data = pd.read_csv('psi_eps.csv')

# # Создание фигур с графиками
# fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
# fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(20, 8))

# # Цвета для графиков
# colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# # 1. График: psi_eps vs Mach
# unique_mach = data['Mach'].unique()
# if len(unique_mach) > 0:
#     mach_avg = data.groupby('Mach')['psi_eps'].mean()
#     ax1.plot(mach_avg.index, mach_avg.values, 
#              color='blue', linewidth=2, marker='o', markersize=4)
#     ax1.set_xlabel('M', fontsize=14)
#     ax1.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax1.set_title(r'Зависимость $\psi_{\varepsilon}$ от числа Маха', fontsize=14)
#     ax1.grid(True, alpha=0.3)

# # 2. График: psi_eps vs alpha_p
# unique_alpha_p = data['alpha_p'].unique()
# if len(unique_alpha_p) > 0:
#     alpha_p_avg = data.groupby('alpha_p')['psi_eps'].mean()
#     ax2.plot(alpha_p_avg.index, alpha_p_avg.values, 
#              color='red', linewidth=2, marker='s', markersize=4)
#     ax2.set_xlabel(r'$\alpha_p$', fontsize=14)
#     ax2.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax2.set_title(r'Зависимость $\psi_{\varepsilon}$ от $\alpha_p$', fontsize=14)
#     ax2.grid(True, alpha=0.3)

# # 3. График: psi_eps vs phi_alpha
# unique_phi_alpha = data['phi_alpha'].unique()
# if len(unique_phi_alpha) > 0:
#     phi_alpha_avg = data.groupby('phi_alpha')['psi_eps'].mean()
#     ax3.plot(phi_alpha_avg.index, phi_alpha_avg.values, 
#              color='green', linewidth=2, marker='^', markersize=4)
#     ax3.set_xlabel(r'$\varphi_{\alpha}$', fontsize=14)
#     ax3.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax3.set_title(r'Зависимость $\psi_{\varepsilon}$ от $\varphi_{\alpha}$', fontsize=14)
#     ax3.grid(True, alpha=0.3)

# # 4. График: psi_eps vs psi_I
# unique_psi_I = data['psi_I'].unique()
# if len(unique_psi_I) > 0:
#     psi_I_avg = data.groupby('psi_I')['psi_eps'].mean()
#     ax4.plot(psi_I_avg.index, psi_I_avg.values, 
#              color='orange', linewidth=2, marker='d', markersize=4)
#     ax4.set_xlabel(r'$\psi_I$', fontsize=14)
#     ax4.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax4.set_title(r'Зависимость $\psi_{\varepsilon}$ от $\psi_I$', fontsize=14)
#     ax4.grid(True, alpha=0.3)

# # 5. График: psi_eps vs psi_II
# unique_psi_II = data['psi_II'].unique()
# if len(unique_psi_II) > 0:
#     psi_II_avg = data.groupby('psi_II')['psi_eps'].mean()
#     ax5.plot(psi_II_avg.index, psi_II_avg.values, 
#              color='purple', linewidth=2, marker='v', markersize=4)
#     ax5.set_xlabel(r'$\psi_{II}$', fontsize=14)
#     ax5.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax5.set_title(r'Зависимость $\psi_{\varepsilon}$ от $\psi_{II}$', fontsize=14)
#     ax5.grid(True, alpha=0.3)

# # 6. График: psi_eps vs z_v
# unique_z_v = data['z_v'].unique()
# if len(unique_z_v) > 0:
#     z_v_avg = data.groupby('z_v')['psi_eps'].mean()
#     ax6.plot(z_v_avg.index, z_v_avg.values, 
#              color='brown', linewidth=2, marker='*', markersize=6)
#     ax6.set_xlabel(r'$z_\text{В}$', fontsize=14)
#     ax6.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16)
#     ax6.set_title(r'Зависимость $\psi_{\varepsilon}$ от $z_\text{В}$', fontsize=14)
#     ax6.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()






# data = pd.read_csv('eps_alpha_sr.csv')

# # Создание фигур с графиками
# fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
# fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(20, 16))

# # Цвета для графиков
# colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# # 1. График: eps_alpha_sr vs Mach
# unique_mach = data['Mach'].unique()
# if len(unique_mach) > 0:
#     mach_avg = data.groupby('Mach')['eps_alpha_sr'].mean()
#     ax1.plot(mach_avg.index, mach_avg.values, 
#              color='blue', linewidth=2, marker='o', markersize=4)
#     ax1.set_xlabel('M', fontsize=14)
#     ax1.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax1.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от числа Маха', fontsize=14)
#     ax1.grid(True, alpha=0.3)

# # 2. График: eps_alpha_sr vs alpha_p
# unique_alpha_p = data['alpha_p'].unique()
# if len(unique_alpha_p) > 0:
#     alpha_p_avg = data.groupby('alpha_p')['eps_alpha_sr'].mean()
#     ax2.plot(alpha_p_avg.index, alpha_p_avg.values, 
#              color='red', linewidth=2, marker='s', markersize=4)
#     ax2.set_xlabel(r'$\alpha_p$', fontsize=14)
#     ax2.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax2.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $\alpha_p$', fontsize=14)
#     ax2.grid(True, alpha=0.3)

# # 3. График: eps_alpha_sr vs phi_alpha
# unique_phi_alpha = data['phi_alpha'].unique()
# if len(unique_phi_alpha) > 0:
#     phi_alpha_avg = data.groupby('phi_alpha')['eps_alpha_sr'].mean()
#     ax3.plot(phi_alpha_avg.index, phi_alpha_avg.values, 
#              color='green', linewidth=2, marker='^', markersize=4)
#     ax3.set_xlabel(r'$\phi_{\alpha}$', fontsize=14)
#     ax3.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax3.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $\phi_{\alpha}$', fontsize=14)
#     ax3.grid(True, alpha=0.3)

# # 4. График: eps_alpha_sr vs psi_I
# unique_psi_I = data['psi_I'].unique()
# if len(unique_psi_I) > 0:
#     psi_I_avg = data.groupby('psi_I')['eps_alpha_sr'].mean()
#     ax4.plot(psi_I_avg.index, psi_I_avg.values, 
#              color='orange', linewidth=2, marker='d', markersize=4)
#     ax4.set_xlabel(r'$\psi_I$', fontsize=14)
#     ax4.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax4.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $\psi_I$', fontsize=14)
#     ax4.grid(True, alpha=0.3)

# # 5. График: eps_alpha_sr vs psi_II
# unique_psi_II = data['psi_II'].unique()
# if len(unique_psi_II) > 0:
#     psi_II_avg = data.groupby('psi_II')['eps_alpha_sr'].mean()
#     ax5.plot(psi_II_avg.index, psi_II_avg.values, 
#              color='purple', linewidth=2, marker='v', markersize=4)
#     ax5.set_xlabel(r'$\psi_{II}$', fontsize=14)
#     ax5.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax5.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $\psi_{II}$', fontsize=14)
#     ax5.grid(True, alpha=0.3)

# # 6. График: eps_alpha_sr vs z_v
# unique_z_v = data['z_v'].unique()
# if len(unique_z_v) > 0:
#     z_v_avg = data.groupby('z_v')['eps_alpha_sr'].mean()
#     ax6.plot(z_v_avg.index, z_v_avg.values, 
#              color='brown', linewidth=2, marker='*', markersize=6)
#     ax6.set_xlabel(r'$z_v$', fontsize=14)
#     ax6.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax6.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $z_v$', fontsize=14)
#     ax6.grid(True, alpha=0.3)

# # 7. График: eps_alpha_sr vs psi_eps
# unique_psi_eps = data['psi_eps'].unique()
# if len(unique_psi_eps) > 0:
#     psi_eps_avg = data.groupby('psi_eps')['eps_alpha_sr'].mean()
#     ax7.plot(psi_eps_avg.index, psi_eps_avg.values, 
#              color='cyan', linewidth=2, marker='x', markersize=4)
#     ax7.set_xlabel(r'$\psi_{\varepsilon}$', fontsize=14)
#     ax7.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax7.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $\psi_{\varepsilon}$', fontsize=14)
#     ax7.grid(True, alpha=0.3)

# # 8. График: eps_alpha_sr vs i_v
# unique_i_v = data['i_v'].unique()
# if len(unique_i_v) > 0:
#     i_v_avg = data.groupby('i_v')['eps_alpha_sr'].mean()
#     ax8.plot(i_v_avg.index, i_v_avg.values, 
#              color='magenta', linewidth=2, marker='+', markersize=6)
#     ax8.set_xlabel(r'$i_v$', fontsize=14)
#     ax8.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax8.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $i_v$', fontsize=14)
#     ax8.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # Дополнительно: статистика по данным
# print("Статистика по eps_alpha_sr:")
# print(f"Минимум: {data['eps_alpha_sr'].min():.6f}")
# print(f"Максимум: {data['eps_alpha_sr'].max():.6f}")
# print(f"Среднее: {data['eps_alpha_sr'].mean():.6f}")
# print(f"Стандартное отклонение: {data['eps_alpha_sr'].std():.6f}")

# # Графики для коэффициентов интерференции
# fig3, (ax9, ax10) = plt.subplots(1, 2, figsize=(20, 8))

# # 9. График: eps_alpha_sr vs k_aa
# unique_k_aa = data['k_aa'].unique()
# if len(unique_k_aa) > 0:
#     k_aa_avg = data.groupby('k_aa')['eps_alpha_sr'].mean()
#     ax9.plot(k_aa_avg.index, k_aa_avg.values, 
#              color='navy', linewidth=2, marker='o', markersize=4)
#     ax9.set_xlabel(r'$k_{\alpha\alpha}$', fontsize=14)
#     ax9.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax9.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $k_{\alpha\alpha}$', fontsize=14)
#     ax9.grid(True, alpha=0.3)

# # 10. График: eps_alpha_sr vs K_aa_rl
# unique_K_aa_rl = data['K_aa_rl'].unique()
# if len(unique_K_aa_rl) > 0:
#     K_aa_rl_avg = data.groupby('K_aa_rl')['eps_alpha_sr'].mean()
#     ax10.plot(K_aa_rl_avg.index, K_aa_rl_avg.values, 
#               color='darkred', linewidth=2, marker='s', markersize=4)
#     ax10.set_xlabel(r'$K_{\alpha\alpha}^{рл}$', fontsize=14)
#     ax10.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)
#     ax10.set_title(r'Зависимость $\varepsilon_{\alpha}^{cp}$ от $K_{\alpha\alpha}^{рл}$', fontsize=14)
#     ax10.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()








# data = pd.read_csv('kappa_q_nos.csv')

# # Создание графика
# fig, ax = plt.subplots(figsize=(12, 8))

# # Построение графика
# ax.plot(data['Mach'], data['kappa_q'], 
#         color='blue', linewidth=2, marker='o', markersize=4,
#         label=r'$\kappa_q$')

# # Настройки графика
# ax.set_xlabel('M', fontsize=14)
# ax.set_ylabel(r'$\kappa_{q \text{нос}}$', fontsize=16)
# ax.set_title(r'Зависимость $\kappa_{q \text{нос}}$ от числа Маха', fontsize=16)
# ax.grid(True, alpha=0.3)
# ax.legend(loc='best', fontsize=12)

# # Добавление сетки
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


# data = pd.read_csv('kappa_q.csv')

# # Создание графика
# fig, ax = plt.subplots(figsize=(12, 8))

# # Построение графика
# ax.plot(data['Mach'], data['kappa_q'], 
#         color='blue', linewidth=2, marker='o', markersize=4,
#         label=r'$\kappa_q$')

# # Настройки графика
# ax.set_xlabel('M', fontsize=14)
# ax.set_ylabel(r'$\kappa_q$', fontsize=16)
# ax.set_title(r'Зависимость $\kappa_q$ от числа Маха', fontsize=16)
# ax.grid(True, alpha=0.3)
# ax.legend(loc='best', fontsize=12)

# # Добавление сетки
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()







data = pd.read_csv('c_y_alpha_sum.csv')

# Создание фигуры с двумя графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Цвета для графиков
colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

angles = [-10, -5, 0, 5, 10]

# ЛЕВЫЙ ГРАФИК: c_y_alpha_sum vs M для разных α
for i, angle in enumerate(angles):
    col = f'alpha_{angle}'
    
    ax1.plot(data['Mach'], data[col], 
             color=colors[i % len(colors)], 
             linewidth=2,
             linestyle='-',
             label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('M', fontsize=15)
ax1.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
ax1.set_title(r'Зависимость $c_y^{\alpha}$ от числа Маха для разных углов атаки', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

# ПРАВЫЙ ГРАФИК: c_y_alpha_sum vs α для разных M
selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

print(f"Выбранные числа Маха: {selected_mach}")

# Строим правый график
for j, mach in enumerate(selected_mach):
    # Находим ближайшее число Маха в данных
    idx = (data['Mach'] - mach).abs().idxmin()
    closest_mach = data['Mach'].iloc[idx]
    
    # Данные для текущего числа Маха
    mach_data = data.iloc[idx]
    cy_values = []
    for angle in angles:
        col = f'alpha_{angle}'
        cy_values.append(mach_data[col])
    
    ax2.plot(angles, cy_values, 
             color=mach_colors[j % len(mach_colors)], 
             linewidth=2, 
             linestyle='-',
             marker='o',
             markersize=5,
             label=fr'$M = {mach}$')

ax2.set_xlabel(r'$\alpha$, град', fontsize=16)
ax2.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
ax2.set_title(r'Зависимость $c_y^{\alpha}$ от угла атаки для разных чисел Маха', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10) 

# Добавляем линию при α = 0 для наглядности
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# Чтение данных для коэффициента c_y_delta_1
data = pd.read_csv('c_y_delta_1.csv')

# Создание графика
fig, ax = plt.subplots(figsize=(12, 8))

# Построение графика
ax.plot(data['Mach'], data['c_y_delta_1'], 
        color='red', 
        linewidth=3,
        linestyle='-',
        marker='o',
        markersize=4,
        label=r'$c_y^{\delta_1}$')

# Настройки графика
ax.set_xlabel('M', fontsize=14)
ax.set_ylabel(r'$c_y^{\delta_1}$', fontsize=16)
ax.set_title(r'Зависимость коэффициента $c_y^{\delta_1}$ от числа Маха', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12)

# Добавляем линию при y=0 для наглядности
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()




data = pd.read_csv('c_y_delta_2.csv')
    
    # Создание графика
fig, ax = plt.subplots(figsize=(12, 8))
    
    # Построение графика
ax.plot(data['Mach'], data['c_y_delta_2'], 
            color='blue', 
            linewidth=3,
            linestyle='-',
            marker='s',
            markersize=4,
            label=r'$c_y^{\delta_2}$')
    
    # Настройки графика
ax.set_xlabel('M', fontsize=14)
ax.set_ylabel(r'$c_y^{\delta_2}$', fontsize=16)
ax.set_title(r'Зависимость коэффициента $c_y^{\delta_2}$ от числа Маха', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12)
    
    # Добавляем линию при y=0 для наглядности
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
plt.tight_layout()
plt.show()



df = pd.read_csv('c_y_f.csv')

# График 1: c_y_f от угла атаки для выбранных чисел Маха
plt.figure(figsize=(12, 8))

for mach in selected_mach:
    # Фильтруем данные для текущего числа Маха
    mach_data = df[df['Mach'] == mach]
    # Сортируем по углу атаки для красивого графика
    mach_data = mach_data.sort_values('angle')
    
    plt.plot(mach_data['angle'], mach_data['c_y_f'], 
             marker='o', linewidth=2, markersize=4, 
             label=f'M = {mach}')

plt.xlabel('Угол атаки, градусы')
plt.ylabel(r'$c_\text{y.ф}$')
plt.title(r'Коэффициент подъемной силы $c_\text{y.ф}$ в зависимости от угла атаки')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# График 2: c_y_f от числа Маха для разных углов атаки
plt.figure(figsize=(12, 8))

# Получаем уникальные углы атаки
unique_angles = sorted(df['angle'].unique())

for angle in unique_angles:
    # Фильтруем данные для текущего угла атаки
    angle_data = df[df['angle'] == angle]
    # Сортируем по числу Маха для красивого графика
    angle_data = angle_data.sort_values('Mach')
    
    plt.plot(angle_data['Mach'], angle_data['c_y_f'], 
             marker='s', linewidth=2, markersize=4,
             label=f'α = {angle}°')

plt.xlabel('Число Маха')
plt.ylabel(r'$c_\text{y.ф}$')
plt.title(r'Коэффициент подъемной силы $c_\text{y.ф}$ в зависимости от числа Маха')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================================
# ВИЗУАЛИЗАЦИЯ c_y_sum_big_delta для больших углов и дельт (2D)
# ============================================================================

print("\n" + "="*80)
print("ВИЗУАЛИЗАЦИЯ C_y_sum_big_delta для больших углов и дельт")
print("="*80)


data = pd.read_csv('c_y_sum_big_delta.csv')
print(f"Загружено {len(data)} записей")

# Определяем углы атаки и дельты из данных
angles = sorted(data['alpha_deg'].unique())
deltas_II = sorted(data['delta_II_deg'].unique())

print(f"Углы атаки: {angles}")
print(f"Углы отклонения рулей: {deltas_II}")

# ГРАФИК 1: Для положительных углов атаки
pos_angles = [a for a in angles if a > 0]
neg_angles = [a for a in angles if a < 0]

# ============ ПОЛОЖИТЕЛЬНЫЕ УГЛЫ АТАКИ ============
if pos_angles:
    print(f"\nГрафики для положительных углов: {pos_angles}")
    
    # Фильтруем данные для δ_II = 0 (рули не отклонены)
    data_delta0 = data[data['delta_II_deg'] == 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. График: C_y vs M для разных α при δ=0
    for alpha in pos_angles:
        alpha_data = data_delta0[data_delta0['alpha_deg'] == alpha]
        alpha_data = alpha_data.sort_values('Mach')
        
        ax1.plot(alpha_data['Mach'], alpha_data['c_y_sum_big_delta'], 
                 linewidth=2, marker='o', markersize=4,
                 label=fr'$\alpha = {alpha}^\circ$')
    
    ax1.set_xlabel('M')
    ax1.set_ylabel(r'$C_y$')
    ax1.set_title(r'Зависимость $C_y$ от M ($\delta_{II} = 0^\circ$) (большие углы)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. График: C_y vs M для разных δ_II при α=10°
    alpha_fixed = 10 # фиксируем угол атаки 10°
    data_alpha10 = data[data['alpha_deg'] == alpha_fixed]
    
    # Выбираем несколько дельт для отображения
    selected_deltas = [-25, -15, -10, 0, 10, 15, 25]
    
    for delta in selected_deltas:
        if delta in deltas_II:
            delta_data = data_alpha10[data_alpha10['delta_II_deg'] == delta]
            delta_data = delta_data.sort_values('Mach')
            
            if not delta_data.empty:
                ax2.plot(delta_data['Mach'], delta_data['c_y_sum_big_delta'], 
                         linewidth=2, marker='s', markersize=4,
                         label=fr'$\delta_{{II}} = {delta}^\circ$')
    
    ax2.set_xlabel('M')
    ax2.set_ylabel(r'$C_y$')
    ax2.set_title(fr'Зависимость $C_y$ от M ($\alpha = {alpha_fixed}^\circ$)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ============ ОТРИЦАТЕЛЬНЫЕ УГЛЫ АТАКИ ============
if neg_angles:
    print(f"\nГрафики для отрицательных углов: {neg_angles}")
    
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 3. График: C_y vs M для разных α при δ=0
    for alpha in neg_angles:
        alpha_data = data_delta0[data_delta0['alpha_deg'] == alpha]
        alpha_data = alpha_data.sort_values('Mach')
        
        ax3.plot(alpha_data['Mach'], alpha_data['c_y_sum_big_delta'], 
                 linewidth=2, marker='o', markersize=4,
                 label=fr'$\alpha = {alpha}^\circ$')
    
    ax3.set_xlabel('M')
    ax3.set_ylabel(r'$C_y$')
    ax3.set_title(r'Зависимость $C_y$ от M ($\delta_{II} = 0^\circ$)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 4. График: C_y vs δ_II при разных M при α=-10°
    alpha_fixed_neg = 10  # фиксируем угол атаки -10°
    data_alpha_neg10 = data[data['alpha_deg'] == alpha_fixed_neg]
    
    # Выбираем несколько чисел Маха
    selected_mach = [0.5, 1.0, 2.0, 3.0, 4.0]
    
    for mach in selected_mach:
        mach_data = data_alpha_neg10[np.abs(data_alpha_neg10['Mach'] - mach) < 0.05]
        mach_data = mach_data.sort_values('delta_II_deg')
        
        if not mach_data.empty:
            ax4.plot(mach_data['delta_II_deg'], mach_data['c_y_sum_big_delta'], 
                     linewidth=2, marker='^', markersize=4,
                     label=fr'$M = {mach}$')
    
    ax4.set_xlabel(r'$\delta_{II}$, град')
    ax4.set_ylabel(r'$C_y$')
    ax4.set_title(fr'Зависимость $C_y$ от $\delta_{{II}}$ ($\alpha = {alpha_fixed_neg}^\circ$)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# ВИЗУАЛИЗАЦИЯ c_y для малых углов атаки
# ============================================================================

print("\n" + "="*80)
print("ВИЗУАЛИЗАЦИЯ c_y для малых углов атаки")
print("="*80)

# Чтение данных
data = pd.read_csv('c_y_small_angles.csv')
print(f"\nЗагружено {len(data)} записей")

# ГРАФИК 1: c_y vs M для разных углов при δ_II=0
fig1, ax1 = plt.subplots(figsize=(12, 8))

# Фильтруем данные для δ_II = 0
data_delta0 = data[data['delta_II_deg'] == 0]

# Все углы атаки
angles_deg = sorted(data['alpha_deg'].unique())
colors = plt.cm.rainbow(np.linspace(0, 1, len(angles_deg)))

for i, angle in enumerate(angles_deg):
    angle_data = data_delta0[data_delta0['alpha_deg'] == angle]
    angle_data = angle_data.sort_values('Mach')
    
    ax1.plot(angle_data['Mach'], angle_data['c_y'], 
             color=colors[i], linewidth=2, marker='o', markersize=4,
             label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('M')
ax1.set_ylabel(r'$c_y$')
ax1.set_title(r'Зависимость $c_y$ от M ($\delta_{II} = 0^\circ$)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ГРАФИК 2: c_y vs α для разных M при δ_II=0
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Выбираем несколько чисел Маха
selected_mach = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
colors_mach = plt.cm.plasma(np.linspace(0, 1, len(selected_mach)))

for i, mach in enumerate(selected_mach):
    # Фильтруем данные для данного M и δ_II=0
    mach_data = data_delta0[np.abs(data_delta0['Mach'] - mach) < 0.05]
    mach_data = mach_data.sort_values('alpha_deg')
    
    if not mach_data.empty:
        ax2.plot(mach_data['alpha_deg'], mach_data['c_y'], 
                 color=colors_mach[i], linewidth=2, marker='s', markersize=4,
                 label=fr'$M = {mach}$')

ax2.set_xlabel(r'$\alpha$, град')
ax2.set_ylabel(r'$c_y$')
ax2.set_title(r'Зависимость $c_y$ от $\alpha$ ($\delta_{II} = 0^\circ$)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ГРАФИК 3: c_y vs M для разных δ_II при α=0
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Фильтруем данные для α = 0
data_alpha0 = data[data['alpha_deg'] == 0]

# Выбираем несколько дельт для отображения
deltas_II = sorted(data['delta_II_deg'].unique())
selected_deltas = [-25, -15, -5, 0, 5, 15, 25]  # ключевые значения
colors_delta = plt.cm.viridis(np.linspace(0, 1, len(selected_deltas)))

for i, delta in enumerate(selected_deltas):
    if delta in deltas_II:
        delta_data = data_alpha0[data_alpha0['delta_II_deg'] == delta]
        delta_data = delta_data.sort_values('Mach')
        
        if not delta_data.empty:
            ax3.plot(delta_data['Mach'], delta_data['c_y'], 
                     color=colors_delta[i], linewidth=2, marker='^', markersize=4,
                     label=fr'$\delta_{{II}} = {delta}^\circ$')

ax3.set_xlabel('M')
ax3.set_ylabel(r'$c_y$')
ax3.set_title(r'Зависимость $c_y$ от M ($\alpha = 0^\circ$)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ГРАФИК 4: c_y vs δ_II для разных M при α=0
fig4, ax4 = plt.subplots(figsize=(12, 8))

for i, mach in enumerate(selected_mach):
    mach_data = data_alpha0[np.abs(data_alpha0['Mach'] - mach) < 0.05]
    mach_data = mach_data.sort_values('delta_II_deg')
    
    if not mach_data.empty:
        ax4.plot(mach_data['delta_II_deg'], mach_data['c_y'], 
                 color=colors_mach[i], linewidth=2, marker='d', markersize=4,
                 label=fr'$M = {mach}$')

ax4.set_xlabel(r'$\delta_{II}$, град')
ax4.set_ylabel(r'$c_y$')
ax4.set_title(r'Зависимость $c_y$ от $\delta_{II}$ ($\alpha = 0^\circ$)')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



# //////////////////////////////////////////////////////////////////////////////////////////////////////////
def visualize_c_y_sum_minimal():
    """
    Минимальная визуализация - только два основных графика
    """
    # Чтение данных
    data = pd.read_csv('c_y_sum.csv')
    
    # Фильтруем данные для нулевых отклонений
    data_zero = data[(data['delta_I_deg'] == 0) & (data['delta_II_deg'] == 0)]
    
    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ГРАФИК 1: C_y vs M для разных углов
    angles = sorted(data_zero['alpha_deg'].unique())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, angle in enumerate(angles):
        angle_data = data_zero[data_zero['alpha_deg'] == angle]
        angle_data = angle_data.sort_values('Mach')
        
        ax1.plot(angle_data['Mach'], angle_data['c_y_sum'], 
                 color=colors[i % len(colors)], 
                 linewidth=2, marker='o', markersize=4,
                 label=fr'$\alpha = {angle}^\circ$')
    
    ax1.set_xlabel('M', fontsize=12)
    ax1.set_ylabel(r'$C_y$', fontsize=12)
    ax1.set_title('Зависимость $C_y$ от M при $\delta_{II}$ = 0', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # ГРАФИК 2: C_y vs α для разных M
    selected_mach = [0.5, 1.0, 2.0, 3.0, 4.0]
    markers = ['o', 's', '^', 'v', 'd', '<', '>', 'p']
    
    for i, mach in enumerate(selected_mach):
        mach_data = data_zero[np.abs(data_zero['Mach'] - mach) < 0.05]
        mach_data = mach_data.sort_values('alpha_deg')
        
        if not mach_data.empty:
            ax2.plot(mach_data['alpha_deg'], mach_data['c_y_sum'], 
                     marker=markers[i % len(markers)], 
                     linewidth=2, markersize=5,
                     label=fr'$M = {mach}$')
    
    ax2.set_xlabel(r'$\alpha$, град', fontsize=12)
    ax2.set_ylabel(r'$C_y^{\sum}$', fontsize=12)
    ax2.set_title('Зависимость $C_y$ от угла атаки', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# Запуск визуализации
visualize_c_y_sum_minimal()


# /////////////////////////////////////////////////////////////////////////////\

def visualize_cx_from_csv():
    """
    Минимальная визуализация Cx из CSV файла
    """
    try:
        # Чтение данных
        data = pd.read_csv('c_x_data.csv')
        print(f"Загружено {len(data)} записей")
    except FileNotFoundError:
        print("ОШИБКА: Файл c_x_data.csv не найден")
        print("Сначала запустите расчеты и сохраните данные")
        return
    
    # 1. Cx vs M для разных углов при delta_I=0 и delta_II=0
    plt.figure(figsize=(10, 6))
    
    # Фильтруем данные для delta_I=0 и delta_II=0
    data_zero_delta = data[(data['delta_I_deg'] == 0) & (data['delta_II_deg'] == 0)]
    
    # Получаем уникальные углы
    angles = sorted(data_zero_delta['alpha_deg'].unique())
    
    for angle in angles:
        angle_data = data_zero_delta[data_zero_delta['alpha_deg'] == angle]
        angle_data = angle_data.sort_values('Mach')
        
        plt.plot(angle_data['Mach'], angle_data['c_x'], 
                 linewidth=2, marker='o', markersize=3,
                 label=fr'$\alpha={angle}^\circ$')
    
    plt.xlabel('M')
    plt.ylabel(r'$C_x$')
    plt.title(r'$C_x$ vs M ($\delta_{II}=0^\circ$)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Cx vs α для разных M при delta_I=0 и delta_II=0
    plt.figure(figsize=(10, 6))
    
    # Выбираем несколько чисел Маха
    mach_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    markers = ['o', 's', '^', 'v', 'd', '<', '>', 'p']
    
    for i, mach in enumerate(mach_values):
        mach_data = data_zero_delta[np.abs(data_zero_delta['Mach'] - mach) < 0.05]
        mach_data = mach_data.sort_values('alpha_deg')
        
        if not mach_data.empty:
            plt.plot(mach_data['alpha_deg'], mach_data['c_x'], 
                     marker=markers[i % len(markers)], 
                     linewidth=2, markersize=4,
                     label=fr'$M={mach}$')
    
    plt.xlabel(r'$\alpha$, град')
    plt.ylabel(r'$C_x$')
    plt.title(r'$C_x$ vs $\alpha$ ($\delta_{II}=0^\circ$)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Запуск визуализации
visualize_cx_from_csv()




def visualize_mz_vs_alpha():
    """
    График m_z от alpha при разных дельта при M=2
    """
    try:
        # Чтение данных
        data = pd.read_csv('m_z_small.csv')
        print(f"Загружено {len(data)} записей")
    except FileNotFoundError:
        print("ОШИБКА: Файл m_z_data.csv не найден")
        return
    
    # Фильтруем данные для M=2
    m_fixed = 2.0
    data_m2 = data[np.abs(data['Mach'] - m_fixed) < 0.05]
    
    if data_m2.empty:
        print(f"Нет данных для M={m_fixed}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Получаем уникальные дельты
    deltas = sorted(data_m2['delta_II_deg'].unique())
    
    for delta in deltas:
        delta_data = data_m2[data_m2['delta_II_deg'] == delta]
        delta_data = delta_data.sort_values('alpha_deg')
        
        plt.plot(delta_data['alpha_deg'], delta_data['m_z'], 
                 linewidth=2, marker='o', markersize=4,
                 label=fr'$\delta_{{II}}={delta}^\circ$')
    
    plt.xlabel(r'$\alpha$, град')
    plt.ylabel(r'$m_z$')
    plt.title(fr'$m_z$ vs $\alpha$ при M={m_fixed}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Запуск визуализации
visualize_mz_vs_alpha()


data = pd.read_csv('c_y_alpha_final.csv')

# Создание фигуры с двумя графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Цвета для графиков
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# ЛЕВЫЙ ГРАФИК: c_y_alpha vs M для разных α
for i, angle in enumerate(angles_deg):
    col = f'alpha_{angle}'
    
    ax1.plot(data['Mach'], data[col], 
             color=colors[i % len(colors)], 
             linewidth=2,
             linestyle='-',
             marker='o',
             markersize=3,
             label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('M', fontsize=15)
ax1.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
ax1.set_title(r'Зависимость $c_y^{\alpha}$ от числа Маха', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=12)

# ПРАВЫЙ ГРАФИК: c_y_alpha vs α для разных M
selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Строим правый график
for j, mach in enumerate(selected_mach):
    # Находим ближайшее число Маха в данных
    idx = (data['Mach'] - mach).abs().idxmin()
    closest_mach = data['Mach'].iloc[idx]
    
    # Данные для текущего числа Маха
    mach_data = data.iloc[idx]
    cy_values = []
    for angle in angles:
        col = f'alpha_{angle}'
        cy_values.append(mach_data[col])
    
    ax2.plot(angles, cy_values, 
             color=colors[j % len(colors)], 
             linewidth=2, 
             linestyle='-',
             marker='s',
             markersize=4,
             label=fr'$M = {mach}$')

ax2.set_xlabel(r'$\alpha$, град', fontsize=15)
ax2.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
# ax2.set_title(r'Зависимость $c_y^{\alpha}$ от угла атаки', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=12)

plt.tight_layout()
plt.show()





data = pd.read_csv('c_y_alpha_final.csv')

# Создание фигуры с двумя графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Цвета для графиков
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# ЛЕВЫЙ ГРАФИК: c_y_alpha vs M для разных α
for i, angle in enumerate(angles_deg):
    col = f'alpha_{angle}'
    
    ax1.plot(data['Mach'], data[col], 
             color=colors[i % len(colors)], 
             linewidth=2,
             linestyle='-',
             marker='o',
             markersize=3,
             label=fr'$\alpha = {angle}^\circ$')

ax1.set_xlabel('M', fontsize=15)
ax1.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
ax1.set_title(r'Зависимость $c_y^{\alpha}$ от числа Маха', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=12)

# ПРАВЫЙ ГРАФИК: c_y_alpha vs α для разных M
selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Строим правый график
for j, mach in enumerate(selected_mach):
    # Находим ближайшее число Маха в данных
    idx = (data['Mach'] - mach).abs().idxmin()
    closest_mach = data['Mach'].iloc[idx]
    
    # Данные для текущего числа Маха
    mach_data = data.iloc[idx]
    cy_values = []
    for angle in angles:
        col = f'alpha_{angle}'
        cy_values.append(mach_data[col])
    
    ax2.plot(angles, cy_values, 
             color=colors[j % len(colors)], 
             linewidth=2, 
             linestyle='-',
             marker='s',
             markersize=4,
             label=fr'$M = {mach}$')

ax2.set_xlabel(r'$\alpha$, град', fontsize=15)
ax2.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
# ax2.set_title(r'Зависимость $c_y^{\alpha}$ от угла атаки', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=12)

plt.tight_layout()
plt.show()


# Вывод информации о данных
print(f"Диапазон чисел Маха: от {data['Mach'].min()} до {data['Mach'].max()}")
print(f"Количество точек по M: {len(data)}")
