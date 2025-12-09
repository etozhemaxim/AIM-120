import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# 1. ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ
# --------------------------------------------------------------------
def load_data(filepath):
    """Загрузка данных из CSV файла"""
    return pd.read_csv(filepath)

def load_multiple_datasets(filepaths):
    """Загрузка нескольких наборов данных"""
    datasets = {}
    for name, path in filepaths.items():
        datasets[name] = load_data(path)
    return datasets

# --------------------------------------------------------------------
# 2. ФУНКЦИИ ДЛЯ ОБРАБОТКИ И РАСЧЕТОВ
# --------------------------------------------------------------------
# Функция для расчета коэффициента торможения (оставляем без изменений)
def calculate_kappa_q_nos_con(M, lambda_nos):
    # Проверка граничных условий
    if lambda_nos < 0.7 or lambda_nos > 5.0:
        return 1.0
    
    # Случай дозвуковых скоростей
    if M <= 1.0:
        return 1.0
    
    # Данные для M=2.0
    lambda_values_M2 = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    kappa_values_M2 = [0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.75, 0.74]
    
    # Интерполяция для M=2
    kappa_at_M2 = np.interp(lambda_nos, lambda_values_M2, kappa_values_M2)
    
    # Интерполяция между M=1 и M=2
    if M <= 2.0:
        return 1.0 + (M - 1.0) * (kappa_at_M2 - 1.0)
    else:
        # Для M > 2 используем значение при M=2
        return kappa_at_M2

def apply_kappa_correction(data, lambda_nos, angles):
    """Применение коррекции торможения к данным"""
    data_with_kappa = data.copy()
    
    # Добавляем столбец с коэффициентом торможения
    data_with_kappa['kappa_q'] = data_with_kappa['Mach'].apply(
        lambda M: calculate_kappa_q_nos_con(M, lambda_nos)
    )
    
    # Применяем торможение ко всем коэффициентам подъемной силы
    for angle in angles:
        col = f'alpha_{angle}'
        data_with_kappa[col + '_with_kappa'] = data_with_kappa[col] * data_with_kappa['kappa_q']
    
    return data_with_kappa

def get_closest_mach_data(data, target_mach):
    """Получение данных для ближайшего числа Маха"""
    idx = (data['Mach'] - target_mach).abs().idxmin()
    return data.iloc[idx]

# --------------------------------------------------------------------
# 3. ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ
# --------------------------------------------------------------------
def create_two_panel_figure(width=20, height=8):
    """Создание фигуры с двумя графиками"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
    return fig, ax1, ax2

def plot_cy_vs_mach_comparison(ax, data_original, data_corrected, angles, title_suffix=""):
    """Построение графика C_y vs M с сравнением исходных и скорректированных данных"""
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    
    for i, angle in enumerate(angles):
        col = f'alpha_{angle}'
        
        # Без торможения (исходные данные) - пунктирная линия
        ax.plot(data_original['Mach'], data_original[col], 
                color=colors[i % len(colors)], 
                linewidth=2,
                linestyle='--',
                alpha=0.7,
                label=fr'$\alpha = {angle}^\circ$ (без торможения)')
        
        # С торможением - сплошная линия
        corrected_col = col + '_with_kappa'
        ax.plot(data_corrected['Mach'], data_corrected[corrected_col], 
                color=colors[i % len(colors)], 
                linewidth=3,
                linestyle='-',
                label=fr'$\alpha = {angle}^\circ$ (с торможением)')
    
    ax.set_xlabel('M', fontsize=15)
    ax.set_ylabel(r'$C_y\text{из.ф}$', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    if title_suffix:
        ax.set_title(title_suffix)

def plot_cy_vs_alpha_for_mach(ax, data_original, data_corrected, angles, selected_mach, title_suffix=""):
    """Построение графика C_y vs α для разных чисел Маха"""
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for j, mach in enumerate(selected_mach):
        # Данные без торможения
        mach_data_orig = get_closest_mach_data(data_original, mach)
        cy_values_no_kappa = []
        for angle in angles:
            col = f'alpha_{angle}'
            cy_values_no_kappa.append(mach_data_orig[col])
        
        # Данные с торможением
        mach_data_corr = get_closest_mach_data(data_corrected, mach)
        cy_values_with_kappa = []
        for angle in angles:
            col = f'alpha_{angle}_with_kappa'
            cy_values_with_kappa.append(mach_data_corr[col])
        
        # Без торможения - пунктир
        ax.plot(angles, cy_values_no_kappa, 
                color=mach_colors[j % len(mach_colors)], 
                linewidth=2, 
                linestyle='--',
                alpha=0.7,
                marker='s',
                markersize=4,
                label=fr'$M = {mach}$ (без торможения)')
        
        # С торможением - сплошная
        ax.plot(angles, cy_values_with_kappa, 
                color=mach_colors[j % len(mach_colors)], 
                linewidth=3, 
                linestyle='-',
                marker='o',
                markersize=5,
                label=fr'$M = {mach}$ (с торможением)')
    
    ax.set_xlabel('$\\alpha$', fontsize=16)
    ax.set_ylabel(r'$C_y\text{из.ф}$', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if title_suffix:
        ax.set_title(title_suffix)

def plot_single_component(ax, data, x_col, y_col, color='blue', marker='o', 
                         xlabel='', ylabel='', title='', legend_label=''):
    """Построение графика для одного компонента"""
    ax.plot(data[x_col], data[y_col], 
            color=color, 
            linewidth=2,
            linestyle='-',
            marker=marker,
            markersize=4,
            label=legend_label)
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if legend_label:
        ax.legend(loc='best', fontsize=12)

def plot_dependency_group(data, x_columns, y_column, figsize=(20, 16), nrows=2, ncols=2):
    """Построение группы графиков зависимостей"""
    num_plots = len(x_columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for idx, x_col in enumerate(x_columns):
        if idx < len(axes):
            unique_values = data[x_col].unique()
            if len(unique_values) > 0:
                avg_data = data.groupby(x_col)[y_column].mean()
                axes[idx].plot(avg_data.index, avg_data.values, 
                              color=colors[idx % len(colors)], 
                              linewidth=2, 
                              marker=['o', 's', '^', 'd', 'v', '*', 'x', '+'][idx % 8],
                              markersize=4)
                
                axes[idx].set_xlabel(x_col, fontsize=14)
                axes[idx].set_ylabel(y_column, fontsize=16)
                axes[idx].set_title(f'Зависимость {y_column} от {x_col}', fontsize=14)
                axes[idx].grid(True, alpha=0.3)
    
    # Скрываем пустые оси
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# --------------------------------------------------------------------
# 4. СПЕЦИАЛИЗИРОВАННЫЕ ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# --------------------------------------------------------------------
def visualize_cy_comparison(data_path, angles, lambda_nos):
    """Визуализация сравнения C_y с коррекцией торможения и без"""
    # Загрузка данных
    data = load_data(data_path)
    
    # Применение коррекции торможения
    data_with_kappa = apply_kappa_correction(data, lambda_nos, angles)
    
    # Создание графиков
    fig, ax1, ax2 = create_two_panel_figure()
    
    # Левый график: C_y vs M
    plot_cy_vs_mach_comparison(ax1, data, data_with_kappa, angles)
    
    # Правый график: C_y vs α для разных M
    selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    plot_cy_vs_alpha_for_mach(ax2, data, data_with_kappa, angles, selected_mach)
    
    plt.tight_layout()
    plt.show()
    
    return data_with_kappa

def visualize_simple_cy_plots(data_path, angles, ylabel=r'$C_y\text{из.кр}$'):
    """Визуализация простых графиков C_y"""
    data = load_data(data_path)
    fig, ax1, ax2 = create_two_panel_figure()
    
    # Левый график: C_y vs M для разных α
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    for i, angle in enumerate(angles):
        col = f'alpha_{angle}'
        ax1.plot(data['Mach'], data[col], 
                color=colors[i % len(colors)], 
                linewidth=2,
                linestyle='-',
                label=fr'$\alpha = {angle}^\circ$')
    
    ax1.set_xlabel('M', fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Правый график: C_y vs α для разных M
    selected_mach = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for j, mach in enumerate(selected_mach):
        mach_data = get_closest_mach_data(data, mach)
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
    
    ax2.set_xlabel('$\\alpha$', fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def visualize_big_angles_delta(data_path):
    """Визуализация больших углов и дельт"""
    data = load_data(data_path)
    
    # Определяем углы атаки и дельты из данных
    angles = sorted(data['alpha_deg'].unique())
    deltas_II = sorted(data['delta_II_deg'].unique())
    
    print(f"Углы атаки: {angles}")
    print(f"Углы отклонения рулей: {deltas_II}")
    
    # Положительные углы атаки
    pos_angles = [a for a in angles if a > 0]
    
    if pos_angles:
        print(f"\nГрафики для положительных углов: {pos_angles}")
        
        # Фильтруем данные для δ_II = 0
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
        alpha_fixed = 10
        data_alpha10 = data[data['alpha_deg'] == alpha_fixed]
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

def visualize_mz_vs_alpha(data_path, mach_value=2.0):
    """График m_z от alpha при разных дельта для заданного числа Маха"""
    data = load_data(data_path)
    
    # Фильтруем данные для заданного числа Маха
    data_m2 = data[np.abs(data['Mach'] - mach_value) < 0.05]
    
    if data_m2.empty:
        print(f"Нет данных для M={mach_value}")
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
    plt.title(fr'$m_z$ vs $\alpha$ при M={mach_value}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------
# 6. ПРОСТАЯ ВИЗУАЛИЗАЦИЯ ПОЛЯРЫ ЛИЛИЕНТАЛЯ
# --------------------------------------------------------------------
def visualize_cy_vs_cx(data_path, delta_II_deg, mach):
    """
    График C_y от C_x (поляра) - только положительные углы атаки
    
    Параметры:
    data_path - путь к CSV файлу
    delta_II_deg - значение delta_II_deg для фильтрации
    mach - значение числа Маха для фильтрации
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    delta_I_deg = 0.0
    
    # Фильтрация: только положительные углы атаки, нулевая delta_I, заданная delta_II
    filtered_data = data[
        (data['alpha_deg'] >= 0) &  # ТОЛЬКО положительные углы
        (data['delta_II_deg'] == delta_II_deg) & 
        (data['delta_I_deg'] == delta_I_deg) &
        (np.abs(data['Mach'] - mach) < 0.05)
    ]
    
    if filtered_data.empty:
        print(f"Нет данных для alpha>0, delta_II_deg={delta_II_deg}, Mach={mach}")
        return
    
    print(f"Найдено {len(filtered_data)} точек с alpha>0")
    
    # Создание графика
    plt.figure(figsize=(10, 8))
    
    # Сортируем по углу атаки для красивого графика
    filtered_data = filtered_data.sort_values('alpha_deg')
    
    # Строим график
    plt.plot(filtered_data['c_x'], filtered_data['c_y'], 
            marker='o', linewidth=2,
            label=f'M={mach}')
    
    # Подписываем точки углами атаки
    for _, row in filtered_data.iterrows():
        plt.annotate(f"{row['alpha_deg']}°", 
                   xy=(row['c_x'], row['c_y']), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    
    # Настройки графика
    plt.xlabel('C_x')
    plt.ylabel('C_y')
    plt.title(f'Поляра: C_y от C_x (α>0°, δ_II={delta_II_deg}°, M={mach})')
    plt.grid(True)
    plt.legend()
    plt.show()







# --------------------------------------------------------------------
# 5. ГЛАВНАЯ ФУНКЦИЯ
# --------------------------------------------------------------------
def main():
    """Основная функция для запуска всех визуализаций"""
    # Глобальные параметры
    ANGLES = [-10, -5, 0, 5, 10]
    BIG_ANGLES = [-25, -20, -15, -10, 10, 15, 20, 25]
    L_NOS = 0.47
    D = 0.178
    LAMBDA_NOS = L_NOS / D
    
    print("=" * 80)
    print("НАЧАЛО ВИЗУАЛИЗАЦИИ ДАННЫХ")
    print("=" * 80)
    
    # # 1. Визуализация с коррекцией торможения
    # print("\n1. Визуализация C_y с коррекцией торможения...")
    # visualize_cy_comparison('data/all_angles_data.csv', ANGLES, LAMBDA_NOS)
    
    # # 2. Визуализация изолированного крыла
    # print("\n2. Визуализация изолированного крыла...")
    # visualize_simple_cy_plots('data/krylo_isP.csv', ANGLES, r'$C_y\text{из.кр}$')
    
    # # 3. Визуализация коэффициентов интерференции крыла
    # print("\n3. Визуализация коэффициентов интерференции крыла...")
    # visualize_simple_cy_plots('data/Kappa_aa_kr.csv', ANGLES, r'$K_{\alpha \alpha \text{кр}}$')
    
    # # 4. Визуализация изолированного руля
    # print("\n4. Визуализация изолированного руля...")
    # visualize_simple_cy_plots('data/rul_isP.csv', ANGLES, r'$C^\alpha_y\text{из.рл}$')
    
    # # 5. Визуализация коэффициентов интерференции руля
    # print("\n5. Визуализация коэффициентов интерференции руля...")
    # visualize_simple_cy_plots('data/Kappa_aa_rl.csv', ANGLES, r'$K_{\alpha \alpha \text{рл}}$')
    
    # # 6. Визуализация z_b
    # print("\n6. Визуализация z_b...")
    # data = load_data('data/z_b_v.csv')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_single_component(ax, data, 'Mach', 'z_b', 
    #                      color='blue', marker='s',
    #                      xlabel='M', ylabel=r'$\bar{z}_\text{В}$',
    #                      legend_label=r'$\bar{z}_\text{В}$')
    # plt.tight_layout()
    # plt.show()
    
    # # 7. Визуализация i_v
    # print("\n7. Визуализация i_v...")
    # data = load_data('data/i_v.csv')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_single_component(ax, data, 'Mach', 'i_v', 
    #                      color='red', marker='s',
    #                      xlabel='M', ylabel=r'$i_\text{В}$',
    #                      legend_label=r'$i_\text{В}$')
    # plt.tight_layout()
    # plt.show()
    
    # # 8. Визуализация psi_eps зависимостей
    # print("\n8. Визуализация psi_eps зависимостей...")
    # data = load_data('data/psi_eps.csv')
    # x_columns = ['Mach', 'alpha_p', 'phi_alpha', 'psi_I', 'psi_II', 'z_v']
    # plot_dependency_group(data, x_columns, 'psi_eps', figsize=(20, 16), nrows=3, ncols=2)
    # plt.show()
    
    # # 9. Визуализация eps_alpha_sr зависимостей
    # print("\n9. Визуализация eps_alpha_sr зависимостей...")
    # data = load_data('data/eps_alpha_sr.csv')
    # x_columns = ['Mach', 'alpha_p', 'phi_alpha', 'psi_I', 'psi_II', 
    #             'z_v', 'psi_eps', 'i_v', 'k_aa', 'K_aa_rl']
    # plot_dependency_group(data, x_columns[:8], 'eps_alpha_sr', figsize=(20, 16), nrows=4, ncols=2)
    # plt.show()
    
    # plot_dependency_group(data, x_columns[8:], 'eps_alpha_sr', figsize=(20, 8), nrows=1, ncols=2)
    # plt.show()
    
    # # 10. Визуализация kappa_q
    # print("\n10. Визуализация коэффициента торможения...")
    # data = load_data('data/kappa_q_nos.csv')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_single_component(ax, data, 'Mach', 'kappa_q', 
    #                      color='blue', marker='o',
    #                      xlabel='M', ylabel=r'$\kappa_{q \text{нос}}$',
    #                      title=r'Зависимость $\kappa_{q \text{нос}}$ от числа Маха',
    #                      legend_label=r'$\kappa_{q \text{нос}}$')
    # plt.tight_layout()
    # plt.show()
    
    # 11. Визуализация c_y_alpha_sum
    print("\n11. Визуализация суммарного коэффициента подъемной силы...")
    visualize_simple_cy_plots('data/c_y_alpha_sum.csv', ANGLES, r'$c_y^{\alpha}$')
    
    # # 12. Визуализация c_y_delta_1
    # print("\n12. Визуализация коэффициента c_y_delta_1...")
    # data = load_data('data/c_y_delta_1.csv')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_single_component(ax, data, 'Mach', 'c_y_delta_1', 
    #                      color='red', marker='o',
    #                      xlabel='M', ylabel=r'$c_y^{\delta_1}$',
    #                      title=r'Зависимость коэффициента $c_y^{\delta_1}$ от числа Маха',
    #                      legend_label=r'$c_y^{\delta_1}$')
    # ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    
    # # 13. Визуализация c_y_delta_2
    # print("\n13. Визуализация коэффициента c_y_delta_2...")
    # data = load_data('data/c_y_delta_2.csv')
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_single_component(ax, data, 'Mach', 'c_y_delta_2', 
    #                      color='blue', marker='s',
    #                      xlabel='M', ylabel=r'$c_y^{\delta_2}$',
    #                      title=r'Зависимость коэффициента $c_y^{\delta_2}$ от числа Маха',
    #                      legend_label=r'$c_y^{\delta_2}$')
    # ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    
    # # 14. Визуализация больших углов и дельт
    # print("\n14. Визуализация больших углов и дельт...")
    # visualize_big_angles_delta('data/c_y_sum_big_delta.csv')
    
    #####15. Визуализация момента m_z
    print("\n15. Визуализация момента m_z...")
    visualize_mz_vs_alpha('data/m_z_small.csv', mach_value=2.0)
    
    # # 16. Визуализация итогового c_y_alpha
    # print("\n16. Визуализация итогового c_y_alpha...")
    # visualize_simple_cy_plots('data/c_y_alpha_final.csv', ANGLES, r'$c_y^{\alpha}$')

     # 17. Простая визуализация поляры Лилиенталя
    print("\n17. Визуализация поляры Лилиенталя...")
    visualize_cy_vs_cx('data/polar_lilienthal.csv', delta_II_deg=0, mach=1.5)
    
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)

# --------------------------------------------------------------------
# 6. ЗАПУСК ПРОГРАММЫ
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Инициализация глобальных переменных (если нужны)
    angles_big_deg = [-25, -20, -15, -10, 10, 15, 20, 25]
    deltas = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    M_values = np.arange(0.5, 4.1, 0.1)
    
    # Запуск основной функции
    main()