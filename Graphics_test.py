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

def visualize_K_vs_alpha(data_path, delta_II_deg=0, mach_list=[0.5, 1.0 , 3.0]):
    """
    График аэродинамического качества K от угла атаки alpha
    """
    data = pd.read_csv(data_path)
    
    # Фильтруем для заданной delta_II и delta_I=0
    filtered_data = data[
        (data['delta_II_deg'] == delta_II_deg) &
        (data['delta_I_deg'] == 0) &
        (data['K'] >= 0)
    ]
    
    if filtered_data.empty:
        print(f"Нет данных для delta_II={delta_II_deg}")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(mach_list)))
    
    for i, mach in enumerate(mach_list):
        # Берем ближайшее значение M из данных
        mach_data = filtered_data[np.abs(filtered_data['Mach'] - mach) < 0.1]
        if not mach_data.empty:
            # Группируем по углу и усредняем
            grouped = mach_data.groupby('alpha_deg')['K'].mean()
            plt.plot(grouped.index, grouped.values, 
                    color=colors[i], 
                    linewidth=2,
                    label=f'M={mach:.1f}')
    
    plt.xlabel(r'$\alpha$, град', fontsize=18)
    plt.ylabel('K ', fontsize=18)
    plt.title(f'Аэродинамическое качество K ($\delta_{{II}}={delta_II_deg}^\circ$)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # 2. График: K vs α для всех delta_II при фиксированном Mach
    print("\n" + "="*60)
    print("График 2: K vs α для всех delta_II при фиксированном Mach")
    print("="*60)
    
    # Определяем доступные значения Mach
    available_mach = sorted(data['Mach'].unique())
    print(f"Доступные значения Mach: {available_mach}")
    
    # Фиксируем Mach (можно изменить на нужное)
    fixed_mach = 1.0 if available_mach else 3.0
    if fixed_mach not in available_mach and available_mach:
        # Берем ближайшее доступное значение
        fixed_mach = min(available_mach, key=lambda x: abs(x - fixed_mach))
    
    print(f"Используемое значение Mach: {fixed_mach:.1f}")
    
    # Фильтруем данные для фиксированного Mach и delta_I=0
    mach_fixed_data = data[
        (np.abs(data['Mach'] - fixed_mach) < 0.1) &
        (data['delta_I_deg'] == 0) &
        (data['K'] >= 0)
    ]
    
    # Фильтруем alpha в диапазоне -10 до 10
    mach_fixed_data = mach_fixed_data[
        (mach_fixed_data['alpha_deg'] >= -10) & 
        (mach_fixed_data['alpha_deg'] <= 10)
    ]
    
    # Получаем все уникальные delta_II
    all_deltas_II = sorted(mach_fixed_data['delta_II_deg'].unique())
    print(f"Найдено delta_II: {all_deltas_II}")
    
    if len(all_deltas_II) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Определяем цвета для разных delta_II
        colors_delta = plt.cm.viridis(np.linspace(0, 1, len(all_deltas_II)))
        
        for delta, color in zip(all_deltas_II, colors_delta):
            delta_data = mach_fixed_data[mach_fixed_data['delta_II_deg'] == delta]
            
            if not delta_data.empty:
                # Сортируем по alpha_deg
                delta_data = delta_data.sort_values('alpha_deg')
                
                # Определяем стиль линии
                linestyle = '-' if delta >= 0 else '--'
                linewidth = 2 if delta == 0 else 1.5
                
                ax.plot(delta_data['alpha_deg'], delta_data['K'],
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        marker='o' if len(delta_data) < 10 else '',
                        markersize=4,
                        label=fr'$\delta_{{II}}={delta:.0f}^\circ$')
        
        ax.set_xlabel(r'$\alpha$, град', fontsize=18)
        ax.set_ylabel('K', fontsize=18)
        ax.set_title(fr'Зависимость K от $\alpha$ ($M={fixed_mach:.1f}$, $\delta_I=0^\circ$)', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Легенда с заголовком
        ax.legend(fontsize=10, title=fr'$\delta_{{II}}$, град', 
                 title_fontsize=11, loc='upper right')
        
        # Добавляем линии сетки
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Ограничиваем диапазон оси X
        ax.set_xlim(-10, 10)
        
        # Добавляем вертикальные линии для ключевых углов
        for angle in [-10, -5, 0, 5, 10]:
            ax.axvline(x=angle, color='lightgray', linestyle=':', alpha=0.3)
        
        # Добавляем подпись с фиксированным Mach
        ax.text(0.02, 0.98, f'M = {fixed_mach:.1f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 3. Дополнительно: можно построить несколько графиков для разных Mach
        print("\n" + "="*60)
        print("Дополнительно: графики для нескольких значений Mach")
        print("="*60)
        
        # Выбираем несколько значений Mach для отображения
        if len(available_mach) > 1:
            selected_mach = [available_mach[0], 
                           available_mach[len(available_mach)//2], 
                           available_mach[-1]]
            selected_mach = selected_mach[:min(3, len(available_mach))]
            
            for plot_mach in selected_mach:
                if plot_mach != fixed_mach:  # Не повторять уже построенный
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    
                    mach_data_for_plot = data[
                        (np.abs(data['Mach'] - plot_mach) < 0.1) &
                        (data['delta_I_deg'] == 0) &
                        (data['K'] >= 0) &
                        (data['alpha_deg'] >= -10) & 
                        (data['alpha_deg'] <= 10)
                    ]
                    
                    plot_deltas = sorted(mach_data_for_plot['delta_II_deg'].unique())
                    
                    if len(plot_deltas) > 0:
                        colors_plot = plt.cm.viridis(np.linspace(0, 1, len(plot_deltas)))
                        
                        for delta, color in zip(plot_deltas, colors_plot):
                            delta_plot_data = mach_data_for_plot[mach_data_for_plot['delta_II_deg'] == delta]
                            if not delta_plot_data.empty:
                                delta_plot_data = delta_plot_data.sort_values('alpha_deg')
                                linestyle = '-' if delta >= 0 else '--'
                                linewidth = 2 if delta == 0 else 1.5
                                
                                ax2.plot(delta_plot_data['alpha_deg'], delta_plot_data['K'],
                                        color=color,
                                        linewidth=linewidth,
                                        linestyle=linestyle,
                                        label=fr'$\delta_{{II}}={delta:.0f}^\circ$')
                        
                        ax2.set_xlabel(r'$\alpha$, град', fontsize=18)
                        ax2.set_ylabel('K', fontsize=18)
                        ax2.set_title(fr'Зависимость K от $\alpha$ ($M={plot_mach:.1f}$, $\delta_I=0^\circ$)', fontsize=16)
                        ax2.grid(True, alpha=0.3)
                        ax2.legend(fontsize=10, title=fr'$\delta_{{II}}$, град', 
                                 title_fontsize=11, loc='upper right')
                        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                        ax2.set_xlim(-10, 10)
                        plt.tight_layout()
                        plt.show()
    else:
        print("Нет данных для построения графика 2")
    

def visualize_big_angles_delta(data_path, mach_value=None):
    """Визуализация больших углов и дельт"""
    data = load_data(data_path)
    
    # Определяем углы атаки и дельты из данных
    angles = sorted(data['alpha_deg'].unique())
    deltas_II = sorted(data['delta_II_deg'].unique())
    
    print(f"Все углы атаки: {angles}")
    print(f"Углы отклонения рулей: {deltas_II}")
    print(f"Доступные значения M: {sorted(data['Mach'].unique())}")
    
    # Фильтр для alpha_deg
    selected_angles = [-10.0, -0.5, 0.0, 5.0, 10.0]
    # Оставляем только те углы, которые есть в данных
    available_angles = [angle for angle in selected_angles if angle in angles]
    
    if not available_angles:
        print("Выбранные углы атаки не найдены в данных!")
        return
    
    print(f"\nВыбранные углы атаки для анализа: {available_angles}")
    
    # Положительные углы атаки из выбранных
    pos_angles = [a for a in available_angles if a > 0]
    
    if not pos_angles:
        print("Нет положительных углов среди выбранных!")

    # 3. ГРАФИК: C_y vs δ_II для фиксированного α в диапазоне -10 до 10
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Фиксируем несколько углов атаки в диапазоне -10 до 10
    fixed_alphas = [-10.0, -5.0, 0.0, 5.0, 10.0]
    # Проверяем, какие углы есть в данных
    available_fixed_alphas = [alpha for alpha in fixed_alphas if alpha in angles]
    
    if not available_fixed_alphas:
        print("Нет углов атаки в диапазоне -10 до 10 в данных!")
    else:
        # Выбираем одно или несколько значений M для отображения
        mach_values = sorted(data['Mach'].unique())
        # Выбираем 2-3 значения M для отображения
        selected_mach = mach_values[:3] if len(mach_values) >= 3 else mach_values
        
        # Определяем цвета для разных α
        colors = plt.cm.Set1(np.linspace(0, 1, len(available_fixed_alphas)))
        
        # Для каждого M строим линии для разных α
        for mach in selected_mach:
            mach_data = data[data['Mach'] == mach]
            
            for alpha, color in zip(available_fixed_alphas, colors):
                alpha_data = mach_data[mach_data['alpha_deg'] == alpha]
                if not alpha_data.empty:
                    # Сортируем по δ_II
                    alpha_data = alpha_data.sort_values('delta_II_deg')
                    
                    ax3.plot(alpha_data['delta_II_deg'], alpha_data['c_y_sum_big_delta'],
                            linewidth=2,
                            color=color,
                            marker='o',
                            markersize=5,
                            label=fr'M={mach:.1f}, $\alpha={alpha:.1f}^\circ$')
        
        ax3.set_xlabel(r'$\delta_{II}$, град', fontsize=18)
        ax3.set_ylabel(r'${c_{y_{a}}}$', fontsize=18)
        ax3.set_title(r'Зависимость $c_{y_a}$ от $\delta_{II}$', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Добавляем вертикальные линии для выбранных дельт
        if deltas_II:
            for delta in [-10, 0, 10]:
                if delta in deltas_II:
                    ax3.axvline(x=delta, color='lightgray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 4. ГРАФИК: C_y vs α для ВСЕХ δ_II при ФИКСИРОВАННОМ M (ручной выбор)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Определяем фиксированное значение M
    if mach_value is None:
        # Если не указано, берем среднее значение
        fixed_mach = mach_values[len(mach_values)//2] if mach_values else 0.8
    else:
        fixed_mach = mach_value
    
    # Проверяем, есть ли такое M в данных
    if fixed_mach not in mach_values:
        print(f"Внимание: M={fixed_mach} нет в данных!")
        # Берем ближайшее доступное
        fixed_mach = min(mach_values, key=lambda x: abs(x - fixed_mach))
        print(f"Используем ближайшее доступное M={fixed_mach}")
    
    print(f"\nФиксированное значение M для 4-го графика: {fixed_mach}")
    
    # Фильтруем данные по фиксированному M
    mach_data_fixed = data[data['Mach'] == fixed_mach]
    
    # Рисуем линии для каждой δ_II
    for delta in sorted(deltas_II):
        delta_data = mach_data_fixed[mach_data_fixed['delta_II_deg'] == delta]
        if not delta_data.empty:
            delta_data = delta_data.sort_values('alpha_deg')
            
            # Определяем стиль линии в зависимости от знака дельты
            linestyle = '-' if delta >= 0 else '--'
            linewidth = 2 if delta == 0 else 1.5
            
            ax4.plot(delta_data['alpha_deg'], delta_data['c_y_sum_big_delta'],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=fr'$\delta_{{II}}={delta:.0f}^\circ$')
    
    ax4.set_xlabel(r'$\alpha$, град', fontsize=18)
    ax4.set_ylabel(r'${c_{y_{a}}}$', fontsize=18)
    ax4.set_title(fr' $M={fixed_mach:.1f}$', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, title=fr'$\delta_{{II}}$, град', title_fontsize=10)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlim(-10, 10)  # Ограничиваем диапазон по оси X от -10 до 10

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
    
    plt.xlabel(r'$\alpha$, град' , fontsize  = 18)
    plt.ylabel(r'$m_z$' , fontsize  = 18)
    plt.title(r'$x_\text{цм}$ = 1.914 м, M=0.5', fontsize  = 18)
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
    plt.xlabel(r'$c_x$', fontsize = 18)
    plt.ylabel(r'$c_y$', fontsize = 18)
    plt.title(f'Поляра: α>0°, δII={delta_II_deg}°, M={mach}')
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_ny_rasp_simple(data_path, delta_II=0):
    """
    Простая визуализация располагаемой перегрузки
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    
    # Фильтруем для delta_I=0, delta_II=заданного
    data = data[(data['delta_I_deg'] == 0) & (data['delta_II_deg'] == delta_II)]
    
    if data.empty:
        print(f"Нет данных для delta_II={delta_II}")
        return
    
    # 1. График: n_y_rasp vs M для разных alpha
    plt.figure(figsize=(12, 5))
    
    # Получаем уникальные углы атаки
    angles = sorted(data['alpha_deg'].unique())
    
    for alpha in angles:
        angle_data = data[data['alpha_deg'] == alpha]
        angle_data = angle_data.sort_values('Mach')
        
        plt.plot(angle_data['Mach'], angle_data['n_y_rasp'], 
                marker='o', linewidth=2,
                label=f'α={alpha}°')
    
    plt.xlabel('M')
    plt.ylabel('n_y_rasp')
    plt.title(f'Располагаемая перегрузка (δ_II={delta_II}°)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 2. График: n_y_rasp vs alpha для разных M
    plt.figure(figsize=(12, 5))
    
    # Выбираем несколько чисел Маха
    mach_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for M in mach_list:
        # Берем ближайшее значение M из данных
        mach_data = data[np.abs(data['Mach'] - M) < 0.1]
        if not mach_data.empty:
            # Группируем по углу и усредняем
            grouped = mach_data.groupby('alpha_deg')['n_y_rasp'].mean()
            plt.plot(grouped.index, grouped.values, 
                    marker='s', linewidth=2,
                    label=f'M≈{M}')
    
    plt.xlabel('α, град')
    plt.ylabel('n_y_rasp')
    plt.title(f'Располагаемая перегрузка vs α (δ_II={delta_II}°)')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_ny_rasp_by_delta(data_path, mach_value=2.0, alpha_value=5):
    """
    Визуализация как меняется n_y_rasp при изменении delta_II
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    
    # Фильтруем
    data = data[
        (np.abs(data['Mach'] - mach_value) < 0.1) &
        (np.abs(data['alpha_deg'] - alpha_value) < 0.5) &
        (data['delta_I_deg'] == 0)
    ]
    
    if data.empty:
        print(f"Нет данных для M≈{mach_value}, α≈{alpha_value}")
        return
    
    # Сортируем по delta_II
    data = data.sort_values('delta_II_deg')
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['delta_II_deg'], data['n_y_rasp'], 
            linewidth=2)
    
    plt.xlabel('δ_II, град')
    plt.ylabel('n_y_rasp')
    plt.title(f'n_y_rasp vs δ_II (M≈{mach_value}, α≈{alpha_value}°)')
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.show()

def visualize_K_vs_alpha(data_path, delta_II_deg=0, mach_list=[3.0]):
    """
    График аэродинамического качества K от угла атаки alpha
    """
    data = pd.read_csv(data_path)
    
    # 1. Первый график: K vs α для заданной delta_II и нескольких Mach
    plt.figure(figsize=(12, 8))
    
    # Фильтруем для заданной delta_II и delta_I=0
    filtered_data = data[
        (data['delta_II_deg'] == delta_II_deg) &
        (data['delta_I_deg'] == 0)
    ]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(mach_list)))
    
    for i, mach in enumerate(mach_list):
        # Берем ближайшее значение M из данных
        mach_data = filtered_data[np.abs(filtered_data['Mach'] - mach) < 0.1]
        if not mach_data.empty:
            mach_data = mach_data.sort_values('alpha_deg')
            plt.plot(mach_data['alpha_deg'], mach_data['K'], 
                    color=colors[i], 
                    linewidth=2,
                    label=f'M={mach:.1f}')
    
    plt.xlabel(r'$\alpha$, град', fontsize=18)
    plt.ylabel('K ', fontsize=18)
    plt.title(f'K vs $\\alpha$ ($\\delta_{{II}}={delta_II_deg}^\\circ$)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # 2. Второй график: K vs α для фиксированного Mach и всех delta_II
    plt.figure(figsize=(12, 8))
    
    # Фиксируем Mach
    fixed_mach = 4.0
    print(f"\nВторой график: M = {fixed_mach:.1f}")
    
    # Фильтруем данные по фиксированному Mach и delta_I=0
    mach_data = data[
        (np.abs(data['Mach'] - fixed_mach) < 0.1) &
        (data['delta_I_deg'] == 0)
    ]
    
    # Фильтруем альфу в диапазоне -10 до 10
    mach_data = mach_data[
        (mach_data['alpha_deg'] >= -10) &
        (mach_data['alpha_deg'] <= 10)
    ]
    
    # Получаем все уникальные delta_II
    all_deltas = sorted(mach_data['delta_II_deg'].unique())
    print(f"Найдено delta_II: {all_deltas}")
    
    # Строим график для каждой delta_II
    for delta in all_deltas:
        delta_data = mach_data[mach_data['delta_II_deg'] == delta]
        if not delta_data.empty:
            delta_data = delta_data.sort_values('alpha_deg')
            plt.plot(delta_data['alpha_deg'], delta_data['K'],
                    linewidth=2,
                    label=fr'$\delta_{{II}}={delta:.0f}^\circ$')
    
    plt.xlabel(r'$\alpha$, град', fontsize=18)
    plt.ylabel('K', fontsize=18)
    plt.title(fr'$M={fixed_mach:.1f}$', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, title=fr'$\delta_{{II}}$', title_fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlim(-10, 10)
    plt.tight_layout()
    plt.show()
    

def visualize_balancing_params(data_path, mach_values=[0.5, 1.0, 4.0], fixed_delta_II_deg=0, fixed_alpha_deg=0):
    df = pd.read_csv(data_path)
    
    # 2. Фильтрация
    df = df[df['delta_I_deg'] == 0]
    df = df[df['delta_II_deg'] == fixed_delta_II_deg]
    df = df[np.abs(df['alpha_deg'] - fixed_alpha_deg) < 0.1]
    
    # 3. Создание ПЕРВОГО графика
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # 4. Построение для каждого M (только delta_bal)
    for mach in mach_values:
        mach_df = df[np.abs(df['Mach'] - mach) < 0.05]
        
        if mach_df.empty:
            continue
            
        # Добавляем точку (0,0) если ее нет
        if 0 not in mach_df['n_y_rasp'].values:
            zero_point = pd.DataFrame({
                'n_y_rasp': [0],
                'delta_bal': [0],
                'alpha_bal': [0]
            })
            mach_df = pd.concat([mach_df, zero_point])
        
        # Сортируем
        mach_df = mach_df.sort_values('n_y_rasp')
        
        # ТОЛЬКО ПЕРВЫЙ ГРАФИК
        ax1.plot(mach_df['n_y_rasp'], mach_df['delta_bal'], 
                label=f'M={mach}')
    
    # 5. Настройка первого графика
    ax1.set_xlabel(r'$n_\text{y потр}$', fontsize = 18)
    ax1.set_ylabel(r'$\delta_\text{бал}$, град', fontsize = 18)
    ax1.grid(True)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.axvline(x=0, color='gray', linestyle='--')
    plt.title(r'$x_\text{цм}$ = 1.914 м, H = 30000 м  ', fontsize=18)
    plt.tight_layout()
    plt.show()

    # 6. Создание ВТОРОГО графика
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    # 7. Построение для каждого M (только alpha_bal)
    for mach in mach_values:
        mach_df = df[np.abs(df['Mach'] - mach) < 0.05]
        
        if mach_df.empty:
            continue
            
        # Добавляем точку (0,0) если ее нет (повторно для второго графика)
        if 0 not in mach_df['n_y_rasp'].values:
            zero_point = pd.DataFrame({
                'n_y_rasp': [0],
                'delta_bal': [0],
                'alpha_bal': [0]
            })
            mach_df = pd.concat([mach_df, zero_point])
        
        # Сортируем
        mach_df = mach_df.sort_values('n_y_rasp')
        
        # ТОЛЬКО ВТОРОЙ ГРАФИК
        ax2.plot(mach_df['n_y_rasp'], mach_df['alpha_bal'],
                label=f'M={mach}')
    
    # 8. Настройка второго графика
    ax2.set_xlabel(r'$n_\text{y потр}$', fontsize=16)
    ax2.set_ylabel(r'$\alpha_\text{бал}$, град', fontsize=16)
    ax2.grid(True)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axvline(x=0, color='gray', linestyle='--')
    plt.title(r'$x_\text{цм}$ = 1.914 м, H = 30000 м  ', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_mach_curves(filename, alpha, delta, mach_list):
    df = pd.read_csv(filename)
    tolerance=1e-10
    # 2. Фильтруем данные
    filtered = df[
        (df['alpha_deg'] == alpha) & 
        (df['delta_II_deg'] == delta)
    ].copy()  # Делаем копию
    
    if filtered.empty:
        print("Нет данных для указанных параметров!")
        return
    
    # 3. Создаём график
    plt.figure(figsize=(10, 6))
    
    # 4. Рисуем кривые для каждого числа Маха
    for target_mach in sorted(mach_list):
        # Ищем данные с близкими значениями Mach
        mach_data = filtered[np.abs(filtered['Mach'] - target_mach) < tolerance].copy()
        
        if not mach_data.empty:
            # Сортируем по n_y_rasp
            mach_data = mach_data.sort_values('n_y_rasp')
            
            # Добавляем точку (0,0) если её нет
            if not ((0, 0) in zip(mach_data['n_y_rasp'], mach_data['M_sh'])):
                zero_point = pd.DataFrame({
                    'n_y_rasp': [0],
                    'M_sh': [0],
                    'Mach': [target_mach],  # Добавляем Mach для совместимости
                    'alpha_deg': [alpha],
                    'delta_II_deg': [delta]
                })
                mach_data = pd.concat([zero_point, mach_data], ignore_index=True)
                mach_data = mach_data.sort_values('n_y_rasp')
            
            # Рисуем линию и точки
            plt.plot(mach_data['n_y_rasp'], mach_data['M_sh'], 
                    linewidth=2, 
                    label=f'Mach = {target_mach}')
    
    # 5. Добавляем линии осей
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 6. Настраиваем график
    plt.xlabel(r'$n_\text{y потр}$', fontsize=18)
    plt.ylabel(r'$M_\text{ш}$', fontsize=18)
    plt.title(r'$x_\text{цм}$ = 1.914 м, H = 30000 м  ', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 7. Показываем график
    plt.tight_layout()
    plt.show()

    
    

def visualize_x_Fa_f_simple_reversed(file_path):
    """Простой график: x_Fa_f по оси X, Mach по оси Y"""
    df = pd.read_csv(file_path)
    df = df.sort_values('x_Fa_f')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['x_Fa_f'], df['Mach'], 'b-o', linewidth=2)
    plt.xlabel('x_Fa_f (м)')
    plt.ylabel('Mach')
    plt.grid(True)
    plt.title('Mach от x_Fa_f')
    plt.tight_layout()
    plt.show()

def visualize_delta_c_simple(file_path):
    """
    Простая визуализация delta_c от Height для разных Mach
    """
    # Загружаем данные из файла
    df = pd.read_csv(file_path)
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    
    # Для каждого числа Маха строим свою линию
    for mach in df['Mach'].unique():
        mach_data = df[df['Mach'] == mach].sort_values('Height')
        plt.plot(mach_data['Height'], mach_data['delta_c'], 
                marker='o', label=f'M={mach}')
    
    plt.xlabel('Высота, м')
    plt.ylabel('delta_c')
    plt.title('delta_c от высоты для разных чисел Маха')
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
    
    # #1. Визуализация с коррекцией торможения
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
    
    # # 11. Визуализация c_y_alpha_sum
    # print("\n11. Визуализация суммарного коэффициента подъемной силы...")
    # visualize_simple_cy_plots('data/c_y_alpha_sum.csv', ANGLES, r'$c_y^{\alpha}$')
    
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
    
    # 14. Визуализация больших углов и дельт
    # print("\n14. Визуализация больших углов и дельт...")
    # visualize_big_angles_delta('data/c_y_sum_big_delta.csv', 4.0)

    # visualize_small_angles_delta('data/c_y_small_angles.csv')
    
#     #####15. Визуализация момента m_z
#     print("\n15. Визуализация момента m_z...")
#     visualize_mz_vs_alpha('data/m_z_small.csv', mach_value=1.5)
    
#     # # 16. Визуализация итогового c_y_alpha
#     # print("\n16. Визуализация итогового c_y_alpha...")
#     # visualize_simple_cy_plots('data/c_y_alpha_final.csv', ANGLES, r'$c_y^{\alpha}$')

#      # 17. Простая визуализация поляры Лилиенталя
#     print("\n17. Визуализация поляры Лилиенталя...")
#     visualize_cy_vs_cx('data/polar_lilienthal.csv', delta_II_deg=0, mach=1.5)

#     # 18. Визуализация располагаемой перегрузки
#     print("\n18. Визуализация располагаемой перегрузки...")
#     visualize_ny_rasp_simple('data/n_y_rasp.csv', delta_II=0)

    # #19.
    visualize_K_vs_alpha('data/K_quality.csv', delta_II_deg=25, mach_list=[0.5, 1.5, 3.0])
    
#     # #21
#     visualize_balancing_params('data/n_y_rasp_test.csv', mach_values=[0.5, 1.5, 4.0], fixed_delta_II_deg=-5, fixed_alpha_deg=-5)

    #22
    # Загружаем данные из CSV файла
    visualize_x_Fa_f_simple_reversed('data/x_Fa.csv')  # Предполагаю, что файл имеет расширение .csv

#     #22
#     plot_mach_curves(
#     filename='data/M_sh.csv',
#     alpha=5,        # фиксированный угол атаки
#     delta=-10,        # фиксированный угол отклонения
#     mach_list=[0.5, 1.5, 4.0]  # 3 числа Маха для сравнения
# )
    # #23
    # visualize_delta_c_simple('data/delta_c_df.csv')


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