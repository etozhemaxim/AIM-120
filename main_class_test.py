# graphics_oo.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Общие настройки стиля
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = False

# Наборы значений, как в исходнике
ANGLES_SMALL = [-10, -5, 0, 5, 10]
ANGLES_BIG = [-25, -20, -15, -10, 10, 15, 20, 25]
DELTAS_II = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
SELECTED_MACH = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# ---------- Утилиты загрузки ----------

def _get_df(name: str, dfs: Optional[Dict[str, pd.DataFrame]], from_csv: bool) -> pd.DataFrame:
    """
    Возвращает DataFrame либо из словаря dfs[name], либо читает CSV name из диска.
    """
    if dfs is not None and not from_csv:
        return dfs[name]
    return pd.read_csv(name)

# ---------- Рисующие функции (повторяют “парный” стиль) ----------

def plot_all_angles_data(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    """
    all_angles_data.csv: фюзеляж (изолированный), с наложением торможения потока kappa_q (как в исходнике)
    """
    df = _get_df('all_angles_data.csv', dfs, from_csv)

    # внутренняя модель торможения (оставляю, как у вас)
    l_nos, D = 0.47, 0.178
    lambda_nos = l_nos / D

    def calculate_kappa_q_nos_con(M, lambda_nos):
        if lambda_nos < 0.7 or lambda_nos > 5.0:
            return 1.0
        if M <= 1.0:
            return 1.0
        lambda_values_M2 = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        kappa_values_M2 = [0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.75, 0.74]
        kappa_at_M2 = np.interp(lambda_nos, lambda_values_M2, kappa_values_M2)
        if M <= 2.0:
            return 1.0 + (M - 1.0) * (kappa_at_M2 - 1.0)
        else:
            return kappa_at_M2

    data = df.copy()
    data['kappa_q'] = data['Mach'].apply(lambda M: calculate_kappa_q_nos_con(M, lambda_nos))
    for angle in ANGLES_SMALL:
        col = f'alpha_{angle}'
        data[col + '_with_kappa'] = data[col] * data['kappa_q']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # Левый: C_y vs M для разных α (без/с торможением)
    for i, angle in enumerate(ANGLES_SMALL):
        col = f'alpha_{angle}'
        ax1.plot(df['Mach'], df[col],
                 color=colors[i % len(colors)], lw=2, ls='--', alpha=0.7,
                 label=fr'$\alpha = {angle}^\circ$ (без торможения)')
        ax1.plot(data['Mach'], data[col + '_with_kappa'],
                 color=colors[i % len(colors)], lw=3, ls='-',
                 label=fr'$\alpha = {angle}^\circ$ (с торможением)')
    ax1.set_xlabel('M', fontsize=15)
    ax1.set_ylabel(r'$C_y\ \mathrm{из.ф}$', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Правый: C_y vs α для разных M (без/с торможением)
    for j, mach in enumerate(SELECTED_MACH):
        idx = (df['Mach'] - mach).abs().idxmin()
        cy_no = [df.iloc[idx][f'alpha_{a}'] for a in ANGLES_SMALL]
        cy_wk = [data.iloc[idx][f'alpha_{a}_with_kappa'] for a in ANGLES_SMALL]
        ax2.plot(ANGLES_SMALL, cy_no, color=mach_colors[j % len(mach_colors)], lw=2, ls='--',
                 alpha=0.7, marker='s', ms=4, label=fr'$M = {mach}$ (без торможения)')
        ax2.plot(ANGLES_SMALL, cy_wk, color=mach_colors[j % len(mach_colors)], lw=3, ls='-',
                 marker='o', ms=5, label=fr'$M = {mach}$ (с торможением)')
    ax2.set_xlabel(r'$\alpha$', fontsize=16)
    ax2.set_ylabel(r'$C_y\ \mathrm{из.ф}$', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_isolated_wing(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True, with_intrf: bool = False):
    """
    krylo_isP.csv или krylo_isP_Intrf.csv
    """
    name = 'krylo_isP_Intrf.csv' if with_intrf else 'krylo_isP.csv'
    data = _get_df(name, dfs, from_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # Левый: C_y vs M для разных α
    for i, angle in enumerate(ANGLES_SMALL):
        col = f'alpha_{angle}'
        ax1.plot(data['Mach'], data[col], color=colors[i % len(colors)], lw=2, ls='-',
                 label=fr'$\alpha = {angle}^\circ$')
    ax1.set_xlabel('M', fontsize=15)
    ax1.set_ylabel(r'$C_y\ \mathrm{из.кр}$', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Правый: C_y vs α для разных M
    for j, mach in enumerate(SELECTED_MACH):
        idx = (data['Mach'] - mach).abs().idxmin()
        cy = [data.iloc[idx][f'alpha_{a}'] for a in ANGLES_SMALL]
        ax2.plot(ANGLES_SMALL, cy, color=mach_colors[j % len(mach_colors)], lw=2, ls='-',
                 marker='o', ms=5, label=fr'$M = {mach}$')
    ax2.set_xlabel(r'$\alpha$', fontsize=16)
    ax2.set_ylabel(r'$C_y\ \mathrm{из.кр}$', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_Kappa_aa(name_csv: str, ylabel_tex: str,
                  dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    """
    Kappa_aa_kr.csv / Kappa_aa_rl.csv
    """
    data = _get_df(name_csv, dfs, from_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    for i, angle in enumerate(ANGLES_SMALL):
        col = f'alpha_{angle}'
        ax1.plot(data['Mach'], data[col], color=colors[i % len(colors)], lw=2, ls='-',
                 label=fr'$\alpha = {angle}^\circ$')
    ax1.set_xlabel('M', fontsize=15)
    ax1.set_ylabel(ylabel_tex, fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    for j, mach in enumerate(SELECTED_MACH):
        idx = (data['Mach'] - mach).abs().idxmin()
        cy = [data.iloc[idx][f'alpha_{a}'] for a in ANGLES_SMALL]
        ax2.plot(ANGLES_SMALL, cy, color=mach_colors[j % len(mach_colors)], lw=2, ls='-',
                 marker='o', ms=5, label=fr'$M = {mach}$')
    ax2.set_xlabel(r'$\alpha$', fontsize=16)
    ax2.set_ylabel(ylabel_tex, fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_scalar_vs_M(name_csv: str, yname: str, ylabel_tex: str,
                     dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df(name_csv, dfs, from_csv)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['Mach'], data[yname], color='blue', lw=2, marker='o', ms=4, label=ylabel_tex)
    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel(ylabel_tex, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_psi_eps(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df('psi_eps.csv', dfs, from_csv)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. psi_eps vs Mach
    mach_avg = data.groupby('Mach')['psi_eps'].mean()
    ax1.plot(mach_avg.index, mach_avg.values, color='blue', lw=2, marker='o', ms=4)
    ax1.set_xlabel('M', fontsize=14); ax1.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax1.grid(True, alpha=0.3)

    # 2. vs alpha_p
    alpha_p_avg = data.groupby('alpha_p')['psi_eps'].mean()
    ax2.plot(alpha_p_avg.index, alpha_p_avg.values, color='red', lw=2, marker='s', ms=4)
    ax2.set_xlabel(r'$\alpha_p$', fontsize=14); ax2.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax2.grid(True, alpha=0.3)

    # 3. vs phi_alpha
    phi_alpha_avg = data.groupby('phi_alpha')['psi_eps'].mean()
    ax3.plot(phi_alpha_avg.index, phi_alpha_avg.values, color='green', lw=2, marker='^', ms=4)
    ax3.set_xlabel(r'$\varphi_{\alpha}$', fontsize=14); ax3.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax3.grid(True, alpha=0.3)

    # 4. vs psi_I
    psi_I_avg = data.groupby('psi_I')['psi_eps'].mean()
    ax4.plot(psi_I_avg.index, psi_I_avg.values, color='orange', lw=2, marker='d', ms=4)
    ax4.set_xlabel(r'$\psi_I$', fontsize=14); ax4.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax4.grid(True, alpha=0.3)

    # 5. vs psi_II
    psi_II_avg = data.groupby('psi_II')['psi_eps'].mean()
    ax5.plot(psi_II_avg.index, psi_II_avg.values, color='purple', lw=2, marker='v', ms=4)
    ax5.set_xlabel(r'$\psi_{II}$', fontsize=14); ax5.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax5.grid(True, alpha=0.3)

    # 6. vs z_v
    z_v_avg = data.groupby('z_v')['psi_eps'].mean()
    ax6.plot(z_v_avg.index, z_v_avg.values, color='brown', lw=2, marker='*', ms=6)
    ax6.set_xlabel(r'$z_\mathrm{В}$', fontsize=14); ax6.set_ylabel(r'$\psi_{\varepsilon}$', fontsize=16); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_eps_alpha_sr(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df('eps_alpha_sr.csv', dfs, from_csv)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(20, 16))

    def _group_line(ax, xname, yname, color, marker):
        avg = data.groupby(xname)[yname].mean()
        ax.plot(avg.index, avg.values, color=color, lw=2, marker=marker, ms=4)
        ax.grid(True, alpha=0.3)

    _group_line(ax1, 'Mach', 'eps_alpha_sr', 'blue', 'o')
    ax1.set_xlabel('M', fontsize=14); ax1.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax2, 'alpha_p', 'eps_alpha_sr', 'red', 's')
    ax2.set_xlabel(r'$\alpha_p$', fontsize=14); ax2.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax3, 'phi_alpha', 'eps_alpha_sr', 'green', '^')
    ax3.set_xlabel(r'$\phi_{\alpha}$', fontsize=14); ax3.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax4, 'psi_I', 'eps_alpha_sr', 'orange', 'd')
    ax4.set_xlabel(r'$\psi_I$', fontsize=14); ax4.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax5, 'psi_II', 'eps_alpha_sr', 'purple', 'v')
    ax5.set_xlabel(r'$\psi_{II}$', fontsize=14); ax5.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax6, 'z_v', 'eps_alpha_sr', 'brown', '*')
    ax6.set_xlabel(r'$z_v$', fontsize=14); ax6.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax7, 'psi_eps', 'eps_alpha_sr', 'cyan', 'x')
    ax7.set_xlabel(r'$\psi_{\varepsilon}$', fontsize=14); ax7.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    _group_line(ax8, 'i_v', 'eps_alpha_sr', 'magenta', '+')
    ax8.set_xlabel(r'$i_v$', fontsize=14); ax8.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16)

    plt.tight_layout()
    plt.show()

    # Доп. графики по интерференции
    fig3, (ax9, ax10) = plt.subplots(1, 2, figsize=(20, 8))
    k_aa_avg = data.groupby('k_aa')['eps_alpha_sr'].mean()
    ax9.plot(k_aa_avg.index, k_aa_avg.values, color='navy', lw=2, marker='o', ms=4)
    ax9.set_xlabel(r'$k_{\alpha\alpha}$', fontsize=14); ax9.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16); ax9.grid(True, alpha=0.3)

    K_aa_rl_avg = data.groupby('K_aa_rl')['eps_alpha_sr'].mean()
    ax10.plot(K_aa_rl_avg.index, K_aa_rl_avg.values, color='darkred', lw=2, marker='s', ms=4)
    ax10.set_xlabel(r'$K_{\alpha\alpha}^{\mathrm{рл}}$', fontsize=14); ax10.set_ylabel(r'$\varepsilon_{\alpha}^{cp}$', fontsize=16); ax10.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_cy_alpha_sum(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df('c_y_alpha_sum.csv', dfs, from_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    mach_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    for i, angle in enumerate(ANGLES_SMALL):
        col = f'alpha_{angle}'
        ax1.plot(data['Mach'], data[col], color=colors[i % len(colors)], lw=2, ls='-',
                 label=fr'$\alpha = {angle}^\circ$')
    ax1.set_xlabel('M', fontsize=15); ax1.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
    ax1.set_title(r'$c_y^{\alpha}$ vs M', fontsize=14); ax1.grid(True, alpha=0.3); ax1.legend(loc='upper right', fontsize=10)

    for j, mach in enumerate(SELECTED_MACH):
        idx = (data['Mach'] - mach).abs().idxmin()
        cy = [data.iloc[idx][f'alpha_{a}'] for a in ANGLES_SMALL]
        ax2.plot(ANGLES_SMALL, cy, color=mach_colors[j % len(mach_colors)], lw=2, ls='-',
                 marker='o', ms=5, label=fr'$M = {mach}$')
    ax2.set_xlabel(r'$\alpha$, град', fontsize=16); ax2.set_ylabel(r'$c_y^{\alpha}$', fontsize=16)
    ax2.grid(True, alpha=0.3); ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5); ax2.axvline(0, color='gray', ls='--', alpha=0.5)

    plt.tight_layout(); plt.show()

def plot_cy_delta_12(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    # delta_1
    d1 = _get_df('c_y_delta_1.csv', dfs, from_csv)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(d1['Mach'], d1['c_y_delta_1'], color='red', lw=3, ls='-', marker='o', ms=4, label=r'$c_y^{\delta_1}$')
    ax.set_xlabel('M', fontsize=14); ax.set_ylabel(r'$c_y^{\delta_1}$', fontsize=16)
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=12); ax.axhline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    # delta_2
    d2 = _get_df('c_y_delta_2.csv', dfs, from_csv)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(d2['Mach'], d2['c_y_delta_2'], color='blue', lw=3, ls='-', marker='s', ms=4, label=r'$c_y^{\delta_2}$')
    ax.set_xlabel('M', fontsize=14); ax.set_ylabel(r'$c_y^{\delta_2}$', fontsize=16)
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=12); ax.axhline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

def plot_cy_f_big(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    df = _get_df('c_y_f.csv', dfs, from_csv)
    # 1) c_y_f vs angle (для выбранных М)
    plt.figure(figsize=(12, 8))
    for mach in SELECTED_MACH:
        mach_data = df[np.abs(df['Mach'] - mach) < 1e-6].sort_values('angle')
        if not mach_data.empty:
            plt.plot(mach_data['angle'], mach_data['c_y_f'], marker='o', lw=2, ms=4, label=f'M = {mach}')
    plt.xlabel('Угол атаки, градусы'); plt.ylabel(r'$c_{y.\mathrm{ф}}$'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

    # 2) c_y_f vs M для фиксированных углов
    plt.figure(figsize=(12, 8))
    for angle in sorted(df['angle'].unique()):
        angle_data = df[np.abs(df['angle'] - angle) < 1e-6].sort_values('Mach')
        if not angle_data.empty:
            plt.plot(angle_data['Mach'], angle_data['c_y_f'], marker='s', lw=2, ms=4, label=f'α = {angle}°')
    plt.xlabel('Число Маха'); plt.ylabel(r'$c_{y.\mathrm{ф}}$'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

def plot_cy_sum_big_delta(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df('c_y_sum_big_delta.csv', dfs, from_csv)
    print(f"Загружено {len(data)} записей для c_y_sum_big_delta")

    angles = sorted(data['alpha_deg'].unique())
    deltas_II = sorted(data['delta_II_deg'].unique())
    pos_angles = [a for a in angles if a > 0]
    neg_angles = [a for a in angles if a < 0]
    data_delta0 = data[data['delta_II_deg'] == 0]

    # Положительные углы: (C_y vs M при δ=0) и (C_y vs M при α=10° для разных δ_II)
    if pos_angles:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        for alpha in pos_angles:
            alpha_data = data_delta0[data_delta0['alpha_deg'] == alpha].sort_values('Mach')
            ax1.plot(alpha_data['Mach'], alpha_data['c_y_sum_big_delta'], lw=2, marker='o', ms=4, label=fr'$\alpha = {alpha}^\circ$')
        ax1.set_xlabel('M'); ax1.set_ylabel(r'$C_y$'); ax1.grid(True, alpha=0.3); ax1.legend(); ax1.axhline(0, color='gray', ls='--', alpha=0.5)

        alpha_fixed = 10
        data_alpha10 = data[data['alpha_deg'] == alpha_fixed]
        selected_deltas = [-25, -15, -5, 0, 5, 15, 25]
        for delta in selected_deltas:
            if delta in deltas_II:
                delta_data = data_alpha10[data_alpha10['delta_II_deg'] == delta].sort_values('Mach')
                if not delta_data.empty:
                    ax2.plot(delta_data['Mach'], delta_data['c_y_sum_big_delta'], lw=2, marker='s', ms=4,
                             label=fr'$\delta_{{II}} = {delta}^\circ$')
        ax2.set_xlabel('M'); ax2.set_ylabel(r'$C_y$'); ax2.grid(True, alpha=0.3); ax2.legend(); ax2.axhline(0, color='gray', ls='--', alpha=0.5)
        plt.tight_layout(); plt.show()

    # Отрицательные углы: (C_y vs M при δ=0) и (C_y vs δ_II при α=-10° для разных М)
    if neg_angles:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
        for alpha in neg_angles:
            alpha_data = data_delta0[data_delta0['alpha_deg'] == alpha].sort_values('Mach')
            ax3.plot(alpha_data['Mach'], alpha_data['c_y_sum_big_delta'], lw=2, marker='o', ms=4, label=fr'$\alpha = {alpha}^\circ$')
        ax3.set_xlabel('M'); ax3.set_ylabel(r'$C_y$'); ax3.grid(True, alpha=0.3); ax3.legend(); ax3.axhline(0, color='gray', ls='--', alpha=0.5)

        alpha_fixed_neg = -10
        data_alpha_neg10 = data[data['alpha_deg'] == alpha_fixed_neg]
        for mach in [0.5, 1.0, 2.0, 3.0, 4.0]:
            mach_data = data_alpha_neg10[np.abs(data_alpha_neg10['Mach'] - mach) < 0.05].sort_values('delta_II_deg')
            if not mach_data.empty:
                ax4.plot(mach_data['delta_II_deg'], mach_data['c_y_sum_big_delta'], lw=2, marker='^', ms=4, label=fr'$M = {mach}$')
        ax4.set_xlabel(r'$\delta_{II}$, град'); ax4.set_ylabel(r'$C_y$'); ax4.grid(True, alpha=0.3); ax4.legend()
        ax4.axhline(0, color='gray', ls='--', alpha=0.5); ax4.axvline(0, color='gray', ls='--', alpha=0.5)
        plt.tight_layout(); plt.show()

def plot_cy_small_angles(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    data = _get_df('c_y_small_angles.csv', dfs, from_csv)
    print(f"Загружено {len(data)} записей для c_y_small_angles")

    # 1) c_y vs M при δ_II=0 для всех α
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    data_delta0 = data[data['delta_II_deg'] == 0]
    angles_deg = sorted(data['alpha_deg'].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(angles_deg)))
    for i, angle in enumerate(angles_deg):
        ad = data_delta0[data_delta0['alpha_deg'] == angle].sort_values('Mach')
        ax1.plot(ad['Mach'], ad['c_y'], color=colors[i], lw=2, marker='o', ms=4, label=fr'$\alpha = {angle}^\circ$')
    ax1.set_xlabel('M'); ax1.set_ylabel(r'$c_y$'); ax1.grid(True, alpha=0.3); ax1.legend(); ax1.axhline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    # 2) c_y vs α при δ_II=0 для разных M
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors_mach = plt.cm.plasma(np.linspace(0, 1, len(SELECTED_MACH)))
    for i, mach in enumerate(SELECTED_MACH):
        md = data_delta0[np.abs(data_delta0['Mach'] - mach) < 0.05].sort_values('alpha_deg')
        if not md.empty:
            ax2.plot(md['alpha_deg'], md['c_y'], color=colors_mach[i], lw=2, marker='s', ms=4, label=fr'$M = {mach}$')
    ax2.set_xlabel(r'$\alpha$, град'); ax2.set_ylabel(r'$c_y$'); ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.axhline(0, color='gray', ls='--', alpha=0.5); ax2.axvline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    # 3) c_y vs M при α=0 для разных δ_II
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    data_alpha0 = data[data['alpha_deg'] == 0]
    selected_deltas = [-25, -15, -5, 0, 5, 15, 25]
    colors_delta = plt.cm.viridis(np.linspace(0, 1, len(selected_deltas)))
    for i, delta in enumerate(selected_deltas):
        if delta in data_alpha0['delta_II_deg'].unique():
            dd = data_alpha0[data_alpha0['delta_II_deg'] == delta].sort_values('Mach')
            if not dd.empty:
                ax3.plot(dd['Mach'], dd['c_y'], color=colors_delta[i], lw=2, marker='^', ms=4, label=fr'$\delta_{{II}} = {delta}^\circ$')
    ax3.set_xlabel('M'); ax3.set_ylabel(r'$c_y$'); ax3.grid(True, alpha=0.3); ax3.legend(); ax3.axhline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    # 4) c_y vs δ_II при α=0 для разных M
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    for i, mach in enumerate(SELECTED_MACH):
        md = data_alpha0[np.abs(data_alpha0['Mach'] - mach) < 0.05].sort_values('delta_II_deg')
        if not md.empty:
            ax4.plot(md['delta_II_deg'], md['c_y'], lw=2, marker='d', ms=4, label=fr'$M = {mach}$')
    ax4.set_xlabel(r'$\delta_{II}$, град'); ax4.set_ylabel(r'$c_y$'); ax4.grid(True, alpha=0.3); ax4.legend()
    ax4.axhline(0, color='gray', ls='--', alpha=0.5); ax4.axvline(0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()

# ---------- Главный агрегатор ----------

def draw_all(dfs: Optional[Dict[str, pd.DataFrame]] = None, from_csv: bool = True):
    """
    Построить все графики. Если dfs=None или from_csv=True — читаем CSV. Иначе — используем dfs из модели.
    """
    plot_all_angles_data(dfs, from_csv)
    plot_isolated_wing(dfs, from_csv, with_intrf=False)
    plot_isolated_wing(dfs, from_csv, with_intrf=True)
    plot_Kappa_aa('Kappa_aa_kr.csv', ylabel_tex=r'$K_{\alpha\alpha,\ \mathrm{кр}}$', dfs=dfs, from_csv=from_csv)
    plot_Kappa_aa('Kappa_aa_rl.csv', ylabel_tex=r'$K_{\alpha\alpha,\ \mathrm{рл}}$', dfs=dfs, from_csv=from_csv)
    plot_scalar_vs_M('z_b_v.csv', 'z_b', r'$\bar{z}_{\mathrm{В}}$', dfs, from_csv)
    plot_scalar_vs_M('i_v.csv', 'i_v', r'$i_{\mathrm{В}}$', dfs, from_csv)
    plot_psi_eps(dfs, from_csv)
    plot_eps_alpha_sr(dfs, from_csv)
    plot_scalar_vs_M('kappa_q_nos.csv', 'kappa_q', r'$\kappa_{q,\ \mathrm{нос}}$', dfs, from_csv)
    plot_scalar_vs_M('kappa_q.csv', 'kappa_q', r'$\kappa_q$', dfs, from_csv)
    plot_cy_alpha_sum(dfs, from_csv)
    plot_cy_delta_12(dfs, from_csv)
    plot_cy_f_big(dfs, from_csv)
    plot_cy_sum_big_delta(dfs, from_csv)
    plot_cy_small_angles(dfs, from_csv)

# ---------- CLI ----------

if __name__ == "__main__":
    # По умолчанию работаем из CSV (как ваш старый Graphics.py)
    draw_all(dfs=None, from_csv=True)
