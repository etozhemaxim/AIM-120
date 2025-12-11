from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from atmosphere import atmo

from numba import njit

import numpy as np

# Внешняя библиотека проекта
import AeroBDSM

# Опционально для сборки таблиц и (локального) сохранения CSV
try:
    import pandas as pd
except ImportError:
    pd = None


# ========= 1) Параметры и сетки =========

@dataclass
class Geometry:
    """Геометрические параметры ЛА"""
    l_f: float = 3.906
    D: float = 0.178
    l_nos: float = 0.47
    l_korm: float = 0.375
    b_b_kr: float = 0.2
    b_b_rl: float = 0.316
    l_raszmah_kr: float = 0.498
    l_raszmah_rl: float = 0.651
    S_kr: float = 0.0996
    S_rl: float = 0.2125
    S_f: float = 0.0248
    L_xv_kr: float = 2  # Расстояние от середины бортовой хорды крыла до кормового среза НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    L_xv_rl: float = 0.300    # Расстояние от середины бортовой хорды руля до кормового среза НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    # Исходя из геометрии моего крыла и рис. 5.1 стр. 253
    x_Ak: float  = 0    #НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    b_Ak_r: float = 0.26 #НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    b_Ak_kr : float= 0.8 #НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!

    x_b_kr: float = 1.632 # координата начала бортовой хорды крыла (от носа) НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    x_b_rl: float = 3.632 # координата начала бортовой хорды руля (от носа) НАДО ПОМЕНЯТЬ ЭТО НЕ МОЕ ЗНАЧЕНИЕ !!!!!!!!!
    x_t : float = 0

    chi_0 : float = 0.896
    chi_0_rl: float = 0.729
    psi : float = 0.0


    
    @property
    def S_m(self) -> float:
        return math.pi * (self.D ** 2) / 4.0

    @property
    def S_total(self) -> float:
        return self.S_f + self.S_kr + self.S_rl

    @property
    def S_f_bar(self) -> float:
        return self.S_f / self.S_total

    @property
    def S_1_bar(self) -> float:
        return self.S_kr / self.S_total

    @property
    def S_2_bar(self) -> float:
        return self.S_rl / self.S_total

    @property
    def D_bar(self) -> float:
        return self.D / self.l_f

    @property
    def S_b_Nos(self) -> float:
        return math.pi * self.D**2 / 8.0

    @property
    def S_bok(self) -> float:
        # носовая боковая + цилиндрическая боковая (как в main.py)
        return self.S_b_Nos + math.pi * self.D * (self.l_f - self.D / 2.0)
    @property
    def S_cx(self) -> float:
        return self.S_kr + self.S_rl + self.S_f
    
    # def S_b_F_unshaded(self, M: float) -> float:
    #     S_b_Nos = self.g.S_b_Nos
    #     ds = [self.g.D, self.g.D]  # одна цилиндрическая часть
    #     Ls = [self.g.l_f - self.g.l_nos]  # длина цилиндрической части
    #     Ms = [M, M]       # M для крыла и для рулей (простая заглушка)
    #     b_bs = [self.g.b_b_kr, self.g.b_b_rl]
    #     L_hvs = [1.701, 1.701]  # ваши хвостовые расстояния
    #     # В Python-версии get_S_b_F принимает списки или числа
    #     return float(AeroBDSM.get_S_b_F(S_b_Nos, ds, Ls, Ms, b_bs, L_hvs))

@dataclass
class AerodynamicParams:
    """Аэродинамические параметры и постоянные"""
    lambda_kr: float = 5.0
    lambda_rl: float = 2.805
    chi_05_kr: float = 0.510
    chi_05_rl: float = 0.420
    bar_c_kr: float = 0.224
    zeta_kr: float = 0.2
    zeta_rl: float = 0.0348
    eta_k: float = 1.0
    nu: float = 15.1e-6
    c_profile: float = 0.004
    x_b: float = 1.0
    x_M: float = 0.85
    chi_p: float = 0.0
    L_A: float = 1.0
    b_A: float = 1.5

    # psi_eps-пакет (имена оставляю как у вас, чтобы не ломать совместимость)
    x_zI_II: float = 1.0  # это L_vI_bII по Manual (расстояние вдоль Ох от вихря до начала б/х II)
    d_II: Optional[float] = None  # если None — подставим D из Geometry
    l_1c_II: float = 0.231
    zeta_II: float = 0.0348
    b_b_II: float = 0.316
    chi_0_II: float = 0.420


@dataclass
class FlowSetup:
    """Сетки расчета"""
    M: np.ndarray
    small_angles_deg: np.ndarray
    big_angles_deg: np.ndarray
    deltas_I_deg: np.ndarray
    deltas_II_deg: np.ndarray
    phi_alpha_values: np.ndarray
    psi_I_values: np.ndarray
    psi_II_values: np.ndarray

    @staticmethod
    def default() -> FlowSetup:
        M = np.arange(0.5, 4.1, 0.1)
        small_angles_deg = np.array([-10, -5, 0, 5, 10], dtype=float)
        big_angles_deg = np.array([-25, -20, -15, -5,-10,0, 5,10, 15, 20, 25], dtype=float)
        deltas_I_deg = np.array([0.0], dtype=float)  # крыло (пояс I)
        deltas_II_deg = np.arange(-25, 26, 5, dtype=float)  # рули (пояс II)
        phi_alpha = np.linspace(0, 0.436332, 10)
        psi_I = np.linspace(0, 0.436332, 10)
        psi_II = np.linspace(0, 0.436332, 10)
        return FlowSetup(M, small_angles_deg, big_angles_deg, deltas_I_deg, deltas_II_deg,
                         phi_alpha, psi_I, psi_II)


# ========= 2) Обертки AeroBDSM =========

class AeroBDSMWrapper:
    @staticmethod
    def _val(res) -> float:
        try:
            if hasattr(res, 'ErrorCode'):
                return float(res.Value) if res.ErrorCode == 0 else np.nan
            return float(res)
        except Exception:
            return np.nan

    @staticmethod
    def c_y_alpha_NosCil_Par(M: float, lambda_Nos: float, lambda_Cil: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_c_y_alpha_NosCil_Par(M, lambda_Nos, lambda_Cil))

    @staticmethod
    def c_y_alpha_IsP(M: float, lam: float, bar_c: float, chi_05: float, zeta: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_c_y_alpha_IsP(M, lam, bar_c, chi_05, zeta))

    @staticmethod
    def bar_z_v(M: float, lam: float, chi_05: float, zeta: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_bar_z_v(M, lam, chi_05, zeta))

    @staticmethod
    def i_v(zeta: float, D: float, l_r: float, y_v: float, z_v: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_i_v(zeta, D, l_r, y_v, z_v))

    @staticmethod
    def psi_eps(M: float, alpha_p: float, phi_alpha: float,
                psi_I: float, psi_II: float, z_v: float, y_v: float,
                x_zI_II: float, d_II: float, l_1c_II: float, zeta_II: float,
                b_b_II: float, chi_0_II: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_psi_eps(
            M, alpha_p, phi_alpha, psi_I, psi_II, z_v, y_v, x_zI_II,
            d_II, l_1c_II, zeta_II, b_b_II, chi_0_II
        ))

    @staticmethod
    def kappa_q_Nos_Con(M: float, lambda_Nos: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_kappa_q_Nos_Con(M, lambda_Nos))

    @staticmethod
    def kappa_q_IsP(M: float, L_A: float, b_A: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_kappa_q_IsP(M, L_A, b_A))

    @staticmethod
    def c_yPerp_Cil(My: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_c_yPerp_Cil(My))

    @staticmethod
    def c_x0_p_Nos_Par(M: float, lambda_Nos: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_c_x0_p_Nos_Par(M, lambda_Nos))

    @staticmethod
    def A_IsP(M: float, zeta: float, c_y_alpha_is: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_A_IsP(M, zeta, c_y_alpha_is))
    
    @staticmethod
    def c_x0_w_IsP_Rmb(M: float, c: float, zeta: float, chi_05: float, lam: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_c_x0_w_IsP_Rmb(M, c, zeta, chi_05, lam))

    @staticmethod
    def sigma_cp_Nos_Par(M: float, lambda_Nos: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_sigma_cp_Nos_Par(M, lambda_Nos))
        
    @staticmethod
    def Delta_bar_x_Falpha_NosCil(M: float, lambda_Nos: float, lambda_Cil: float ) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_Delta_bar_x_Falpha_NosCil(M, lambda_Nos , lambda_Cil ))
        
    @staticmethod
    def bar_x_Falpha_IsP(M: float, lambda_: float, chi_05: float, zeta: float) -> float:
        # скалярная обёртка поверх библиотеки
        return AeroBDSMWrapper._val(AeroBDSM.get_bar_x_Falpha_IsP(M, lambda_, chi_05, zeta))

    @staticmethod
    def bar_x_Falpha_IsP_vec(M_arr: np.ndarray, lambda_: float, chi_05: float, zeta: float) -> np.ndarray:
        # удобный векторизатор
        return AeroBDSMWrapper.vec(lambda Mi: AeroBDSMWrapper.bar_x_Falpha_IsP(Mi, lambda_, chi_05, zeta), M_arr)
    
    @staticmethod
    def  Delta_bar_z_Falpha_iC(d_bar: float) -> float:
        return AeroBDSMWrapper._val(AeroBDSM.get_Delta_bar_z_Falpha_iC(d_bar))

    # Векторизатор
    @staticmethod
    def vec(fn, *args):
        vfn = np.vectorize(fn, otypes=[float])
        return vfn(*args)



# ========= 3) Интерференция =========

class InterferenceCalculator:
    def __init__(self, geom: Geometry, aero: AerodynamicParams):
        self.g = geom
        self.a = aero
        self.D_bar = self.g.D_bar

    def _sqrt_term(self, M: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(M**2 - 1.0, 0.0))

    def _F_L(self, b_b: float, L_xv: float, M: np.ndarray) -> np.ndarray:
        st = self._sqrt_term(M)
        denom = (math.pi / 2.0) * self.g.D * np.where(st > 0, st, np.nan)
        b_bar = np.where(st > 0, b_b / denom, 0.0)
        L_bar = np.where(st > 0, L_xv / denom, 0.0)

        c = (4 + (1 / self.a.eta_k) ) * (1 + 8 * self.D_bar**2) #тут не толщина профиля а 3.29!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        out = np.ones_like(M, dtype=float)
        mask = (b_bar > 0) & (c > 0)
        if np.any(mask):
            from math import erf
            Phi = lambda x: 0.5 * (1.0 + np.vectorize(erf)(x / math.sqrt(2.0)))
            t1 = (b_bar + L_bar) * math.sqrt(2*c)
            t2 = (L_bar) * math.sqrt(2*c)
            out[mask] = 1.0 - (math.sqrt(math.pi) / (2.0 * b_bar[mask] * math.sqrt(2 * c))) * (Phi(t1[mask]) - Phi(t2[mask]))
        return out

    def _delta_zvz_bar(self, M: np.ndarray, L_1: float) -> np.ndarray:
        V = 342.0 * M
        ReL = (V * L_1) / self.a.nu
        return (0.093 / (ReL ** (1/5))) * (L_1 / self.g.D) * (1 + 0.4 * M + 0.147 * M**2 - 0.006 * M**3)

    def _x_ps(self, delta_zvz_bar: np.ndarray, eta_k: float) -> np.ndarray:
        term1 = (1 - (2 * self.D_bar)/(1 - self.D_bar**2) * delta_zvz_bar)
        term2 = (1 - (self.D_bar * (eta_k - 1)) / ((1 - self.D_bar) * (eta_k + 1)) * delta_zvz_bar)
        return term1 * term2

    def _x_nos(self, L1_bar: float) -> float:
        return 0.6 + 0.4 * (1 - math.exp(-0.5 * L1_bar))

    def _k_aa_zvz(self, eta_k: float) -> float:
        D_bar = self.g.D_bar
        K_aa_t = (1 + D_bar) ** 2
        return K_aa_t * ((1 + 3 * D_bar - (1 / eta_k) * D_bar * (1 - D_bar)) / (1 + D_bar) ** 2)

    def _K_aa_zvz(self, eta_k: float) -> float:
        D_bar = self.g.D_bar
        return 1 + 3 * D_bar - ((D_bar * (1 - D_bar)) / eta_k)

    def kK_aa_for_surface(self,
                          M: np.ndarray,
                          b_b: float,
                          L_xv: float,
                          L_1: float,
                          eta_k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        F_L = self._F_L(b_b=b_b, L_xv=L_xv, M=M)
        delta_bar = self._delta_zvz_bar(M, L_1=L_1)
        x_ps = self._x_ps(delta_bar, eta_k=eta_k)
        L1_bar = L_1 / self.g.D
        x_nos = self._x_nos(L1_bar=L1_bar)

        k_aa_zvz = self._k_aa_zvz(eta_k=eta_k)
        K_aa_zvz = self._K_aa_zvz(eta_k=eta_k)

        k_aa = K_aa_zvz * x_ps * self.a.x_M * x_nos

        st = self._sqrt_term(M)
        sup_mask = (M >= 1.0) & (st > 0)
        K_aa = np.empty_like(M, dtype=float)
        K_aa[sup_mask] = (k_aa_zvz + (K_aa_zvz - k_aa_zvz) * F_L[sup_mask]) * x_ps[sup_mask] * self.a.x_M * x_nos
        sub_mask = ~sup_mask
        K_aa[sub_mask] = K_aa_zvz * F_L[sub_mask] * x_ps[sub_mask] * self.a.x_M * x_nos

        return k_aa, K_aa, F_L, delta_bar, x_ps


# ========= 4) Основная модель =========

class AerodynamicsModel:
    def __init__(self, geom: Geometry, aero: AerodynamicParams, grid: FlowSetup, resistance: Optional['Resistance']=None):
        self.g = geom
        self.a = aero
        self.grid = grid
        self.inf = InterferenceCalculator(geom, aero)

        self.lambda_Nos = self.g.l_nos / self.g.D
        self.lambda_Cil = self.g.l_f / self.g.D
        self.L_1_kr = self.a.x_b + (self.g.b_b_kr / 2.0)
        self.L_1_rl = self.a.x_b + (self.g.b_b_rl / 2.0)
        self.d_II = self.a.d_II if self.a.d_II is not None else self.g.D

        # если не передали — создаём сами, передавая self
        self.resistance = resistance or Resistance(self.g, self.a, self.grid, self)

    # ---- AeroBDSM векторно ----

    def c_ya_f(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.c_y_alpha_NosCil_Par(M, self.lambda_Nos, self.lambda_Cil)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def c_ya_is_kr(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.c_y_alpha_IsP(M, self.a.lambda_kr, self.a.bar_c_kr, self.a.chi_05_kr, self.a.zeta_kr)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def c_ya_is_rl(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.c_y_alpha_IsP(M, self.a.lambda_rl, self.a.bar_c_kr, self.a.chi_05_rl, self.a.zeta_rl)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def z_bar(self) -> np.ndarray:
        """Безразмерная bar_z_v (B.1.4)."""
        fn = lambda M: AeroBDSMWrapper.bar_z_v(M, self.a.lambda_kr, self.a.chi_05_kr, self.a.zeta_kr)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def z_v_meters(self) -> np.ndarray:
        """Перевод bar_z_v в метры: z_v = (D + z_bar*(l_span - D))/2."""
        zbar = self.z_bar()
        D = self.g.D
        l_span = self.g.l_raszmah_kr  # передняя поверхность = крыло
        return (D + zbar * (l_span - D)) / 2.0

    def i_v(self, z_v: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Универсальный вызов get_i_v: если z_v не задан, берём метрический z_v для передней поверхности.
        """
        if z_v is None:
            z_v = self.z_v_meters()
        fn = lambda zv_: AeroBDSMWrapper.i_v(self.a.zeta_rl, self.g.D, self.g.l_raszmah_rl, 1.0, zv_)
        return AeroBDSMWrapper.vec(fn, z_v)

    def kappa_q_nos(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.kappa_q_Nos_Con(M, self.lambda_Nos)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def kappa_q_isp(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.kappa_q_IsP(M, self.a.L_A, self.a.b_A)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    # ---- Интерференция ----

    def compute_interference(self) -> Dict[str, np.ndarray]:
        k_aa_kr, K_aa_kr, F_L_kr, delta_zvz_bar_kr, x_ps_kr = self.inf.kK_aa_for_surface(
            M=self.grid.M, b_b=self.g.b_b_kr, L_xv=self.g.L_xv_kr, L_1=self.L_1_kr, eta_k=self.a.eta_k
        )
        k_aa_rl, K_aa_rl, F_L_rl, delta_zvz_bar_rl, x_ps_rl = self.inf.kK_aa_for_surface(
            M=self.grid.M, b_b=self.g.b_b_rl, L_xv=self.g.L_xv_rl, L_1=self.L_1_rl, eta_k=self.a.eta_k
        )
        return dict(
            k_aa_kr=k_aa_kr, K_aa_kr=K_aa_kr, F_L_kr=F_L_kr,
            k_aa_rl=k_aa_rl, K_aa_rl=K_aa_rl, F_L_rl=F_L_rl,
            delta_zvz_bar_kr=delta_zvz_bar_kr, delta_zvz_bar_rl=delta_zvz_bar_rl,
            x_ps_kr=x_ps_kr, x_ps_rl=x_ps_rl
        )

    # ---- eps_alpha_sr и сетка psi_eps ----

    def compute_psi_eps_grid(self) -> np.ndarray:
        """
        Полная сетка psi_eps(M, alpha_p, phi_alpha, psi_I, psi_II).
        Возвращает массив shape (N, 7).
        """
        rows = []
        for psi_II in self.grid.psi_II_values:
            for psi_I in self.grid.psi_I_values:
                for phi_alpha in self.grid.phi_alpha_values:
                    for alpha_p in self.grid.phi_alpha_values:
                        for M in self.grid.M:
                            # bar_z_v -> z_v (м)
                            zbar = AeroBDSMWrapper.bar_z_v(M, self.a.lambda_kr, self.a.chi_05_kr, self.a.zeta_kr)
                            z_v = (self.g.D + zbar * (self.g.l_raszmah_kr - self.g.D)) / 2.0
                            psi = AeroBDSMWrapper.psi_eps(
                                M, alpha_p, phi_alpha, psi_I, psi_II, z_v, 1.0,
                                self.a.x_zI_II, self.d_II, self.a.l_1c_II, self.a.zeta_II, self.a.b_b_II, self.a.chi_0_II
                            )
                            rows.append((M, alpha_p, phi_alpha, psi_I, psi_II, z_v, psi))
        return np.array(rows, dtype=float)

    def compute_eps_alpha_sr(self, K_aa_rl: np.ndarray, k_aa_kr: np.ndarray) -> np.ndarray:
        z_v = self.z_v_meters()          # м
        i_v = self.i_v(z_v)              # согласован с тем же z_v
        c_ya_kr = self.c_ya_is_kr()
        # psi_eps при нулях
        psi_eps = AeroBDSMWrapper.vec(
            lambda M, zv: AeroBDSMWrapper.psi_eps(
                M, 0.0, 0.0, 0.0, 0.0, zv, 1.0,
                self.a.x_zI_II, self.d_II, self.a.l_1c_II, self.a.zeta_II, self.a.b_b_II, self.a.chi_0_II
            ),
            self.grid.M, z_v
        )
        eps = 1 * (i_v / np.maximum(z_v, 1e-12)) \
              * (self.g.l_raszmah_kr / self.g.l_raszmah_rl) \
              * (c_ya_kr / self.a.lambda_kr) \
              * (np.maximum(k_aa_kr, 1e-12) / np.maximum(K_aa_rl, 1e-12)) \
              * psi_eps
        return eps

    def compute_eps_alpha_sr_grid(self) -> np.ndarray:
        """
        Сетка eps_alpha_sr по M, alpha_p, phi_alpha, psi_I, psi_II.
        Возвращает shape (N, 11).
        """
        intr = self.compute_interference()

        rows = []
        for psi_II in self.grid.psi_II_values:
            for psi_I in self.grid.psi_I_values:
                for phi_alpha in self.grid.phi_alpha_values:
                    for alpha_p in self.grid.phi_alpha_values:
                        for idx, M in enumerate(self.grid.M):
                            zbar = AeroBDSMWrapper.bar_z_v(M, self.a.lambda_kr, self.a.chi_05_kr, self.a.zeta_kr)
                            z_v = (self.g.D + zbar * (self.g.l_raszmah_kr - self.g.D)) / 2.0  # м
                            i_v_val = AeroBDSMWrapper.i_v(self.a.zeta_rl, self.g.D, self.g.l_raszmah_rl, 1.0, z_v)
                            psi_eps = AeroBDSMWrapper.psi_eps(
                                M, alpha_p, phi_alpha, psi_I, psi_II, z_v, 1.0,
                                self.a.x_zI_II, self.d_II, self.a.l_1c_II, self.a.zeta_II, self.a.b_b_II, self.a.chi_0_II
                            )
                            c_ya_kr = AeroBDSMWrapper.c_y_alpha_IsP(M, self.a.lambda_kr, self.a.bar_c_kr, self.a.chi_05_kr, self.a.zeta_kr)
                            eps = (1/(2*math.pi)) * (i_v_val/max(z_v,1e-12)) \
                                  * (self.g.l_raszmah_kr/self.g.l_raszmah_rl) \
                                  * (c_ya_kr/self.a.lambda_kr) \
                                  * (max(intr['k_aa_kr'][idx],1e-12)/max(intr['K_aa_rl'][idx],1e-12)) \
                                  * psi_eps
                            rows.append((eps, M, alpha_p, phi_alpha, psi_I, psi_II, z_v, psi_eps, i_v_val,
                                         intr['k_aa_kr'][idx], intr['K_aa_rl'][idx]))
        return np.array(rows, dtype=float)

    # ---- Малые углы ----

    def compute_small_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        intr = self.compute_interference()
        k_aa_kr, K_aa_kr = intr['k_aa_kr'], intr['K_aa_kr']
        K_aa_rl = intr['K_aa_rl']

        eps_alpha_sr = self.compute_eps_alpha_sr(K_aa_rl=K_aa_rl, k_aa_kr=k_aa_kr)

        cya_f = self.c_ya_f()
        cya_kr = self.c_ya_is_kr()
        cya_rl = self.c_ya_is_rl()
        kq_nos = self.kappa_q_nos()
        kq_isp = self.kappa_q_isp()

        cya_sum = (cya_f * self.g.S_f_bar
                   + cya_kr * K_aa_kr * (1.0 - eps_alpha_sr /57.3) * self.g.S_1_bar * kq_nos
                   + cya_rl * K_aa_rl * self.g.S_2_bar * kq_isp)

        alphas_rad = np.radians(self.grid.small_angles_deg)
        c_y_small = cya_sum[:, None] * alphas_rad[None, :]
        return cya_sum, c_y_small

    # ---- Производные по дельтам ----

    def _k_delta_0_zvz(self, k_aa_zvz: float, K_aa_zvz: float) -> Tuple[float, float]:
        k_delta_0 = (k_aa_zvz ** 2) / K_aa_zvz
        K_delta_0 = k_aa_zvz
        return k_delta_0, K_delta_0

    def _x_ps_bar(self, delta_zvz_bar: np.ndarray) -> np.ndarray:
        D_bar, eta_k = self.g.D_bar, self.a.eta_k
        num = (1 - D_bar * (1 + delta_zvz_bar)) * (1 - (eta_k - 1) * D_bar * (1 + delta_zvz_bar) / (eta_k + 1 - 2 * D_bar))
        den = (1 - D_bar) * (1 - (eta_k - 1) * D_bar / (eta_k + 1 - 2 * D_bar))
        return num / den

    def _kappa_M(self, M: np.ndarray) -> np.ndarray:
        return np.ones_like(M)

    def _k_sch(self, M: np.ndarray) -> np.ndarray:
        out = np.where(M > 1.4, 0.85 + (M - 1.4) * (1.0 - 0.85) / (self.grid.M[-1] - 1.4), 0.85)
        return out

    def _n_eff(self, M: np.ndarray) -> np.ndarray:
        return self._k_sch(M) * math.cos(self.a.chi_p)

    def compute_delta_derivatives(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        intr = self.compute_interference()
        k_aa_kr, K_aa_kr, F_L_kr = intr['k_aa_kr'], intr['K_aa_kr'], intr['F_L_kr']
        k_aa_rl, K_aa_rl, F_L_rl = intr['k_aa_rl'], intr['K_aa_rl'], intr['F_L_rl']
        delta_kr, delta_rl = intr['delta_zvz_bar_kr'], intr['delta_zvz_bar_rl']
        x_ps_bar_kr = self._x_ps_bar(delta_kr)
        x_ps_bar_rl = self._x_ps_bar(delta_rl)
        kappa_M = self._kappa_M(self.grid.M)

        k_aa_zvz_kr = self.inf._k_aa_zvz(self.a.eta_k)
        K_aa_zvz_kr = self.inf._K_aa_zvz(self.a.eta_k)
        k_aa_zvz_rl = k_aa_zvz_kr
        K_aa_zvz_rl = K_aa_zvz_kr

        k_delta0_zvz_kr, K_delta0_zvz_kr = self._k_delta_0_zvz(k_aa_zvz_kr, K_aa_zvz_kr)
        k_delta0_zvz_rl, K_delta0_zvz_rl = self._k_delta_0_zvz(k_aa_zvz_rl, K_aa_zvz_rl)

        k_delta_0_kr = k_delta0_zvz_kr * x_ps_bar_kr * kappa_M
        K_delta_0_kr = (k_delta0_zvz_kr + (K_delta0_zvz_kr - k_delta0_zvz_kr) * F_L_kr) * x_ps_bar_kr * kappa_M

        k_delta_0_rl = k_delta0_zvz_rl * x_ps_bar_rl * kappa_M
        K_delta_0_rl = (k_delta0_zvz_rl + (K_delta0_zvz_rl - k_delta0_zvz_rl) * F_L_rl) * x_ps_bar_rl * kappa_M

        return k_delta_0_kr, K_delta_0_kr, k_delta_0_rl, K_delta_0_rl

    def compute_eps_delta_sr(self,
                            K_aa_rl: np.ndarray,
                            k_delta_0_kr: np.ndarray,
                            alpha_p_deg: float = 0.0,
                            y_v: float = 0.0) -> np.ndarray:
        """
        eps_delta_sr(M) для крыла (аналог eps_alpha_sr, но вместо k_aa — k_delta_0).
        По умолчанию считаем при alpha_p=0, phi_alpha=psi_I=psi_II=0, y_v=0.
        Возвращает массив shape (M,).
        """
        import math
        M = self.grid.M                                   # (M,)
        n_vals = self._n_eff(M)                           # (M,)
        z_v = self.z_v_meters()                           # (M,)
        i_v = self.i_v(z_v)                               # (M,)
        c_ya_kr = self.c_ya_is_kr()                       # (M,)
        alpha_p = math.radians(alpha_p_deg)

        psi_eps = AeroBDSMWrapper.vec(
            lambda Mi, zv: AeroBDSMWrapper.psi_eps(
                Mi, alpha_p, 0.0, 0.0, 0.0, zv, y_v,
                self.a.x_zI_II, self.d_II, self.a.l_1c_II,
                self.a.zeta_II, self.a.b_b_II, self.a.chi_0_II
            ),
            M, z_v
        )  # (M,)

        eps = (1.0/(2*math.pi)) * (i_v/np.maximum(z_v, 1e-12)) \
            * (self.g.l_raszmah_kr/self.g.l_raszmah_rl) \
            * (c_ya_kr/self.a.lambda_kr) \
            * ((k_delta_0_kr*n_vals)/np.maximum(K_aa_rl, 1e-12)) \
            * psi_eps
        return eps  # (M,)


    def compute_cy_delta_1_2(self) -> Tuple[np.ndarray, np.ndarray]:
        intr = self.compute_interference()
        K_aa_rl = intr['K_aa_rl']

        k_delta_0_kr, K_delta_0_kr, k_delta_0_rl, K_delta_0_rl = self.compute_delta_derivatives()

        n_vals = self._n_eff(self.grid.M)
        cya_kr = self.c_ya_is_kr()
        cya_rl = self.c_ya_is_rl()
        kq_nos = self.kappa_q_nos()
        kq_isp = self.kappa_q_isp()

        eps_delta_sr = self.compute_eps_delta_sr(K_aa_rl=K_aa_rl, k_delta_0_kr=k_delta_0_kr)

        # δ1 — крыло (пояс I) стр.176 3.50
        c_y_delta_1 = (cya_kr * K_delta_0_kr * n_vals * self.g.S_1_bar * kq_nos) \
                      - (cya_rl * K_aa_rl * self.g.S_2_bar * kq_isp) * eps_delta_sr

        # δ2 — рули (пояс II) 3.52
        c_y_delta_2 = (cya_rl * K_delta_0_rl * n_vals * self.g.S_2_bar * kq_isp)

        return c_y_delta_1, c_y_delta_2

    # финальная функция c_y для малых ушлов (3.2)
    def compute_c_y_sum(self) -> np.ndarray:
        """
        Малые углы: c_y(M, alpha, delta_I, delta_II) =
            c_y_alpha_sum(M)*alpha + c_y_delta_1(M)*delta_I + c_y_delta_2(M)*delta_II
        Возвращает массив shape (M, A, DI, DII)
        """
        # из модели
        M = self.grid.M
        alphas_rad = np.radians(self.grid.small_angles_deg)        # (A,)
        dI_rad     = np.radians(self.grid.deltas_I_deg)            # (DI,)
        dII_rad    = np.radians(self.grid.deltas_II_deg)           # (DII,)

        # данные по производным
        c_y_delta_1, c_y_delta_2 = self.compute_cy_delta_1_2()     # (M,), (M,)
        c_y_alpha_sum, _ = self.compute_small_angles()              # (M,), (M,A) но берем суммарную производную

        # приводим к размерностям через broadcasting
        # (M,1,1,1)
        cya = c_y_alpha_sum[:, None, None, None]
        d1  = c_y_delta_1[:,   None, None, None]
        d2  = c_y_delta_2[:,   None, None, None]

        # (1,A,1,1), (1,1,DI,1), (1,1,1,DII)
        A   = alphas_rad[None, :, None, None]
        DI  = dI_rad[None, None, :, None]
        DII = dII_rad[None, None, None, :]

        # итог: (M,A,DI,DII)
        c_y_sum = cya*A + d1*DI + d2*DII
        return c_y_sum
        

    # ---- Большие углы ----

    def _kappa_alpha(self, angles_deg: np.ndarray) -> np.ndarray:
        M = self.grid.M[:, None]
        ang = np.abs(angles_deg[None, :])
        return 1.0 - 0.45 * (1 - np.exp(-0.06 * M**2)) * (1 - np.exp(-0.12 * np.radians(ang)))

    def _M_y(self, angles_deg: np.ndarray) -> np.ndarray:
        M = self.grid.M[:, None]
        ang = np.radians(np.abs(angles_deg))[None, :]
        return M * np.sin(ang)

    def c_y_f_big(self) -> np.ndarray:
        angles = self.grid.big_angles_deg
        cya_f = self.c_ya_f()[:, None] / 1.0
        kappa = self._kappa_alpha(angles)
        My = self._M_y(angles)
        c_y_perp = AeroBDSMWrapper.vec(lambda My_: AeroBDSMWrapper.c_yPerp_Cil(My_), My.ravel()).reshape(My.shape)
        ang_rad = np.radians(angles)[None, :]
        sign = np.sign(angles)[None, :]
                # ГЕОМЕТРИЯ ДЛЯ S_b_F
        L_cyl = max(self.g.l_f - self.g.l_nos - self.g.l_korm, 0.0)
        L_tail = self.g.l_korm
        d_tail_end = 0.0

        # L_hv от конца бортовой хорды до донного среза
        x_b = self.a.x_b
        x_TE_kr = x_b + self.g.b_b_kr/2.0
        x_TE_rl = x_b + self.g.b_b_rl/2.0
        L_hv_kr = max(self.g.l_f - x_TE_kr, 0.0)
        L_hv_rl = max(self.g.l_f - x_TE_rl, 0.0)

        def sbf(M):  
            return float(AeroBDSM.get_S_b_F(self.g.S_b_Nos,
                                            [self.g.D, self.g.D],
                                            [L_cyl - L_tail],
                                            [M, M],
                                            [self.g.b_b_kr, self.g.b_b_rl],
                                            [self.g.L_xv_kr, self.g.L_xv_rl]))


        S_bF = np.vectorize(sbf)(self.grid.M)[:, None]  # (M,)
        term1 = 1.0 * cya_f * kappa * np.sin(ang_rad) * np.cos(ang_rad)
        term2 = (4 * S_bF) / (math.pi * self.g.D**2) * c_y_perp * (np.sin(ang_rad)**2) * sign
        return term1 + term2

    def _A_is(self, cya_is: np.ndarray, for_wing: bool) -> np.ndarray:
        zeta = self.a.zeta_kr if for_wing else self.a.zeta_rl
        M = self.grid.M
        return AeroBDSMWrapper.vec(lambda Mi, c_: AeroBDSMWrapper.A_IsP(Mi, zeta, c_), M, cya_is)

    def alpha_eff(self, angles_big_deg: np.ndarray, deltas_deg: np.ndarray,
                  k_aa: np.ndarray, k_delta_0: np.ndarray, n_vals: np.ndarray,
                  eps: Optional[np.ndarray] = None, sqrt2: bool = True) -> np.ndarray:
        alpha = np.radians(angles_big_deg)[None, :, None]
        delta = np.radians(deltas_deg)[None, None, :]
        kaa = k_aa[:, None, None]
        kd = k_delta_0[:, None, None]
        n = n_vals[:, None, None]
        divider = math.sqrt(2) if sqrt2 else 1.0
        eps_term = 0.0 if eps is None else eps[:, None, None]
        return kaa * ((alpha - eps_term) / divider) + kd * n * delta

    def c_n_from_alpha_eff(self, alpha_eff: np.ndarray, cya_is: np.ndarray, A_is: np.ndarray) -> np.ndarray:
        sin_ae = np.sin(alpha_eff)
        return 1.0 * cya_is[:, None, None] * sin_ae * np.cos(alpha_eff) + A_is[:, None, None] * (sin_ae**2) * np.sign(alpha_eff)

    def c_x0_fuselage(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.c_x0_p_Nos_Par(M, self.lambda_Nos)
        return AeroBDSMWrapper.vec(fn, self.grid.M)

    def c_y_I_II_big(self,
                    alpha_eff_I: np.ndarray, alpha_eff_II: np.ndarray,
                    cya_is_kr: np.ndarray, cya_is_rl: np.ndarray,
                    A_kr: np.ndarray, A_rl: np.ndarray,
                    K_aa_kr: np.ndarray, k_aa_kr: np.ndarray,
                    K_aa_rl: np.ndarray, k_aa_rl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        c_y_I — для крыла (пояс I), c_y_II — для рулей (пояс II) на больших углах.
        """
        c_nI = self.c_n_from_alpha_eff(alpha_eff_I,  cya_is_kr, A_kr)  # крыло
        c_nII = self.c_n_from_alpha_eff(alpha_eff_II, cya_is_rl, A_rl) # рули

        c_x_0 = self.c_x0_fuselage()

        alpha = np.radians(self.grid.big_angles_deg)[None, :, None]
        deltas_I  = np.radians(self.grid.deltas_I_deg)[None, None, :]
        deltas_II = np.radians(self.grid.deltas_II_deg)[None, None, :]

        sqrt2 = math.sqrt(2)

        c_y_I = c_nI * ((K_aa_kr[:, None, None] / np.maximum(k_aa_kr[:, None, None], 1e-12))
                        * np.cos(alpha) * np.cos(deltas_I)) * sqrt2 \
                - (2.0 * c_x_0[:, None, None] * np.sin(alpha + deltas_I))

        c_y_II = c_nII * ((K_aa_rl[:, None, None] / np.maximum(k_aa_rl[:, None, None], 1e-12))
                        * np.cos(alpha) * np.cos(deltas_II)) * sqrt2 \
                - (2.0 * c_x_0[:, None, None] * np.sin(alpha + deltas_II))

        return c_y_I, c_y_II

    def compute_big_angles(self) -> Dict[str, np.ndarray]:
        angles = self.grid.big_angles_deg

        intr = self.compute_interference()
        k_aa_kr, K_aa_kr = intr['k_aa_kr'], intr['K_aa_kr']   # крыло
        k_aa_rl, K_aa_rl = intr['k_aa_rl'], intr['K_aa_rl']   # рули

        c_y_f = self.c_y_f_big()

        cya_is_kr = self.c_ya_is_kr()
        cya_is_rl = self.c_ya_is_rl()
        A_kr = self._A_is(cya_is_kr, for_wing=True)
        A_rl = self._A_is(cya_is_rl, for_wing=False)

        n_vals = self._n_eff(self.grid.M)

        k_delta_0_kr, K_delta_0_kr, k_delta_0_rl, K_delta_0_rl = self.compute_delta_derivatives()

        alpha_eff_I0 = self.alpha_eff(angles, self.grid.deltas_I_deg,
                                      k_aa=k_aa_kr, k_delta_0=np.zeros_like(k_aa_kr), n_vals=n_vals,
                                      eps=None, sqrt2=True)
        c_nI0 = self.c_n_from_alpha_eff(alpha_eff_I0, cya_is_kr, A_kr).squeeze(-1)

        z_v = self.z_v_meters()            # м
        i_v = self.i_v(z_v)
        psi_eps_zeros = AeroBDSMWrapper.vec(
            lambda Mi, zv: AeroBDSMWrapper.psi_eps(
                Mi, 0.0, 0.0, 0.0, 0.0, zv, 1.0,
                self.a.x_zI_II, self.d_II, self.a.l_1c_II, self.a.zeta_II, self.a.b_b_II, self.a.chi_0_II
            ),
            self.grid.M, z_v
        )
        c_nI_mean = np.mean(c_nI0, axis=1)

        eps_delta_sr = 1.0 * (i_v / np.maximum(z_v, 1e-12)) \
                       * (self.g.l_raszmah_kr / self.g.l_raszmah_rl) \
                       * (c_nI_mean / self.a.lambda_kr) \
                       * ((k_delta_0_kr * n_vals) / np.maximum(K_aa_rl, 1e-12)) \
                       * psi_eps_zeros

        alpha_eff_I = self.alpha_eff(angles, self.grid.deltas_I_deg,
                                     k_aa=k_aa_kr, k_delta_0=k_delta_0_kr, n_vals=n_vals,
                                     eps=None, sqrt2=True)
        alpha_eff_II = self.alpha_eff(angles, self.grid.deltas_II_deg,
                                      k_aa=k_aa_rl, k_delta_0=k_delta_0_rl, n_vals=n_vals,
                                      eps=eps_delta_sr, sqrt2=True)

        c_y_I, c_y_II = self.c_y_I_II_big(alpha_eff_I, alpha_eff_II,
                                          cya_is_kr, cya_is_rl, A_kr, A_rl,
                                          K_aa_kr, k_aa_kr, K_aa_rl, k_aa_rl)

        kq_nos = self.kappa_q_nos()
        kq_isp = self.kappa_q_isp()
        c_y_sum_big = (c_y_f[:, :, None, None] * self.g.S_f_bar
                       + c_y_I[:, :, :, None]  * self.g.S_1_bar * k_aa_kr[:, None, None, None]
                       + c_y_II[:, :, None, :] * self.g.S_2_bar * k_aa_rl[:, None, None, None])

        return dict(
            c_y_f=c_y_f,
            A_kr=A_kr, A_rl=A_rl,
            alpha_eff_I=alpha_eff_I, alpha_eff_II=alpha_eff_II,
            c_y_I=c_y_I, c_y_II=c_y_II,
            c_y_sum_big=c_y_sum_big
        )

    # поляра Лилиенталя
    def Polar(self) -> Tuple[np.ndarray , np.ndarray]:

        test = self.compute_big_angles()
        c_y = test['c_y_sum_big']

        c_x = self.resistance.c_x_tensor()

        return c_y ,  c_x



    # ---- Сборщики DataFrame ----

    def assemble_all_angles_data_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        angles = self.grid.small_angles_deg
        cya_f = self.c_ya_f()
        data = {"Mach": M}
        for a in angles:
            data[f"alpha_{int(a)}"] = cya_f * math.radians(a)
        return pd.DataFrame(data)

    def assemble_krylo_isP_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        angles = self.grid.small_angles_deg
        cya = self.c_ya_is_kr()
        data = {"Mach": M}
        for a in angles:
            data[f"alpha_{int(a)}"] = cya * math.radians(a)
        return pd.DataFrame(data)

    def assemble_rul_isP_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        angles = self.grid.small_angles_deg
        cya = self.c_ya_is_rl()
        data = {"Mach": M}
        for a in angles:
            data[f"alpha_{int(a)}"] = cya * math.radians(a)
        return pd.DataFrame(data)

    def assemble_Kappa_aa_kr_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        intr = self.compute_interference()
        M = self.grid.M
        K_aa = intr['K_aa_kr']
        angles = self.grid.small_angles_deg
        data = {"Mach": M, "K_aa": K_aa}
        for a in angles:
            data[f"alpha_{int(a)}"] = K_aa
        return pd.DataFrame(data)

    def assemble_Kappa_aa_rl_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        intr = self.compute_interference()
        M = self.grid.M
        K_aa = intr['K_aa_rl']
        angles = self.grid.small_angles_deg
        data = {"Mach": M, "K_aa": K_aa}
        for a in angles:
            data[f"alpha_{int(a)}"] = K_aa
        return pd.DataFrame(data)

    def assemble_z_b_v_df(self) -> 'pd.DataFrame':
        """
        bar_z_v (как ожидает ваш Graphics.py: колонка 'z_b' = безразмерная величина)
        """
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        z_bar = self.z_bar()
        return pd.DataFrame({"Mach": M, "z_b": z_bar})

    def assemble_i_v_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        z_v = self.z_v_meters()
        i_v = self.i_v(z_v)
        return pd.DataFrame({"Mach": M, "i_v": i_v})

    def assemble_psi_eps_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        arr = self.compute_psi_eps_grid()
        return pd.DataFrame(arr, columns=["Mach", "alpha_p", "phi_alpha", "psi_I", "psi_II", "z_v", "psi_eps"])

    def assemble_eps_alpha_sr_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        arr = self.compute_eps_alpha_sr_grid()
        cols = ["eps_alpha_sr", "Mach", "alpha_p", "phi_alpha", "psi_I", "psi_II", "z_v", "psi_eps", "i_v", "k_aa", "K_aa_rl"]
        return pd.DataFrame(arr, columns=cols)

    def assemble_kappa_q_nos_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        kq = self.kappa_q_nos()
        return pd.DataFrame({"Mach": M, "kappa_q": kq})

    def assemble_kappa_q_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        M = self.grid.M
        kq = self.kappa_q_isp()
        return pd.DataFrame({"Mach": M, "kappa_q": kq})

    def assemble_c_y_alpha_sum_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        cya_sum, _ = self.compute_small_angles()
        M = self.grid.M
        angles = self.grid.small_angles_deg
        data = {"Mach": M}
        for a in angles:
            data[f"alpha_{int(a)}"] = cya_sum * math.radians(a)
        return pd.DataFrame(data)

    def assemble_c_y_delta_1_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        c1, _ = self.compute_cy_delta_1_2()
        return pd.DataFrame({"Mach": self.grid.M, "c_y_delta_1": c1})

    def assemble_c_y_delta_2_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        _, c2 = self.compute_cy_delta_1_2()
        return pd.DataFrame({"Mach": self.grid.M, "c_y_delta_2": c2})

    def assemble_c_y_f_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        big = self.compute_big_angles()
        M = np.repeat(self.grid.M, len(self.grid.big_angles_deg))
        angles = np.tile(self.grid.big_angles_deg, len(self.grid.M))
        c_y_f = big['c_y_f'].reshape(-1)
        return pd.DataFrame({"Mach": M, "angle": angles, "c_y_f": c_y_f})

    def assemble_c_y_small_angles_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("пandas не установлен.")
        _, c_y_small = self.compute_small_angles()
        M = np.repeat(self.grid.M, len(self.grid.small_angles_deg)*len(self.grid.deltas_I_deg)*len(self.grid.deltas_II_deg))
        alpha_list, delta_I_list, delta_II_list, cy_list = [], [], [], []
        for i, _Mi in enumerate(self.grid.M):
            for alpha in self.grid.small_angles_deg:
                for dI in self.grid.deltas_I_deg:
                    for dII in self.grid.deltas_II_deg:
                        alpha_list.append(alpha)
                        delta_I_list.append(dI)
                        delta_II_list.append(dII)
                        cy_list.append(c_y_small[i, self.grid.small_angles_deg.tolist().index(alpha)])
        return pd.DataFrame({"Mach": M,
                             "alpha_deg": alpha_list, "delta_I_deg": delta_I_list, "delta_II_deg": delta_II_list,
                             "c_y": cy_list})

    def assemble_c_y_sum_big_delta_df(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        big = self.compute_big_angles()
        c_y_sum = big['c_y_sum_big']  # (M,A,Di,D)
        M_list, alpha_list, delta_I_list, delta_II_list, cy_list = [], [], [], [], []
        for i, M in enumerate(self.grid.M):
            for ai, alpha in enumerate(self.grid.big_angles_deg):
                for di, dI in enumerate(self.grid.deltas_I_deg):
                    for dj, dII in enumerate(self.grid.deltas_II_deg):
                        M_list.append(M)
                        alpha_list.append(alpha)
                        delta_I_list.append(dI)
                        delta_II_list.append(dII)
                        cy_list.append(c_y_sum[i, ai, di, dj])
        return pd.DataFrame({"Mach": M_list,
                             "alpha_deg": alpha_list, "delta_I_deg": delta_I_list, "delta_II_deg": delta_II_list,
                             "c_y_sum_big_delta": cy_list})
    

    def assemble_c_y_sum(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        
        # Получаем массив с коэффициентом подъемной силы для малых углов
        c_y_sum = self.compute_c_y_sum()  # (M, A, DI, DII)
        
        # Создаем списки для DataFrame
        M_list, alpha_list, delta_I_list, delta_II_list, cy_list = [], [], [], [], []
        
        # Заполняем списки значениями из массива
        for i, M in enumerate(self.grid.M):
            for ai, alpha in enumerate(self.grid.small_angles_deg):
                for di, dI in enumerate(self.grid.deltas_I_deg):
                    for dj, dII in enumerate(self.grid.deltas_II_deg):
                        M_list.append(M)
                        alpha_list.append(alpha)
                        delta_I_list.append(dI)
                        delta_II_list.append(dII)
                        cy_list.append(c_y_sum[i, ai, di, dj])
        
        # Создаем DataFrame
        df = pd.DataFrame({
            "Mach": M_list,
            "alpha_deg": alpha_list,
            "delta_I_deg": delta_I_list,
            "delta_II_deg": delta_II_list,
            "c_y_sum": cy_list
        })
        
        return df
    


    def assemble_c_x(self) -> 'pd.DataFrame':
        if pd is None:
            raise RuntimeError("pandas не установлен.")

        cx = self.resistance.c_x_tensor()  # (M,A,DI,DII)

        M_list, alpha_list, dI_list, dII_list, cx_list = [], [], [], [], []
        for i, Mval in enumerate(self.grid.M):
            for ai, alpha in enumerate(self.grid.big_angles_deg):
                for di, dI in enumerate(self.grid.deltas_I_deg):   # DI обычно 1 (0°)
                    for dj, dII in enumerate(self.grid.deltas_II_deg):
                        M_list.append(Mval)
                        alpha_list.append(alpha)
                        dI_list.append(dI)
                        dII_list.append(dII)
                        cx_list.append(cx[i, ai, di, dj])

        return pd.DataFrame({
            "Mach": M_list,
            "alpha_deg": alpha_list,
            "delta_I_deg": dI_list,
            "delta_II_deg": dII_list,
            "c_x": cx_list
        })
    
    def assemble_m_z_df(self) -> 'pd.DataFrame':
            """
            Собирает момент тангажа для малых углов в виде таблицы:
            колонки: Mach, alpha_deg, delta_I_deg, delta_II_deg, m_z
            """
            if pd is None:
                raise RuntimeError("pandas не установлен.")

            # создаём Moments локально (без хранения в self)
            moments = Moments(self.grid, self, self.g, self.a, AeroBDSMWrapper)

            # тензор (M,A,DI,DII)
            mz = moments.m_z()

            M_list, alpha_list, dI_list, dII_list, mz_list = [], [], [], [], []
            for i, Mval in enumerate(self.grid.M):
                for ai, alpha in enumerate(self.grid.small_angles_deg):
                    for di, dI in enumerate(self.grid.deltas_I_deg):
                        for dj, dII in enumerate(self.grid.deltas_II_deg):
                            M_list.append(Mval)
                            alpha_list.append(alpha)
                            dI_list.append(dI)
                            dII_list.append(dII)
                            mz_list.append(mz[i, ai, di, dj])

            return pd.DataFrame({
                "Mach": M_list,
                "alpha_deg": alpha_list,
                "delta_I_deg": dI_list,
                "delta_II_deg": dII_list,
                "m_z": mz_list
            })
    
    def assemble_polar_lilienthal_df(self) -> 'pd.DataFrame':
        """
        Собирает данные для построения поляры Лилиенталя (C_y и C_x отдельно)
        Колонки: Mach, alpha_deg, delta_I_deg, delta_II_deg, c_y, c_x, Polar_ratio
        """
        if pd is None:
            raise RuntimeError("pandas не установлен.")

        # Получаем C_y и C_x отдельно
        c_y_array, c_x_array = self.Polar()  # оба массива формы (M,A,DI,DII)

        # Создаем списки для DataFrame
        M_list, alpha_list, delta_I_list, delta_II_list = [], [], [], []
        cy_list, cx_list, polar_list = [], [], []

        # Заполняем списки значениями из массивов
        for i, M in enumerate(self.grid.M):
            for ai, alpha in enumerate(self.grid.big_angles_deg):
                for di, dI in enumerate(self.grid.deltas_I_deg):
                    for dj, dII in enumerate(self.grid.deltas_II_deg):
                        M_list.append(M)
                        alpha_list.append(alpha)
                        delta_I_list.append(dI)
                        delta_II_list.append(dII)
                        
                        # Получаем значения
                        cy_val = c_y_array[i, ai, di, dj]
                        cx_val = c_x_array[i, ai, di, dj]
                        
                        cy_list.append(cy_val)
                        cx_list.append(cx_val)
                        
                        # Вычисляем поляру (отношение C_y/C_x)
                        # Защита от деления на ноль и очень маленьких значений
                        if abs(cx_val) < 1e-10:
                            polar_val = np.nan
                        else:
                            polar_val = cy_val / cx_val
                        
                        polar_list.append(polar_val)

        # Создаем DataFrame
        df = pd.DataFrame({
            "Mach": M_list,
            "alpha_deg": alpha_list,
            "delta_I_deg": delta_I_list,
            "delta_II_deg": delta_II_list,
            "c_y": cy_list,
            "c_x": cx_list,
            "Polar_ratio": polar_list  # отношение C_y/C_x
        })

        return df


# ========= 5) Глава IV "Лобовое сопротивление " =========

class Resistance :
    def __init__(self,geom: Geometry, aero: AerodynamicParams, grid: FlowSetup, aeroModel :AerodynamicsModel):
        self.geom = geom
        self.aero= aero
        self.grid = grid
        self.aeroModel = aeroModel
        self.a = 340   #скорость звука
        self.eta_c = 1.13  #рисунок 4.28

# ------------ 1. Расчет c_x_0 -------------------------

# расчет числа Рейнольдса Re стр.204
    def Re(self) -> np.ndarray:
        M = self.grid.M
        V = M * self.a  # скорость потока, м/с
        return V * (self.geom.l_f/ self.aero.nu)

# расчет кэфа c_f В.1.5 в документации на AeroBDSM
    def c_f (self) -> np.ndarray:
        Re_values = self.Re()
        fn = lambda Re_values: AeroBDSMWrapper._val(AeroBDSM.get_c_f0(Re_values, 0))
        return AeroBDSMWrapper.vec(fn, Re_values)
# Расчет c_x_tr 4.4
    def c_x_tr(self) -> np.ndarray:
        c_f_values = self.c_f()
        return ((2 *c_f_values) / 2 ) * ((self.geom.S_f - (( np.pi * self.geom.D**2) / 4 )) / self.geom.S_f )

# коэффициент сопротивления давления параболи ческой носовой части при нулевом угле атаки.
    def c_x0_p_Nos_Par(self) -> np.ndarray:
        fn = lambda M: AeroBDSMWrapper.c_x0_p_Nos_Par(M, self.aeroModel.lambda_Nos)
        return AeroBDSMWrapper.vec(fn, self.grid.M)
    '''
    # кэф донного сопрпотивления c_x_dn 4.39 !!!!!!!!!!!!!!! здесь формула справедлива для  M <= 0.8 , надо проправить потом
    '''
    def c_x_dn(self) -> np.ndarray: 
        c_f_values = self.c_f()
        return ( 0.0155 / np.sqrt( self.aeroModel.lambda_Cil * c_f_values) ) * self.aero.eta_k * ( ((self.geom.D**2) / 4 ) / self.geom.S_f )
    
    # c_x_0_f -- кэф лобового сопротивления корпуса (4.3) 
    # по идее в формуле отстустует c_korm , как его считать я не знаю
    def c_x_0_f(self) -> np.ndarray:
        c_x_tr = self.c_x_tr()
        c_x_nos = self.c_x0_p_Nos_Par()
        c_x_dn = self.c_x_dn()
        return c_x_tr + c_x_nos + c_x_dn

# ------------ 2.Коэф. лобового сопротивления несущих поверхностей при alpha = delta = 0 -------------------------

# ------------2.1 Профильное сопротивление--------------------------------------
# стр.231

# расчет кэфа профильного сопротивления c_x_p_I/II
#  4.44 стр. 231
    def c_x_p(self) -> Tuple[np.ndarray, np.ndarray]:
        c_f = self.c_f()
        arr  = 2 * c_f * self.eta_c #оба профиля считаются по одной и той же формуле, нагло списжено у Шихинсона
        return  arr, arr #присваиваем один массив двум переменным

# ------------2.2 Волновое сопротивление--------------------------------------
# стр 233

    def c_x_v(self) -> Tuple[np.ndarray, np.ndarray] :
        M = self.grid.M
        c = self.aero.bar_c_kr 
        zeta_kr = self.aero.zeta_kr
        zeta_rl = self.aero.zeta_rl
        chi_05_kr = self.aero.chi_05_kr
        chi_05_rl = self.aero.chi_05_rl
        lambda_kr = self.aero.lambda_kr
        lambda_rl = self.aero.lambda_rl
        # Инициализируем массивы для результатов
        c_x_v_I = np.zeros_like(M, dtype=float)
        c_x_v_II = np.zeros_like(M, dtype=float)
        
        # Инициализируем массивы для результатов
        c_x_v_I = np.zeros_like(M, dtype=float)
        c_x_v_II = np.zeros_like(M, dtype=float)
        
        # Определяем маски
        mask_supersonic = M >= 1.0  # Сверхзвуковые режимы
        mask_subsonic = M < 1.0     # Дозвуковые режимы
        
        # Волновое сопротивление существует только при M >= 1
        if np.any(mask_supersonic):
            # Создаем функции для расчета
            fn_I = lambda M_val: AeroBDSMWrapper._val(
                AeroBDSM.get_c_x0_w_IsP_Rmb(M_val, c, zeta_kr, chi_05_kr, lambda_kr)
            )
            fn_II = lambda M_val: AeroBDSMWrapper._val(
                AeroBDSM.get_c_x0_w_IsP_Rmb(M_val, c, zeta_rl, chi_05_rl, lambda_rl)
            )
            
            # Применяем функции только для сверхзвуковых M
            M_supersonic = M[mask_supersonic]
            c_x_v_I[mask_supersonic] = AeroBDSMWrapper.vec(fn_I, M_supersonic)
            c_x_v_II[mask_supersonic] = AeroBDSMWrapper.vec(fn_II, M_supersonic)
        
        # Для дозвуковых режимов остается 0 (уже инициализировано)
        
        return c_x_v_I, c_x_v_II
        
    # собираем  вместе
    def c_x_0_I_II(self) -> Tuple[np.ndarray, np.ndarray]:

        c_x_v_I, c_x_v_II = self.c_x_v()
        c_x_p_I, c_x_p_II = self.c_x_p()

        return c_x_v_I + c_x_p_I , c_x_v_II + c_x_p_II
    
    # финальная формула 4.2
    def c_x0_total(self) -> np.ndarray:
        cx0_f  = self.c_x_0_f()                    # (M,)
        cx0_I, cx0_II = self.c_x_0_I_II()          # (M,), (M,)
        S = self.geom.S_cx
        return 1.05 * (cx0_f * (self.geom.S_f/S) + cx0_I * (self.geom.S_kr/S) + cx0_II * (self.geom.S_rl/S))  # (M,)

        
    # ======================= 3. Индуктивое сопротивление ========================

    # найдем сигму 
    def sigma(self) -> np.ndarray:
        M = self.grid.M
        lambda_Nos = self.aeroModel.lambda_Nos
        fn = lambda M: AeroBDSMWrapper._val(AeroBDSM.get_sigma_cp_Nos_Par(M, lambda_Nos))
        return AeroBDSMWrapper.vec(fn, M)

    # найдем delta_c_x1 из 4.55 по формуле 4.57
    def delta_c_x1(self) -> np.ndarray:
        sigma = self.sigma()
        alpha = self.grid.big_angles_deg
        return 2 * sigma * np.sin(np.radians(alpha))**2
    
    # c_x_i_f индуктивный кэф сопротивления для фезюляжа 4.56
    def c_xif(self) -> np.ndarray:
        c_yf = self.aeroModel.c_y_f_big()        # (M,A)
        c_x_0_f = self.c_x_0_f()[:, None]        # (M,1)
        sigma = self.sigma()[:, None]            # (M,1)
        alpha = np.radians(self.grid.big_angles_deg)[None, :]  # (1,A)
        # ΔCx1 по 4.57: 2*sigma*sin^2(alpha)
        delta_c_x1 = 2.0 * sigma * np.sin(alpha)**2              # (M,A)
        return (c_yf + c_x_0_f * np.sin(alpha)) * np.tan(alpha) + delta_c_x1 * np.cos(alpha)
    
    # индуктивное сопротивление несущих поверхностей 4.63 и 4.69
    # кси равна нулю из-за ромбовидного профиля (впадлу считать еще че там будет если не ромбовидный я не знаю какой в реале в моей ракете)
    def c_xi_I_II(self) -> Tuple[np.ndarray, np.ndarray]:
        # Углы
        A_deg   = self.grid.big_angles_deg
        DI_deg  = self.grid.deltas_I_deg    # обычно [0]
        DII_deg = self.grid.deltas_II_deg

        alpha = np.radians(A_deg)[None, :, None]    # (1,A,1)
        dI    = np.radians(DI_deg)[None, None, :]   # (1,1,DI)
        dII   = np.radians(DII_deg)[None, None, :]  # (1,1,DII)

        # Данные больших углов из модели (один вызов)
        big = self.aeroModel.compute_big_angles()
        A_kr        = big["A_kr"]            # (M,)
        A_rl        = big["A_rl"]            # (M,)
        alpha_eff_I = big["alpha_eff_I"]     # (M,A,DI)
        alpha_eff_II= big["alpha_eff_II"]    # (M,A,DII)

        # Производные изолированных поверхностей
        cya_is_kr = self.aeroModel.c_ya_is_kr()   # (M,)
        cya_is_rl = self.aeroModel.c_ya_is_rl()   # (M,)

        # Нормальные силы
        c_nI  = self.aeroModel.c_n_from_alpha_eff(alpha_eff_I,  cya_is_kr, A_kr)  # (M,A,DI)
        c_nII = self.aeroModel.c_n_from_alpha_eff(alpha_eff_II, cya_is_rl, A_rl)  # (M,A,DII)

        # Интерференция (с приведением форм)
        intr  = self.aeroModel.compute_interference()
        kaa_kr = intr["k_aa_kr"][:, None, None]   # (M,1,1)
        Kaa_kr = intr["K_aa_kr"][:, None, None]   # (M,1,1)
        kaa_rl = intr["k_aa_rl"][:, None, None]   # (M,1,1)
        Kaa_rl = intr["K_aa_rl"][:, None, None]

        # Формулы для индуктивного сопротивления (обобщенные)
        # Для I: δ_I присутствует (DI-ось), для II: δ_II присутствует (DII-ось)
        Cxi_I  = c_nI  * ( np.sin(alpha + dI)  + ((Kaa_kr - kaa_kr)/np.maximum(kaa_kr,1e-12)) * np.sin(alpha) * np.cos(dI) )
        Cxi_II = c_nII * ( np.sin(alpha + dII) + ((Kaa_rl - kaa_rl)/np.maximum(kaa_rl,1e-12)) * np.sin(alpha) * np.cos(dII) )

        return Cxi_I, Cxi_II  # (M,A,DI), (M,A,DII)

# финальная  4.2 вроде

    def c_x_tensor(self) -> np.ndarray:
        """
        Cx(M,A,DI,DII) = Cx0(M) + [ Sf*Cxi_f(M,A) + S1*Cxi_I(M,A,DI) + S2*Cxi_II(M,A,DII) ]
        """
        cx0 = self.c_x0_total()[:, None, None, None]   # (M,1,1,1)
        Cxi_f  = self.c_xif()[:, :, None, None]        # (M,A,1,1)
        Cxi_I, Cxi_II = self.c_xi_I_II()               # (M,A,DI), (M,A,DII)
        Cxi_I  = Cxi_I[:, :, :, None]                  # (M,A,DI,1)
        Cxi_II = Cxi_II[:, :, None, :]                 # (M,A,1,DII)

        Sf = self.geom.S_f / self.geom.S_cx
        S1 = self.geom.S_kr / self.geom.S_cx
        S2 = self.geom.S_rl / self.geom.S_cx

        return cx0 + (Sf*Cxi_f + S1*Cxi_I + S2*Cxi_II)  # (M,A,DI,DII)

# =============== 5) Глава V "Моменты тангажа и рыскания" =================
class Moments:
    def __init__(self , grid: FlowSetup, model : AerodynamicsModel , geom: Geometry, aero: AerodynamicParams , w : AeroBDSMWrapper):
        self.grid = grid
        self.model = model 
        self.geom = geom
        self.aero = aero
        self.W = w 
        # расчет геометрии для mz: 5.13 - 5.15
        self.chi_0 = self.geom.chi_0
        self.l = self.geom.l_raszmah_kr
        self.S = self.geom.S_kr
        self.eta_k = self.aero.eta_k
        self.psi = self.geom.psi
        self.zeta = self.aero.zeta_kr


        self.l_rl = self.geom.l_raszmah_rl
        self.chi_0_rl = self.geom.chi_0_rl
        self.S_rl = self.geom.S_rl
        self.eta_k_rl = self.aero.eta_k
        self.psi_rl = self.geom.psi
        self.zeta_rl = self.aero.zeta_rl
        
        # для крыльев
        self.b_a = (4/3) * (self.S/self.l) * (1 - (self.eta_k / (self.zeta + 1 )**2))
        self.x_a = ( self.l / 6 ) * ((self.eta_k + 2) / (self.zeta + 1) ) * np.tan(self.chi_0)
        self.y_a = ( self.l / 6 ) * ((self.eta_k + 2) / (self.zeta + 1) ) * np.tan(self.psi)
        # для рулей
        self.b_a_rl = (4/3) * (self.S/self.l) * (1 - (self.eta_k / (self.zeta_rl + 1 )**2))
        self.x_a_rl = ( self.l / 6 ) * ((self.zeta_rl + 2) / (self.zeta_rl + 1) ) * np.tan(self.chi_0_rl)
        self.y_a_rl = ( self.l / 6 ) * ((self.zeta_rl + 2) / (self.zeta_rl + 1) ) * np.tan(self.psi_rl)

# Работаем в связанной системе координат, начало координат совпадает с центром масс

# ....................4 Расчет координаты фокуса ЛА по углу атаки.......................
# ------------------------4.1  Фокус корпуса------------------------
    def delta_x_F(self) -> np.ndarray:
        fn = lambda M: self.W.Delta_bar_x_Falpha_NosCil(M, self.model.lambda_Nos, self.model.lambda_Cil)
        return self.W.vec(fn, self.grid.M)  # (M,)
    
    def c_yPerp_Cil(self, angles_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Возвращает c_y⊥_цилиндра для набора углов angles_deg:
        выход: (len(M), len(angles_deg))
        """
        if angles_deg is None:
            angles_deg = self.grid.big_angles_deg

        M_y = self.model._M_y(angles_deg)         # (M,A)
        vfn = np.vectorize(self.W.c_yPerp_Cil, otypes=[float])
        return vfn(M_y)                            # (M,A)
    
    # (5.34)
    def x_Fa_f(self) -> np.ndarray:

        L_nos = self.geom.l_nos
        W_nos = ((np.pi * self.geom.D**2)/8.0) * self.geom.l_nos #надо будет пересчитать объем черехз тройной интеграл пока я объем носа представил как объем цилиндра!!!
        S_f = self.geom.S_m
        L_korm = self.geom.l_korm
        L_f = self.geom.l_f
        delta_x_F = self.delta_x_F() [:, None]#(M,)
        W_korm = ((np.pi * self.geom.D**2)/4.0) * self.geom.l_korm
        S_dn = ((np.pi * self.geom.D**2)/4.0)

        # 5.36
        x_Fa_nos_cil = L_nos - (W_nos/S_f) + delta_x_F
    
        # 5.37
        x_Fa_korm =  L_f - 0.5 * L_korm              #L_f - ((S_f * L_korm - W_korm) / (S_f - S_dn))

        c_ya_f_nos_cil = self.model.c_ya_f()[:,None] #(M,1)
        c_ya_korm  = np.zeros_like(c_ya_f_nos_cil) # (M,1) — нет модели для кормы
        c_ya_f = c_ya_korm + c_ya_f_nos_cil [:,:]# (M,A)
        # 5.34
        return (1/c_ya_f) * (c_ya_f_nos_cil * x_Fa_nos_cil + c_ya_korm * x_Fa_korm )
    
    # 5.49
    def x_Fi_f_I_II (self) -> Tuple[np.ndarray, np.ndarray]:
        
        x_b_kr = self.geom.x_b_kr
        b_b_kr = self.geom.b_b_kr
        x_b_rl = self.geom.x_b_rl
        b_b_rl = self.geom.b_b_rl

        intr = self.model.compute_interference()
        F_L_xv_kr = intr["F_L_kr"]  # (M,)
        F_L_xv_rl = intr["F_L_rl"]  # (M,)

        from math import erf, sqrt, pi

        M     = self.grid.M
        eta_k = self.aero.eta_k
        D     = self.geom.D
        D_bar = self.geom.D_bar

        c = (4.0 + 1.0/eta_k) * (1 + 8*D_bar**2)

        
        st = np.sqrt(np.maximum(M**2 - 1.0, 0.0))
        denom = (np.pi/2)*D*np.where(st>0, st, 1.0)  # чтобы не делить на 0
        L_xv_kr_bar = np.where(st>0, self.geom.L_xv_kr/denom, 0.0)
        L_xv_rl_bar = np.where(st>0, self.geom.L_xv_rl/denom, 0.0)
        b_bar_kr    = self.geom.b_b_kr / D
        b_bar_rl    = self.geom.b_b_rl / D

        Phi = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

        F1_I = 1.0 - (1.0/(c*b_bar_kr**2)) * (np.exp(-c*L_xv_kr_bar**2) - np.exp(-c*(b_bar_kr + L_xv_kr_bar)**2)) \
            + np.sqrt(np.pi)/(b_bar_kr*np.sqrt(c)) * Phi(L_xv_kr_bar*np.sqrt(2*c))

        F1_II = 1.0 - (1.0/(c*b_bar_rl**2)) * (np.exp(-c*L_xv_rl_bar**2) - np.exp(-c*(b_bar_rl + L_xv_rl_bar)**2)) \
                + np.sqrt(np.pi)/(b_bar_rl*np.sqrt(c)) * Phi(L_xv_rl_bar*np.sqrt(2*c))

        
        x_F_is_kr_bar = self.W.vec(
            lambda Mi: self.W.bar_x_Falpha_IsP(Mi, self.aero.lambda_kr, self.aero.chi_05_kr, self.aero.zeta_kr),
            M
        )
        x_F_is_rl_bar = self.W.vec(
            lambda Mi: self.W.bar_x_Falpha_IsP(Mi, self.aero.lambda_rl, self.aero.chi_05_rl, self.aero.zeta_rl),
            M
        )
        x_F_b_bar_kr  = x_F_is_kr_bar + 0.02 * self.aero.lambda_kr * np.tan(self.aero.chi_05_kr)
        x_F_b_bar_rl  = x_F_is_rl_bar + 0.02 * self.aero.lambda_rl * np.tan(self.aero.chi_05_rl)

        # Сверхзвук: используем хвостовую коррекцию, дозвук: fallback к x_F_is (в метрах)
        sup = (M >= 1.0)

        x_Fi_f_I  = np.empty_like(M, dtype=float)
        x_Fi_f_II = np.empty_like(M, dtype=float)

        x_Fi_f_I[sup]  = x_b_kr + b_b_kr * x_F_b_bar_kr[sup] * F_L_xv_kr[sup] * F1_I[sup]
        x_Fi_f_II[sup] = x_b_rl + b_b_rl * x_F_b_bar_rl[sup] * F_L_xv_rl[sup] * F1_II[sup]

        # дозвук: просто точка x_F_is в метрах
        x_Ak   = self.geom.x_Ak
        
        b_Ak_kr= self.geom.b_Ak_kr
        x_F_is_kr = self.x_a + b_Ak_kr * x_F_is_kr_bar
        x_F_is_rl = self.x_a_rl + self.geom.b_Ak_r * x_F_is_rl_bar  # если нужна своя база для рулей

        x_Fi_f_I[~sup]  = x_F_is_kr[~sup]
        x_Fi_f_II[~sup] = x_F_is_rl[~sup]

        return x_Fi_f_I, x_Fi_f_II  # (M,), (M,)
    
    # 5.48
    #5.39 5.40 
    def x_F_a_I_II(self ,kaa_kr: np.ndarray,Kaa_kr: np.ndarray,kaa_rl: np.ndarray,Kaa_rl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = self.grid.M

        x_Fi_f_I, x_Fi_f_II = self.x_Fi_f_I_II()  # (M,), (M,)

        x_Ak    = self.geom.x_Ak
        b_Ak_kr = self.geom.b_Ak_kr
        x_F_is_kr_bar = self.W.vec(
            lambda Mi: self.W.bar_x_Falpha_IsP(Mi, self.aero.lambda_kr, self.aero.chi_05_kr, self.aero.zeta_kr),
            M
        )
        x_F_is_rl_bar = self.W.vec(
            lambda Mi: self.W.bar_x_Falpha_IsP(Mi, self.aero.lambda_rl, self.aero.chi_05_rl, self.aero.zeta_rl),
            M
        )
        x_F_is_kr     = self.x_a + self.b_a * x_F_is_kr_bar  # (M,)
        x_F_is_rl     = self.x_a_rl + self.b_a * x_F_is_rl_bar  # (M,)

        f = self.W.Delta_bar_z_Falpha_iC(1.0)  # скаляр (d_bar = 1 для постоянного диаметра)
        x_F_delta_kr = x_F_is_kr - f * np.tan(self.aero.chi_05_kr)  # (M,)
        x_F_delta_rl = x_F_is_rl - f * np.tan(self.aero.chi_05_rl)  # (M,) 

        
        x_F_a_I  = (x_F_is_kr + (kaa_kr - 1.0) * x_F_delta_kr + (Kaa_kr - kaa_kr)*x_Fi_f_I) / np.maximum(Kaa_kr, 1e-12)
        x_F_a_II = (x_F_is_rl + (kaa_rl - 1.0) * x_F_delta_rl + (Kaa_rl - kaa_rl)*x_Fi_f_II) / np.maximum(Kaa_rl, 1e-12)
        return x_F_a_I, x_F_a_II  # (M,), (M,)

# ---------------------5. Расчет координат фокусов ЛА по углам отклонения несущих поверхностей-------------------
# 5.56
    def x_F_delta_I_II(self) -> Tuple[np.ndarray, np.ndarray]:
        M = self.grid.M

        k_d0_kr, K_d0_kr, k_d0_rl, K_d0_rl = self.model.compute_delta_derivatives()
        intr  = self.model.compute_interference()
        Kaa_rl = intr["K_aa_rl"]  # (M,)
        kaa_kr = intr["k_aa_kr"]; Kaa_kr = intr["K_aa_kr"]
        kaa_rl = intr["k_aa_rl"]; Kaa_rl_all = intr["K_aa_rl"]

        # x_F_delta1 от формул 5.39–5.40, но с k_delta/K_delta
        x_F_delta1, _ = self.x_F_a_I_II(k_d0_kr, K_d0_kr, k_d0_rl, K_d0_rl)

        # x_F_a_II для "обычных" kaa/Kaa
        _, x_F_a_II = self.x_F_a_I_II(kaa_kr, Kaa_kr, kaa_rl, Kaa_rl_all)

        c_y_delta_1, _ = self.model.compute_cy_delta_1_2()  # (M,), (M,)

        eps = self.model.compute_eps_delta_sr(K_aa_rl=Kaa_rl, k_delta_0_kr=k_d0_kr)  # (M,)
        c_ya_is_kr = self.model.c_ya_is_kr()  # (M,)
        n_vals = self.model._n_eff(M)        # (M,)

        S_bar_I  = self.geom.S_kr / self.geom.S_f
        S_bar_II = self.geom.S_rl / self.geom.S_f
        k_I  = self.model.kappa_q_nos()  # (M,)
        k_II = self.model.kappa_q_isp()  # (M,)

        # x_F_is_kr
        x_Ak = self.geom.x_Ak; b_Ak_kr = self.geom.b_Ak_kr
        x_F_is_kr_bar = self.W.bar_x_Falpha_IsP_vec(
    M, self.aero.lambda_kr, self.aero.chi_05_kr, self.aero.zeta_kr)  # (M,)
        x_F_is_kr     = x_Ak + b_Ak_kr * x_F_is_kr_bar

        # x_Fi_f_II
        _, x_Fi_f_II = self.x_Fi_f_I_II()

        # 5.56 
        x_F_delta_I = ( (c_ya_is_kr*K_d0_kr*n_vals*S_bar_I*k_I)*x_F_delta1
                        - (c_ya_is_kr*Kaa_rl*S_bar_II*k_II)*eps*x_F_a_II ) / np.maximum(c_y_delta_1, 1e-12)

        # 5.58
        x_F_delta_II = ( k_d0_rl*x_F_is_kr + (K_d0_rl - k_d0_rl)*x_Fi_f_II ) / np.maximum(K_d0_rl, 1e-12)

        return x_F_delta_I, x_F_delta_II  # (M,), (M,)
    
    # ----------------------считаем финальный m_z ----------------------
    # 5.22
    def m_z(self) -> np.ndarray:
        """
        Моменты по малым углам. Возвращает (M,A,DI,DII)
        m_z = m_z0 + c_y * ((x_t - x_F)/L), где x_F берется соответственно:
        - по α: x_F_a (тут: фюзеляжный фокус, можно также учесть пояса)
        - по δ_I: x_F_delta_I
        - по δ_II: x_F_delta_II
        """
        # углы
        A   = np.radians(self.grid.small_angles_deg)[None, :, None, None]
        DI  = np.radians(self.grid.deltas_I_deg)[None, None, :, None]
        DII = np.radians(self.grid.deltas_II_deg)[None, None, None, :]

        # подъемная сила (линейная)
        c_y = self.model.compute_c_y_sum()  # (M,A,DI,DII)

        # фокусы
        x_Fa = self.x_Fa_f()  # (M,1)
        x_F_delta_I, x_F_delta_II = self.x_F_delta_I_II()  # (M,), (M,)

        x_t = self.geom.x_t
        L   = self.geom.l_f

        # Растянем в тензорные формы
        xFa_b  = x_Fa[:, None, None, :] if x_Fa.ndim == 2 else x_Fa[:, None, None, None]  # (M,1,1,1)
        xF1_b  = x_F_delta_I[:, None, None, None]  # (M,1,1,1)
        xF2_b  = x_F_delta_II[:, None, None, None]

        # Линейные вклады: m_z = m_z0 + cy*угол*((x_t - xF)/L)
        m_z0 = 0 #как у макса шихана надо поменять потом
        m_z_a      =  (self.model.compute_small_angles()[0][:, None, None, None] * A) * ((x_t - xFa_b)/L)
        m_z_deltaI = 0
        m_z_deltaII= (self.model.compute_cy_delta_1_2()[1][:, None, None, None] * DII) * ((x_t - xF2_b)/L)

        return  m_z0 + m_z_a + m_z_deltaI + m_z_deltaII  # (M,A,DI,DII)
    
    def get_m_z_cy(self) -> float:
        intr  = self.model.compute_interference()
        Kaa_rl = intr["K_aa_rl"]  # (M,)
        kaa_kr = intr["k_aa_kr"]; Kaa_kr = intr["K_aa_kr"]
        kaa_rl = intr["k_aa_rl"]; 
        x_t = self.geom.x_t
        x_Fa, _ = self.x_F_a_I_II(kaa_kr, Kaa_kr, kaa_rl, Kaa_rl)
        L: float = 3.906
        return -((x_Fa - x_t )/ L)


#================================ СИМУЛЯЦИЯ ПОЛЕТА================================================

@dataclass
class SimParams:
        dt: float      = 0.1 # шаг по времени
        r_por: float   = 15 #радиус поражения [м]
        t_max: float   = 300 # макс. время полета [c]
        n_y_max: float = 40 # максимальная перегрузка
        y_min: float   = 0 # минимальная высота [м]
        v_max: float   = 1360 # максимальая скорость полета (4 Маха) [м/с]
        
@dataclass  
class InitialConditions:
    """Начальные условия"""
    # Ракета
    x_r0: float     = 0           # начальная X ракеты [м]
    y_r0: float     = 10000       # начальная Y ракеты [м] 
    v_r0: float     = 700         # начальная скорость ракеты [м/с]
    theta_r0: float = -999    # начальный угол (-999 = вычислить)
    
    # Цель
    x_c0: float     = 30000       # начальная X цели [м]
    y_c0: float     = 10000       # начальная Y цели [м]
    v_c0: float     = 600         # скорость цели [м/с]
    theta_c0: float = 0       # угол цели [рад]
    
    # Двигатель
    mu_0: float     = 0.2         # относительный запас топлива
    eta_0: float    = 15         # тяговооружённость
    k_m: float      = 2.5          # распределение массы
    k_p: float      = 0.5          # распределение тяги
    m_start: float  = 161.5    # стартовая масса [кг]
    m_bch: float    = 100.4      # масса боевой части [кг]
    diameter: float = 0.18    # диаметр [м]
    I_ud: float     = 2500        # удельный импульс [м/с]
    length: float   = 3.7       # длина [м]

@dataclass
class GuidanceParams:
    """Параметры наведения"""
    method: int         = 8   # 8 = пропорциональное сближение
    k: float            = 8.0 # коэффициент метода
    phi_upr: float      = 0   # угол управления [рад]
    theta_r_init: float = 0   # начальный угол [рад]

# ============ КЛАСС АТМОСФЕРЫ ============
class Atmosphere:
    def __init__(self):
        self.atm = atmo()
    
    def density(self, h):
        return self.atm.rho(h)
    
    def pressure(self, h):
        return self.atm.p(h)
    
    def sound_speed(self, h):
        return 340  

# ============ КЛАСС ГЕОМЕТРИЧЕСКИХ ФУНКЦИЙ ============
class GeometryCalculator:
    def __init__(self):
        self.g = 9.85066

    # Функции вычисления геометрических параметров
    @njit(fastmath=True)
    def r(x_r, y_r, x_c, y_c):
        return np.sqrt((y_c - y_r) ** 2 + (x_c - x_r) ** 2)

    @njit(fastmath=True)
    def dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c):
        r_rc = self.r(x_r, y_r, x_c, y_c)
        Delta_x = x_c - x_r
        Delta_y = y_c - y_r
        Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)
        Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)
        return (Delta_y * Delta_dot_y + Delta_x * Delta_dot_x) / r_rc

    @njit(fastmath=True)
    def double_dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c, a_xa_c, a_ya_c, a_xa_r, a_ya_r):
        r_rc = self.r(x_r, y_r, x_c, y_c)
        dot_r_rc = self.dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c)
        Delta_x = x_c - x_r
        Delta_y = y_c - y_r
        Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)
        Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)
        Delta_double_dot_x = a_xa_c * np.cos(Theta_c) - a_xa_r * np.cos(Theta_r) - a_ya_c * np.sin(
            Theta_c) + a_ya_r * np.sin(Theta_r)
        Delta_double_dot_y = a_xa_c * np.sin(Theta_c) - a_xa_r * np.sin(Theta_r) + a_ya_c * np.cos(
            Theta_c) - a_ya_r * np.cos(Theta_r)
        return (
                    Delta_dot_x ** 2 + Delta_x * Delta_double_dot_x + Delta_y * Delta_double_dot_y + Delta_dot_y ** 2 - dot_r_rc ** 2) / r_rc

    @njit(fastmath=True)
    def epsilon(x_r, y_r, x_c, y_c):
        if (x_c - x_r > 0):
            return np.atan((y_c - y_r) / (x_c - x_r))

        elif ((x_c - x_r < 0) and (y_c - y_r >= 0)):
            return np.atan((y_c - y_r) / (x_c - x_r)) + np.pi

        elif ((x_c - x_r < 0) and (y_c - y_r < 0)):
            return np.atan((y_c - y_r) / (x_c - x_r)) - np.pi

        elif ((x_c == x_r) and (y_c >= y_r)):
            return np.pi / 2

        elif ((x_c == x_r) and (y_c < y_r)):
            return -np.pi / 2

    @njit(fastmath=True)
    def dot_epsilon(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c):
        r_rc = self.r(x_r, y_r, x_c, y_c)
        Delta_x = x_c - x_r
        Delta_y = y_c - y_r
        Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)
        Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)

        return (Delta_x * Delta_dot_y - Delta_y * Delta_dot_x) / (r_rc ** 2)

    @njit(fastmath=True)
    def double_dot_epsilon(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c, a_xa_c, a_ya_c, a_xa_r, a_ya_r):
        r_rc = self.r(x_r, y_r, x_c, y_c)
        dot_r_rc = self.dot_r(x_r, y_r, x_c, y_c, v_r, v_c, Theta_r, Theta_c)
        Delta_x = x_c - x_r
        Delta_y = y_c - y_r
        Delta_dot_x = v_c * np.cos(Theta_c) - v_r * np.cos(Theta_r)
        Delta_dot_y = v_c * np.sin(Theta_c) - v_r * np.sin(Theta_r)
        Delta_double_dot_x = a_xa_c * np.cos(Theta_c) - a_xa_r * np.cos(Theta_r) - a_ya_c * np.sin(Theta_c) + a_ya_r * np.sin(Theta_r)
        Delta_double_dot_y = a_xa_c * np.sin(Theta_c) - a_xa_r * np.sin(Theta_r) + a_ya_c * np.cos(Theta_c) - a_ya_r * np.cos(Theta_r)
        Add_1 = (Delta_x * Delta_double_dot_y - Delta_y * Delta_double_dot_x) / (r_rc ** 2)
        Add_2 = 2 * ((Delta_y * Delta_dot_y + Delta_x * Delta_dot_x) * (Delta_x * Delta_dot_y - Delta_y * Delta_dot_x)) / (r_rc ** 4)

        return (Add_1 - Add_2)

    @njit(fastmath=True)
    def dpar_i(n_xa_i, v_i, Theta_i):
        dv_i = (n_xa_i - np.sin(Theta_i)) * self.g
        dx_i = v_i * np.cos(Theta_i)
        dy_i = v_i * np.sin(Theta_i)
        return dv_i, dx_i, dy_i

    @njit(fastmath=True)
    def par_i1(Theta_i, dTheta_i, dt, v_i, dv_i, x_i, dx_i, y_i, dy_i):
        Theta_i1 = Theta_i + dTheta_i * dt
        v_i1 = v_i + dv_i * dt
        x_i1 = x_i + dx_i * dt
        y_i1 = y_i + dy_i * dt
        return Theta_i1, v_i1, x_i1, y_i1

    @njit(fastmath=True)
    def value_c(v_c, eps_c, r_nc, delta_r_dot):
        return -(v_c * np.sin(eps_c)) / (r_nc * delta_r_dot)
    
    @njit(fastmath=True)
    def n_xa_tiaga(X_a, P, alf, m_r):
        return (-X_a + P)/ (m_r * self.g)

    def n_ya_tiaga(Y_a, P, alf, m_r):
        return (Y_a + P * np.sin(alf)) / (m_r * self.g)

    @njit(fastmath=True)
    def n_ya_pot_ParallelNav(epsilon_rc, v_r, v_c, Theta_r, Theta_c, a_xa_r, a_xa_c, a_ya_c):
        num_1 = a_ya_c * v_r * np.cos(epsilon_rc - Theta_c)
        num_2 = (a_xa_c * v_r - a_xa_r * v_c) * np.sin(epsilon_rc - Theta_c)
        denum = g * v_r * np.sqrt(1 - ((v_c / v_r) * np.sin(epsilon_rc - Theta_c)) ** 2)
        return (num_1 - num_2) / denum + np.cos(Theta_r)

    @njit(fastmath=True)
    def n_ya_pot_ProportionalNavConst(v_r, dot_epsilon_rc, Theta_r, k):
        return (v_r / g * k * dot_epsilon_rc + np.cos(Theta_r))
    
class CalcTragectory:

    def X(self, c_x):
        pass
        return 
    
    def Y(self, c_y):
        pass
        return 



# ========= 6) Экспорт  =========

class DataExporter:
    def __init__(self, model: AerodynamicsModel, include_heavy: bool = False):
        self.model = model
        self.include_heavy = include_heavy  # флаг

    def build_all(self) -> Dict[str, 'pd.DataFrame']:
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        m = self.model

        dfs = {
            "all_angles_data.csv": m.assemble_all_angles_data_df(),
            "krylo_isP.csv": m.assemble_krylo_isP_df(),
            "rul_isP.csv": m.assemble_rul_isP_df(),
            "Kappa_aa_kr.csv": m.assemble_Kappa_aa_kr_df(),
            "Kappa_aa_rl.csv": m.assemble_Kappa_aa_rl_df(),
            "z_b_v.csv": m.assemble_z_b_v_df(),
            "i_v.csv": m.assemble_i_v_df(),
            "kappa_q_nos.csv": m.assemble_kappa_q_nos_df(),
            "kappa_q.csv": m.assemble_kappa_q_df(),
            "c_y_alpha_sum.csv": m.assemble_c_y_alpha_sum_df(),
            "c_y_delta_1.csv": m.assemble_c_y_delta_1_df(),
            "c_y_delta_2.csv": m.assemble_c_y_delta_2_df(),
            "c_y_f.csv": m.assemble_c_y_f_df(),
            "c_y_small_angles.csv": m.assemble_c_y_small_angles_df(), # это c_y_a для малых углов
            "c_y_sum_big_delta.csv": m.assemble_c_y_sum_big_delta_df(),
            "c_y_sum.csv": m.assemble_c_y_sum(), # c_y для малых углов атаки
            "c_x_data.csv": m.assemble_c_x(),
            "m_z_small.csv": m.assemble_m_z_df(),
            "polar_lilienthal.csv": m.assemble_polar_lilienthal_df(),
            
        }

        if self.include_heavy:
            dfs["psi_eps.csv"] = m.assemble_psi_eps_df()          # тяжелое
            dfs["eps_alpha_sr.csv"] = m.assemble_eps_alpha_sr_df()# очень тяжелое

        return dfs
    
    def save_all(self, dfs: Dict[str, 'pd.DataFrame'], folder: Optional[str] = None) -> None:
        if pd is None:
            raise RuntimeError("pandas не установлен.")
        base = "" if folder is None else folder.rstrip("/") + "/"
        for fname, df in dfs.items():
            df.to_csv(base + fname, index=False)


# ========= 6) Пример использования =========

if __name__ == "__main__":
    geom = Geometry()
    aero = AerodynamicParams()
    grid = FlowSetup.default()
    w    = AeroBDSMWrapper()
    model = AerodynamicsModel(geom, aero, grid)  # resistance создаётся внутри модели
    moments = Moments(grid, model  , geom, aero , w )
    m_z_cy = moments.get_m_z_cy()
    print("Готово")
    print(60*'=')
    print('Запас статической устойчивости:')
    print(m_z_cy)

    if pd is not None:
        exporter = DataExporter(model, include_heavy = False)
        dfs = exporter.build_all()
        exporter.save_all(dfs, folder= 'data')
        print("DataFrames готовы и сохранены.")

