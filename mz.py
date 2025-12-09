# stability_sweep.py
from dataclasses import replace
from itertools import product
import numpy as np
import pandas as pd
from tqdm.auto import tqdm 

from main_class import Geometry, AerodynamicParams, FlowSetup, AerodynamicsModel, Moments

def nearest_idx(arr: np.ndarray, value: float) -> int:
    arr = np.asarray(arr, dtype=float)
    return int(np.argmin(np.abs(arr - value)))

def evaluate_stability_mz(model: AerodynamicsModel,
                          alpha_deg_target: float = -10.0,
                          delta_I_deg: float = 0.0,
                          delta_II_deg: float = 0.0,
                          check_mode: str = "at_mach",  # "at_mach" или "all"
                          mach_probe: float = 2.0) -> dict:
    """
    Считает тензор m_z и оценивает устойчивость:
    - при alpha=-10°, delta_I=0°, delta_II=0°
    - check_mode:
        - "at_mach": проверка только в ближайшем к mach_probe числе Маха
        - "all": проверка для всех M на сетке
    Возвращает словарь с метриками и флаг stable (True/False).
    """
    # рассчитываем m_z на малых углах
    moments = Moments(model.grid, model, model.g, model.a, model.__class__.__mro__[0])  # AeroBDSMWrapper доступен в main_class
    # лучше использовать уже созданный Moments через сборку в модели (если добавите), а пока создадим как в assemble_m_z_df
    # но нам нужен реальный AeroBDSMWrapper; возьмем его из main_class:
    from main_class import AeroBDSMWrapper
    moments = Moments(model.grid, model, model.g, model.a, AeroBDSMWrapper)

    mz = moments.m_z()  # (M,A,DI,DII)

    # индексы среза
    iA   = nearest_idx(model.grid.small_angles_deg, alpha_deg_target)
    iDI  = nearest_idx(model.grid.deltas_I_deg,    delta_I_deg)
    iDII = nearest_idx(model.grid.deltas_II_deg,   delta_II_deg)

    mz_slice = mz[:, iA, iDI, iDII]  # (M,)

    if check_mode == "at_mach":
        iM = nearest_idx(model.grid.M, mach_probe)
        mz_point = float(mz_slice[iM])
        stable = bool(mz_point > 0.0)
    elif check_mode == "all":
        stable = bool(np.all(mz_slice > 0.0))
        # для отчета полезно знать минимальный запас
        mz_point = float(np.min(mz_slice))
    else:
        raise ValueError("check_mode должен быть 'at_mach' или 'all'.")

    return dict(
        stable=stable,
        mz_min=float(np.min(mz_slice)),
        mz_max=float(np.max(mz_slice)),
        mz_at_check=float(mz_point),
    )



def run_stability_sweep(
    # Варьируемые геометрические параметры (оставляйте None, если не варьируете)
    L_xv_kr_list=None,
    L_xv_rl_list=None,
    x_b_kr_list=None,
    x_b_rl_list=None,
    x_Ak_list=None,
    b_Ak_r_list=None,
    b_Ak_kr_list=None,
    # Параметры аэродинамики (если надо варьировать)
    lambda_kr_list=None,
    lambda_rl_list=None,
    chi_05_kr_list=None,
    chi_05_rl_list=None,
    zeta_kr_list=None,
    zeta_rl_list=None,
    # Режим проверки устойчивости
    alpha_deg_target=-10.0,
    delta_I_deg=0.0,
    delta_II_deg=0.0,
    check_mode="at_mach",   # "at_mach" или "all"
    mach_probe=2.0,
    # Сетка потока
    grid: FlowSetup = None,
    # Управление выводом
    verbose: bool = False,
    use_progress: bool = True,
    tqdm_desc: str = "Stability sweep",
):
    """
    Перебирает декартово произведение параметров, оценивает устойчивость по m_z,
    печатает каждую комбинацию при verbose=True и показывает прогресс-бар при use_progress=True.
    """
    if grid is None:
        grid = FlowSetup.default()

    base_geom = Geometry()
    base_aero = AerodynamicParams()

    def as_list(lst, fallback):
        if lst is None:
            return [fallback]
        return list(lst)

    # Подготовим списки
    L_xv_kr_list = as_list(L_xv_kr_list, base_geom.L_xv_kr)
    L_xv_rl_list = as_list(L_xv_rl_list, base_geom.L_xv_rl)
    x_b_kr_list  = as_list(x_b_kr_list,  base_geom.x_b_kr)
    x_b_rl_list  = as_list(x_b_rl_list,  base_geom.x_b_rl)
    x_Ak_list    = as_list(x_Ak_list,    base_geom.x_Ak)
    b_Ak_r_list  = as_list(b_Ak_r_list,  base_geom.b_Ak_r)
    b_Ak_kr_list = as_list(b_Ak_kr_list, base_geom.b_Ak_kr)

    lambda_kr_list = as_list(lambda_kr_list, base_aero.lambda_kr)
    lambda_rl_list = as_list(lambda_rl_list, base_aero.lambda_rl)
    chi_05_kr_list = as_list(chi_05_kr_list, base_aero.chi_05_kr)
    chi_05_rl_list = as_list(chi_05_rl_list, base_aero.chi_05_rl)
    zeta_kr_list   = as_list(zeta_kr_list,   base_aero.zeta_kr)
    zeta_rl_list   = as_list(zeta_rl_list,   base_aero.zeta_rl)

    # Общее число комбинаций
    lengths = [
        len(L_xv_kr_list), len(L_xv_rl_list), len(x_b_kr_list), len(x_b_rl_list),
        len(x_Ak_list), len(b_Ak_r_list), len(b_Ak_kr_list),
        len(lambda_kr_list), len(lambda_rl_list),
        len(chi_05_kr_list), len(chi_05_rl_list),
        len(zeta_kr_list), len(zeta_rl_list),
    ]
    total = int(np.prod(lengths)) if lengths else 0

    rows = []
    counter = 0

    iterator = product(
        L_xv_kr_list, L_xv_rl_list, x_b_kr_list, x_b_rl_list, x_Ak_list, b_Ak_r_list, b_Ak_kr_list,
        lambda_kr_list, lambda_rl_list, chi_05_kr_list, chi_05_rl_list, zeta_kr_list, zeta_rl_list
    )

    # Оборачиваем итератор в tqdm, если нужно
    if use_progress:
        iterator = tqdm(iterator, total=total, desc=tqdm_desc, dynamic_ncols=True)

    for (L_xv_kr, L_xv_rl, x_b_kr, x_b_rl, x_Ak, b_Ak_r, b_Ak_kr,
         lam_kr, lam_rl, chi05_kr, chi05_rl, zeta_kr, zeta_rl) in iterator:

        counter += 1

        geom = replace(Geometry(),
                       L_xv_kr=L_xv_kr, L_xv_rl=L_xv_rl,
                       x_b_kr=x_b_kr, x_b_rl=x_b_rl,
                       x_Ak=x_Ak, b_Ak_r=b_Ak_r, b_Ak_kr=b_Ak_kr)
        aero = replace(AerodynamicParams(),
                       lambda_kr=lam_kr, lambda_rl=lam_rl,
                       chi_05_kr=chi05_kr, chi_05_rl=chi05_rl,
                       zeta_kr=zeta_kr, zeta_rl=zeta_rl)

        model = AerodynamicsModel(geom, aero, grid)

        stab = evaluate_stability_mz(
            model,
            alpha_deg_target=alpha_deg_target,
            delta_I_deg=delta_I_deg,
            delta_II_deg=delta_II_deg,
            check_mode=check_mode,
            mach_probe=mach_probe
        )

        if verbose:
            line = (
                f"[{counter}/{total}] "
                f"Lxv_kr={L_xv_kr:.3f}, Lxv_rl={L_xv_rl:.3f}, "
                f"x_b_kr={x_b_kr:.3f}, x_b_rl={x_b_rl:.3f}, "
                f"x_Ak={x_Ak:.3f}, b_Ak_r={b_Ak_r:.3f}, b_Ak_kr={b_Ak_kr:.3f} | "
                f"lam_kr={lam_kr:.3f}, lam_rl={lam_rl:.3f}, "
                f"chi05_kr={chi05_kr:.3f}, chi05_rl={chi05_rl:.3f}, "
                f"zeta_kr={zeta_kr:.4f}, zeta_rl={zeta_rl:.4f} || "
                f"stable={stab['stable']}, "
                f"mz_at_check={stab['mz_at_check']:.6g}, "
                f"mz_min={stab['mz_min']:.6g}, "
                f"mz_max={stab['mz_max']:.6g}"
            )
            if use_progress:
                # корректный вывод рядом с прогресс-баром
                tqdm.write(line)
            else:
                print(line)

        row = dict(
            L_xv_kr=L_xv_kr, L_xv_rl=L_xv_rl,
            x_b_kr=x_b_kr, x_b_rl=x_b_rl,
            x_Ak=x_Ak, b_Ak_r=b_Ak_r, b_Ak_kr=b_Ak_kr,
            lambda_kr=lam_kr, lambda_rl=lam_rl,
            chi_05_kr=chi05_kr, chi_05_rl=chi05_rl,
            zeta_kr=zeta_kr, zeta_rl=zeta_rl,
            stable=stab["stable"],
            mz_min=stab["mz_min"],
            mz_max=stab["mz_max"],
            mz_at_check=stab["mz_at_check"],
            check_mode=check_mode,
            Mach_probe=mach_probe if check_mode=="at_mach" else np.nan,
            alpha_deg_target=alpha_deg_target,
            delta_I_deg=delta_I_deg,
            delta_II_deg=delta_II_deg,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = run_stability_sweep(
        L_xv_kr_list=[2.0],
        L_xv_rl_list=[0.300],
        x_b_kr_list=[1.632],
        x_b_rl_list=[3.632],
        x_Ak_list=[0],
        b_Ak_r_list=np.arange( 0.1, 1.0, 0.05).tolist(),
        b_Ak_kr_list=np.arange( 0.5, 2.0, 0.1).tolist(),
        lambda_kr_list=[5.0],
        chi_05_kr_list=[0.48, 0.51, 0.54],
        alpha_deg_target=-10.0,
        delta_I_deg=0.0,
        delta_II_deg=0.0,
        check_mode="at_mach",
        mach_probe=2.0,
        verbose=False,       # можно True, строки уйдут через tqdm.write
        use_progress=True,   # ВКЛ прогресс-бар
        tqdm_desc="m_z stability sweep",
    )

    df_stable = df[df["stable"]].copy()
    print("Всего комбинаций:", len(df))
    print("Устойчивых:", len(df_stable))

    # Сохранение CSV у вас локально
    # df.to_csv("stability_scan_all.csv", index=False)
    df_stable.to_csv("stability_scan_stable_only.csv", index=True)


