# save_csv.py
from main_class import Geometry, AerodynamicParams, FlowSetup, AerodynamicsModel, DataExporter

def main():
    geom = Geometry()
    aero = AerodynamicParams()
    grid = FlowSetup.default()

    model = AerodynamicsModel(geom, aero, grid, i_is_wing=False, front_is_wing=False)

    exporter = DataExporter(model, include_heavy=False)
    dfs = exporter.build_all()
    exporter.save_all(dfs, folder=None)

    print("Готово! Все CSV сохранены в текущей директории.")

if __name__ == "__main__":
    main()
