"""Limpia dataset_total_16_11: elimina DUDOSO y muestras de margen en cada cambio de etiqueta.

Uso: python limpiar_dataset.py [--margen 2] [--entrada RUTA] [--salida RUTA]
"""
import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent / "data" / "real"
ENTRADA = BASE / "dataset_total_16_11.xlsx - Dataset.csv"
SALIDA = BASE / "dataset_limpio.csv"


def limpiar(df: pd.DataFrame, margen: int) -> pd.DataFrame:
    df = df.sort_values(["sesion", "t_s"]).reset_index(drop=True)
    cambio = (df.etiqueta != df.groupby("sesion").etiqueta.shift()) & (
        df.groupby("sesion").cumcount() > 0
    )
    # Cada cambio de etiqueta empieza un nuevo tramo
    tramo = cambio.groupby(df.sesion).cumsum()
    pos = df.groupby([df.sesion, tramo]).cumcount()
    n = df.groupby([df.sesion, tramo]).etiqueta.transform("size")
    # Margen solo en las fronteras entre tramos, no en inicio/fin de sesión
    hay_previo = tramo > 0
    hay_siguiente = tramo < tramo.groupby(df.sesion).transform("max")
    borde = (hay_previo & (pos < margen)) | (hay_siguiente & (pos >= n - margen))
    return df[(df.etiqueta != "DUDOSO") & ~borde].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--margen", type=int, default=2)
    a = ap.parse_args()

    df = pd.read_csv(a.entrada, decimal=",")
    out = limpiar(df, a.margen)
    out.to_csv(a.salida, index=False)
    print(f"{len(df)} -> {len(out)} filas ({len(df) - len(out)} eliminadas)")
    print(out.etiqueta.value_counts().to_string())
    print(f"Guardado en {a.salida}")


if __name__ == "__main__":
    main()
