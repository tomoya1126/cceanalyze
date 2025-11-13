import re
import pandas as pd
import numpy as np
from glob import glob

files = sorted(glob("2dz*.log"))
all_data = []

for f in files:
    # ファイル名から "2dz" の後ろの数字（例: 410, 412.5, 430）を正確に抽出
    m = re.search(r"2dz([0-9]+(?:\.5)?)", f)
    if not m:
        raise ValueError(f"ファイル名からz値を抽出できません: {f}")
    z_value_um = float(m.group(1))   # µm単位
    z_value_m = z_value_um * 1e-6    # メートルに変換
    print(f"Reading {f} as z={z_value_um:.1f} µm ...")

    df = pd.read_csv(f, sep=r"\s+", header=0)
    df["Z[m]"] = z_value_m
    all_data.append(df)

# 結合
df_all = pd.concat(all_data, ignore_index=True)

# 必要列だけ残す
cols = ["X[m]", "Y[m]", "Z[m]", "V[V]", "Ex[V/m]", "Ey[V/m]", "Ez[V/m]"]
df_all = df_all[cols]

# 軽量化
df_all = df_all.astype(np.float32)

# 保存
np.savez_compressed("field_410to430um.npz",
    x=df_all["X[m]"].to_numpy(),
    y=df_all["Y[m]"].to_numpy(),
    z=df_all["Z[m]"].to_numpy(),
    V=df_all["V[V]"].to_numpy(),
    Ex=df_all["Ex[V/m]"].to_numpy(),
    Ey=df_all["Ey[V/m]"].to_numpy(),
    Ez=df_all["Ez[V/m]"].to_numpy(),
)

print("Saved to field_410to430um.npz")
