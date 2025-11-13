#!/usr/bin/env python3
"""
OpenSTF電界データを読み込んでnpz形式に変換するスクリプト

処理内容：
1. くし形電界/ と 横型電界/ から元npzを読み込む
2. 統一フォーマット（X,Y,Z,V,Ex,Ey,Ez）に変換
3. V から E = -∇V を再計算
4. 電界/ に保存
5. z=430μm位置の電場分布図を出力
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==============================
#  基本設定
# ==============================

CASES = [
    {
        "label": "kushigata",
        "src_dir": "くし形電界",
    },
    {
        "label": "yokogata",
        "src_dir": "横型電界",
    },
]

OUT_DIR = "電界"


# ==============================
#  ユーティリティ
# ==============================

def diff_nonuniform_1d(f, x):
    """非一様格子上の1次元微分（中央差分・端は片側差分）"""
    f = np.asarray(f, dtype=float)
    x = np.asarray(x, dtype=float)
    N = len(x)
    df = np.zeros_like(f)

    for i in range(N):
        if i == 0:
            df[i] = (f[i+1] - f[i]) / (x[i+1] - x[i])
        elif i == N-1:
            df[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
        else:
            dxp = x[i+1] - x[i]
            dxm = x[i]   - x[i-1]
            df[i] = (dxp*(f[i] - f[i-1]) + dxm*(f[i+1] - f[i])) / (dxp*dxm*(dxp+dxm))
    return df


def find_source_npz(src_dir):
    """
    指定フォルダから元npzを探す
    - *.npz を全部拾う
    - 'merged' や 'field' を含むファイルを優先
    """
    pattern = os.path.join(src_dir, "*.npz")
    cand = glob.glob(pattern)
    if not cand:
        raise FileNotFoundError(f"{src_dir} に npz が見つかりません")

    def score(path):
        name = os.path.basename(path)
        s = 0
        if "merged" in name: s += 2
        if "field" in name:  s += 1
        return -s

    cand_sorted = sorted(cand, key=score)
    chosen = cand_sorted[0]
    print(f"  使用する元 npz: {chosen}")
    return chosen


def ensure_grid_from_npz(npz_path):
    """
    元npzから統一フォーマットのグリッドデータを構成

    パターン1: X,Y,Z (1D), V, Ex,Ey,Ez (Nz,Ny,Nx)
    パターン2: x,y,z,V,Ex,Ey,Ez (全て1D) → grid化
    """
    d = np.load(npz_path)

    # パターン1: 既にグリッド形式
    if all(k in d.files for k in ["X", "Y", "Z", "V"]):
        X = d["X"]; Y = d["Y"]; Z = d["Z"]
        V = d["V"]
        Ex = d["Ex"] if "Ex" in d.files else None
        Ey = d["Ey"] if "Ey" in d.files else None
        Ez = d["Ez"] if "Ez" in d.files else None
        return X, Y, Z, V, Ex, Ey, Ez

    # パターン2: フラット形式
    if not all(k in d.files for k in ["x", "y", "z", "V"]):
        raise KeyError(f"{npz_path} に x,y,z,V または X,Y,Z,V が含まれていません")

    x = d["x"]; y = d["y"]; z = d["z"]
    V_flat  = d["V"]
    Ex_flat = d["Ex"] if "Ex" in d.files else None
    Ey_flat = d["Ey"] if "Ey" in d.files else None
    Ez_flat = d["Ez"] if "Ez" in d.files else None

    Xu = np.unique(x); Yu = np.unique(y); Zu = np.unique(z)
    Nx, Ny, Nz = len(Xu), len(Yu), len(Zu)

    ix = np.searchsorted(Xu, x)
    iy = np.searchsorted(Yu, y)
    iz = np.searchsorted(Zu, z)

    V = np.empty((Nz, Ny, Nx), dtype=float)
    V[iz, iy, ix] = V_flat

    def fill3d(flat):
        if flat is None:
            return None
        F3 = np.empty_like(V)
        F3[iz, iy, ix] = flat
        return F3

    Ex3 = fill3d(Ex_flat)
    Ey3 = fill3d(Ey_flat)
    Ez3 = fill3d(Ez_flat)

    return Xu, Yu, Zu, V, Ex3, Ey3, Ez3


def recompute_E_from_V(X, Y, Z, V):
    """V から E = -∇V を計算（非一様格子対応）"""
    Nz, Ny, Nx = V.shape
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)
    dVdz = np.zeros_like(V)

    # x方向
    for k in range(Nz):
        for j in range(Ny):
            dVdx[k, j, :] = diff_nonuniform_1d(V[k, j, :], X)

    # y方向
    for k in range(Nz):
        for i in range(Nx):
            dVdy[k, :, i] = diff_nonuniform_1d(V[k, :, i], Y)

    # z方向
    for j in range(Ny):
        for i in range(Nx):
            dVdz[:, j, i] = diff_nonuniform_1d(V[:, j, i], Z)

    Ex = -dVdx
    Ey = -dVdy
    Ez = -dVdz
    return Ex, Ey, Ez


def plot_field_at_430um(npz_path, out_fig_dir, label):
    """z=430μm位置の電場分布図を出力"""
    os.makedirs(out_fig_dir, exist_ok=True)

    print(f"\n=== 電場プロット: {label} ===")
    data = np.load(npz_path)

    X = data["X"]; Y = data["Y"]; Z = data["Z"]
    V = data["V"]; Ex = data["Ex"]; Ey = data["Ey"]; Ez = data["Ez"]

    # z=430μm に最も近いインデックスを探す
    target_z_um = 430.0
    target_z = target_z_um * 1e-6  # m
    iz = int(np.argmin(np.abs(Z - target_z)))
    actual_z = Z[iz] * 1e6  # μm

    print(f"  z = {actual_z:.2f} μm (index {iz})")

    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # |E| 分布図
    plt.figure(figsize=(8, 6))
    plt.imshow(Emag[iz, :, :], origin="lower",
               extent=[X.min()*1e6, X.max()*1e6, Y.min()*1e6, Y.max()*1e6],
               aspect="auto", cmap='viridis')
    plt.colorbar(label='|E| [V/m]')
    plt.title(f'{label}: |E| at z={actual_z:.2f} μm')
    plt.xlabel('x [μm]')
    plt.ylabel('y [μm]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f'{label}_Emag_z430um.png'), dpi=150)
    plt.close()

    # ベクトル図
    xx, yy = np.meshgrid(X * 1e6, Y * 1e6)  # μm単位
    step = max(1, len(X) // 20)

    plt.figure(figsize=(8, 6))
    plt.quiver(xx[::step, ::step], yy[::step, ::step],
               Ex[iz, ::step, ::step], Ey[iz, ::step, ::step],
               color='blue', alpha=0.7)
    plt.title(f'{label}: E vector at z={actual_z:.2f} μm')
    plt.xlabel('x [μm]')
    plt.ylabel('y [μm]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f'{label}_Evector_z430um.png'), dpi=150)
    plt.close()

    print(f"  保存: {out_fig_dir}/{label}_*_z430um.png")


def main():
    print("\n" + "="*70)
    print("OpenSTF Field Data Converter")
    print("="*70)
    print(f"Current directory: {os.getcwd()}")

    os.makedirs(OUT_DIR, exist_ok=True)

    for case in CASES:
        label   = case["label"]
        src_dir = case["src_dir"]

        print(f"\n{'='*70}")
        print(f"Processing: {label} ({src_dir})")
        print('='*70)

        # 1) 元npzを探す
        src_npz = find_source_npz(src_dir)

        # 2) グリッド形式に統一
        X, Y, Z, V, Ex_orig, Ey_orig, Ez_orig = ensure_grid_from_npz(src_npz)
        print(f"  Grid: {len(X)} x {len(Y)} x {len(Z)}")
        print(f"  X: [{X.min()*1e6:.2f}, {X.max()*1e6:.2f}] μm")
        print(f"  Y: [{Y.min()*1e6:.2f}, {Y.max()*1e6:.2f}] μm")
        print(f"  Z: [{Z.min()*1e6:.2f}, {Z.max()*1e6:.2f}] μm")

        # 3) V から E を再計算
        Ex, Ey, Ez = recompute_E_from_V(X, Y, Z, V)

        # 4) 保存
        out_npz_name = f"{label}_field.npz"
        out_npz_path = os.path.join(OUT_DIR, out_npz_name)
        np.savez_compressed(
            out_npz_path,
            X=X, Y=Y, Z=Z,
            V=V, Ex=Ex, Ey=Ey, Ez=Ez
        )
        print(f"  ✓ 保存: {out_npz_path}")

        # 5) z=430μm の電場プロット
        fig_dir = os.path.join(OUT_DIR, f"figures_{label}")
        plot_field_at_430um(out_npz_path, fig_dir, label)

    print("\n" + "="*70)
    print("✓ 全ケース処理完了")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
