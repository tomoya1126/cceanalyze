#!/usr/bin/env python3
"""
cce 配下の
  - くし形電界/
  - 横型電界/
にある「元 npz」から

  cce/電界/

の中に
  - kushigata_field.npz
  - yokogata_field.npz

を作成し、それぞれの電界の妥当性検証 + 可視化まで
一括で行うスクリプト。
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==============================
#  基本設定（必要なら名前を修正）
# ==============================

CASES = [
    {
        "label": "kushigata",
        "src_dir": "くし形電界",  # cce 直下のフォルダ名
    },
    {
        "label": "yokogata",
        "src_dir": "横型電界",    # cce 直下のフォルダ名
    },
]

OUT_DIR = "電界"  # 結果をまとめて保存するフォルダ


# ==============================
#  ユーティリティ
# ==============================

def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a*a)))

def diff_nonuniform_1d(f, x):
    """
    非一様格子上の1次元微分（中央差分・端は片側差分）。
    f: (N,)  x: (N,)
    """
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
            # 非一様格子用の2次精度中心差分
            df[i] = (dxp*(f[i] - f[i-1]) + dxm*(f[i+1] - f[i])) / (dxp*dxm*(dxp+dxm))
    return df


def find_source_npz(src_dir):
    """
    指定フォルダの中から「元 npz」を1つ選ぶ。
    ルール：
      - *.npz を全部拾う
      - 'merged' や 'field' を含むファイルを優先
      - それでも1つに決まらなければ最初の1個
    """
    pattern = os.path.join(src_dir, "*.npz")
    cand = glob.glob(pattern)
    if not cand:
        raise FileNotFoundError(f"{src_dir} に npz が見つかりません")

    # 優先度付きソート
    def score(path):
        name = os.path.basename(path)
        s = 0
        if "merged" in name: s += 2
        if "field" in name:  s += 1
        return -s  # 大きいもの優先（ソートでは小さい順なのでマイナス）
    cand_sorted = sorted(cand, key=score)

    chosen = cand_sorted[0]
    print(f"  使用する元 npz: {chosen}")
    return chosen


def ensure_grid_from_npz(npz_path):
    """
    元 npz から統一フォーマットのグリッドデータを構成する。

    期待する元パターン：
      1) 既にグリッド化されている：
         X,Y,Z (1D), V, Ex,Ey,Ez (Nz,Ny,Nx)
      2) フラット形式：
         x,y,z,V,Ex,Ey,Ez (全て1D, 長さ N) → grid に並べ替え

    戻り値:
      X,Y,Z,V,Ex,Ey,Ez （X,Y,Z は1D、他は3D）
    """
    d = np.load(npz_path)

    # パターン1: 既にグリッド形式
    if all(k in d.files for k in ["X", "Y", "Z", "V"]):
        X = d["X"]; Y = d["Y"]; Z = d["Z"]
        V = d["V"]
        # E がない場合は None
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

    # インデックスに変換
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
    """
    V から非一様格子用の中央差分で E = -∇V を計算。
    """
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


def validate_and_plot(npz_path, out_fig_dir, label):
    """
    field_xxx.npz を読み込み、妥当性検証 + 代表断面の画像保存。
    追加: z=430 µm 付近の断面（V, |E|, curlE, LapV, quiver）も出力
    """
    os.makedirs(out_fig_dir, exist_ok=True)

    print(f"\n=== 検証開始: {label} ({npz_path}) ===")
    data = np.load(npz_path)

    X = data["X"]; Y = data["Y"]; Z = data["Z"]
    V = data["V"]; Ex = data["Ex"]; Ey = data["Ey"]; Ez = data["Ez"]

    Nz, Ny, Nx = V.shape

    # --- -∇V を再計算して E と比較 ---
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)
    dVdz = np.zeros_like(V)

    for k in range(Nz):
        for j in range(Ny):
            dVdx[k, j, :] = diff_nonuniform_1d(V[k, j, :], X)
    for k in range(Nz):
        for i in range(Nx):
            dVdy[k, :, i] = diff_nonuniform_1d(V[k, :, i], Y)
    for j in range(Ny):
        for i in range(Nx):
            dVdz[:, j, i] = diff_nonuniform_1d(V[:, j, i], Z)

    Ex_err = Ex + dVdx
    Ey_err = Ey + dVdy
    Ez_err = Ez + dVdz

    print("[E vs -∇V] RMS abs error:")
    print(f"  Ex = {rms(Ex_err):.3e}")
    print(f"  Ey = {rms(Ey_err):.3e}")
    print(f"  Ez = {rms(Ez_err):.3e}")

    print("\n[E vs -∇V] RMS rel error:")
    print(f"  Ex = {rms(Ex_err)/max(rms(Ex),1e-30):.3e}")
    print(f"  Ey = {rms(Ey_err)/max(rms(Ey),1e-30):.3e}")
    print(f"  Ez = {rms(Ez_err)/max(rms(Ez),1e-30):.3e}")

    # --- curl(E) ---
    curlx = np.zeros_like(V)
    curly = np.zeros_like(V)
    curlz = np.zeros_like(V)

    # ∂Ez/∂y - ∂Ey/∂z
    for k in range(Nz):
        for i in range(Nx):
            dEzdy = diff_nonuniform_1d(Ez[k, :, i], Y)
            curlx[k, :, i] = dEzdy
    for j in range(Ny):
        for i in range(Nx):
            dEydz = diff_nonuniform_1d(Ey[:, j, i], Z)
            curlx[:, j, i] -= dEydz

    # ∂Ex/∂z - ∂Ez/∂x
    for j in range(Ny):
        for i in range(Nx):
            dExdz = diff_nonuniform_1d(Ex[:, j, i], Z)
            curly[:, j, i] = dExdz
    for k in range(Nz):
        for j in range(Ny):
            dEzdx = diff_nonuniform_1d(Ez[k, j, :], X)
            curly[k, j, :] -= dEzdx

    # ∂Ey/∂x - ∂Ex/∂y
    for k in range(Nz):
        for j in range(Ny):
            dEydx = diff_nonuniform_1d(Ey[k, j, :], X)
            curlz[k, j, :] = dEydx
    for k in range(Nz):
        for i in range(Nx):
            dExdy = diff_nonuniform_1d(Ex[k, :, i], Y)
            curlz[k, :, i] -= dExdy

    curl_mag = np.sqrt(curlx**2 + curly**2 + curlz**2)
    print(f"\n[curl E] RMS(|curl E|) = {rms(curl_mag):.3e}")

    # --- Laplacian(V) ---
    lapV = np.zeros_like(V)
    for k in range(Nz):
        for j in range(Ny):
            lapV[k, j, :] += diff_nonuniform_1d(dVdx[k, j, :], X)
    for k in range(Nz):
        for i in range(Nx):
            lapV[k, :, i] += diff_nonuniform_1d(dVdy[k, :, i], Y)
    for j in range(Ny):
        for i in range(Nx):
            lapV[:, j, i] += diff_nonuniform_1d(dVdz[:, j, i], Z)

    print(f"[Laplacian V] RMS(∇²V) = {rms(lapV):.3e}")

    # --- E 統計 ---
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    p50, p90, p99 = np.percentile(Emag, [50, 90, 99])
    print("\n[|E| Statistics]")
    print(f"  max  = {np.max(Emag):.3e}")
    print(f"  mean = {np.mean(Emag):.3e}")
    print(f"  p50  = {p50:.3e}, p90 = {p90:.3e}, p99 = {p99:.3e}")

    # ---------- 可視化 ----------
    def imsave(field2d, title, fname):
        plt.figure(figsize=(6,5))
        plt.imshow(field2d, origin="lower",
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_fig_dir, fname))
        plt.close()

    # 中央 z 断面
    midz = Nz // 2
    imsave(V[midz,:,:],       f"V (XY, z index={midz})",       "V_xy_mid.png")
    imsave(Emag[midz,:,:],    f"|E| (XY, z index={midz})",     "Emag_xy_mid.png")
    imsave(curl_mag[midz,:,:],f"|curl E| (XY, z index={midz})","curl_xy_mid.png")
    imsave(lapV[midz,:,:],    f"Laplacian(V) (XY, z={midz})",  "lapV_xy_mid.png")

    # ★ ここから z = 430 µm 付近の断面 ★
    target_z_um = 430.0
    target_z = target_z_um * 1e-6  # m に変換
    iz_430 = int(np.argmin(np.abs(Z - target_z)))
    print(f"\n[Slice] z ≈ {target_z_um} µm (index {iz_430}, Z = {Z[iz_430]:.3e} m)")

    imsave(V[iz_430,:,:],
           f"V (XY, z≈{target_z_um} µm)",
           f"V_xy_z{int(target_z_um)}um.png")
    imsave(Emag[iz_430,:,:],
           f"|E| (XY, z≈{target_z_um} µm)",
           f"Emag_xy_z{int(target_z_um)}um.png")

    imsave(curl_mag[iz_430,:,:],
           f"|curl E| (XY, z≈{target_z_um} µm)",
           f"curl_xy_z{int(target_z_um)}um.png")
    imsave(lapV[iz_430,:,:],
           f"Laplacian(V) (XY, z≈{target_z_um} µm)",
           f"lapV_xy_z{int(target_z_um)}um.png")

    # ベクトル図（Ex,Ey） z = 430 µm
    xx, yy = np.meshgrid(X, Y)
    plt.figure(figsize=(6,5))
    step = max(1, len(X)//32)
    plt.quiver(xx[::step,::step], yy[::step,::step],
               Ex[iz_430,::step,::step], Ey[iz_430,::step,::step],
               color="blue", alpha=0.7)
    plt.title(f"E vectors (XY, z≈{target_z_um} µm)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir,
                             f"E_quiver_xy_z{int(target_z_um)}um.png"))
    plt.close()

    print(f"\n  図は {out_fig_dir} に保存しました。")



    X = data["X"]; Y = data["Y"]; Z = data["Z"]
    V = data["V"]; Ex = data["Ex"]; Ey = data["Ey"]; Ez = data["Ez"]

    Nz, Ny, Nx = V.shape

    # --- -∇V を再計算して E と比較 ---
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)
    dVdz = np.zeros_like(V)

    for k in range(Nz):
        for j in range(Ny):
            dVdx[k, j, :] = diff_nonuniform_1d(V[k, j, :], X)
    for k in range(Nz):
        for i in range(Nx):
            dVdy[k, :, i] = diff_nonuniform_1d(V[k, :, i], Y)
    for j in range(Ny):
        for i in range(Nx):
            dVdz[:, j, i] = diff_nonuniform_1d(V[:, j, i], Z)

    Ex_err = Ex + dVdx
    Ey_err = Ey + dVdy
    Ez_err = Ez + dVdz

    print("[E vs -∇V] RMS abs error:")
    print(f"  Ex = {rms(Ex_err):.3e}")
    print(f"  Ey = {rms(Ey_err):.3e}")
    print(f"  Ez = {rms(Ez_err):.3e}")

    print("\n[E vs -∇V] RMS rel error:")
    print(f"  Ex = {rms(Ex_err)/max(rms(Ex),1e-30):.3e}")
    print(f"  Ey = {rms(Ey_err)/max(rms(Ey),1e-30):.3e}")
    print(f"  Ez = {rms(Ez_err)/max(rms(Ez),1e-30):.3e}")

    # --- curl(E) ---
    curlx = np.zeros_like(V)
    curly = np.zeros_like(V)
    curlz = np.zeros_like(V)

    # ∂Ez/∂y - ∂Ey/∂z
    for k in range(Nz):
        for i in range(Nx):
            dEzdy = diff_nonuniform_1d(Ez[k, :, i], Y)
            curlx[k, :, i] = dEzdy
    for j in range(Ny):
        for i in range(Nx):
            dEydz = diff_nonuniform_1d(Ey[:, j, i], Z)
            curlx[:, j, i] -= dEydz

    # ∂Ex/∂z - ∂Ez/∂x
    for j in range(Ny):
        for i in range(Nx):
            dExdz = diff_nonuniform_1d(Ex[:, j, i], Z)
            curly[:, j, i] = dExdz
    for k in range(Nz):
        for j in range(Ny):
            dEzdx = diff_nonuniform_1d(Ez[k, j, :], X)
            curly[k, j, :] -= dEzdx

    # ∂Ey/∂x - ∂Ex/∂y
    for k in range(Nz):
        for j in range(Ny):
            dEydx = diff_nonuniform_1d(Ey[k, j, :], X)
            curlz[k, j, :] = dEydx
    for k in range(Nz):
        for i in range(Nx):
            dExdy = diff_nonuniform_1d(Ex[k, :, i], Y)
            curlz[k, :, i] -= dExdy

    curl_mag = np.sqrt(curlx**2 + curly**2 + curlz**2)
    print(f"\n[curl E] RMS(|curl E|) = {rms(curl_mag):.3e}")

    # --- Laplacian(V) ---
    lapV = np.zeros_like(V)
    for k in range(Nz):
        for j in range(Ny):
            lapV[k, j, :] += diff_nonuniform_1d(dVdx[k, j, :], X)
    for k in range(Nz):
        for i in range(Nx):
            lapV[k, :, i] += diff_nonuniform_1d(dVdy[k, :, i], Y)
    for j in range(Ny):
        for i in range(Nx):
            lapV[:, j, i] += diff_nonuniform_1d(dVdz[:, j, i], Z)

    print(f"[Laplacian V] RMS(∇²V) = {rms(lapV):.3e}")

    # --- E 統計 ---
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    p50, p90, p99 = np.percentile(Emag, [50, 90, 99])
    print("\n[|E| Statistics]")
    print(f"  max  = {np.max(Emag):.3e}")
    print(f"  mean = {np.mean(Emag):.3e}")
    print(f"  p50  = {p50:.3e}, p90 = {p90:.3e}, p99 = {p99:.3e}")

    # --- XY 断面（中央 z）を描画 ---
    midz = Nz // 2

    def imsave(field2d, title, fname):
        plt.figure(figsize=(6,5))
        plt.imshow(field2d, origin="lower",
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_fig_dir, fname))
        plt.close()

    imsave(V[midz,:,:],       f"V (XY, z index={midz})",      "V_xy.png")
    imsave(Emag[midz,:,:],    f"|E| (XY, z index={midz})",    "Emag_xy.png")
    imsave(curl_mag[midz,:,:],f"|curl E| (XY, z index={midz})","curl_xy.png")
    imsave(lapV[midz,:,:],    f"Laplacian(V) (XY, z={midz})", "lapV_xy.png")

    # --- ベクトル図（Ex,Ey） ---
    xx, yy = np.meshgrid(X, Y)
    plt.figure(figsize=(6,5))
    step = max(1, len(X)//32)  # 間引き
    plt.quiver(xx[::step,::step], yy[::step,::step],
               Ex[midz,::step,::step], Ey[midz,::step,::step],
               color="blue", alpha=0.7)
    plt.title(f"E vectors (XY, z index={midz})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, "E_quiver_xy.png"))
    plt.close()

    print(f"\n  図は {out_fig_dir} に保存しました。")
    print(f"=== 検証終了: {label} ===\n")


def main():
    # cce 直下で走らせる前提
    here = os.getcwd()
    print("Current directory:", here)

    os.makedirs(OUT_DIR, exist_ok=True)

    for case in CASES:
        label   = case["label"]
        src_dir = case["src_dir"]

        print(f"\n=== ケース処理開始: {label} ({src_dir}) ===")

        # 1) 元 npz を選ぶ
        src_npz = find_source_npz(src_dir)

        # 2) グリッド形式に統一
        X, Y, Z, V, Ex_orig, Ey_orig, Ez_orig = ensure_grid_from_npz(src_npz)

        # 3) V から E を再計算（原則はこちらを採用）
        Ex, Ey, Ez = recompute_E_from_V(X, Y, Z, V)

        # 4) 結果を cce/電界 に保存
        out_npz_name = f"{label}_field.npz"
        out_npz_path = os.path.join(OUT_DIR, out_npz_name)
        np.savez_compressed(
            out_npz_path,
            X=X, Y=Y, Z=Z,
            V=V, Ex=Ex, Ey=Ey, Ez=Ez
        )
        print(f"  保存しました: {out_npz_path}")

        # 5) 妥当性検証 + 可視化
        fig_dir = os.path.join(OUT_DIR, f"figures_{label}")
        validate_and_plot(out_npz_path, fig_dir, label)

    print("\n=== 全ケース処理完了 ===")


if __name__ == "__main__":
    main()
