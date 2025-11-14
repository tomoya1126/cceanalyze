#!/usr/bin/env python3
"""
OpenSTF電界データを読み込んで「共通NPZフォーマット」に変換するスクリプト（GUI版）

処理内容：
1. GUI で元 npz ファイルをユーザーに選択させる
   - 対応フォーマット：
     - 旧形式1: X, Y, Z (1D), V(3D)
     - 旧形式2: x, y, z, V (全て1Dフラット)
     - 新形式:  x, y, z, potential (3D)
2. グリッド形式（x, y, z, V）に統一
3. V から E = -∇V を再計算（非一様格子対応）
4. GUI で指定された出力パスに npz として保存：
   - メタデータ: title, nx, ny, nz, eps0, voltages, dielectrics, charges, iteration_steps, residuals, iteration_indices
   - グリッド:   x, y, z, potential
   - 互換用:     X, Y, Z, V
   - 電場:       Ex, Ey, Ez
5. 出力と同じフォルダに figures/ を作成し、z=430μm 位置の電場分布図（|E| とベクトル図）を保存
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


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


def ensure_grid_from_npz_dict(d):
    """
    入力 npz（dict様オブジェクト）からグリッド形式の (x, y, z, V, Ex, Ey, Ez) を構成する。

    対応パターン：
    - パターンA: 新形式
        x, y, z (1D), potential (3D)
    - パターンB: 旧グリッド形式
        X, Y, Z (1D), V (3D)
    - パターンC: フラット形式
        x, y, z, V (全て1D) → grid 化
    """

    files = set(d.files)

    # パターンA: 新形式（指定フォーマット）
    if {"x", "y", "z", "potential"} <= files:
        x = d["x"]
        y = d["y"]
        z = d["z"]
        V = d["potential"]
        Ex = d["Ex"] if "Ex" in files else None
        Ey = d["Ey"] if "Ey" in files else None
        Ez = d["Ez"] if "Ez" in files else None
        return x, y, z, V, Ex, Ey, Ez

    # パターンB: 既にグリッド形式 (X,Y,Z,V)
    if {"X", "Y", "Z", "V"} <= files:
        X = d["X"]
        Y = d["Y"]
        Z = d["Z"]
        V = d["V"]
        Ex = d["Ex"] if "Ex" in files else None
        Ey = d["Ey"] if "Ey" in files else None
        Ez = d["Ez"] if "Ez" in files else None
        # このスクリプト内では x,y,z に統一したいので名前だけ変える
        return X, Y, Z, V, Ex, Ey, Ez

    # パターンC: フラット形式 (x,y,z,V) をグリッド化
    if not {"x", "y", "z", "V"} <= files:
        raise KeyError("npz ファイルに x,y,z, V または X,Y,Z,V, potential が含まれていません。")

    x_flat = d["x"]
    y_flat = d["y"]
    z_flat = d["z"]
    V_flat = d["V"]
    Ex_flat = d["Ex"] if "Ex" in files else None
    Ey_flat = d["Ey"] if "Ey" in files else None
    Ez_flat = d["Ez"] if "Ez" in files else None

    xu = np.unique(x_flat)
    yu = np.unique(y_flat)
    zu = np.unique(z_flat)
    Nx, Ny, Nz = len(xu), len(yu), len(zu)

    ix = np.searchsorted(xu, x_flat)
    iy = np.searchsorted(yu, y_flat)
    iz = np.searchsorted(zu, z_flat)

    V = np.empty((Nz, Ny, Nx), dtype=float)
    V[iz, iy, ix] = V_flat

    def fill3d(flat):
        if flat is None:
            return None
        F3 = np.empty_like(V)
        F3[iz, iy, ix] = flat
        return F3

    Ex = fill3d(Ex_flat)
    Ey = fill3d(Ey_flat)
    Ez = fill3d(Ez_flat)

    return xu, yu, zu, V, Ex, Ey, Ez


def recompute_E_from_V(x, y, z, V):
    """V から E = -∇V を計算（非一様格子対応）"""
    Nz, Ny, Nx = V.shape
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)
    dVdz = np.zeros_like(V)

    # x方向
    for k in range(Nz):
        for j in range(Ny):
            dVdx[k, j, :] = diff_nonuniform_1d(V[k, j, :], x)

    # y方向
    for k in range(Nz):
        for i in range(Nx):
            dVdy[k, :, i] = diff_nonuniform_1d(V[k, :, i], y)

    # z方向
    for j in range(Ny):
        for i in range(Nx):
            dVdz[:, j, i] = diff_nonuniform_1d(V[:, j, i], z)

    Ex = -dVdx
    Ey = -dVdy
    Ez = -dVdz
    return Ex, Ey, Ez


def plot_field_at_430um(x, y, z, Ex, Ey, Ez, out_fig_dir, label):
    """z=430μm位置の電場分布図を出力"""
    os.makedirs(out_fig_dir, exist_ok=True)

    print(f"\n=== 電場プロット: {label} ===")

    target_z_um = 430.0
    target_z = target_z_um * 1e-6  # m
    iz = int(np.argmin(np.abs(z - target_z)))
    actual_z = z[iz] * 1e6  # μm

    print(f"  z = {actual_z:.2f} μm (index {iz})")

    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # |E| 分布図
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Emag[iz, :, :],
        origin="lower",
        extent=[x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="|E| [V/m]")
    plt.title(f"{label}: |E| at z={actual_z:.2f} μm")
    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f"{label}_Emag_z430um.png"), dpi=150)
    plt.close()

    # ベクトル図
    xx, yy = np.meshgrid(x * 1e6, y * 1e6)  # μm単位
    step = max(1, len(x) // 20)

    plt.figure(figsize=(8, 6))
    plt.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        Ex[iz, ::step, ::step],
        Ey[iz, ::step, ::step],
        color="blue",
        alpha=0.7,
    )
    plt.title(f"{label}: E vector at z={actual_z:.2f} μm")
    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f"{label}_Evector_z430um.png"), dpi=150)
    plt.close()

    print(f"  保存: {out_fig_dir}/{label}_*_z430um.png")


# ==============================
#  GUI アプリ本体
# ==============================

class FieldConverterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("OpenSTF 電界データ NPZ 変換ツール")

        self.input_npz_path = None
        self.output_npz_path = None

        self._build_widgets()

    def _build_widgets(self):
        btn_in = tk.Button(self.master, text="入力 NPZ ファイルを選択", command=self.select_input_npz)
        btn_in.pack(padx=10, pady=10, fill="x")

        self.label_in = tk.Label(self.master, text="入力: 未選択")
        self.label_in.pack(padx=10, pady=5, anchor="w")

        self.text_info = tk.Text(self.master, height=10, width=70)
        self.text_info.pack(padx=10, pady=5)
        self.text_info.insert("1.0", "まず入力 NPZ ファイルを選んでください。\n")
        self.text_info.config(state="disabled")

        btn_out = tk.Button(self.master, text="出力先とファイル名を選択", command=self.select_output_npz)
        btn_out.pack(padx=10, pady=10, fill="x")

        self.label_out = tk.Label(self.master, text="出力: 未選択")
        self.label_out.pack(padx=10, pady=5, anchor="w")

        btn_run = tk.Button(self.master, text="変換実行", command=self.run_conversion)
        btn_run.pack(padx=10, pady=15, fill="x")

    def select_input_npz(self):
        path = filedialog.askopenfilename(
            title="入力 NPZ ファイルを選択",
            filetypes=[("NumPy NPZ", "*.npz"), ("すべてのファイル", "*.*")]
        )
        if not path:
            return

        try:
            d = np.load(path)
        except Exception as e:
            messagebox.showerror("エラー", f"NPZ ファイルを開けませんでした。\n{e}")
            return

        try:
            x, y, z, V, Ex0, Ey0, Ez0 = ensure_grid_from_npz_dict(d)
        except Exception as e:
            messagebox.showerror("エラー", f"NPZ の形式が想定と異なります。\n{e}")
            return

        self.input_npz_path = path
        self.label_in.config(text=f"入力: {self.input_npz_path}")

        info_lines = []
        title = str(d["title"]) if "title" in d.files else os.path.basename(path)
        info_lines.append(f"title: {title}")
        info_lines.append(f"x.shape = {x.shape}")
        info_lines.append(f"y.shape = {y.shape}")
        info_lines.append(f"z.shape = {z.shape}")
        info_lines.append(f"V.shape = {V.shape}")
        info_lines.append(f"V range: [{V.min():.3e}, {V.max():.3e}] V")
        if "eps0" in d.files:
            info_lines.append(f"eps0 (from file) = {float(d['eps0']):.5e} F/m")

        self.text_info.config(state="normal")
        self.text_info.delete("1.0", "end")
        self.text_info.insert("1.0", "\n".join(info_lines))
        self.text_info.config(state="disabled")

    def select_output_npz(self):
        default_name = "field_converted.npz"
        path = filedialog.asksaveasfilename(
            title="出力 NPZ ファイルを指定",
            defaultextension=".npz",
            initialfile=default_name,
            filetypes=[("NumPy NPZ", "*.npz"), ("すべてのファイル", "*.*")]
        )
        if not path:
            return

        self.output_npz_path = path
        self.label_out.config(text=f"出力: {self.output_npz_path}")

    def run_conversion(self):
        if self.input_npz_path is None:
            messagebox.showwarning("警告", "入力 NPZ ファイルを先に選択してください。")
            return
        if self.output_npz_path is None:
            messagebox.showwarning("警告", "出力先とファイル名を指定してください。")
            return

        try:
            d = np.load(self.input_npz_path)
            x, y, z, V, Ex0, Ey0, Ez0 = ensure_grid_from_npz_dict(d)
        except Exception as e:
            messagebox.showerror("エラー", f"入力ファイルの読み込みに失敗しました。\n{e}")
            return

        title = str(d["title"]) if "title" in d.files else os.path.basename(self.input_npz_path)
        eps0 = float(d["eps0"]) if "eps0" in d.files else 8.8541878128e-12  # [F/m]

        voltages = d["voltages"] if "voltages" in d.files else np.array([], dtype=float)
        dielectrics = d["dielectrics"] if "dielectrics" in d.files else np.array([], dtype=float)
        charges = d["charges"] if "charges" in d.files else np.array([], dtype=float)
        iteration_steps = d["iteration_steps"] if "iteration_steps" in d.files else np.array([], dtype=int)
        residuals = d["residuals"] if "residuals" in d.files else np.array([], dtype=float)
        iteration_indices = d["iteration_indices"] if "iteration_indices" in d.files else np.array([], dtype=int)

        Nz, Ny, Nx = V.shape
        nx = Nx - 1
        ny = Ny - 1
        nz = Nz - 1

        print("\n" + "="*70)
        print("変換開始")
        print("="*70)
        print(f"入力ファイル: {self.input_npz_path}")
        print(f"出力ファイル: {self.output_npz_path}")
        print(f"Grid: {Nx} x {Ny} x {Nz} (points) → nx,ny,nz = {nx}, {ny}, {nz}")
        print(f"x: [{x.min():.3e}, {x.max():.3e}] m")
        print(f"y: [{y.min():.3e}, {y.max():.3e}] m")
        print(f"z: [{z.min():.3e}, {z.max():.3e}] m")

        print("E = -∇V を再計算中...")
        Ex, Ey, Ez = recompute_E_from_V(x, y, z, V)

        out_dir = os.path.dirname(self.output_npz_path)
        os.makedirs(out_dir, exist_ok=True)

        # ★ここで X,Y,Z,V も出力（X,Y,Z は x,y,z と同じ1D配列）
        np.savez_compressed(
            self.output_npz_path,
            title=title,
            nx=nx,
            ny=ny,
            nz=nz,
            eps0=eps0,
            # 新仕様
            x=x,
            y=y,
            z=z,
            potential=V,
            # 旧仕様互換
            X=x,
            Y=y,
            Z=z,
            V=V,
            # 電場
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            # 収束・その他
            voltages=voltages,
            dielectrics=dielectrics,
            charges=charges,
            iteration_steps=iteration_steps,
            residuals=residuals,
            iteration_indices=iteration_indices,
        )
        print(f"  ✓ 保存: {self.output_npz_path}")

        fig_dir = os.path.join(out_dir, "figures")
        base_label = os.path.splitext(os.path.basename(self.output_npz_path))[0]
        plot_field_at_430um(x, y, z, Ex, Ey, Ez, fig_dir, base_label)

        print("="*70)
        print("✓ 変換完了")
        print("="*70)

        messagebox.showinfo("完了", f"変換が完了しました。\n{self.output_npz_path}")


def main():
    root = tk.Tk()
    app = FieldConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
