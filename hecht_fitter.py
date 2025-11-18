#!/usr/bin/env python3
"""
Hecht Equation Fitting Tool

電界データとSRIMデータから、ヘクトの式を使ってμτをフィッティングするGUIツール。
入射方向: +Y方向から-Y方向（深さ方向）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Tuple, List
import threading
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ========== 物理定数 ==========
E_PAIR = 7.8  # eV - 電子正孔対生成エネルギー
Q_E = 1.602e-19  # C - 電気素量


class FieldData:
    """電界データの管理クラス"""

    def __init__(self, field_file: str):
        """
        電界データを読み込む

        Parameters
        ----------
        field_file : str
            電界データのnpzファイルパス
        """
        data = np.load(field_file)

        self.X = data['X']  # 1D array [m]
        self.Y = data['Y']  # 1D array [m] (深さ方向)
        self.Z = data['Z']  # 1D array [m]
        self.Ex = data['Ex']  # 3D array (nz, ny, nx) [V/m]
        self.Ey = data['Ey']  # 3D array [V/m]
        self.Ez = data['Ez']  # 3D array [V/m]

        # 電界強度を計算
        self.E_mag = np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)

    def get_info(self) -> str:
        """データ情報を文字列で返す"""
        info = []
        info.append(f"Grid size: Nx={len(self.X)}, Ny={len(self.Y)}, Nz={len(self.Z)}")
        info.append(f"X range: [{self.X.min()*1e6:.2f}, {self.X.max()*1e6:.2f}] um")
        info.append(f"Y range (depth): [{self.Y.min()*1e6:.2f}, {self.Y.max()*1e6:.2f}] um")
        info.append(f"Z range: [{self.Z.min()*1e6:.2f}, {self.Z.max()*1e6:.2f}] um")
        info.append(f"E_mag range: [{self.E_mag.min():.2e}, {self.E_mag.max():.2e}] V/m")
        return "\n".join(info)


class SRIMData:
    """SRIMデータの管理クラス"""

    def __init__(self, srim_file: str):
        """
        SRIMファイルを読み込む

        Parameters
        ----------
        srim_file : str
            SRIMデータファイルパス
        """
        data = self._load_srim_file(srim_file)

        # 深さとイオン化エネルギー密度
        self.depth_angstrom = data[:, 0]  # Angstrom
        self.ionization_eV_per_angstrom = data[:, 1]  # eV/Angstrom

        # 単位変換: Angstrom -> m
        self.depth_m = self.depth_angstrom * 1e-10

    def get_info(self) -> str:
        """データ情報を文字列で返す"""
        info = []
        info.append(f"Depth range: [0, {self.depth_m[-1]*1e6:.2f}] um")
        info.append(f"Max ionization: {self.ionization_eV_per_angstrom.max():.2f} eV/Angstrom")
        info.append(f"Number of points: {len(self.depth_angstrom)}")
        return "\n".join(info)

    def _load_srim_file(self, filename: str) -> np.ndarray:
        """SRIMファイルを読み込む（ヘッダー処理付き）"""
        data_rows = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        depth = float(parts[0])
                        ionization = float(parts[1])
                        data_rows.append([depth, ionization])
                    except ValueError:
                        continue

        if not data_rows:
            raise ValueError(f"No valid data found in {filename}")

        return np.array(data_rows)


class HechtAnalyzer:
    """ヘクト式による解析クラス"""

    def __init__(self, field_data: FieldData, srim_data: SRIMData):
        """
        Parameters
        ----------
        field_data : FieldData
            電界データ
        srim_data : SRIMData
            SRIMデータ
        """
        self.field = field_data
        self.srim = srim_data

    def calculate_cce_vs_depth(self, ix: int, iz: int, mu_tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定(x,z)位置での深さごとのCCEを計算

        Parameters
        ----------
        ix : int
            x方向のインデックス
        iz : int
            z方向のインデックス
        mu_tau : float
            μτ積 [m²/V]

        Returns
        -------
        depths : np.ndarray
            深さ配列 [m]
        cce_values : np.ndarray
            各深さでのCCE値
        """
        # Y軸方向が深さ（+Y方向から入射）
        y_surface = self.field.Y.max()  # 表面

        depths = []
        cce_values = []

        # SRIMデータの各深さ点について
        for i in range(len(self.srim.depth_m)):
            depth = self.srim.depth_m[i]

            # 検出器内のy座標（表面から深さ分引く）
            y_current = y_surface - depth

            # y座標が電界グリッド範囲内かチェック
            if y_current < self.field.Y[0] or y_current > self.field.Y[-1]:
                continue

            # y座標に最も近いグリッドインデックスを見つける
            iy = np.argmin(np.abs(self.field.Y - y_current))

            # この位置での電界強度 [V/m]
            E_mag = self.field.E_mag[iz, iy, ix]

            # ドリフト長 λ = μτE [m]
            lambda_drift = mu_tau * E_mag

            # 電極までの距離（残りの厚さ）[m]
            # 電極は底面（Y.min()）にあると仮定
            distance_to_electrode = y_current - self.field.Y.min()

            # ヘクトの式でCCE計算
            if lambda_drift > 0:
                cce = (lambda_drift / distance_to_electrode) * (1 - np.exp(-distance_to_electrode / lambda_drift))
            else:
                cce = 0.0

            # CCEは最大1.0
            cce = min(cce, 1.0)

            depths.append(depth)
            cce_values.append(cce)

        return np.array(depths), np.array(cce_values)

    def calculate_total_cce(self, ix: int, iz: int, mu_tau: float) -> float:
        """
        指定(x,z)位置での全体のCCEを計算（電荷量で重み付け）

        Parameters
        ----------
        ix : int
            x方向のインデックス
        iz : int
            z方向のインデックス
        mu_tau : float
            μτ積 [m²/V]

        Returns
        -------
        total_cce : float
            全体のCCE
        """
        depths, cce_values = self.calculate_cce_vs_depth(ix, iz, mu_tau)

        if len(depths) == 0:
            return 0.0

        # 各深さでの電荷生成量
        charges = []
        for i, depth in enumerate(depths):
            # この深さでの電子正孔対密度 [pairs/Angstrom]
            idx = np.argmin(np.abs(self.srim.depth_m - depth))
            pair_density = self.srim.ionization_eV_per_angstrom[idx] / E_PAIR

            # dz [Angstrom]
            if i < len(depths) - 1:
                dz_angstrom = (depths[i+1] - depths[i]) * 1e10
            else:
                dz_angstrom = (depths[i] - depths[i-1]) * 1e10

            # 電荷数
            n_pairs = pair_density * dz_angstrom
            charges.append(n_pairs)

        charges = np.array(charges)

        # 電荷量で重み付けした平均CCE
        total_charge = np.sum(charges)
        if total_charge > 0:
            total_cce = np.sum(charges * cce_values) / total_charge
        else:
            total_cce = 0.0

        return total_cce

    def fit_mu_tau(self, ix: int, iz: int, target_cce: float,
                   initial_guess: float = 1e-10) -> Tuple[float, float]:
        """
        目標CCEに合うようにμτをフィッティング

        Parameters
        ----------
        ix : int
            x方向のインデックス
        iz : int
            z方向のインデックス
        target_cce : float
            目標CCE値
        initial_guess : float
            μτの初期推定値 [m²/V]

        Returns
        -------
        fitted_mu_tau : float
            フィットされたμτ [m²/V]
        fitted_cce : float
            実際に得られたCCE
        """
        from scipy.optimize import minimize_scalar

        def objective(mu_tau):
            cce = self.calculate_total_cce(ix, iz, mu_tau)
            return abs(cce - target_cce)

        result = minimize_scalar(objective, bounds=(1e-15, 1e-5), method='bounded')
        fitted_mu_tau = result.x
        fitted_cce = self.calculate_total_cce(ix, iz, fitted_mu_tau)

        return fitted_mu_tau, fitted_cce

    def calculate_drift_trajectory(self, ix: int, iz: int, start_depth_um: float,
                                   mu_tau: float, n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定深さからのドリフト軌跡を計算

        Parameters
        ----------
        ix : int
            x方向のインデックス
        iz : int
            z方向のインデックス
        start_depth_um : float
            開始深さ [um]
        mu_tau : float
            μτ積 [m²/V]
        n_steps : int
            計算ステップ数

        Returns
        -------
        positions : np.ndarray
            位置の配列 (n_steps, 3) [x, y, z] [m]
        times : np.ndarray
            時間の配列 [s]
        """
        start_depth_m = start_depth_um * 1e-6
        y_surface = self.field.Y.max()
        y_start = y_surface - start_depth_m

        # 初期位置
        x_pos = self.field.X[ix]
        y_pos = y_start
        z_pos = self.field.Z[iz]

        positions = [[x_pos, y_pos, z_pos]]
        times = [0.0]

        dt = 1e-12  # 時間ステップ [s]

        for step in range(n_steps):
            # 現在位置でのインデックス
            iy = np.argmin(np.abs(self.field.Y - y_pos))

            # 境界チェック
            if y_pos <= self.field.Y.min() or y_pos >= self.field.Y.max():
                break

            # 電界取得
            Ey = self.field.Ey[iz, iy, ix]
            E_mag = self.field.E_mag[iz, iy, ix]

            # 移動度（簡易的にμ = μτ/τを使用、τは適当に設定）
            tau = 1e-8  # 仮の寿命 [s]
            mu = mu_tau / tau

            # ドリフト速度 v = μE
            vy = mu * Ey

            # 位置更新
            y_pos += vy * dt

            positions.append([x_pos, y_pos, z_pos])
            times.append(times[-1] + dt)

            # 時間制限
            if times[-1] > 1e-6:  # 1us
                break

        return np.array(positions), np.array(times)


class HechtFitterGUI(tk.Tk):
    """GUIアプリケーション"""

    def __init__(self):
        super().__init__()
        self.title("Hecht Equation Fitting Tool")
        self.geometry("1200x800")

        self.field_file = None
        self.srim_file = None
        self.analyzer = None
        self.running = False

        # 現在選択されているグリッド位置
        self.current_ix = 0
        self.current_iz = 0

        self._build_widgets()

    def _build_widgets(self):
        """ウィジェット構築"""
        # メインフレーム
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # 左側：コントロールパネル
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        row = 0

        # === ファイル選択 ===
        ttk.Label(control_frame, text="Field File (npz):").grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        self.field_file_label = ttk.Label(control_frame, text="Not selected", foreground="gray")
        self.field_file_label.grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Button(control_frame, text="Browse...", command=self.browse_field_file).grid(
            row=row, column=2, padx=5
        )
        row += 1

        ttk.Label(control_frame, text="SRIM File (txt):").grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        self.srim_file_label = ttk.Label(control_frame, text="Not selected", foreground="gray")
        self.srim_file_label.grid(row=row, column=1, sticky=tk.W, padx=10)
        ttk.Button(control_frame, text="Browse...", command=self.browse_srim_file).grid(
            row=row, column=2, padx=5
        )
        row += 1

        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # === グリッド位置選択 ===
        ttk.Label(control_frame, text="Grid Position:", font=('', 9, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        row += 1

        ttk.Label(control_frame, text="X index:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ix_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.ix_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        ttk.Label(control_frame, text="Z index:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.iz_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.iz_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # === フィッティングパラメータ ===
        ttk.Label(control_frame, text="Fitting Parameters:", font=('', 9, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        row += 1

        ttk.Label(control_frame, text="Target CCE:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.target_cce_var = tk.StringVar(value="0.5")
        ttk.Entry(control_frame, textvariable=self.target_cce_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        ttk.Label(control_frame, text="mu*tau [cm²/V]:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mu_tau_var = tk.StringVar(value="1e-6")
        ttk.Entry(control_frame, textvariable=self.mu_tau_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        # === ドリフト軌跡パラメータ ===
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        ttk.Label(control_frame, text="Drift Trajectory:", font=('', 9, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        row += 1

        ttk.Label(control_frame, text="Start depth [um]:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.start_depth_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.start_depth_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        # === ボタン ===
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        ttk.Button(control_frame, text="Load Data", command=self.load_data).grid(
            row=row, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E)
        )
        row += 1

        ttk.Button(control_frame, text="Plot CCE vs Depth", command=self.plot_cce_vs_depth).grid(
            row=row, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E)
        )
        row += 1

        ttk.Button(control_frame, text="Fit mu*tau", command=self.fit_mu_tau).grid(
            row=row, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E)
        )
        row += 1

        ttk.Button(control_frame, text="Show Drift Trajectory", command=self.show_drift_trajectory).grid(
            row=row, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E)
        )
        row += 1

        # === ログエリア ===
        log_frame = ttk.LabelFrame(control_frame, text="Log", padding="5")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        control_frame.rowconfigure(row, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=40, height=10, font=("Courier", 8)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 右側：プロットエリア
        plot_frame = ttk.LabelFrame(main_frame, text="Plots", padding="10")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message: str):
        """ログメッセージを表示"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def browse_field_file(self):
        """電界ファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select Electric Field File",
            filetypes=[("NumPy Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.field_file = filename
            self.field_file_label.config(text=os.path.basename(filename), foreground="black")
            self.log(f"Selected field file: {filename}")

    def browse_srim_file(self):
        """SRIMファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select SRIM File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            self.srim_file = filename
            self.srim_file_label.config(text=os.path.basename(filename), foreground="black")
            self.log(f"Selected SRIM file: {filename}")

    def load_data(self):
        """データを読み込む"""
        if self.field_file is None or self.srim_file is None:
            messagebox.showwarning("Warning", "Please select both field and SRIM files.")
            return

        try:
            self.log("\n" + "="*50)
            self.log("Loading data...")

            field_data = FieldData(self.field_file)
            self.log("Field data loaded")
            self.log(field_data.get_info())

            srim_data = SRIMData(self.srim_file)
            self.log("\nSRIM data loaded")
            self.log(srim_data.get_info())

            self.analyzer = HechtAnalyzer(field_data, srim_data)
            self.log("\nAnalyzer initialized successfully!")

            messagebox.showinfo("Success", "Data loaded successfully!")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def plot_cce_vs_depth(self):
        """CCE vs 深さをプロット"""
        if self.analyzer is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            ix = int(self.ix_var.get())
            iz = int(self.iz_var.get())
            mu_tau_cm2_V = float(self.mu_tau_var.get())
            mu_tau = mu_tau_cm2_V * 1e-4  # cm²/V -> m²/V

            self.log(f"\nCalculating CCE vs depth at (ix={ix}, iz={iz})...")
            self.log(f"mu*tau = {mu_tau_cm2_V} cm²/V")

            depths, cce_values = self.analyzer.calculate_cce_vs_depth(ix, iz, mu_tau)

            # プロット
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            ax.plot(depths * 1e6, cce_values, 'b-', linewidth=2, label=f'mu*tau = {mu_tau_cm2_V} cm²/V')
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='CCE = 1.0')
            ax.set_xlabel('Depth [um]', fontsize=12)
            ax.set_ylabel('CCE', fontsize=12)
            ax.set_title(f'CCE vs Depth (ix={ix}, iz={iz})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])

            self.canvas.draw()

            total_cce = self.analyzer.calculate_total_cce(ix, iz, mu_tau)
            self.log(f"Total CCE (weighted): {total_cce:.4f}")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            messagebox.showerror("Error", f"Failed to plot:\n{e}")

    def fit_mu_tau(self):
        """μτをフィッティング"""
        if self.analyzer is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            ix = int(self.ix_var.get())
            iz = int(self.iz_var.get())
            target_cce = float(self.target_cce_var.get())

            self.log(f"\nFitting mu*tau for target CCE = {target_cce:.4f}...")
            self.log(f"Position: (ix={ix}, iz={iz})")

            fitted_mu_tau, fitted_cce = self.analyzer.fit_mu_tau(ix, iz, target_cce)
            fitted_mu_tau_cm2_V = fitted_mu_tau * 1e4  # m²/V -> cm²/V

            self.log(f"Fitted mu*tau = {fitted_mu_tau_cm2_V:.4e} cm²/V")
            self.log(f"Achieved CCE = {fitted_cce:.4f}")

            # 結果を入力欄に反映
            self.mu_tau_var.set(f"{fitted_mu_tau_cm2_V:.4e}")

            # プロット
            depths, cce_values = self.analyzer.calculate_cce_vs_depth(ix, iz, fitted_mu_tau)

            self.fig.clear()
            ax = self.fig.add_subplot(111)

            ax.plot(depths * 1e6, cce_values, 'b-', linewidth=2,
                   label=f'Fitted: mu*tau = {fitted_mu_tau_cm2_V:.4e} cm²/V')
            ax.axhline(y=target_cce, color='r', linestyle='--', alpha=0.5,
                      label=f'Target CCE = {target_cce:.4f}')
            ax.axhline(y=fitted_cce, color='g', linestyle=':', alpha=0.5,
                      label=f'Achieved CCE = {fitted_cce:.4f}')
            ax.set_xlabel('Depth [um]', fontsize=12)
            ax.set_ylabel('CCE', fontsize=12)
            ax.set_title(f'Fitted CCE vs Depth (ix={ix}, iz={iz})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])

            self.canvas.draw()

            messagebox.showinfo("Success",
                              f"Fitted mu*tau = {fitted_mu_tau_cm2_V:.4e} cm²/V\n"
                              f"Achieved CCE = {fitted_cce:.4f}")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            messagebox.showerror("Error", f"Failed to fit:\n{e}")

    def show_drift_trajectory(self):
        """ドリフト軌跡を表示"""
        if self.analyzer is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            ix = int(self.ix_var.get())
            iz = int(self.iz_var.get())
            start_depth_um = float(self.start_depth_var.get())
            mu_tau_cm2_V = float(self.mu_tau_var.get())
            mu_tau = mu_tau_cm2_V * 1e-4  # cm²/V -> m²/V

            self.log(f"\nCalculating drift trajectory from depth = {start_depth_um} um...")

            positions, times = self.analyzer.calculate_drift_trajectory(
                ix, iz, start_depth_um, mu_tau
            )

            # プロット
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            y_positions = positions[:, 1] * 1e6  # m -> um
            ax.plot(times * 1e9, y_positions, 'b-', linewidth=2)
            ax.axhline(y=self.analyzer.field.Y.min() * 1e6, color='r',
                      linestyle='--', alpha=0.5, label='Electrode')
            ax.set_xlabel('Time [ns]', fontsize=12)
            ax.set_ylabel('Y Position [um]', fontsize=12)
            ax.set_title(f'Drift Trajectory (start depth = {start_depth_um} um)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            self.canvas.draw()

            drift_distance = (positions[0, 1] - positions[-1, 1]) * 1e6
            drift_time = times[-1] * 1e9
            self.log(f"Drift distance: {drift_distance:.2f} um")
            self.log(f"Drift time: {drift_time:.2f} ns")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            messagebox.showerror("Error", f"Failed to show trajectory:\n{e}")


def main():
    """メイン関数"""
    app = HechtFitterGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
