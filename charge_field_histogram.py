#!/usr/bin/env python3
"""
Charge-Field Product Histogram Analyzer

各(x,y)位置でイオン入射時に生成される電荷数×電界強度の積分値を計算し、
その分布をヒストグラムで表示するGUIアプリケーション。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Tuple
import threading

# ========== 物理定数 ==========
E_PAIR = 7.8  # eV - 電子正孔対生成エネルギー


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
        self.Y = data['Y']  # 1D array [m]
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
        info.append(f"X range: [{self.X.min()*1e6:.2f}, {self.X.max()*1e6:.2f}] μm")
        info.append(f"Y range: [{self.Y.min()*1e6:.2f}, {self.Y.max()*1e6:.2f}] μm")
        info.append(f"Z range: [{self.Z.min()*1e6:.2f}, {self.Z.max()*1e6:.2f}] μm")
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
        # SRIMファイルの読み込み
        data = self._load_srim_file(srim_file)

        # 深さとイオン化エネルギー密度
        self.depth_angstrom = data[:, 0]  # Angstrom
        self.ionization_eV_per_angstrom = data[:, 1]  # eV/Angstrom

        # 単位変換: Angstrom -> m
        self.depth_m = self.depth_angstrom * 1e-10

    def get_info(self) -> str:
        """データ情報を文字列で返す"""
        info = []
        info.append(f"Depth range: [0, {self.depth_m[-1]*1e6:.2f}] μm")
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


class ChargeFieldAnalyzer:
    """電荷×電界の積分値を計算するクラス"""

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
        self.results = None

    def calculate_all_positions(self, progress_callback=None) -> np.ndarray:
        """
        全(x,y)位置について計算を実行

        Parameters
        ----------
        progress_callback : callable, optional
            進捗報告用のコールバック関数 (current, total, message)

        Returns
        -------
        np.ndarray
            各(x,y)位置での計算結果 (ny, nx)
        """
        nx = len(self.field.X)
        ny = len(self.field.Y)

        results = np.zeros((ny, nx))
        total_positions = nx * ny

        # 各(x,y)位置について計算
        for iy in range(ny):
            for ix in range(nx):
                value = self._calculate_single_position(ix, iy)
                results[iy, ix] = value

                current = iy * nx + ix + 1
                if progress_callback is not None:
                    progress_callback(current, total_positions,
                                    f"Position ({ix+1}/{nx}, {iy+1}/{ny})")

        self.results = results
        return results

    def _calculate_single_position(self, ix: int, iy: int) -> float:
        """
        単一(x,y)位置での計算

        Parameters
        ----------
        ix : int
            x方向のインデックス
        iy : int
            y方向のインデックス

        Returns
        -------
        float
            Σ(電子正孔対数(z) × 電界強度(x,y,z))
        """
        # z軸は上から下（z_max → z_min）に入射
        # SRIMの深さは0から始まるので、z_maxから引いていく

        z_surface = self.field.Z[-1]  # z_max（検出器上面）

        # z方向について積分
        integral_sum = 0.0

        # SRIMデータの各深さ点について
        for i in range(len(self.srim.depth_m)):
            # 現在の深さ
            depth = self.srim.depth_m[i]

            # 検出器内のz座標（上面から深さ分引く）
            z_current = z_surface - depth

            # z座標が電界グリッド範囲内かチェック
            if z_current < self.field.Z[0] or z_current > self.field.Z[-1]:
                continue

            # z座標に最も近いグリッドインデックスを見つける
            iz = np.argmin(np.abs(self.field.Z - z_current))

            # この深さでの電子正孔対密度 [pairs/Angstrom]
            pair_density = self.srim.ionization_eV_per_angstrom[i] / E_PAIR

            # dz（積分の刻み幅）[Angstrom]
            if i < len(self.srim.depth_m) - 1:
                dz_angstrom = self.srim.depth_angstrom[i+1] - self.srim.depth_angstrom[i]
            else:
                # 最後の点
                dz_angstrom = self.srim.depth_angstrom[i] - self.srim.depth_angstrom[i-1]

            # この区間での電子正孔対数 [pairs]
            n_pairs = pair_density * dz_angstrom

            # この位置(ix, iy, iz)での電界強度 [V/m]
            E_mag = self.field.E_mag[iz, iy, ix]

            # 積分に加算
            integral_sum += n_pairs * E_mag

        return integral_sum


class ChargeFieldHistogramGUI(tk.Tk):
    """GUIアプリケーション"""

    def __init__(self):
        super().__init__()
        self.title("Charge-Field Product Histogram Analyzer")
        self.geometry("1000x700")

        self.field_file = None
        self.srim_file = None
        self.analyzer = None
        self.running = False

        self._build_widgets()

    def _build_widgets(self):
        """ウィジェット構築"""
        # メインフレーム
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # === ファイル選択エリア ===
        file_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # 実電界ファイル
        ttk.Label(file_frame, text="Electric Field (npz):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.field_file_label = ttk.Label(
            file_frame, text="Not selected", foreground="gray"
        )
        self.field_file_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Button(
            file_frame, text="Browse...", command=self.browse_field_file
        ).grid(row=0, column=2, padx=5)

        # SRIMファイル
        ttk.Label(file_frame, text="SRIM File (txt):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.srim_file_label = ttk.Label(
            file_frame, text="Not selected", foreground="gray"
        )
        self.srim_file_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Button(
            file_frame, text="Browse...", command=self.browse_srim_file
        ).grid(row=1, column=2, padx=5)

        # === ボタンエリア ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=10)

        self.calc_button = ttk.Button(
            button_frame, text="Calculate", command=self.run_calculation
        )
        self.calc_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(
            button_frame, text="Save Histogram", command=self.save_histogram,
            state="disabled"
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Quit", command=self.quit).pack(
            side=tk.LEFT, padx=5
        )

        # === 進捗表示エリア ===
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(
            progress_frame, mode='determinate', length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # === ログエリア ===
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=80, height=8, font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # === ヒストグラム表示エリア ===
        hist_frame = ttk.LabelFrame(main_frame, text="Histogram", padding="10")
        hist_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(4, weight=1)

        # matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=hist_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初期プロット
        self.ax.text(0.5, 0.5, 'No data yet',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=self.ax.transAxes)
        self.ax.set_xlabel('Charge×Field Product')
        self.ax.set_ylabel('Counts')
        self.ax.set_title('Histogram')
        self.canvas.draw()

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
            self.field_file_label.config(
                text=os.path.basename(filename), foreground="black"
            )
            self.log(f"Selected field file: {filename}")

    def browse_srim_file(self):
        """SRIMファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select SRIM File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            self.srim_file = filename
            self.srim_file_label.config(
                text=os.path.basename(filename), foreground="black"
            )
            self.log(f"Selected SRIM file: {filename}")

    def run_calculation(self):
        """計算を実行"""
        # ファイルチェック
        if self.field_file is None or self.srim_file is None:
            messagebox.showwarning(
                "Warning", "Please select both field and SRIM files."
            )
            return

        if self.running:
            messagebox.showwarning("Warning", "Calculation already running.")
            return

        # 別スレッドで計算実行
        thread = threading.Thread(target=self._calculation_thread)
        thread.daemon = True
        thread.start()

    def _calculation_thread(self):
        """計算スレッド"""
        self.running = True
        self.calc_button.config(state="disabled")

        try:
            # データ読み込み
            self.log("\n" + "="*70)
            self.log("Loading data...")

            field_data = FieldData(self.field_file)
            self.log("Field data loaded")
            self.log(field_data.get_info())

            srim_data = SRIMData(self.srim_file)
            self.log("\nSRIM data loaded")
            self.log(srim_data.get_info())

            # Analyzerの作成
            self.analyzer = ChargeFieldAnalyzer(field_data, srim_data)

            # 計算実行
            self.log("\nStarting calculation...")
            self.progress_bar['value'] = 0

            def progress_callback(current, total, message):
                # GUI更新
                progress = (current / total) * 100
                self.progress_bar['value'] = progress
                self.progress_label.config(text=f"{message} ({current}/{total})")
                self.update_idletasks()

            results = self.analyzer.calculate_all_positions(progress_callback)

            self.log("\nCalculation completed!")
            self.log(f"Result shape: {results.shape}")
            self.log(f"Result range: [{results.min():.2e}, {results.max():.2e}]")
            self.log(f"Mean: {results.mean():.2e}")
            self.log(f"Std: {results.std():.2e}")

            # ヒストグラムを表示
            self._plot_histogram(results)

            self.save_button.config(state="normal")
            self.progress_label.config(text="Completed!")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            messagebox.showerror("Error", f"Calculation failed:\n{e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            self.calc_button.config(state="normal")

    def _plot_histogram(self, results: np.ndarray):
        """ヒストグラムをプロット"""
        self.ax.clear()

        # 1次元配列に変換
        data = results.flatten()

        # ヒストグラム
        self.ax.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        self.ax.axvline(data.mean(), color='red', linestyle='--',
                       label=f'Mean = {data.mean():.2e}')

        self.ax.set_xlabel('Charge×Field Product [pairs·V/m]', fontsize=12)
        self.ax.set_ylabel('Counts', fontsize=12)
        self.ax.set_title(
            f'Histogram (N={len(data)}, Mean={data.mean():.2e})',
            fontsize=14
        )
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def save_histogram(self):
        """ヒストグラムを保存"""
        if self.analyzer is None or self.analyzer.results is None:
            messagebox.showwarning("Warning", "No results to save.")
            return

        # 保存先を選択
        filename = filedialog.asksaveasfilename(
            title="Save Histogram",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")]
        )

        if filename:
            try:
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                self.log(f"Histogram saved: {filename}")
                messagebox.showinfo("Success", f"Histogram saved:\n{filename}")
            except Exception as e:
                self.log(f"ERROR saving histogram: {e}")
                messagebox.showerror("Error", f"Failed to save:\n{e}")


def main():
    """メイン関数"""
    app = ChargeFieldHistogramGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
