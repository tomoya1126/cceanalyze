#!/usr/bin/env python3
"""
SiC検出器のCCE(電荷収集効率)シミュレーションコード

α線入射時の電子・正孔の軌道を追跡し、電極で収集される電荷量を計算する。
OpenSTFで計算した電界データとSRIMの電子正孔対分布データを使用。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from typing import Tuple, Optional, List, Dict
import warnings


# ========== 物理定数 ==========
E_PAIR = 7.8  # eV - 電子正孔対生成エネルギー
E_ALPHA = 5.486e6  # eV - α線エネルギー (Am-241)
Q_E = 1.602e-19  # C - 電気素量
MU_E = 800e-4  # m²/Vs - 電子移動度 (800 cm²/Vs)
MU_H = 40e-4  # m²/Vs - 正孔移動度 (40 cm²/Vs)
BIAS_VOLTAGE = 100.0  # V - バイアス電圧


class FieldInterpolator:
    """電界データの補間クラス"""

    def __init__(self, field_file: str):
        """
        電界データを読み込んで補間器を初期化

        Parameters:
        -----------
        field_file : str
            電界データのnpzファイルパス
        """
        print(f"Loading field data from: {field_file}")
        data = np.load(field_file)

        # 座標軸とフィールドデータ
        self.X = data['X']  # 1D array [m]
        self.Y = data['Y']  # 1D array [m]
        self.Z = data['Z']  # 1D array [m]
        self.V = data['V']  # 3D array (Nz, Ny, Nx) [V]
        self.Ex = data['Ex']  # 3D array [V/m]
        self.Ey = data['Ey']  # 3D array [V/m]
        self.Ez = data['Ez']  # 3D array [V/m]

        # 計算領域の境界
        self.x_min, self.x_max = self.X.min(), self.X.max()
        self.y_min, self.y_max = self.Y.min(), self.Y.max()
        self.z_min, self.z_max = self.Z.min(), self.Z.max()

        print(f"  Grid size: Nx={len(self.X)}, Ny={len(self.Y)}, Nz={len(self.Z)}")
        print(f"  X range: [{self.x_min*1e6:.2f}, {self.x_max*1e6:.2f}] μm")
        print(f"  Y range: [{self.y_min*1e6:.2f}, {self.y_max*1e6:.2f}] μm")
        print(f"  Z range: [{self.z_min*1e6:.2f}, {self.z_max*1e6:.2f}] μm")
        print(f"  V range: [{self.V.min():.2f}, {self.V.max():.2f}] V")

        # 補間器の作成 (Z, Y, X の順序に注意)
        self.V_interp = RegularGridInterpolator(
            (self.Z, self.Y, self.X), self.V,
            bounds_error=False, fill_value=None
        )
        self.Ex_interp = RegularGridInterpolator(
            (self.Z, self.Y, self.X), self.Ex,
            bounds_error=False, fill_value=0.0
        )
        self.Ey_interp = RegularGridInterpolator(
            (self.Z, self.Y, self.X), self.Ey,
            bounds_error=False, fill_value=0.0
        )
        self.Ez_interp = RegularGridInterpolator(
            (self.Z, self.Y, self.X), self.Ez,
            bounds_error=False, fill_value=0.0
        )

        # 電極位置の特定
        self._identify_electrodes()

    def _identify_electrodes(self):
        """電位分布から電極位置を特定"""
        # 100V電極（高電位側）と0V電極（低電位側）を特定
        v_max = self.V.max()
        v_min = self.V.min()

        print(f"\nElectrode identification:")
        print(f"  Maximum potential: {v_max:.2f} V")
        print(f"  Minimum potential: {v_min:.2f} V")

        # 100V電極は高電位側、0V電極は低電位側
        self.v_high = v_max  # 100V電極の電位
        self.v_low = v_min   # 0V電極の電位

        # 電極判定の閾値（少し余裕を持たせる）
        self.v_high_threshold = self.v_high - 1.0  # 99V以上
        self.v_low_threshold = self.v_low + 1.0    # 1V以下

        print(f"  High electrode (100V side): V > {self.v_high_threshold:.2f} V")
        print(f"  Low electrode (0V side): V < {self.v_low_threshold:.2f} V")

    def get_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        指定位置での電界を取得

        Parameters:
        -----------
        x, y, z : float
            位置座標 [m]

        Returns:
        --------
        Ex, Ey, Ez : float
            電界成分 [V/m]
        """
        point = np.array([[z, y, x]])  # (Z, Y, X) の順序
        Ex = float(self.Ex_interp(point)[0])
        Ey = float(self.Ey_interp(point)[0])
        Ez = float(self.Ez_interp(point)[0])
        return Ex, Ey, Ez

    def get_potential(self, x: float, y: float, z: float) -> float:
        """
        指定位置での電位を取得

        Parameters:
        -----------
        x, y, z : float
            位置座標 [m]

        Returns:
        --------
        V : float
            電位 [V]
        """
        point = np.array([[z, y, x]])  # (Z, Y, X) の順序
        return float(self.V_interp(point)[0])

    def is_in_bounds(self, x: float, y: float, z: float) -> bool:
        """位置が計算領域内かチェック"""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def check_electrode_reached(self, x: float, y: float, z: float,
                               carrier_type: str) -> bool:
        """
        キャリアが電極に到達したかチェック

        Parameters:
        -----------
        x, y, z : float
            現在位置 [m]
        carrier_type : str
            'electron' or 'hole'

        Returns:
        --------
        reached : bool
            電極に到達したかどうか
        """
        if not self.is_in_bounds(x, y, z):
            return False

        V = self.get_potential(x, y, z)

        if carrier_type == 'electron':
            # 電子は高電位（100V）側に到達
            return V >= self.v_high_threshold
        elif carrier_type == 'hole':
            # 正孔は低電位（0V）側に到達
            return V <= self.v_low_threshold
        else:
            raise ValueError(f"Unknown carrier type: {carrier_type}")


class SRIMDataLoader:
    """SRIMデータの読み込みと処理"""

    def __init__(self, srim_file: str):
        """
        SRIMファイルを読み込む

        Parameters:
        -----------
        srim_file : str
            SRIMデータファイルパス
        """
        print(f"\nLoading SRIM data from: {srim_file}")

        # SRIMファイルの読み込み (ヘッダーをスキップ)
        self.data = self._load_srim_file(srim_file)

        # 深さとイオン化エネルギー
        self.depth_angstrom = self.data[:, 0]  # Angstrom
        self.ionization_eV_per_angstrom = self.data[:, 1]  # eV/Angstrom

        # 単位変換: Angstrom -> m
        self.depth_m = self.depth_angstrom * 1e-10

        # イオン化エネルギー: eV/Angstrom -> eV/m
        self.ionization_eV_per_m = self.ionization_eV_per_angstrom * 1e10

        # 総エネルギーとブラッグピーク
        self.total_energy = np.trapezoid(self.ionization_eV_per_m, self.depth_m)
        self.bragg_peak_depth = self.depth_m[np.argmax(self.ionization_eV_per_m)]

        print(f"  Depth range: [0, {self.depth_m[-1]*1e6:.2f}] μm")
        print(f"  Total ionization energy: {self.total_energy/1e6:.3f} MeV")
        print(f"  Bragg peak at: {self.bragg_peak_depth*1e6:.2f} μm")
        print(f"  Expected e-h pairs: {self.total_energy/E_PAIR:.3e}")

    def _load_srim_file(self, filename: str) -> np.ndarray:
        """SRIMファイルを読み込む（ヘッダー処理付き）"""
        data_rows = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # 数値データ行を探す
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

    def generate_eh_pairs(self, n_alpha: int = 1,
                         incident_x: Optional[float] = None,
                         incident_y: Optional[float] = None,
                         incident_z: Optional[float] = None,
                         field_interp: Optional[FieldInterpolator] = None,
                         sampling_ratio: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        電子正孔対の初期位置を生成

        α線はz_max（検出器上面）から-z方向（深さ方向）に進む

        Parameters:
        -----------
        n_alpha : int
            α線の入射数
        incident_x, incident_y, incident_z : float, optional
            入射位置 [m]。Noneの場合はランダムまたは自動設定
        field_interp : FieldInterpolator, optional
            入射位置の範囲決定に使用
        sampling_ratio : float
            サブサンプリング比率（0.0-1.0）。1.0=全キャリア、0.01=1%をサンプリング

        Returns:
        --------
        positions : ndarray
            電子正孔対の初期位置 (N_sampled, 3) [m]
            各行は [x, y, z]
        total_pairs : int
            実際の総電子正孔対数（サンプリング前）
        """
        # 各深さでの電子正孔対数を計算
        depth_step = np.diff(self.depth_m)
        depth_step = np.append(depth_step, depth_step[-1])

        # 各深さでの生成エネルギー [eV]
        energy_per_depth = self.ionization_eV_per_m * depth_step

        # 各深さでの電子正孔対数
        pairs_per_depth = (energy_per_depth / E_PAIR).astype(int)

        total_pairs = np.sum(pairs_per_depth)
        n_sampled = int(total_pairs * sampling_ratio)

        print(f"\nGenerating e-h pairs:")
        print(f"  Total pairs: {total_pairs:,}")
        print(f"  Sampling ratio: {sampling_ratio*100:.1f}%")
        print(f"  Sampled pairs: {n_sampled:,}")

        positions = []

        for alpha_idx in range(n_alpha):
            # 入射位置の決定（x, y方向はランダム、z方向は固定）
            if incident_x is None or incident_y is None:
                if field_interp is not None:
                    # フィールドの全範囲でランダム
                    x_range = field_interp.x_max - field_interp.x_min
                    y_range = field_interp.y_max - field_interp.y_min
                    x0 = field_interp.x_min + np.random.rand() * x_range
                    y0 = field_interp.y_min + np.random.rand() * y_range
                else:
                    x0, y0 = 0.0, 0.0
            else:
                x0, y0 = incident_x, incident_y

            # z入射位置（デフォルトは電界範囲の最大値 = 検出器上面）
            if incident_z is None and field_interp is not None:
                z0 = field_interp.z_max  # 上面から入射
            elif incident_z is not None:
                z0 = incident_z
            else:
                z0 = 0.0

            # 各深さで電子正孔対を生成（サブサンプリング）
            for i, n_pairs in enumerate(pairs_per_depth):
                if n_pairs == 0:
                    continue

                # サンプリング数
                n_sample = int(n_pairs * sampling_ratio)
                if n_sample == 0 and n_pairs > 0 and np.random.rand() < sampling_ratio:
                    n_sample = 1  # 最低1個は生成

                if n_sample == 0:
                    continue

                # α線の飛程（SRIMの深さ）
                depth = self.depth_m[i]

                # 実際のz座標：z0から-z方向に進む
                z = z0 - depth

                # 小さなランダムオフセットを追加（横方向の広がり）
                for _ in range(n_sample):
                    dx = np.random.normal(0, 10e-9)  # 10 nm std
                    dy = np.random.normal(0, 10e-9)  # 10 nm std
                    positions.append([x0 + dx, y0 + dy, z])

        positions = np.array(positions)
        if len(positions) > 0:
            print(f"  Position range: x=[{positions[:,0].min()*1e6:.3f}, {positions[:,0].max()*1e6:.3f}] μm")
            print(f"                  y=[{positions[:,1].min()*1e6:.3f}, {positions[:,1].max()*1e6:.3f}] μm")
            print(f"                  z=[{positions[:,2].min()*1e6:.3f}, {positions[:,2].max()*1e6:.3f}] μm")

        return positions, total_pairs


class CarrierTracker:
    """キャリア軌道追跡"""

    def __init__(self, field_interp: FieldInterpolator):
        """
        Parameters:
        -----------
        field_interp : FieldInterpolator
            電界補間器
        """
        self.field = field_interp
        self.max_steps = 100000  # 最大ステップ数
        self.max_time = 1e-6  # 最大時間 [s] = 1 μs

    def track_carrier(self, x0: float, y0: float, z0: float,
                     carrier_type: str,
                     dt_initial: float = 1e-12,
                     save_trajectory: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        単一キャリアの軌道を追跡

        Parameters:
        -----------
        x0, y0, z0 : float
            初期位置 [m]
        carrier_type : str
            'electron' or 'hole'
        dt_initial : float
            初期時間ステップ [s]
        save_trajectory : bool
            軌道データを保存するか

        Returns:
        --------
        collected : bool
            電極に到達したか
        trajectory : ndarray or None
            軌道データ (N_steps, 4) [x, y, z, t] または None
        """
        # 移動度の選択
        if carrier_type == 'electron':
            mu = MU_E
            sign = -1  # 電子は電界と逆方向
        elif carrier_type == 'hole':
            mu = MU_H
            sign = 1  # 正孔は電界と同方向
        else:
            raise ValueError(f"Unknown carrier type: {carrier_type}")

        # 初期化
        x, y, z = x0, y0, z0
        t = 0.0
        dt = dt_initial

        trajectory = [[x, y, z, t]] if save_trajectory else None

        for step in range(self.max_steps):
            # 境界チェック
            if not self.field.is_in_bounds(x, y, z):
                return False, np.array(trajectory) if save_trajectory else None

            # 電極到達チェック
            if self.field.check_electrode_reached(x, y, z, carrier_type):
                if save_trajectory:
                    trajectory.append([x, y, z, t])
                return True, np.array(trajectory) if save_trajectory else None

            # 時間制限チェック
            if t > self.max_time:
                return False, np.array(trajectory) if save_trajectory else None

            # Runge-Kutta 4次法で位置を更新
            x_new, y_new, z_new, dt_new = self._rk4_step(
                x, y, z, dt, mu, sign
            )

            # 時間とステップサイズの更新
            t += dt
            dt = dt_new

            # 位置の更新
            x, y, z = x_new, y_new, z_new

            # 軌道の保存（間引き）
            if save_trajectory and (step % 10 == 0):
                trajectory.append([x, y, z, t])

        # 最大ステップ数に達した
        return False, np.array(trajectory) if save_trajectory else None

    def _rk4_step(self, x: float, y: float, z: float, dt: float,
                  mu: float, sign: int) -> Tuple[float, float, float, float]:
        """
        Runge-Kutta 4次法で1ステップ進める

        Returns:
        --------
        x_new, y_new, z_new : float
            新しい位置 [m]
        dt_new : float
            次のステップサイズ [s]
        """
        # k1
        Ex1, Ey1, Ez1 = self.field.get_field(x, y, z)
        vx1 = sign * mu * Ex1
        vy1 = sign * mu * Ey1
        vz1 = sign * mu * Ez1

        # k2
        x2 = x + 0.5 * dt * vx1
        y2 = y + 0.5 * dt * vy1
        z2 = z + 0.5 * dt * vz1
        if not self.field.is_in_bounds(x2, y2, z2):
            return x + dt * vx1, y + dt * vy1, z + dt * vz1, dt

        Ex2, Ey2, Ez2 = self.field.get_field(x2, y2, z2)
        vx2 = sign * mu * Ex2
        vy2 = sign * mu * Ey2
        vz2 = sign * mu * Ez2

        # k3
        x3 = x + 0.5 * dt * vx2
        y3 = y + 0.5 * dt * vy2
        z3 = z + 0.5 * dt * vz2
        if not self.field.is_in_bounds(x3, y3, z3):
            return x + dt * vx1, y + dt * vy1, z + dt * vz1, dt

        Ex3, Ey3, Ez3 = self.field.get_field(x3, y3, z3)
        vx3 = sign * mu * Ex3
        vy3 = sign * mu * Ey3
        vz3 = sign * mu * Ez3

        # k4
        x4 = x + dt * vx3
        y4 = y + dt * vy3
        z4 = z + dt * vz3
        if not self.field.is_in_bounds(x4, y4, z4):
            return x + dt * vx1, y + dt * vy1, z + dt * vz1, dt

        Ex4, Ey4, Ez4 = self.field.get_field(x4, y4, z4)
        vx4 = sign * mu * Ex4
        vy4 = sign * mu * Ey4
        vz4 = sign * mu * Ez4

        # 位置の更新
        x_new = x + (dt / 6.0) * (vx1 + 2*vx2 + 2*vx3 + vx4)
        y_new = y + (dt / 6.0) * (vy1 + 2*vy2 + 2*vy3 + vy4)
        z_new = z + (dt / 6.0) * (vz1 + 2*vz2 + 2*vz3 + vz4)

        # 適応的時間ステップ（電界が強い場所では小さく）
        E_mag = np.sqrt(Ex1**2 + Ey1**2 + Ez1**2)
        v_mag = mu * E_mag

        if v_mag > 0:
            # 1ステップで進む距離を適切に制限
            # グリッドサイズの1/10程度を目安
            grid_size = min(
                self.field.X[1] - self.field.X[0] if len(self.field.X) > 1 else 1e-6,
                self.field.Y[1] - self.field.Y[0] if len(self.field.Y) > 1 else 1e-6,
                self.field.Z[1] - self.field.Z[0] if len(self.field.Z) > 1 else 1e-6
            )
            target_step = 0.1 * grid_size
            dt_new = min(target_step / v_mag, dt * 2.0, 1e-10)
            dt_new = max(dt_new, 1e-14)  # 最小ステップサイズ
        else:
            dt_new = dt

        return x_new, y_new, z_new, dt_new


class CCESimulator:
    """CCE シミュレーター"""

    def __init__(self, field_file: str, srim_file: str):
        """
        Parameters:
        -----------
        field_file : str
            電界データのnpzファイル
        srim_file : str
            SRIMデータファイル
        """
        self.field = FieldInterpolator(field_file)
        self.srim = SRIMDataLoader(srim_file)
        self.tracker = CarrierTracker(self.field)

    def simulate_single_alpha(self,
                             incident_x: Optional[float] = None,
                             incident_y: Optional[float] = None,
                             incident_z: Optional[float] = None,
                             n_sample_trajectories: int = 5,
                             sampling_ratio: float = 0.001) -> Dict:
        """
        単一α線イベントのシミュレーション

        Parameters:
        -----------
        incident_x, incident_y, incident_z : float, optional
            入射位置 [m]
        n_sample_trajectories : int
            軌道を保存するキャリア数
        sampling_ratio : float
            キャリアサンプリング比率（デフォルト0.1% = 0.001）

        Returns:
        --------
        result : dict
            シミュレーション結果
        """
        print("\n" + "="*60)
        print("Single alpha event simulation")
        print("="*60)

        # 電子正孔対の生成
        positions, total_pairs = self.srim.generate_eh_pairs(
            n_alpha=1,
            incident_x=incident_x,
            incident_y=incident_y,
            incident_z=incident_z,
            field_interp=self.field,
            sampling_ratio=sampling_ratio
        )

        n_sampled = len(positions)

        # カウンター
        n_electron_collected = 0
        n_hole_collected = 0
        n_electron_lost = 0
        n_hole_lost = 0

        # サンプル軌道
        electron_trajectories = []
        hole_trajectories = []

        print(f"\nTracking carriers...")

        # 電子の追跡
        for i, pos in enumerate(positions):
            save_traj = (i < n_sample_trajectories)
            collected, trajectory = self.tracker.track_carrier(
                pos[0], pos[1], pos[2],
                carrier_type='electron',
                save_trajectory=save_traj
            )

            if collected:
                n_electron_collected += 1
                if save_traj and trajectory is not None:
                    electron_trajectories.append(trajectory)
            else:
                n_electron_lost += 1

        # 正孔の追跡
        for i, pos in enumerate(positions):
            save_traj = (i < n_sample_trajectories)
            collected, trajectory = self.tracker.track_carrier(
                pos[0], pos[1], pos[2],
                carrier_type='hole',
                save_trajectory=save_traj
            )

            if collected:
                n_hole_collected += 1
                if save_traj and trajectory is not None:
                    hole_trajectories.append(trajectory)
            else:
                n_hole_lost += 1

        # 実際の数に補正（サンプリング比率で割る）
        n_electron_collected_actual = int(n_electron_collected / sampling_ratio)
        n_hole_collected_actual = int(n_hole_collected / sampling_ratio)
        n_electron_lost_actual = int(n_electron_lost / sampling_ratio)
        n_hole_lost_actual = int(n_hole_lost / sampling_ratio)

        # 収集電荷の計算（実際の数で）
        q_collected = (n_electron_collected_actual + n_hole_collected_actual) * Q_E
        q_total = total_pairs * 2 * Q_E
        cce = (n_electron_collected_actual + n_hole_collected_actual) / (2 * total_pairs) if total_pairs > 0 else 0

        # 結果の表示
        print("\n" + "-"*60)
        print("Results:")
        print("-"*60)
        print(f"Total e-h pairs: {total_pairs:,}")
        print(f"Sampled pairs: {n_sampled:,} ({sampling_ratio*100:.2f}%)")
        print(f"\nEstimated from sampling:")
        print(f"  Electrons collected: {n_electron_collected_actual:,} ({n_electron_collected_actual/total_pairs*100:.1f}%)")
        print(f"  Holes collected: {n_hole_collected_actual:,} ({n_hole_collected_actual/total_pairs*100:.1f}%)")
        print(f"  Total charge: {q_collected*1e15:.2f} fC")
        print(f"  CCE: {cce*100:.2f}%")
        print("-"*60)

        return {
            'n_total': total_pairs,
            'n_sampled': n_sampled,
            'n_electron_collected': n_electron_collected_actual,
            'n_hole_collected': n_hole_collected_actual,
            'n_electron_lost': n_electron_lost_actual,
            'n_hole_lost': n_hole_lost_actual,
            'q_collected': q_collected,
            'q_total': q_total,
            'cce': cce,
            'electron_trajectories': electron_trajectories,
            'hole_trajectories': hole_trajectories,
            'initial_positions': positions,
            'sampling_ratio': sampling_ratio
        }

    def simulate_multiple_alphas(self, n_events: int = 100,
                                 sampling_ratio: float = 0.001) -> List[Dict]:
        """
        複数α線イベントのシミュレーション

        Parameters:
        -----------
        n_events : int
            イベント数
        sampling_ratio : float
            キャリアサンプリング比率

        Returns:
        --------
        results : list of dict
            各イベントの結果
        """
        print("\n" + "="*60)
        print(f"Multiple events simulation ({n_events} events)")
        print("="*60)

        results = []

        for i in range(n_events):
            print(f"\n--- Event {i+1}/{n_events} ---")
            result = self.simulate_single_alpha(
                n_sample_trajectories=0,
                sampling_ratio=sampling_ratio
            )
            results.append(result)

        # 統計情報
        cce_values = [r['cce'] for r in results]
        q_values = [r['q_collected'] for r in results]

        print("\n" + "="*60)
        print(f"Statistics ({n_events} events):")
        print("="*60)
        print(f"CCE: {np.mean(cce_values)*100:.2f}% ± {np.std(cce_values)*100:.2f}%")
        print(f"Charge: {np.mean(q_values)*1e15:.2f} ± {np.std(q_values)*1e15:.2f} fC")
        print("="*60)

        return results


class Visualizer:
    """可視化クラス"""

    @staticmethod
    def plot_trajectories(result: Dict, output_file: str = 'trajectory_plot.png'):
        """軌道の3Dプロット"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 電子軌道
        for traj in result['electron_trajectories']:
            ax.plot(traj[:, 0]*1e6, traj[:, 1]*1e6, traj[:, 2]*1e6,
                   'b-', alpha=0.6, linewidth=1, label='Electron' if traj is result['electron_trajectories'][0] else '')

        # 正孔軌道
        for traj in result['hole_trajectories']:
            ax.plot(traj[:, 0]*1e6, traj[:, 1]*1e6, traj[:, 2]*1e6,
                   'r-', alpha=0.6, linewidth=1, label='Hole' if traj is result['hole_trajectories'][0] else '')

        # 初期位置
        pos = result['initial_positions'][:100]  # 最初の100個だけプロット
        ax.scatter(pos[:, 0]*1e6, pos[:, 1]*1e6, pos[:, 2]*1e6,
                  c='green', marker='o', s=1, alpha=0.3, label='Initial positions')

        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        ax.set_zlabel('Z (depth) [μm]')
        ax.set_title(f'Carrier Trajectories (CCE = {result["cce"]*100:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved trajectory plot: {output_file}")

    @staticmethod
    def plot_charge_histogram(results: List[Dict],
                             experimental_data: Optional[pd.DataFrame] = None,
                             output_file: str = 'charge_histogram.png'):
        """収集電荷のヒストグラム"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 収集電荷 [fC]
        q_collected = np.array([r['q_collected'] for r in results]) * 1e15

        # シミュレーション結果
        ax1.hist(q_collected, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(q_collected.mean(), color='red', linestyle='--',
                   label=f'Mean = {q_collected.mean():.2f} fC')
        ax1.set_xlabel('Collected Charge [fC]')
        ax1.set_ylabel('Counts')
        ax1.set_title('Simulation: Collected Charge Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CCE分布
        cce_values = np.array([r['cce'] for r in results]) * 100
        ax2.hist(cce_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(cce_values.mean(), color='red', linestyle='--',
                   label=f'Mean = {cce_values.mean():.2f}%')
        ax2.set_xlabel('CCE [%]')
        ax2.set_ylabel('Counts')
        ax2.set_title('CCE Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved charge histogram: {output_file}")

    @staticmethod
    def plot_experiment_comparison(results: List[Dict],
                                   experimental_data: pd.DataFrame,
                                   output_file: str = 'experiment_comparison.png'):
        """実験データとの比較"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # シミュレーション結果
        q_sim = np.array([r['q_collected'] for r in results]) * 1e15  # fC
        cce_sim = np.array([r['cce'] for r in results]) * 100  # %

        # 実験データ（PeakHeightをfCに変換する必要がある）
        # ここでは仮にPeakHeightが直接電圧[V]として記録されていると仮定
        # 実際の変換係数はプリアンプのゲインに依存
        if 'PeakHeight' in experimental_data.columns:
            peak_height_V = experimental_data['PeakHeight'].values

            # プリアンプの感度を仮定（例: 1 V/pC = 1000 V/fC）
            # この値は実験セットアップに応じて調整が必要
            sensitivity = 1000  # V/fC
            q_exp = peak_height_V * 1000 / sensitivity  # fC

            # ヒストグラム比較
            ax1 = axes[0, 0]
            ax1.hist(q_sim, bins=30, alpha=0.5, color='blue', label='Simulation', density=True)
            ax1.hist(q_exp, bins=30, alpha=0.5, color='red', label='Experiment', density=True)
            ax1.set_xlabel('Collected Charge [fC]')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Charge Distribution Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 統計比較
            ax2 = axes[0, 1]
            labels = ['Simulation', 'Experiment']
            means = [q_sim.mean(), q_exp.mean()]
            stds = [q_sim.std(), q_exp.std()]
            x_pos = np.arange(len(labels))
            ax2.bar(x_pos, means, yerr=stds, alpha=0.7, color=['blue', 'red'],
                   capsize=10, edgecolor='black')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Collected Charge [fC]')
            ax2.set_title('Mean and Std Comparison')
            ax2.grid(True, alpha=0.3, axis='y')

            # Q-Qプロット
            ax3 = axes[1, 0]
            q_sim_sorted = np.sort(q_sim)
            q_exp_sorted = np.sort(q_exp)
            # 長さを合わせる
            n_min = min(len(q_sim_sorted), len(q_exp_sorted))
            ax3.scatter(q_exp_sorted[:n_min], q_sim_sorted[:n_min], alpha=0.5)
            lims = [min(q_exp_sorted.min(), q_sim_sorted.min()),
                   max(q_exp_sorted.max(), q_sim_sorted.max())]
            ax3.plot(lims, lims, 'r--', alpha=0.7, label='y=x')
            ax3.set_xlabel('Experiment [fC]')
            ax3.set_ylabel('Simulation [fC]')
            ax3.set_title('Q-Q Plot')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 差の分布
            ax4 = axes[1, 1]
            # 平均値の差
            diff = q_sim.mean() - q_exp.mean()
            relative_diff = diff / q_exp.mean() * 100
            ax4.text(0.5, 0.7, f'Simulation mean: {q_sim.mean():.2f} fC',
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.5, 0.5, f'Experiment mean: {q_exp.mean():.2f} fC',
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.5, 0.3, f'Difference: {diff:.2f} fC ({relative_diff:.1f}%)',
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('Statistical Summary')
            ax4.axis('off')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved experiment comparison: {output_file}")

    @staticmethod
    def save_statistics(results: List[Dict], output_file: str = 'cce_statistics.txt'):
        """統計情報をテキストファイルに保存"""
        cce_values = np.array([r['cce'] for r in results]) * 100
        q_values = np.array([r['q_collected'] for r in results]) * 1e15

        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CCE Simulation Statistics\n")
            f.write("="*60 + "\n\n")

            f.write(f"Number of events: {len(results)}\n\n")

            f.write("CCE Statistics:\n")
            f.write(f"  Mean: {cce_values.mean():.2f}%\n")
            f.write(f"  Std:  {cce_values.std():.2f}%\n")
            f.write(f"  Min:  {cce_values.min():.2f}%\n")
            f.write(f"  Max:  {cce_values.max():.2f}%\n\n")

            f.write("Collected Charge Statistics:\n")
            f.write(f"  Mean: {q_values.mean():.3f} fC\n")
            f.write(f"  Std:  {q_values.std():.3f} fC\n")
            f.write(f"  Min:  {q_values.min():.3f} fC\n")
            f.write(f"  Max:  {q_values.max():.3f} fC\n\n")

            f.write("Carrier Collection Efficiency:\n")
            for i, r in enumerate(results[:10]):  # 最初の10イベント
                f.write(f"  Event {i+1}: ")
                f.write(f"e={r['n_electron_collected']}/{r['n_total']} ")
                f.write(f"h={r['n_hole_collected']}/{r['n_total']} ")
                f.write(f"CCE={r['cce']*100:.2f}%\n")

            if len(results) > 10:
                f.write(f"  ... (and {len(results)-10} more events)\n")

        print(f"Saved statistics: {output_file}")


def main():
    """メイン関数"""

    # ==================== パラメータ設定 ====================

    # ファイルパス設定（必要に応じて変更してください）
    # Linux/Mac の場合
    field_file = '電界/yokogata_field.npz'
    srim_file = 'data/5486keVαinSiCIONIZ.txt'
    exp_file = 'data/SiC2_500_10_clear_α_20250124_142116_100.0V.csv'

    # Windows の場合は以下のようにフルパスで指定（コメント外して使用）
    # field_file = r'C:\Users\discu\デスクトップ\python\cce\電界\yokogata_field.npz'
    # srim_file = r'C:\Users\discu\デスクトップ\python\cce\5486keVαinSiCIONIZ.txt'
    # 横型検出器の場合：
    # exp_file = r'C:\Users\discu\デスクトップ\python\cce\実験データ\SiC2_500_10_clear_α_20250124_142116_100.0V.csv'
    # くし形検出器の場合：
    # exp_file = r'C:\Users\discu\デスクトップ\python\cce\実験データ\くし形100V_204222.csv'

    # シミュレーションパラメータ
    sampling_ratio = 0.001  # 0.1% をサンプリング（メモリ節約）
                            # 1.0 = 全キャリア、0.01 = 1%、0.001 = 0.1%
    n_events = 10           # イベント数

    # ==================== ファイルチェック ====================

    # ファイルの存在チェック
    if not os.path.exists(field_file):
        print(f"ERROR: Field file not found: {field_file}")
        print("\nPlease either:")
        print("  1. Run generate_test_field.py to create test data")
        print("  2. Run build_and_validate_fields.py with real OpenSTF data")
        print("  3. Update the field_file path in main()")
        return

    if not os.path.exists(srim_file):
        print(f"ERROR: SRIM file not found: {srim_file}")
        print("\nPlease place the SRIM ionization file at: {srim_file}")
        print("Or update the srim_file path in main().")
        return

    # ==================== シミュレーション開始 ====================

    print("\n" + "="*70)
    print("SiC Detector CCE Simulation")
    print("="*70)
    print(f"Sampling ratio: {sampling_ratio*100:.2f}%")
    print(f"Number of events: {n_events}")

    simulator = CCESimulator(field_file, srim_file)

    # === ステップ1: 単一イベントのテスト ===
    print("\n" + "="*70)
    print("STEP 1: Single event test with trajectory visualization")
    print("="*70)

    result_single = simulator.simulate_single_alpha(
        n_sample_trajectories=5,
        sampling_ratio=sampling_ratio
    )

    # 軌道の可視化
    Visualizer.plot_trajectories(result_single, 'trajectory_plot.png')

    # === ステップ2: 複数イベントのシミュレーション ===
    print("\n" + "="*70)
    print("STEP 2: Multiple events simulation")
    print("="*70)

    results_multiple = simulator.simulate_multiple_alphas(
        n_events=n_events,
        sampling_ratio=sampling_ratio
    )

    # ヒストグラムの作成
    Visualizer.plot_charge_histogram(results_multiple, output_file='charge_histogram.png')

    # === ステップ3: 実験データとの比較 ===
    if os.path.exists(exp_file):
        print("\n" + "="*70)
        print("STEP 3: Comparison with experimental data")
        print("="*70)

        # タブ区切り（TSV）またはカンマ区切り（CSV）を自動判定
        try:
            # まずタブ区切りを試す
            exp_data = pd.read_csv(exp_file, sep='\t')
            if len(exp_data.columns) == 1:
                # 1列しかない場合はカンマ区切りを試す
                exp_data = pd.read_csv(exp_file, sep=',')
        except:
            # カンマ区切りで再試行
            exp_data = pd.read_csv(exp_file, sep=',')

        print(f"Loaded experimental data: {len(exp_data)} events")
        print(f"Columns: {list(exp_data.columns)}")

        Visualizer.plot_experiment_comparison(
            results_multiple, exp_data,
            output_file='experiment_comparison.png'
        )
    else:
        print(f"\nWARNING: Experimental data file not found: {exp_file}")
        print("Skipping experimental comparison.")

    # 統計情報の保存
    Visualizer.save_statistics(results_multiple, 'cce_statistics.txt')

    print("\n" + "="*70)
    print("Simulation completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - trajectory_plot.png")
    print("  - charge_histogram.png")
    if os.path.exists(exp_file):
        print("  - experiment_comparison.png")
    print("  - cce_statistics.txt")
    print("\n")


if __name__ == '__main__':
    main()
