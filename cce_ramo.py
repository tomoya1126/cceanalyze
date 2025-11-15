#!/usr/bin/env python3
"""
Shockley-Ramo定理に基づくSiC検出器CCE計算（Numba高速化版）

α線入射時の電荷収集効率（CCE）を、重み電位（weighting potential）を用いて
高速に計算します。キャリアの個別追跡は行わず、Shockley-Ramoの定理から
直接誘導電荷を計算します。

高速化技術：
- Numba JITコンパイル（SOR法の内部ループ）
- マルチスレッド並列化（nb.prange）
- 重み電位のnpzキャッシュ

物理定数：
- SiC 平均電子正孔対生成エネルギー: W_eh = 7.8 eV
- 電気素量: e = 1.602e-19 C
- α線エネルギー: 5.486 MeV (Am-241)
"""

import argparse
import os
from typing import Optional
import hashlib
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: numba not available. Falling back to pure numpy (very slow).")
    print("Install numba for 10-100x speedup: pip install numba")

# ========== 物理定数 ==========
W_EH = 7.8  # eV - SiC の平均電子正孔対生成エネルギー
Q_E = 1.602e-19  # C - 電気素量
E_ALPHA = 5.486e6  # eV - α線エネルギー


# ========== 電界データ読み込み ==========

def load_field_npz(path: str) -> dict:
    """
    電界データnpzファイルを読み込む。

    Parameters
    ----------
    path : str
        npzファイルパス

    Returns
    -------
    dict
        'V': 3D電位 [V], shape (nz, ny, nx) (optional)
        'Ex', 'Ey', 'Ez': 3D電界 [V/m]
        'X', 'Y', 'Z': 1D座標軸 [m]

    Notes
    -----
    WARNING: 実際のnpzファイルはα線入射表面領域（z=410〜430 μm付近）のみを
             含んでいます。バルク領域（z=0〜410 μm）は含まれていません。
             ramo_drift モードでは、この制約を考慮した近似が使われます。

    'V' (電位) はオプショナルです。含まれていない場合は None が返されます。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Field file not found: {path}")

    data = np.load(path)

    print(f"Loading field file: {path}")
    print(f"  Available keys: {list(data.keys())}")

    # キー名の確認と取得
    result = {}

    # 座標軸（X, Y, Z または x, y, z）
    for key in ['X', 'x']:
        if key in data.files:
            result['X'] = data[key]
            break
    for key in ['Y', 'y']:
        if key in data.files:
            result['Y'] = data[key]
            break
    for key in ['Z', 'z']:
        if key in data.files:
            result['Z'] = data[key]
            break

    # 必須キーのチェック
    if 'X' not in result or 'Y' not in result or 'Z' not in result:
        raise KeyError(f"Field file must contain X, Y, Z coordinates. Found keys: {list(data.keys())}")

    # 電位（オプショナル）
    if 'V' in data.files:
        result['V'] = data['V']
    elif 'v' in data.files:
        result['V'] = data['v']
    else:
        print("  WARNING: 'V' (potential) not found in field file")
        result['V'] = None

    # 電界成分（必須）
    required_fields = ['Ex', 'Ey', 'Ez']
    for field in required_fields:
        if field in data.files:
            result[field] = data[field]
        elif field.lower() in data.files:
            result[field] = data[field.lower()]
        else:
            raise KeyError(f"Required field '{field}' not found in {path}. Available: {list(data.keys())}")

    print(f"  Grid: {len(result['X'])} x {len(result['Y'])} x {len(result['Z'])}")
    print(f"  X: [{result['X'].min()*1e6:.2f}, {result['X'].max()*1e6:.2f}] μm")
    print(f"  Y: [{result['Y'].min()*1e6:.2f}, {result['Y'].max()*1e6:.2f}] μm")
    print(f"  Z: [{result['Z'].min()*1e6:.2f}, {result['Z'].max()*1e6:.2f}] μm")

    if result['V'] is not None:
        print(f"  V: [{result['V'].min():.2f}, {result['V'].max():.2f}] V")

    # Z範囲の警告（表面領域のみの場合）
    z_min_um = result['Z'].min() * 1e6
    if z_min_um > 50:  # 50 μm 以上から始まる場合は表面領域のみ
        print(f"  NOTE: This field only covers the alpha-incident surface region.")
        print(f"        Bulk region (z < {z_min_um:.0f} μm) is NOT included in this npz.")
        print(f"        For ramo_drift mode, E-field in bulk is approximated from surface values.")

    return result


# ========== SRIM IONIZ読み込み ==========

def load_srim_ioniz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    SRIM IONIZ ファイルを読み込む。

    Parameters
    ----------
    path : str
        IONIZ ファイルパス

    Returns
    -------
    z_array : np.ndarray
        深さ [m]
    dE_array : np.ndarray
        各セグメントでのエネルギー付与 [eV]

    Notes
    -----
    TODO: 実際のSRIM出力の列構成に合わせて調整すること。
          現在は「1列目=深さ[Å]、2列目=イオン化[eV/Å]」と仮定。
    """
    data_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # TODO: 列インデックスを実際のSRIM出力に合わせて調整
                    depth_angstrom = float(parts[0])  # 深さ [Å]
                    ioniz_eV_per_angstrom = float(parts[1])  # [eV/Å]
                    data_rows.append([depth_angstrom, ioniz_eV_per_angstrom])
                except ValueError:
                    continue

    if not data_rows:
        raise ValueError(f"No valid data found in {path}")

    data = np.array(data_rows)
    depth_angstrom = data[:, 0]
    ioniz_eV_per_angstrom = data[:, 1]

    # 単位変換
    z_array = depth_angstrom * 1e-10  # Å → m

    # 各セグメントでのエネルギー付与を計算
    # ΔE_i ≈ (dE/dz)_i * Δz_i
    dz = np.diff(z_array)
    dz = np.append(dz, dz[-1])  # 最後のセグメントも同じ幅と仮定

    dE_array = ioniz_eV_per_angstrom * (dz * 1e10)  # [eV/Å] * [Å] = [eV]

    print(f"\nLoaded SRIM IONIZ from {path}")
    print(f"  Segments: {len(z_array)}")
    print(f"  Depth range: [0, {z_array[-1]*1e6:.2f}] μm")
    print(f"  Total energy: {dE_array.sum()/1e6:.3f} MeV")
    print(f"  Expected e-h pairs: {dE_array.sum()/W_EH:.3e}")

    return z_array, dE_array


# ========== 電極マスク作成 ==========

def create_electrode_masks(
    V: np.ndarray,
    Z: np.ndarray,
    z_surface: float = 430e-6,
    eps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    z=430μm面の電位分布から、収集電極とグラウンド電極のマスクを作成。

    Parameters
    ----------
    V : np.ndarray
        3D電位 [V], shape (nz, ny, nx)
    Z : np.ndarray
        z座標軸 [m]
    z_surface : float
        電極面のz座標 [m]
    eps : float
        電位判定の許容誤差 [V]

    Returns
    -------
    collect_mask : np.ndarray
        収集電極マスク（True=電極）, shape (nz, ny, nx)
    ground_mask : np.ndarray
        グラウンド電極マスク, shape (nz, ny, nx)
    k_surface : int
        表面のzインデックス

    Notes
    -----
    TODO: 裏面電極など他の電極も考慮する場合はここを拡張すること。
    """
    # z=430μm に最も近いインデックス
    k_surface = int(np.argmin(np.abs(Z - z_surface)))
    z_actual = Z[k_surface] * 1e6  # μm

    print(f"\nCreating electrode masks at z={z_actual:.2f} μm (index {k_surface})")

    # 表面の電位分布
    V_surf = V[k_surface, :, :]
    V_max = V_surf.max()
    V_min = V_surf.min()

    print(f"  Surface potential: [{V_min:.2f}, {V_max:.2f}] V")
    print(f"  Epsilon: {eps} V")

    # 電極マスク作成
    collect_mask = np.zeros_like(V, dtype=bool)
    ground_mask = np.zeros_like(V, dtype=bool)

    # 表面のみ
    collect_mask[k_surface, :, :] = (V_surf > V_max - eps)
    ground_mask[k_surface, :, :] = (V_surf < V_min + eps)

    n_collect = collect_mask.sum()
    n_ground = ground_mask.sum()

    print(f"  Collect electrode cells: {n_collect}")
    print(f"  Ground electrode cells: {n_ground}")

    # TODO: 必要なら裏面など他の電極マスクも追加すること

    return collect_mask, ground_mask, k_surface


# ========== 重み電位計算（Numba高速化版） ==========

if NUMBA_AVAILABLE:
    @nb.njit(parallel=True, fastmath=True)
    def sor_step_numba(phi_w: np.ndarray, fixed: np.ndarray, omega: float) -> float:
        """
        SOR (Successive Over-Relaxation) 1ステップ（Numba JITコンパイル版）

        Parameters
        ----------
        phi_w : np.ndarray
            重み電位, shape (nz, ny, nx)
        fixed : np.ndarray
            固定セルマスク（True=電極）
        omega : float
            緩和係数（1.0=Gauss-Seidel, 1.0-2.0でSOR）

        Returns
        -------
        float
            最大変化量

        Notes
        -----
        ホットループ: nb.prange で並列化。
        均一グリッドを仮定（dx=dy=dz）。

        FIXED: 各スレッドのlocal_max_diffを配列に保存してから max() を取る。
        parallel=True で共有スカラー変数を更新するとrace conditionが発生する。
        """
        nz, ny, nx = phi_w.shape

        # 各z層（スレッド）のmax_diffを保存する配列
        thread_max_diffs = np.zeros(nz, dtype=np.float64)

        # z方向は並列化（データ競合を避けるため）
        for k in nb.prange(1, nz-1):
            local_max_diff = 0.0
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    if fixed[k, j, i]:
                        continue

                    # 6点ステンシル（均一グリッド）
                    phi_old = phi_w[k, j, i]
                    phi_new = (
                        phi_w[k, j, i-1] + phi_w[k, j, i+1] +
                        phi_w[k, j-1, i] + phi_w[k, j+1, i] +
                        phi_w[k-1, j, i] + phi_w[k+1, j, i]
                    ) / 6.0

                    # SOR更新
                    phi_w[k, j, i] = phi_old + omega * (phi_new - phi_old)

                    # 局所最大差分
                    diff = abs(phi_w[k, j, i] - phi_old)
                    if diff > local_max_diff:
                        local_max_diff = diff

            # 各スレッドの結果を配列に保存（これはスレッドセーフ）
            thread_max_diffs[k] = local_max_diff

        # 全スレッドの結果から最大値を取る
        max_diff = thread_max_diffs.max()
        return max_diff


def solve_weighting_potential(
    V: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    collect_mask: np.ndarray,
    ground_mask: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    num_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Laplace方程式 ∇²φ_w = 0 を3D有限差分（SOR法）で解く。

    境界条件:
    - collect_mask == True: φ_w = 1
    - ground_mask == True: φ_w = 0
    - その他: Laplace方程式

    Parameters
    ----------
    V : np.ndarray
        参照用の電位（グリッドサイズ取得用）
    X, Y, Z : np.ndarray
        座標軸 [m]
    collect_mask, ground_mask : np.ndarray
        電極マスク
    max_iter : int
        最大反復回数
    tol : float
        収束判定の閾値
    omega : float
        SOR緩和係数（1.0=Gauss-Seidel, 1.6-1.9が最適）
    num_threads : int | None
        Numbaスレッド数（Noneで自動）

    Returns
    -------
    phi_w : np.ndarray
        重み電位, shape (nz, ny, nx)

    Notes
    -----
    Numba利用可能時は並列SOR、不可時は純粋numpyにフォールバック。
    均一グリッドを仮定。
    """
    nz, ny, nx = V.shape

    # スレッド数設定
    if NUMBA_AVAILABLE and num_threads is not None:
        nb.set_num_threads(num_threads)
        print(f"\nNumba threads: {num_threads}")

    # グリッド間隔（均一と仮定）
    dx = X[1] - X[0] if len(X) > 1 else 1e-6
    dy = Y[1] - Y[0] if len(Y) > 1 else 1e-6
    dz = Z[1] - Z[0] if len(Z) > 1 else 1e-6

    print(f"\nSolving weighting potential (SOR, omega={omega:.2f})...")
    print(f"  Grid: {nx} x {ny} x {nz}")
    print(f"  Grid spacing: dx={dx*1e6:.3f} μm, dy={dy*1e6:.3f} μm, dz={dz*1e6:.3f} μm")
    print(f"  Max iterations: {max_iter}, tolerance: {tol}")
    print(f"  Method: {'Numba JIT (parallel)' if NUMBA_AVAILABLE else 'Pure numpy (slow)'}")

    # 初期化
    phi_w = np.zeros((nz, ny, nx), dtype=np.float64)

    # 境界条件
    phi_w[collect_mask] = 1.0
    phi_w[ground_mask] = 0.0

    # 固定セル（電極）
    fixed = (collect_mask | ground_mask).astype(np.bool_)

    # デバッグ情報
    total_cells = nz * ny * nx
    fixed_cells = fixed.sum()
    free_cells = total_cells - fixed_cells
    print(f"  Total cells: {total_cells}")
    print(f"  Fixed cells (electrodes): {fixed_cells} ({fixed_cells/total_cells*100:.2f}%)")
    print(f"  Free cells (to solve): {free_cells} ({free_cells/total_cells*100:.2f}%)")

    # Z層ごとの固定セル数（nz <= 20 の場合のみ表示）
    if nz <= 20:
        print(f"  Fixed cells by z-layer:")
        for k in range(nz):
            fixed_layer = fixed[k, :, :].sum()
            print(f"    z[{k}] = {Z[k]*1e6:6.2f} μm: {fixed_layer:6d} fixed")

    # SOR反復
    if NUMBA_AVAILABLE:
        # Numba高速版
        for iteration in range(max_iter):
            max_diff = sor_step_numba(phi_w, fixed, omega)

            # 収束判定
            if max_diff < tol:
                print(f"  ✓ Converged after {iteration+1} iterations (max diff={max_diff:.3e})")
                break

            if (iteration + 1) % 500 == 0:
                print(f"  Iteration {iteration+1}/{max_iter}, max diff={max_diff:.3e}")
        else:
            print(f"  ⚠ WARNING: Did not converge after {max_iter} iterations (max diff={max_diff:.3e})")
    else:
        # Pure numpy フォールバック（遅い）
        for iteration in range(max_iter):
            phi_w_old = phi_w.copy()

            for k in range(1, nz-1):
                for j in range(1, ny-1):
                    for i in range(1, nx-1):
                        if fixed[k, j, i]:
                            continue

                        phi_new = (
                            phi_w[k, j, i-1] + phi_w[k, j, i+1] +
                            phi_w[k, j-1, i] + phi_w[k, j+1, i] +
                            phi_w[k-1, j, i] + phi_w[k+1, j, i]
                        ) / 6.0

                        phi_w[k, j, i] = phi_w[k, j, i] + omega * (phi_new - phi_w[k, j, i])

            max_diff = np.abs(phi_w - phi_w_old).max()
            if max_diff < tol:
                print(f"  ✓ Converged after {iteration+1} iterations (max diff={max_diff:.3e})")
                break

            if (iteration + 1) % 500 == 0:
                print(f"  Iteration {iteration+1}/{max_iter}, max diff={max_diff:.3e}")
        else:
            print(f"  ⚠ WARNING: Did not converge after {max_iter} iterations")

    print(f"  φ_w range: [{phi_w.min():.6f}, {phi_w.max():.6f}]")

    # φ_w の統計（nz <= 20 の場合のみz層ごとに表示）
    if nz <= 20:
        print(f"  φ_w statistics by z-layer:")
        for k in range(nz):
            phi_layer = phi_w[k, :, :]
            n_zero = np.sum(phi_layer == 0.0)
            n_one = np.sum(phi_layer == 1.0)
            n_mid = np.sum((phi_layer > 0.0) & (phi_layer < 1.0))
            phi_mid_mean = phi_layer[(phi_layer > 0.0) & (phi_layer < 1.0)].mean() if n_mid > 0 else 0.0
            print(f"    z[{k}] = {Z[k]*1e6:6.2f} μm: 0.0={n_zero:5d}, 1.0={n_one:5d}, mid={n_mid:5d} (mean={phi_mid_mean:.4f})")

    return phi_w


# ========== 三線形補間 ==========

def trilinear_interpolate(
    phi_w: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    3D配列から指定座標の値を三線形補間で取得。

    Parameters
    ----------
    phi_w : np.ndarray
        3D配列, shape (nz, ny, nx)
    X, Y, Z : np.ndarray
        座標軸 [m]
    x, y, z : float
        補間点座標 [m]

    Returns
    -------
    float
        補間値
    """
    # 範囲チェック
    if not (X[0] <= x <= X[-1] and Y[0] <= y <= Y[-1] and Z[0] <= z <= Z[-1]):
        return 0.0  # 範囲外は0と仮定

    # インデックス検索
    ix = np.searchsorted(X, x) - 1
    iy = np.searchsorted(Y, y) - 1
    iz = np.searchsorted(Z, z) - 1

    # 境界処理
    ix = max(0, min(ix, len(X) - 2))
    iy = max(0, min(iy, len(Y) - 2))
    iz = max(0, min(iz, len(Z) - 2))

    # 相対位置
    tx = (x - X[ix]) / (X[ix+1] - X[ix]) if X[ix+1] != X[ix] else 0.0
    ty = (y - Y[iy]) / (Y[iy+1] - Y[iy]) if Y[iy+1] != Y[iy] else 0.0
    tz = (z - Z[iz]) / (Z[iz+1] - Z[iz]) if Z[iz+1] != Z[iz] else 0.0

    # 三線形補間
    c000 = phi_w[iz,   iy,   ix]
    c001 = phi_w[iz,   iy,   ix+1]
    c010 = phi_w[iz,   iy+1, ix]
    c011 = phi_w[iz,   iy+1, ix+1]
    c100 = phi_w[iz+1, iy,   ix]
    c101 = phi_w[iz+1, iy,   ix+1]
    c110 = phi_w[iz+1, iy+1, ix]
    c111 = phi_w[iz+1, iy+1, ix+1]

    c00 = c000 * (1 - tx) + c001 * tx
    c01 = c010 * (1 - tx) + c011 * tx
    c10 = c100 * (1 - tx) + c101 * tx
    c11 = c110 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty

    c = c0 * (1 - tz) + c1 * tz

    return c


# ========== 重み電位キャッシュ ==========

def get_weighting_potential(
    field_path: str,
    cache_path: Optional[str] = None,
    force_recalc: bool = False,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    num_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    重み電位 φ_w をキャッシュから読み込み、またはキャッシュがなければ計算して保存。

    Parameters
    ----------
    field_path : str
        電界npzファイルパス
    cache_path : str | None
        φ_w を保存/読込する npz のパス。None の場合は field_path から自動決定。
    force_recalc : bool
        True の場合、キャッシュを無視して再計算
    max_iter : int
        重み電位計算の最大反復回数
    tol : float
        重み電位計算の収束判定閾値
    omega : float
        SOR緩和パラメータ (1.0 < omega < 2.0, 最適値は1.6~1.9)
    num_threads : int | None
        Numbaスレッド数。Noneの場合は自動。

    Returns
    -------
    phi_w : np.ndarray
        重み電位 (Nz, Ny, Nx)
    X, Y, Z : np.ndarray
        座標軸 [m]

    Notes
    -----
    キャッシュファイル名:
        yokogata_field.npz → yokogata_weighting.npz
        kushigata_field.npz → kushigata_weighting.npz
    """
    # キャッシュパスの決定
    if cache_path is None:
        # field_path から自動生成
        # 例: "電界/yokogata_field.npz" → "電界/yokogata_weighting.npz"
        base_name = os.path.basename(field_path)
        dir_name = os.path.dirname(field_path)

        if "_field.npz" in base_name:
            cache_name = base_name.replace("_field.npz", "_weighting.npz")
        else:
            # フォールバック: .npz の前に _weighting を挿入
            cache_name = base_name.replace(".npz", "_weighting.npz")

        cache_path = os.path.join(dir_name, cache_name)

    print(f"\n{'='*70}")
    print("Weighting Potential")
    print('='*70)
    print(f"  Field: {field_path}")
    print(f"  Cache: {cache_path}")

    # キャッシュ読み込み試行
    if not force_recalc and os.path.exists(cache_path):
        try:
            print(f"  ✓ Loading weighting potential from cache...")
            cache_data = np.load(cache_path)
            phi_w = cache_data['phi_w']
            X = cache_data['X']
            Y = cache_data['Y']
            Z = cache_data['Z']

            print(f"  ✓ Cache loaded successfully!")
            print(f"     Grid: {len(X)} x {len(Y)} x {len(Z)}")
            print(f"     phi_w range: [{phi_w.min():.4f}, {phi_w.max():.4f}]")

            return phi_w, X, Y, Z

        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}")
            print(f"  → Recomputing...")

    # キャッシュなし/無効 → 計算する
    if force_recalc:
        print(f"  → Force recalculation (--force-recalc-weighting)")
    else:
        print(f"  → Cache not found, computing...")

    # 1. 電界データ読み込み
    field_data = load_field_npz(field_path)
    V = field_data['V']
    X = field_data['X']
    Y = field_data['Y']
    Z = field_data['Z']

    # 2. 電極マスク作成
    collect_mask, ground_mask, k_surface = create_electrode_masks(V, Z)

    # 3. 重み電位計算
    phi_w = solve_weighting_potential(
        V, X, Y, Z, collect_mask, ground_mask,
        max_iter=max_iter, tol=tol, omega=omega, num_threads=num_threads
    )

    # 4. キャッシュに保存
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            phi_w=phi_w,
            X=X, Y=Y, Z=Z
        )
        print(f"\n  ✓ Weighting potential saved to cache: {cache_path}")
        print(f"     File size: {os.path.getsize(cache_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"  ⚠ Failed to save cache: {e}")

    return phi_w, X, Y, Z


# ========== 全厚メッシュ生成（OpenSTF風） ==========

def create_fullthickness_mesh(
    field_path: str,
    z_max: float = 430e-6,
    target_dz: float = 2.5e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OpenSTF風の全厚メッシュを生成（z=0 から z_max まで）。

    実電界データから X, Y を取得し、Z は 0 から z_max まで均一刻みで生成。

    Parameters
    ----------
    field_path : str
        電界npzファイルパス（X, Yの範囲取得用）
    z_max : float
        最大z座標 [m]、デフォルトは430μm
    target_dz : float
        目標z刻み幅 [m]、デフォルトは2.5μm

    Returns
    -------
    X, Y, Z : np.ndarray
        座標軸 [m]
    V_template : np.ndarray
        テンプレート配列 shape (nz, ny, nx)、全て0で初期化

    Notes
    -----
    実電界の表面電位分布は別途読み込んで電極マスク作成に使用する。
    """
    # 実電界データから X, Y を取得
    field_data = load_field_npz(field_path)
    X_field = field_data['X']
    Y_field = field_data['Y']

    # X, Y はそのまま使用
    X = X_field.copy()
    Y = Y_field.copy()

    # Z を新たに生成（0 から z_max まで均一刻み）
    nz = int(np.ceil(z_max / target_dz)) + 1
    Z = np.linspace(0, z_max, nz)

    nx = len(X)
    ny = len(Y)

    # テンプレート配列
    V_template = np.zeros((nz, ny, nx), dtype=np.float64)

    print(f"\n{'='*70}")
    print("Full-thickness mesh generation (OpenSTF-style)")
    print('='*70)
    print(f"  X: [{X.min()*1e6:.1f}, {X.max()*1e6:.1f}] μm, n={nx}")
    print(f"  Y: [{Y.min()*1e6:.1f}, {Y.max()*1e6:.1f}] μm, n={ny}")
    print(f"  Z: [{Z.min()*1e6:.1f}, {Z.max()*1e6:.1f}] μm, n={nz}")
    dx = X[1] - X[0] if len(X) > 1 else 0
    dy = Y[1] - Y[0] if len(Y) > 1 else 0
    dz = Z[1] - Z[0] if len(Z) > 1 else 0
    print(f"  Grid spacing: dx={dx*1e6:.3f} μm, dy={dy*1e6:.3f} μm, dz={dz*1e6:.3f} μm")
    print(f"  Total cells: {nx} × {ny} × {nz} = {nx*ny*nz:,}")

    return X, Y, Z, V_template


def create_electrode_masks_fullthickness(
    field_path: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    z_surface: float = 430e-6,
    eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OpenSTF風の電極マスクを作成（裏面 + 表面電極パターン）。

    Parameters
    ----------
    field_path : str
        電界npzファイルパス（表面電位分布取得用）
    X, Y, Z : np.ndarray
        全厚メッシュの座標軸 [m]
    z_surface : float
        電極面のz座標 [m]、デフォルトは430μm
    eps : float
        電極判定の閾値（電位範囲の割合、0-1）
        例: eps=0.1 → 上位10%を収集電極、下位10%をグラウンド電極
        残りの中間領域はNeumann境界条件 (∂φ/∂z=0)

    Returns
    -------
    collect_mask : np.ndarray
        収集電極マスク（True=電極、φ_w=1）, shape (nz, ny, nx)
    ground_mask : np.ndarray
        グラウンド電極マスク（True=電極、φ_w=0）, shape (nz, ny, nx)

    Notes
    -----
    境界条件：
    - 裏面 z=0: Neumann境界条件（∂φ/∂z=0）
    - 表面 z=z_surface: 実電界の電位分布から収集/グラウンド電極を判定
    """
    nz, ny, nx = len(Z), len(Y), len(X)

    # 電極マスク初期化
    collect_mask = np.zeros((nz, ny, nx), dtype=bool)
    ground_mask = np.zeros((nz, ny, nx), dtype=bool)

    print(f"\nCreating electrode masks (OpenSTF-style, full-thickness)")

    # 1. 裏面 z=0: Neumann境界条件（∂φ/∂z=0）に変更
    # 以前は全面グラウンド電極（Dirichlet, φ_w=0）だったが、
    # よりフレキシブルなNeumann条件に変更
    k_back = 0
    print(f"  Backside (z={Z[k_back]*1e6:.2f} μm): Neumann BC (∂φ/∂z=0)")

    # 2. 表面 z=z_surface: 実電界データから電極パターンを取得
    # 実電界データを読み込み
    field_data = load_field_npz(field_path)
    V_field = field_data['V']
    Z_field = field_data['Z']

    if V_field is None:
        raise ValueError(
            f"Field file {field_path} does not contain voltage data ('V').\n"
            "Cannot determine electrode pattern without voltage distribution.\n"
            "Please use a field file that includes 'V' data."
        )

    # 表面に最も近いインデックスを見つける
    k_surface_full = int(np.argmin(np.abs(Z - z_surface)))
    k_surface_field = int(np.argmin(np.abs(Z_field - z_surface)))

    z_actual_full = Z[k_surface_full] * 1e6
    z_actual_field = Z_field[k_surface_field] * 1e6

    print(f"  Surface electrode plane:")
    print(f"    Full mesh: z[{k_surface_full}] = {z_actual_full:.2f} μm")
    print(f"    Field data: z[{k_surface_field}] = {z_actual_field:.2f} μm")

    # 表面の電位分布を取得
    V_surf = V_field[k_surface_field, :, :]
    V_max = V_surf.max()
    V_min = V_surf.min()

    print(f"    Surface potential: [{V_min:.2f}, {V_max:.2f}] V")
    print(f"    Epsilon: {eps} V")

    # 電極パターンの判定（実電界データのサイズに合わせる）
    # 注意: X, Y のサイズが field と full で同じと仮定
    if V_surf.shape != (ny, nx):
        print(f"    WARNING: Size mismatch! V_surf{V_surf.shape} vs full({ny},{nx})")
        print(f"    Using interpolation or padding...")
        # 簡易対応: V_surf を full mesh にコピー（同サイズと仮定）
        V_surf_resized = np.zeros((ny, nx))
        ny_min = min(V_surf.shape[0], ny)
        nx_min = min(V_surf.shape[1], nx)
        V_surf_resized[:ny_min, :nx_min] = V_surf[:ny_min, :nx_min]
        V_surf = V_surf_resized
        V_max = V_surf.max()
        V_min = V_surf.min()

    # 収集電極 (高電位側) とグラウンド電極 (低電位側) を判定
    # 電極パターンのみを固定し、それ以外はNeumann境界条件とする
    V_range = V_max - V_min
    # epsで指定された割合を電極とする
    collect_threshold = V_max - eps * V_range
    ground_threshold = V_min + eps * V_range

    collect_mask[k_surface_full, :, :] = (V_surf > collect_threshold)
    ground_mask[k_surface_full, :, :] = (V_surf < ground_threshold)

    n_collect = collect_mask[k_surface_full, :, :].sum()
    n_ground_surf = ground_mask[k_surface_full, :, :].sum()
    n_neumann = ((ny * nx) - n_collect - n_ground_surf)

    print(f"    Electrode threshold: {eps*100:.1f}% of voltage range ({V_range:.2f} V)")
    print(f"    Collect electrode: {n_collect} cells → φ_w=1 (V > {collect_threshold:.2f})")
    print(f"    Ground electrode: {n_ground_surf} cells → φ_w=0 (V < {ground_threshold:.2f})")
    print(f"    Neumann BC region: {n_neumann} cells (∂φ/∂z=0 at surface)")

    # 統計
    total_collect = collect_mask.sum()
    total_ground = ground_mask.sum()
    print(f"\n  Total electrode cells:")
    print(f"    Collect: {total_collect}")
    print(f"    Ground: {total_ground} (surface only)")
    print(f"  Note: Backside uses Neumann BC (not fixed electrode)")

    return collect_mask, ground_mask


# ========== Neumann境界条件対応SORソルバー ==========

if NUMBA_AVAILABLE:
    @nb.njit(parallel=True, fastmath=True)
    def sor_step_numba_neumann(
        phi_w: np.ndarray,
        fixed: np.ndarray,
        omega: float,
        apply_neumann: bool = True,
    ) -> float:
        """
        SOR 1ステップ（Neumann境界条件対応版）。

        Parameters
        ----------
        phi_w : np.ndarray
            重み電位, shape (nz, ny, nx)
        fixed : np.ndarray
            固定セル（電極）マスク, shape (nz, ny, nx)
        omega : float
            SOR緩和係数
        apply_neumann : bool
            True の場合、外側境界に Neumann 条件 (∂φ/∂n=0) を適用

        Returns
        -------
        float
            最大変化量

        Notes
        -----
        Neumann境界条件 (∂φ/∂n=0):
        - x=0, x=x_max の面: φ[i=0]=φ[i=1], φ[i=nx-1]=φ[i=nx-2]
        - y=0, y=y_max の面: φ[j=0]=φ[j=1], φ[j=ny-1]=φ[j=ny-2]
        - z=z_max の面（電極以外）: φ[k=nz-1]=φ[k=nz-2]
        - z=0 の面（電極以外）: φ[k=0]=φ[k=1] （裏面もNeumann条件）
        """
        nz, ny, nx = phi_w.shape

        # 各スレッドのmax_diff保存用
        thread_max_diffs = np.zeros(nz, dtype=np.float64)

        # 1. 内部セルの更新（z方向並列化）
        for k in nb.prange(1, nz-1):
            local_max_diff = 0.0
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    if fixed[k, j, i]:
                        continue

                    # 6点ステンシル
                    phi_old = phi_w[k, j, i]
                    phi_new = (
                        phi_w[k, j, i-1] + phi_w[k, j, i+1] +
                        phi_w[k, j-1, i] + phi_w[k, j+1, i] +
                        phi_w[k-1, j, i] + phi_w[k+1, j, i]
                    ) / 6.0

                    phi_w[k, j, i] = phi_old + omega * (phi_new - phi_old)

                    diff = abs(phi_w[k, j, i] - phi_old)
                    if diff > local_max_diff:
                        local_max_diff = diff

            thread_max_diffs[k] = local_max_diff

        # 2. Neumann境界条件の適用（電極でない境界セルのみ）
        if apply_neumann:
            # x=0 面
            for k in range(nz):
                for j in range(ny):
                    if not fixed[k, j, 0]:
                        phi_w[k, j, 0] = phi_w[k, j, 1]

            # x=x_max 面
            for k in range(nz):
                for j in range(ny):
                    if not fixed[k, j, nx-1]:
                        phi_w[k, j, nx-1] = phi_w[k, j, nx-2]

            # y=0 面
            for k in range(nz):
                for i in range(nx):
                    if not fixed[k, 0, i]:
                        phi_w[k, 0, i] = phi_w[k, 1, i]

            # y=y_max 面
            for k in range(nz):
                for i in range(nx):
                    if not fixed[k, ny-1, i]:
                        phi_w[k, ny-1, i] = phi_w[k, ny-2, i]

            # z=z_max 面（電極でない部分のみ）
            for j in range(ny):
                for i in range(nx):
                    if not fixed[nz-1, j, i]:
                        phi_w[nz-1, j, i] = phi_w[nz-2, j, i]

            # z=0 面（裏面）にもNeumann条件を適用（電極でない部分のみ）
            for j in range(ny):
                for i in range(nx):
                    if not fixed[0, j, i]:
                        phi_w[0, j, i] = phi_w[1, j, i]

        max_diff = thread_max_diffs.max()
        return max_diff


def solve_weighting_potential_fullthickness(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    collect_mask: np.ndarray,
    ground_mask: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    num_threads: Optional[int] = None,
    use_neumann: bool = True,
) -> np.ndarray:
    """
    全厚メッシュで重み電位を計算（Neumann境界条件対応）。

    Parameters
    ----------
    X, Y, Z : np.ndarray
        座標軸 [m]
    collect_mask, ground_mask : np.ndarray
        電極マスク, shape (nz, ny, nx)
    max_iter : int
        最大反復回数
    tol : float
        収束判定閾値
    omega : float
        SOR緩和係数
    num_threads : int | None
        Numbaスレッド数
    use_neumann : bool
        True の場合、Neumann境界条件を適用

    Returns
    -------
    phi_w : np.ndarray
        重み電位, shape (nz, ny, nx)
    """
    nz, ny, nx = len(Z), len(Y), len(X)

    # スレッド数設定
    if NUMBA_AVAILABLE and num_threads is not None:
        nb.set_num_threads(num_threads)
        print(f"\nNumba threads: {num_threads}")

    # グリッド間隔
    dx = X[1] - X[0] if len(X) > 1 else 1e-6
    dy = Y[1] - Y[0] if len(Y) > 1 else 1e-6
    dz = Z[1] - Z[0] if len(Z) > 1 else 1e-6

    print(f"\nSolving weighting potential (full-thickness, SOR + Neumann, omega={omega:.2f})...")
    print(f"  Grid: {nx} x {ny} x {nz}")
    print(f"  Grid spacing: dx={dx*1e6:.3f} μm, dy={dy*1e6:.3f} μm, dz={dz*1e6:.3f} μm")
    print(f"  Max iterations: {max_iter}, tolerance: {tol}")
    print(f"  Neumann BC: {'enabled' if use_neumann else 'disabled'}")
    print(f"  Method: {'Numba JIT (parallel)' if NUMBA_AVAILABLE else 'Pure numpy (slow)'}")

    # 初期化
    phi_w = np.zeros((nz, ny, nx), dtype=np.float64)

    # Dirichlet境界条件
    phi_w[collect_mask] = 1.0
    phi_w[ground_mask] = 0.0

    # 固定セル
    fixed = (collect_mask | ground_mask).astype(np.bool_)

    # デバッグ情報
    total_cells = nz * ny * nx
    fixed_cells = fixed.sum()
    free_cells = total_cells - fixed_cells
    print(f"  Total cells: {total_cells:,}")
    print(f"  Fixed cells (electrodes): {fixed_cells:,} ({fixed_cells/total_cells*100:.2f}%)")
    print(f"  Free cells (to solve): {free_cells:,} ({free_cells/total_cells*100:.2f}%)")

    # SOR反復
    if NUMBA_AVAILABLE:
        # Numba高速版（Neumann対応）
        for iteration in range(max_iter):
            max_diff = sor_step_numba_neumann(phi_w, fixed, omega, use_neumann)

            # 収束判定
            if max_diff < tol:
                print(f"  ✓ Converged after {iteration+1} iterations (max diff={max_diff:.3e})")
                break

            if (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration+1}/{max_iter}, max diff={max_diff:.3e}")
        else:
            print(f"  ⚠ WARNING: Did not converge after {max_iter} iterations (max diff={max_diff:.3e})")
    else:
        # Pure numpy フォールバック（Neumann境界条件は省略）
        print("  WARNING: Numba not available. Neumann BC not fully supported in numpy mode.")
        for iteration in range(max_iter):
            phi_w_old = phi_w.copy()

            for k in range(1, nz-1):
                for j in range(1, ny-1):
                    for i in range(1, nx-1):
                        if fixed[k, j, i]:
                            continue

                        phi_new = (
                            phi_w[k, j, i-1] + phi_w[k, j, i+1] +
                            phi_w[k, j-1, i] + phi_w[k, j+1, i] +
                            phi_w[k-1, j, i] + phi_w[k+1, j, i]
                        ) / 6.0

                        phi_w[k, j, i] = phi_w[k, j, i] + omega * (phi_new - phi_w[k, j, i])

            max_diff = np.abs(phi_w - phi_w_old).max()
            if max_diff < tol:
                print(f"  ✓ Converged after {iteration+1} iterations (max diff={max_diff:.3e})")
                break

            if (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration+1}/{max_iter}, max diff={max_diff:.3e}")
        else:
            print(f"  ⚠ WARNING: Did not converge after {max_iter} iterations")

    print(f"  φ_w range: [{phi_w.min():.6f}, {phi_w.max():.6f}]")

    # z層ごとの統計（最大20層まで表示）
    print(f"  φ_w statistics by z-layer (showing every {max(1, nz//20)} layer):")
    step = max(1, nz // 20)
    for k in range(0, nz, step):
        phi_layer = phi_w[k, :, :]
        mask_layer = ~fixed[k, :, :]
        if mask_layer.sum() > 0:
            phi_free = phi_layer[mask_layer]
            print(f"    z[{k:3d}] = {Z[k]*1e6:6.2f} μm: "
                  f"min={phi_layer.min():.4f}, "
                  f"mean(free)={phi_free.mean():.4f}, "
                  f"max={phi_layer.max():.4f}")

    return phi_w


def get_weighting_potential_fullthickness(
    field_path: str,
    cache_path: Optional[str] = None,
    force_recalc: bool = False,
    z_max: float = 430e-6,
    target_dz: float = 2.5e-6,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    num_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OpenSTF風の全厚重み電位をキャッシュから読み込み、または計算して保存。

    Parameters
    ----------
    field_path : str
        電界npzファイルパス
    cache_path : str | None
        キャッシュパス。None の場合は自動決定（*_weighting_fullthick.npz）
    force_recalc : bool
        True の場合、キャッシュを無視して再計算
    z_max : float
        最大z座標 [m]
    target_dz : float
        目標z刻み幅 [m]
    max_iter : int
        SOR最大反復回数
    tol : float
        収束判定閾値
    omega : float
        SOR緩和係数
    num_threads : int | None
        Numbaスレッド数

    Returns
    -------
    phi_w : np.ndarray
        重み電位, shape (nz, ny, nx)
    X, Y, Z : np.ndarray
        座標軸 [m]
    """
    # キャッシュパスの決定
    if cache_path is None:
        base_name = os.path.basename(field_path)
        dir_name = os.path.dirname(field_path)

        if "_field.npz" in base_name:
            cache_name = base_name.replace("_field.npz", "_weighting_fullthick.npz")
        else:
            cache_name = base_name.replace(".npz", "_weighting_fullthick.npz")

        cache_path = os.path.join(dir_name, cache_name)

    print(f"\n{'='*70}")
    print("Weighting Potential (Full-thickness, OpenSTF-style)")
    print('='*70)
    print(f"  Field: {field_path}")
    print(f"  Cache: {cache_path}")

    # キャッシュ読み込み試行
    if not force_recalc and os.path.exists(cache_path):
        try:
            print(f"  ✓ Loading weighting potential from cache...")
            cache_data = np.load(cache_path)
            phi_w = cache_data['phi_w']
            X = cache_data['X']
            Y = cache_data['Y']
            Z = cache_data['Z']

            print(f"  ✓ Cache loaded successfully!")
            print(f"     Grid: {len(X)} x {len(Y)} x {len(Z)}")
            print(f"     phi_w range: [{phi_w.min():.4f}, {phi_w.max():.4f}]")

            return phi_w, X, Y, Z

        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}")
            print(f"  → Recomputing...")

    # キャッシュなし/無効 → 計算する
    if force_recalc:
        print(f"  → Force recalculation")
    else:
        print(f"  → Cache not found, computing...")

    # 1. 全厚メッシュ生成
    X, Y, Z, V_template = create_fullthickness_mesh(field_path, z_max, target_dz)

    # 2. 電極マスク作成（裏面 + 表面パターン）
    collect_mask, ground_mask = create_electrode_masks_fullthickness(
        field_path, X, Y, Z, z_surface=z_max
    )

    # 3. 重み電位計算（Neumann境界条件対応）
    phi_w = solve_weighting_potential_fullthickness(
        X, Y, Z, collect_mask, ground_mask,
        max_iter=max_iter, tol=tol, omega=omega, num_threads=num_threads,
        use_neumann=True
    )

    # 4. キャッシュに保存
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            phi_w=phi_w,
            X=X, Y=Y, Z=Z
        )
        print(f"\n  ✓ Weighting potential saved to cache: {cache_path}")
        print(f"     File size: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"  ⚠ Failed to save cache: {e}")

    return phi_w, X, Y, Z


# ========== 1イベントのCCE計算 ==========

def compute_cce_for_one_event(
    phi_w: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    z_seg: np.ndarray,
    dE_seg: np.ndarray,
    rng: np.random.Generator,
    z_incident: float = 430e-6,
) -> float:
    """
    1本のα線イベントについてShockley-RamoでCCEを計算。

    Parameters
    ----------
    phi_w : np.ndarray
        重み電位
    X, Y, Z : np.ndarray
        座標軸 [m]
    z_seg : np.ndarray
        SRIMセグメントの深さ [m]
    dE_seg : np.ndarray
        各セグメントでのエネルギー付与 [eV]
    rng : np.random.Generator
        乱数生成器
    z_incident : float
        入射面のz座標 [m]

    Returns
    -------
    float
        CCE (0~1)

    Notes
    -----
    Shockley-Ramoの定理:
        Q_induced = -q * [φ_w(r_end) - φ_w(r_start)]

    簡略モデル:
    - 電子: collect電極に到達 → φ_w(end)=1
      → Q_e = -(-e)[1 - φ_w(start)] = e[1 - φ_w(start)]
    - 正孔: ground電極に到達 → φ_w(end)=0
      → Q_h = -(+e)[0 - φ_w(start)] = e φ_w(start)
    - 合計: Q_pair = e[1 - φ_w(start)] + e φ_w(start) = e

    実際は φ_w(start) の値により寄与が変わります。
    """
    # (x, y)をランダムサンプリング
    x_event = rng.uniform(X[0], X[-1])
    y_event = rng.uniform(Y[0], Y[-1])

    # 総e-hペア数
    N_total = dE_seg.sum() / W_EH
    Q_gen = N_total * Q_E  # [C]

    # 誘導電荷の計算
    Q_induced = 0.0

    for i in range(len(z_seg)):
        # セグメントi の位置
        # α線はz_incident面から入射し、-z方向（内部）に進む
        z_i = z_incident - z_seg[i]

        # 範囲外チェック
        if z_i < Z[0] or z_i > Z[-1]:
            continue

        # φ_w(r_i) を補間
        phi_w_start = trilinear_interpolate(phi_w, X, Y, Z, x_event, y_event, z_i)

        # このセグメントのe-hペア数
        N_i = dE_seg[i] / W_EH

        # Shockley-Ramoによる寄与
        # 電子: Q_e = e[1 - φ_w(start)]
        # 正孔: Q_h = e φ_w(start)
        # 合計: Q_i = e * N_i (理想的)
        # ただし実際の計算では境界条件により変動
        Q_i = Q_E * N_i  # 簡略化: 完全収集を仮定

        # TODO: より詳細なモデルでは以下のように計算：
        # Q_e_i = Q_E * N_i * (1 - phi_w_start)
        # Q_h_i = Q_E * N_i * phi_w_start
        # Q_i = Q_e_i + Q_h_i

        Q_induced += Q_i

    # CCE = 収集電荷 / 生成電荷
    cce = Q_induced / Q_gen if Q_gen > 0 else 0.0

    return cce


def compute_cce_ramo_ideal(
    z_seg: np.ndarray,
    dE_seg: np.ndarray,
) -> float:
    """
    理想モード: 完全収集を仮定（CCE=1.0）。

    Parameters
    ----------
    z_seg : np.ndarray
        SRIMセグメントの深さ [m]（使用しない）
    dE_seg : np.ndarray
        各セグメントでのエネルギー付与 [eV]（使用しない）

    Returns
    -------
    float
        CCE = 1.0（常に完全収集）

    Notes
    -----
    テスト用の単純化モデル。電子・正孔が必ず電極に到達し、
    再結合や寿命の影響を受けないと仮定します。
    """
    return 1.0


def compute_cce_ramo_drift(
    phi_w: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    X_field: np.ndarray,
    Y_field: np.ndarray,
    Z_field: np.ndarray,
    z_seg: np.ndarray,
    dE_seg: np.ndarray,
    x_event: float,
    y_event: float,
    z_surface: float,
    mu_e: float,
    tau_e: float,
    mu_h: float = 40.0,
    tau_h: float = 1e-8,
) -> float:
    """
    Drift モード: 実電界＋有限寿命を考慮した CCE 計算（3D局所電場ドリフトモデル）。

    Parameters
    ----------
    phi_w : np.ndarray
        重み電位, shape (nz, ny, nx)
    X, Y, Z : np.ndarray
        重み電位の座標軸 [m]
    Ex, Ey, Ez : np.ndarray
        電界成分 [V/m], shape (nz_field, ny_field, nx_field)
    X_field, Y_field, Z_field : np.ndarray
        電界データの座標軸 [m]
    z_seg : np.ndarray
        SRIMセグメントの深さ [m]（入射面からの距離）
    dE_seg : np.ndarray
        各セグメントでのエネルギー付与 [eV]
    x_event, y_event : float
        イベントの (x, y) 位置 [m]
    z_surface : float
        入射面の z 座標 [m]（通常は Z[-1]）
    mu_e : float
        電子移動度 [cm²/Vs]
    tau_e : float
        電子寿命 [s]
    mu_h : float
        正孔移動度 [cm²/Vs]
    tau_h : float
        正孔寿命 [s]

    Returns
    -------
    float
        CCE (0~1)

    Notes
    -----
    【改修後のモデル（3D局所電場ドリフト）】：
    - 各生成点で局所電場ベクトル E = (Ex, Ey, Ez) を補間
    - キャリアは電場方向に沿って寿命 τ の間だけドリフト
      * 電子：-E 方向（電場と逆向き）
      * 正孔：+E 方向（電場と同じ向き）
    - ドリフト速度：v = μ * |E| (μは移動度、|E|は電場の大きさ)
    - ドリフト距離：d = v * τ（寿命の間だけ移動）
    - 終点座標を計算し、そこでの φ_w を補間
    - Shockley-Ramo: Q = N * q * (φ_w_end - φ_w_start)
    - 従来の指数減衰 f_survival は使用せず、寿命を「有効ドリフト時間」として扱う

    【改修前のモデル（1D z方向近似）】との違い：
    - 旧：z方向の距離のみでドリフトを計算（横方向電場を無視）
    - 新：電場ベクトル全体を考慮した3Dドリフト（横型・くし形電極に対応）

    バルク電界（z < Z_field.min()）の処理：
    - 境界値（Z_field[0]）の電界を使用して位置依存性を保持
    """
    # デバッグ用：phi_w 配列全体の範囲を出力
    print(f"[DEBUG] phi_w range: {phi_w.min():.4f} .. {phi_w.max():.4f}")

    # 定数：電場の最小閾値（これ以下の電場では寄与を無視）
    E_MIN = 1e3  # [V/m]

    # 総e-hペア数と生成電荷
    N_i = dE_seg / W_EH
    N_total = N_i.sum()
    Q_gen = N_total * Q_E  # [C]

    if N_total == 0:
        return 0.0

    # 誘導電荷の計算（N_i個のキャリア分を直接計算）
    Q_induced = 0.0

    # バルク領域（z < Z_field.min()）の電界補間用キャッシュ
    # 境界値（Z_field[0]）の電界を使用して位置依存性を保つ
    E_bulk_cache = {}  # (x_event, y_event) → (Ex, Ey, Ez) at Z_field[0]

    # バルク領域のφ_w外挿用キャッシュ
    # 境界値（Z[0]）の重み電位を使用
    phi_w_bulk_cache = {}  # (x_event, y_event) → φ_w at Z[0]

    for i in range(len(z_seg)):
        # セグメント i の生成位置
        # α線は z_surface から入射し、-z 方向（内部）に進む
        z_i = z_surface - z_seg[i]
        x_i = x_event
        y_i = y_event

        # 電界の取得（電界データの座標系 X_field, Y_field, Z_field を使用）
        if z_i >= Z_field[0] and z_i <= Z_field[-1]:
            # z が電界データの範囲内 → 補間で取得
            Ex_i = trilinear_interpolate(Ex, X_field, Y_field, Z_field, x_i, y_i, z_i)
            Ey_i = trilinear_interpolate(Ey, X_field, Y_field, Z_field, x_i, y_i, z_i)
            Ez_i = trilinear_interpolate(Ez, X_field, Y_field, Z_field, x_i, y_i, z_i)
        else:
            # z が電界データの範囲外（バルク領域）→ 境界値外挿
            if (x_event, y_event) not in E_bulk_cache:
                Ex_boundary = trilinear_interpolate(Ex, X_field, Y_field, Z_field, x_event, y_event, Z_field[0])
                Ey_boundary = trilinear_interpolate(Ey, X_field, Y_field, Z_field, x_event, y_event, Z_field[0])
                Ez_boundary = trilinear_interpolate(Ez, X_field, Y_field, Z_field, x_event, y_event, Z_field[0])
                E_bulk_cache[(x_event, y_event)] = (Ex_boundary, Ey_boundary, Ez_boundary)

            Ex_i, Ey_i, Ez_i = E_bulk_cache[(x_event, y_event)]

        # 電場の大きさ
        E_mag = np.sqrt(Ex_i**2 + Ey_i**2 + Ez_i**2)  # [V/m]

        # 電場が小さすぎる場合はスキップ
        if E_mag < E_MIN:
            continue

        # 電場方向の単位ベクトル
        E_unit = np.array([Ex_i, Ey_i, Ez_i]) / E_mag

        # 重み電位（生成位置）
        if z_i >= Z[0] and z_i <= Z[-1]:
            # 範囲内 → 補間
            phi_w_start = trilinear_interpolate(phi_w, X, Y, Z, x_i, y_i, z_i)
            # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
            phi_w_start = float(np.clip(phi_w_start, 0.0, 1.0))
        else:
            # 範囲外（バルク領域、z < Z.min()）
            # 境界値（Z[0]）の重み電位を使用（位置依存性を保持）
            if (x_event, y_event) not in phi_w_bulk_cache:
                phi_w_boundary = trilinear_interpolate(
                    phi_w, X, Y, Z, x_event, y_event, Z[0]
                )
                # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
                phi_w_bulk_cache[(x_event, y_event)] = float(np.clip(phi_w_boundary, 0.0, 1.0))
            phi_w_start = phi_w_bulk_cache[(x_event, y_event)]

        # 生成位置ベクトル
        r_start = np.array([x_i, y_i, z_i])

        # === 電子の寄与 ===
        # ドリフト方向：電場と逆向き（負電荷）
        u_e = -E_unit
        # ドリフト速度の大きさ [m/s]
        v_e = mu_e * 1e-4 * E_mag
        # ドリフト時間：寿命の間だけ移動（表面を超える場合はクリップ）
        t_move_e = tau_e

        # 表面を超えないようにクリップ（u_e の z成分が正の場合）
        if u_e[2] > 1e-10:  # ほぼゼロでない場合
            t_to_surf = (z_surface - z_i) / (u_e[2] * v_e)
            if t_to_surf > 0 and t_to_surf < t_move_e:
                t_move_e = t_to_surf

        # 終点座標
        r_end_e = r_start + u_e * v_e * t_move_e
        x_end_e, y_end_e, z_end_e = r_end_e

        # 終点が重み電位グリッド範囲内かチェック
        if (x_end_e >= X[0] and x_end_e <= X[-1] and
            y_end_e >= Y[0] and y_end_e <= Y[-1] and
            z_end_e >= Z[0] and z_end_e <= Z[-1]):
            # 終点の重み電位を補間
            phi_w_end_e = trilinear_interpolate(phi_w, X, Y, Z, x_end_e, y_end_e, z_end_e)
            # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
            phi_w_end_e = float(np.clip(phi_w_end_e, 0.0, 1.0))
        else:
            # 範囲外の場合は、境界値を使用するか、寄与をゼロにする
            # ここでは簡単のため、z が表面を超えた場合は φ_w = 1.0、
            # その他の境界を超えた場合は最近傍の境界値を使用
            x_clipped = np.clip(x_end_e, X[0], X[-1])
            y_clipped = np.clip(y_end_e, Y[0], Y[-1])
            z_clipped = np.clip(z_end_e, Z[0], Z[-1])
            phi_w_end_e = trilinear_interpolate(phi_w, X, Y, Z, x_clipped, y_clipped, z_clipped)
            # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
            phi_w_end_e = float(np.clip(phi_w_end_e, 0.0, 1.0))

        # Shockley-Ramo: 誘起電荷（f_survival を使わない）
        Q_e_i = N_i[i] * Q_E * (phi_w_end_e - phi_w_start)
        Q_induced += Q_e_i

        # === 正孔の寄与 ===
        # ドリフト方向：電場と同じ向き（正電荷）
        u_h = +E_unit
        # ドリフト速度の大きさ [m/s]
        v_h = mu_h * 1e-4 * E_mag
        # ドリフト時間：寿命の間だけ移動（裏面を超える場合はクリップ）
        t_move_h = tau_h

        # 裏面（z=0）を超えないようにクリップ（u_h の z成分が負の場合）
        if u_h[2] < -1e-10:  # ほぼゼロでない場合
            t_to_back = -z_i / (u_h[2] * v_h)  # z_i から z=0 までの時間
            if t_to_back > 0 and t_to_back < t_move_h:
                t_move_h = t_to_back

        # 終点座標
        r_end_h = r_start + u_h * v_h * t_move_h
        x_end_h, y_end_h, z_end_h = r_end_h

        # 終点が重み電位グリッド範囲内かチェック
        if (x_end_h >= X[0] and x_end_h <= X[-1] and
            y_end_h >= Y[0] and y_end_h <= Y[-1] and
            z_end_h >= Z[0] and z_end_h <= Z[-1]):
            # 終点の重み電位を補間
            phi_w_end_h = trilinear_interpolate(phi_w, X, Y, Z, x_end_h, y_end_h, z_end_h)
            # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
            phi_w_end_h = float(np.clip(phi_w_end_h, 0.0, 1.0))
        else:
            # 範囲外の場合は、境界値を使用
            x_clipped = np.clip(x_end_h, X[0], X[-1])
            y_clipped = np.clip(y_end_h, Y[0], Y[-1])
            z_clipped = np.clip(z_end_h, Z[0], Z[-1])
            phi_w_end_h = trilinear_interpolate(phi_w, X, Y, Z, x_clipped, y_clipped, z_clipped)
            # φ_w を 0〜1 にクリップ（負のCCEを防ぐ）
            phi_w_end_h = float(np.clip(phi_w_end_h, 0.0, 1.0))

        # Shockley-Ramo: 誘起電荷（f_survival を使わない）
        Q_h_i = N_i[i] * Q_E * (phi_w_start - phi_w_end_h)
        Q_induced += Q_h_i

    # CCE = 収集電荷 / 生成電荷
    cce = Q_induced / Q_gen if Q_gen > 0 else 0.0

    # 最終的なCCEも0〜1にクリップ（念のため）
    cce = float(np.clip(cce, 0.0, 1.0))

    return cce


# ========== メインシミュレーション ==========

def simulate_cce(
    detector_type: str = "yoko",
    n_events: int = 1000,
    mode: str = "ramo_ideal",
    mu_e: float = 100.0,
    tau_e: float = 1e-8,
    mu_h: float = 40.0,
    tau_h: float = 1e-8,
    num_threads: Optional[int] = None,
    seed: Optional[int] = None,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    force_recalc_weighting: bool = False,
    use_fullthick: bool = True,
    field_path: Optional[str] = None,
    srim_path: Optional[str] = None,
    stop_check: Optional[callable] = None,
) -> dict:
    """
    n_eventsイベントをシミュレーションしてCCE統計を返す。

    Parameters
    ----------
    detector_type : str
        検出器タイプ: "yoko" (横型) or "kushi" (くし形)
    n_events : int
        シミュレーションするイベント数
    mode : str
        計算モード: "ramo_ideal" (CCE=1テスト) or "ramo_drift" (μ,τ考慮)
    mu_e : float
        電子移動度 [cm^2/Vs] (mode="ramo_drift"時に使用)
    tau_e : float
        電子寿命 [s] (mode="ramo_drift"時に使用)
    num_threads : int | None
        Numbaスレッド数
    seed : int | None
        乱数シード
    max_iter : int
        重み電位計算の最大反復回数
    tol : float
        重み電位計算の収束判定閾値
    omega : float
        SOR緩和パラメータ (1.0 < omega < 2.0)
    force_recalc_weighting : bool
        Trueの場合、キャッシュを無視して重み電位を再計算
    use_fullthick : bool
        Trueの場合、全厚メッシュ（z=0-430μm）で重み電位を計算
    field_path : str | None
        電界npzファイルパス（Noneの場合はdetector_typeから自動決定）
    srim_path : str | None
        SRIM IONIZファイルパス（Noneの場合はデフォルト使用）
    stop_check : callable | None
        停止チェック用コールバック関数。Trueを返すと計算を中断。

    Returns
    -------
    dict
        'cce_list': CCEのリスト
        'mean': 平均CCE
        'std': 標準偏差
        'min': 最小CCE
        'max': 最大CCE
        'n_events': イベント数
        'stopped': 停止された場合True
    """
    # field_path の自動決定
    if field_path is None:
        base_dir = r"C:\Users\discu\デスクトップ\python\cce\電界"
        if detector_type == "yoko":
            field_path = os.path.join(base_dir, "yokogata_field.npz")
        elif detector_type == "kushi":
            field_path = os.path.join(base_dir, "kushigata_field.npz")
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

    # srim_path のデフォルト
    if srim_path is None:
        # デフォルトSRIMファイルを探索（クロスプラットフォーム対応）
        default_srim_candidates = [
            "data/5486keVαinSiCIONIZ.txt",
            "5486keVαinSiCIONIZ.txt",
            r"C:\Users\discu\デスクトップ\python\cce\5486keVαinSiCIONIZ.txt",  # Windows用後方互換
        ]
        for candidate in default_srim_candidates:
            if os.path.exists(candidate):
                srim_path = candidate
                break
        else:
            raise FileNotFoundError(
                "Default SRIM file not found. Please specify srim_path explicitly or place "
                "'5486keVαinSiCIONIZ.txt' in the current directory or 'data/' directory."
            )
    print("="*70)
    print("Shockley-Ramo CCE Simulation")
    print("="*70)

    # 乱数生成器
    rng = np.random.default_rng(seed)

    # 1. 重み電位取得（キャッシュあり）
    if use_fullthick:
        print(f"  Using full-thickness weighting potential (z=0-430 μm)")
        phi_w, X, Y, Z = get_weighting_potential_fullthickness(
            field_path=field_path,
            cache_path=None,  # 自動決定
            force_recalc=force_recalc_weighting,
            z_max=430e-6,
            target_dz=2.5e-6,
            max_iter=max_iter,
            tol=tol,
            omega=omega,
            num_threads=num_threads,
        )
    else:
        print(f"  Using standard weighting potential")
        phi_w, X, Y, Z = get_weighting_potential(
            field_path=field_path,
            cache_path=None,  # 自動決定
            force_recalc=force_recalc_weighting,
            max_iter=max_iter,
            tol=tol,
            omega=omega,
            num_threads=num_threads,
        )

    # 2. SRIM読み込み
    z_seg, dE_seg = load_srim_ioniz(srim_path)

    # 3. CCEシミュレーション
    print(f"\n{'='*70}")
    print(f"CCE Simulation")
    print('='*70)
    print(f"  Mode: {mode}")
    print(f"  Detector: {detector_type}")
    print(f"  Events: {n_events}")

    if mode == "ramo_drift":
        print(f"  μ_e: {mu_e} cm²/Vs")
        print(f"  τ_e: {tau_e} s")

    cce_list = []
    x_list = []  # (x, y) 位置を記録（ramo_driftモードのみ）
    y_list = []

    stopped = False  # 停止フラグ

    if mode == "ramo_ideal":
        # 理想モード: CCE=1.0（テスト用）
        print(f"\n  Mode: Ideal (CCE=1.0, no recombination)")
        for i in range(n_events):
            # 停止チェック
            if stop_check and stop_check():
                print(f"\n  *** Simulation stopped by user at event {i+1}/{n_events} ***")
                stopped = True
                break

            cce = compute_cce_ramo_ideal(z_seg, dE_seg)
            cce_list.append(cce)
            # 理想モードでは位置情報なし
            x_list.append(None)
            y_list.append(None)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Event {i+1}/{n_events}: CCE = {cce:.4f}")

    elif mode == "ramo_drift":
        # ドリフトモード: 実電界＋寿命を考慮
        print(f"\n  Mode: Drift (E-field + lifetime)")

        # 電界データを再読み込み（Ex, Ey, Ez が必要）
        field_data = load_field_npz(field_path)
        Ex = field_data['Ex']
        Ey = field_data['Ey']
        Ez = field_data['Ez']
        X_field = field_data['X']
        Y_field = field_data['Y']
        Z_field = field_data['Z']

        z_surface = Z[-1]  # 入射面（表面）

        for i in range(n_events):
            # 停止チェック
            if stop_check and stop_check():
                print(f"\n  *** Simulation stopped by user at event {i+1}/{n_events} ***")
                stopped = True
                break

            # ランダムな (x, y) 位置をサンプリング
            x_event = rng.uniform(X[0], X[-1])
            y_event = rng.uniform(Y[0], Y[-1])

            cce = compute_cce_ramo_drift(
                phi_w, X, Y, Z, Ex, Ey, Ez, X_field, Y_field, Z_field,
                z_seg, dE_seg,
                x_event, y_event, z_surface,
                mu_e, tau_e, mu_h, tau_h
            )
            cce_list.append(cce)
            x_list.append(x_event)
            y_list.append(y_event)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Event {i+1}/{n_events}: CCE = {cce:.4f}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'ramo_ideal' or 'ramo_drift'.")

    cce_array = np.array(cce_list)

    # 停止された場合やイベントがない場合の処理
    if len(cce_array) == 0:
        print(f"\n{'='*70}")
        print("Results")
        print('='*70)
        print(f"  No events completed.")
        return {
            'cce_list': [],
            'x_list': [],
            'y_list': [],
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'n_events': 0,
            'stopped': stopped,
            'V_electrode': None,
            'X_electrode': None,
            'Y_electrode': None,
            'Z_electrode': None,
        }

    mean_cce = cce_array.mean()
    std_cce = cce_array.std()

    print(f"\n{'='*70}")
    print("Results")
    print('='*70)
    print(f"  Completed events: {len(cce_list)}/{n_events}")
    if stopped:
        print(f"  (Stopped by user)")
    print(f"  Mean CCE: {mean_cce:.4f} ± {std_cce:.4f}")
    print(f"  Min CCE: {cce_array.min():.4f}")
    print(f"  Max CCE: {cce_array.max():.4f}")

    # 電極形状表示用に電位データを返す
    V_electrode = None
    X_electrode = None
    Y_electrode = None
    Z_electrode = None
    if mode == "ramo_drift":
        try:
            # 電位データを読み込み（電極形状表示用）
            if 'V' in field_data and field_data['V'] is not None:
                V_electrode = field_data['V']
                X_electrode = field_data['X']
                Y_electrode = field_data['Y']
                Z_electrode = field_data['Z']
        except Exception as e:
            print(f"  ⚠ Failed to load electrode data: {e}")

    return {
        'cce_list': cce_list,
        'x_list': x_list,
        'y_list': y_list,
        'mean': mean_cce,
        'std': std_cce,
        'min': cce_array.min(),
        'max': cce_array.max(),
        'n_events': len(cce_list),
        'stopped': stopped,
        'V_electrode': V_electrode,
        'X_electrode': X_electrode,
        'Y_electrode': Y_electrode,
        'Z_electrode': Z_electrode,
    }


# ========== ヒストグラム描画 ==========

def plot_cce_histogram(cce_list: list[float], output_file: str = "cce_histogram.png"):
    """CCEヒストグラムを描画して保存。"""
    cce_array = np.array(cce_list)

    plt.figure(figsize=(10, 6))
    plt.hist(cce_array, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(cce_array.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {cce_array.mean():.4f}')
    plt.xlabel('CCE', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title(f'CCE Distribution (N={len(cce_list)})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"\nSaved histogram: {output_file}")


# ========== CCE空間マップ描画 ==========

def plot_cce_map(
    x_list: list[float],
    y_list: list[float],
    cce_list: list[float],
    output_file: Optional[str] = None,
    nx: int = 50,
    ny: int = 50,
    V: Optional[np.ndarray] = None,
    X_electrode: Optional[np.ndarray] = None,
    Y_electrode: Optional[np.ndarray] = None,
    Z_electrode: Optional[np.ndarray] = None,
    z_surface: float = 430e-6,
) -> None:
    """
    (x, y)位置ごとの平均CCEマップを描画。

    Parameters
    ----------
    x_list : list[float]
        各イベントのx座標 [m]
    y_list : list[float]
        各イベントのy座標 [m]
    cce_list : list[float]
        各イベントのCCE値
    output_file : str | None
        保存先ファイル名（Noneの場合は保存せず表示のみ）
    nx : int
        x方向のグリッド数
    ny : int
        y方向のグリッド数
    V : np.ndarray | None
        電位分布 [V], shape (nz, ny, nx)（電極形状表示用）
    X_electrode, Y_electrode, Z_electrode : np.ndarray | None
        電位分布の座標軸 [m]（電極形状表示用）
    z_surface : float
        電極面のz座標 [m]
    """
    # Noneを除外（ramo_idealモードの場合）
    valid_indices = [
        i for i in range(len(x_list))
        if x_list[i] is not None and y_list[i] is not None
    ]

    if len(valid_indices) == 0:
        print("Warning: No valid (x, y) data. CCE map requires ramo_drift mode.")
        return

    x_valid = np.array([x_list[i] for i in valid_indices])
    y_valid = np.array([y_list[i] for i in valid_indices])
    cce_valid = np.array([cce_list[i] for i in valid_indices])

    # x, y の範囲
    x_min, x_max = x_valid.min(), x_valid.max()
    y_min, y_max = y_valid.min(), y_valid.max()

    # グリッド作成
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    # 各グリッドセル内のCCE平均を計算
    cce_map = np.full((ny, nx), np.nan)
    counts = np.zeros((ny, nx), dtype=int)

    for i in range(len(x_valid)):
        # どのグリッドセルに属するか
        ix = np.searchsorted(x_edges, x_valid[i]) - 1
        iy = np.searchsorted(y_edges, y_valid[i]) - 1

        # 範囲チェック
        if 0 <= ix < nx and 0 <= iy < ny:
            if counts[iy, ix] == 0:
                cce_map[iy, ix] = cce_valid[i]
            else:
                # 累積平均
                cce_map[iy, ix] = (
                    cce_map[iy, ix] * counts[iy, ix] + cce_valid[i]
                ) / (counts[iy, ix] + 1)
            counts[iy, ix] += 1

    # プロット
    fig, ax = plt.subplots(figsize=(10, 8))

    # x, y グリッド中心
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2 * 1e6  # μm単位
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2 * 1e6  # μm単位

    # ヒートマップ描画
    im = ax.pcolormesh(
        x_edges * 1e6,
        y_edges * 1e6,
        cce_map,
        cmap='viridis',
        shading='flat',
        vmin=0.0,
        vmax=1.0,
    )

    # カラーバー
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('CCE', fontsize=12)

    # 軸ラベル
    ax.set_xlabel('x [μm]', fontsize=12)
    ax.set_ylabel('y [μm]', fontsize=12)
    ax.set_title(
        f'CCE Spatial Map (N={len(valid_indices)}, Grid={nx}x{ny})',
        fontsize=14
    )
    ax.set_aspect('equal', adjustable='box')

    # 電極形状の重ね描き
    if V is not None and X_electrode is not None and Y_electrode is not None and Z_electrode is not None:
        try:
            # 電極マスクを作成
            collect_mask, ground_mask, k_surface = create_electrode_masks(
                V, Z_electrode, z_surface=z_surface, eps=1.0
            )

            # 表面（k_surface）の電極マスクを取得
            collect_2d = collect_mask[k_surface, :, :]  # shape (ny, nx)
            ground_2d = ground_mask[k_surface, :, :]

            # 電極の境界線を抽出（contour）
            X_grid, Y_grid = np.meshgrid(X_electrode * 1e6, Y_electrode * 1e6)

            # Collect電極の境界線（赤）
            if collect_2d.sum() > 0:
                ax.contour(
                    X_grid, Y_grid, collect_2d.astype(float),
                    levels=[0.5],
                    colors='red',
                    linewidths=0.5,
                    linestyles='-',
                )
                # ラベル追加（最初の1点のみ）
                collect_coords = np.where(collect_2d)
                if len(collect_coords[0]) > 0:
                    ax.plot([], [], 'r-', linewidth=0.5, label='Collect electrode')

            # Ground電極の境界線（青）
            if ground_2d.sum() > 0:
                ax.contour(
                    X_grid, Y_grid, ground_2d.astype(float),
                    levels=[0.5],
                    colors='blue',
                    linewidths=0.5,
                    linestyles='-',
                )
                # ラベル追加
                ground_coords = np.where(ground_2d)
                if len(ground_coords[0]) > 0:
                    ax.plot([], [], 'b-', linewidth=0.5, label='Ground electrode')

            # 凡例を追加
            if collect_2d.sum() > 0 or ground_2d.sum() > 0:
                ax.legend(loc='upper right', fontsize=10)

        except Exception as e:
            print(f"  ⚠ Failed to overlay electrode shapes: {e}")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"\nSaved CCE map: {output_file}")
    else:
        plt.show()


# ========== 結果保存 ==========

def create_results_directory(base_dir: str = "results") -> str:
    """
    タイムスタンプ付き結果保存ディレクトリを作成。

    Parameters
    ----------
    base_dir : str
        ベースディレクトリ名（デフォルト: "results"）

    Returns
    -------
    str
        作成されたディレクトリパス（例: "results/20250114_153045"）
    """
    # タイムスタンプ生成（例: 20250114_153045）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ディレクトリパス作成
    result_dir = os.path.join(base_dir, timestamp)

    # ディレクトリ作成（親ディレクトリも含む）
    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def save_simulation_results(
    results: dict,
    params: dict,
    output_dir: Optional[str] = None,
) -> str:
    """
    シミュレーション結果を保存。

    Parameters
    ----------
    results : dict
        simulate_cce()の返り値
        - 'cce_list': CCE値のリスト
        - 'mean', 'std', 'min', 'max': 統計値
        - 'n_events': イベント数
        - 'stopped': 停止フラグ
    params : dict
        入力パラメータ辞書
        - 'detector_type': 検出器タイプ
        - 'n_events': イベント数
        - 'mode': シミュレーションモード
        - 'alpha_MeV': α線エネルギー
        - 'mu_e': 電子移動度
        - 'tau_e': 電子寿命
        - 'use_fullthick': 全厚weighting potential使用フラグ
        - その他
    output_dir : str | None
        出力先ディレクトリ（Noneの場合は自動生成）

    Returns
    -------
    str
        保存先ディレクトリパス
    """
    # 出力ディレクトリ作成
    if output_dir is None:
        output_dir = create_results_directory()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 1. パラメータをJSONで保存
    params_file = os.path.join(output_dir, "params.json")
    # 保存用にparamsとresultsの統計値を結合
    save_params = params.copy()
    save_params['results_summary'] = {
        'mean': results['mean'],
        'std': results['std'],
        'min': results['min'],
        'max': results['max'],
        'n_events_completed': results['n_events'],
        'stopped': results.get('stopped', False),
    }

    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(save_params, f, indent=2, ensure_ascii=False)

    print(f"Saved parameters: {params_file}")

    # 2. CCE生データをCSVで保存
    cce_data_file = os.path.join(output_dir, "cce_data.csv")
    x_list = results.get('x_list', [])
    y_list = results.get('y_list', [])
    cce_list = results['cce_list']

    # (x, y)位置情報がある場合は3列で保存、ない場合はCCEのみ
    if x_list and y_list and any(x is not None for x in x_list):
        # ramo_driftモード: x, y, cceを保存
        data_array = np.column_stack([
            [x if x is not None else np.nan for x in x_list],
            [y if y is not None else np.nan for y in y_list],
            cce_list
        ])
        np.savetxt(
            cce_data_file,
            data_array,
            delimiter=',',
            header='x[m],y[m],CCE',
            comments='',
            fmt='%.8e,%.8e,%.6f'
        )
    else:
        # ramo_idealモード: CCEのみ
        cce_array = np.array(cce_list)
        np.savetxt(
            cce_data_file,
            cce_array,
            delimiter=',',
            header='CCE',
            comments='',
            fmt='%.6f'
        )

    print(f"Saved CCE data: {cce_data_file}")

    # 3. ヒストグラムを画像で保存
    histogram_file = os.path.join(output_dir, "cce_histogram.png")
    if len(results['cce_list']) > 0:
        plot_cce_histogram(results['cce_list'], output_file=histogram_file)
    else:
        print("Skipped histogram (no data)")

    # 4. CCE空間マップを画像で保存（ramo_driftモードのみ）
    if x_list and y_list and any(x is not None for x in x_list):
        cce_map_file = os.path.join(output_dir, "cce_map.png")
        plot_cce_map(
            x_list, y_list, cce_list,
            output_file=cce_map_file,
            V=results.get('V_electrode'),
            X_electrode=results.get('X_electrode'),
            Y_electrode=results.get('Y_electrode'),
            Z_electrode=results.get('Z_electrode'),
        )
    else:
        print("Skipped CCE map (requires ramo_drift mode)")

    print(f"\n{'='*70}")
    print(f"All results saved to: {output_dir}")
    print('='*70)

    return output_dir


# ========== CLI ==========

def main_cli():
    """CLI モードのメイン関数"""
    parser = argparse.ArgumentParser(
        description="Shockley-Ramo CCE simulation for SiC detector"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yoko",
        choices=["yoko", "kushi"],
        help="Detector type: yoko (横型) or kushi (くし形)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ramo_ideal",
        choices=["ramo_ideal", "ramo_drift"],
        help="Simulation mode: ramo_ideal (CCE=1 test) or ramo_drift (μ,τ with field)"
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=1000,
        help="Number of events to simulate"
    )
    parser.add_argument(
        "--mu-e",
        type=float,
        default=100.0,
        help="Electron mobility [cm^2/Vs] (for ramo_drift mode)"
    )
    parser.add_argument(
        "--tau-e",
        type=float,
        default=1e-8,
        help="Electron lifetime [s] (for ramo_drift mode)"
    )
    parser.add_argument(
        "--mu-h",
        type=float,
        default=40.0,
        help="Hole mobility [cm^2/Vs] (for ramo_drift mode)"
    )
    parser.add_argument(
        "--tau-h",
        type=float,
        default=1e-8,
        help="Hole lifetime [s] (for ramo_drift mode)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of Numba threads (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Max iterations for weighting potential solver"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance for weighting potential solver"
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.8,
        help="SOR relaxation parameter (1.0 < omega < 2.0, optimal: 1.6-1.9)"
    )
    parser.add_argument(
        "--force-recalc-weighting",
        action="store_true",
        help="Force recalculation of weighting potential (ignore cache)"
    )
    parser.add_argument(
        "--use-fullthick",
        action="store_true",
        default=True,
        help="Use full-thickness weighting potential (z=0-430 μm, default: True)"
    )
    parser.add_argument(
        "--no-fullthick",
        action="store_false",
        dest="use_fullthick",
        help="Use standard weighting potential instead of full-thickness"
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Custom path to field npz file (overrides --detector)"
    )
    parser.add_argument(
        "--srim",
        type=str,
        default=None,
        help="Custom path to SRIM IONIZ file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cce_histogram.png",
        help="Output histogram filename"
    )

    args = parser.parse_args()

    # シミュレーション実行
    results = simulate_cce(
        detector_type=args.detector,
        n_events=args.n_events,
        mode=args.mode,
        mu_e=args.mu_e,
        tau_e=args.tau_e,
        mu_h=args.mu_h,
        tau_h=args.tau_h,
        num_threads=args.num_threads,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        omega=args.omega,
        force_recalc_weighting=args.force_recalc_weighting,
        use_fullthick=args.use_fullthick,
        field_path=args.field,
        srim_path=args.srim,
    )

    # ヒストグラム描画
    plot_cce_histogram(results['cce_list'], args.output)

    print("\nSimulation completed successfully!")


# ========== GUI (tkinter) ==========

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import threading
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


if TKINTER_AVAILABLE:
    class CCESimulationGUI(tk.Tk):
        """
        Shockley-Ramo CCE シミュレーション用 GUI
    
        tkinter 標準ライブラリのみを使用したシンプルな GUI。
        パラメータを入力して「Run Simulation」ボタンでシミュレーションを実行。
        """
    
        def __init__(self):
            super().__init__()
            self.title("Shockley-Ramo CCE Simulator")
            self.geometry("1200x800")
    
            self.running = False
            self.stop_requested = False  # 停止リクエストフラグ
            self.last_results = None  # 最後の結果を保存

            # ウェイティングポテンシャルデータ
            self.weight_data = None  # {phi_w, X, Y, Z, Ex, Ey, Ez}
            self.field_path = None
            self.weight_path = None
    
            self._build_widgets()
    
        def _build_widgets(self):
            """ウィジェット配置（タブUI）"""
            # メインフレーム
            main_frame = ttk.Frame(self, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
    
            # タブUIの作成
            self.notebook = ttk.Notebook(main_frame)
            self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(0, weight=1)
    
            # タブ1: CCE Simulation（既存の内容）
            self.tab_cce = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_cce, text="CCE Simulation")
            self._build_cce_tab()
    
            # タブ2: Weighting Potential（新規）
            self.tab_weighting = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_weighting, text="Weighting Potential")
            self._build_weighting_tab()

            # タブ3: Electric Field（新規）
            self.tab_field = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_field, text="Electric Field")
            self._build_field_tab()

            # タブ4: Diagnostics（新規）
            self.tab_diagnostics = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_diagnostics, text="Diagnostics")
            self._build_diagnostics_tab()
    
        def _build_cce_tab(self):
            """CCE Simulationタブの構築（既存の内容）"""
            # メインフレーム（タブ内）
            tab_frame = self.tab_cce
            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(0, weight=1)
    
            # === パラメータ入力エリア ===
            param_frame = ttk.LabelFrame(tab_frame, text="Parameters", padding="10")
            param_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
    
            row = 0
    
            # Detector Type
            ttk.Label(param_frame, text="Detector:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.detector_var = tk.StringVar(value="Yokogata (横型)")
            detector_combo = ttk.Combobox(
                param_frame,
                textvariable=self.detector_var,
                values=["Yokogata (横型)", "Kushigata (くし形)"],
                state="readonly",
                width=25
            )
            detector_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Field File Selection
            ttk.Label(param_frame, text="Field file:").grid(row=row, column=0, sticky=tk.W, pady=2)

            # 利用可能な電界ファイルを探索
            field_dir = "電界"
            available_fields = []
            if os.path.exists(field_dir):
                for f in os.listdir(field_dir):
                    if f.endswith(".npz"):
                        available_fields.append(f)

            # デフォルト値の設定（Autoまたは最初のファイル）
            default_field = "Auto (from detector type)"
            field_values = [default_field] + sorted(available_fields)

            self.field_file_var = tk.StringVar(value=default_field)
            field_combo = ttk.Combobox(
                param_frame,
                textvariable=self.field_file_var,
                values=field_values,
                state="readonly",
                width=25
            )
            field_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Label(param_frame, text="(Auto = detector type)", font=("", 8)).grid(
                row=row, column=2, sticky=tk.W, padx=5
            )
            row += 1

            # SRIM File Selection
            ttk.Label(param_frame, text="SRIM file:").grid(row=row, column=0, sticky=tk.W, pady=2)

            # 利用可能なSRIMファイルを探索
            available_srim = []
            for search_dir in ["data", "."]:
                if os.path.exists(search_dir):
                    for f in os.listdir(search_dir):
                        if f.endswith(".txt") and ("IONIZ" in f or "ioniz" in f or "srim" in f.lower()):
                            available_srim.append(os.path.join(search_dir, f))

            # デフォルト値の設定
            default_srim = "Auto (5486keVαinSiCIONIZ.txt)"
            srim_values = [default_srim] + sorted(set(available_srim))

            self.srim_file_var = tk.StringVar(value=default_srim)
            srim_combo = ttk.Combobox(
                param_frame,
                textvariable=self.srim_file_var,
                values=srim_values,
                state="readonly",
                width=25
            )
            srim_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Label(param_frame, text="(Auto = default)", font=("", 8)).grid(
                row=row, column=2, sticky=tk.W, padx=5
            )
            row += 1

            # Mode
            ttk.Label(param_frame, text="Mode:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.mode_var = tk.StringVar(value="Ideal (CCE=1, test)")
            mode_combo = ttk.Combobox(
                param_frame,
                textvariable=self.mode_var,
                values=["Ideal (CCE=1, test)", "Drift (μ,τ with field)"],
                state="readonly",
                width=25
            )
            mode_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1
    
            # Events
            ttk.Label(param_frame, text="Events:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.events_var = tk.StringVar(value="1000")
            ttk.Entry(param_frame, textvariable=self.events_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            row += 1
    
            # μ_e
            ttk.Label(param_frame, text="μ_e [cm²/Vs]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.mu_e_var = tk.StringVar(value="100.0")  # TODO: 実験値に合わせて調整
            ttk.Entry(param_frame, textvariable=self.mu_e_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            row += 1

            # τ_e
            ttk.Label(param_frame, text="τ_e [s]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.tau_e_var = tk.StringVar(value="1e-8")
            ttk.Entry(param_frame, textvariable=self.tau_e_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            row += 1

            # μ_h
            ttk.Label(param_frame, text="μ_h [cm²/Vs]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.mu_h_var = tk.StringVar(value="40.0")  # SiC typical hole mobility
            ttk.Entry(param_frame, textvariable=self.mu_h_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            row += 1

            # τ_h
            ttk.Label(param_frame, text="τ_h [s]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.tau_h_var = tk.StringVar(value="1e-8")
            ttk.Entry(param_frame, textvariable=self.tau_h_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            row += 1
    
            # Threads
            ttk.Label(param_frame, text="Threads:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.threads_var = tk.StringVar(value="")  # 空欄 = auto
            ttk.Entry(param_frame, textvariable=self.threads_var, width=28).grid(
                row=row, column=1, sticky=(tk.W, tk.E), pady=2
            )
            ttk.Label(param_frame, text="(empty = auto)", font=("", 8)).grid(
                row=row, column=2, sticky=tk.W, padx=5
            )
            row += 1

            # === Weighting potential options ===
            ttk.Label(param_frame, text="Weighting options:", font=("", 9, "bold")).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 2)
            )
            row += 1

            # Use full-thickness
            self.cce_use_fullthick_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                param_frame,
                text="Use full-thickness (0-430 μm)",
                variable=self.cce_use_fullthick_var
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

            # Force recalculation
            self.cce_force_recalc_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                param_frame,
                text="Force recalc weighting potential",
                variable=self.cce_force_recalc_var
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

            # カラム幅調整
            param_frame.columnconfigure(1, weight=1)
    
            # === ボタンエリア ===
            button_frame = ttk.Frame(tab_frame)
            button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
    
            self.run_button = ttk.Button(
                button_frame,
                text="Run Simulation",
                command=self.run_simulation
            )
            self.run_button.pack(side=tk.LEFT, padx=5)

            self.stop_button = ttk.Button(
                button_frame,
                text="Stop Simulation",
                command=self.stop_simulation,
                state="disabled"  # 最初は無効
            )
            self.stop_button.pack(side=tk.LEFT, padx=5)

            self.hist_button = ttk.Button(
                button_frame,
                text="Show Histogram",
                command=self.show_histogram,
                state="disabled"  # 最初は無効
            )
            self.hist_button.pack(side=tk.LEFT, padx=5)

            self.map_button = ttk.Button(
                button_frame,
                text="Show CCE Map",
                command=self.show_cce_map,
                state="disabled"  # 最初は無効
            )
            self.map_button.pack(side=tk.LEFT, padx=5)

            self.save_button = ttk.Button(
                button_frame,
                text="Save Results",
                command=self.save_results,
                state="disabled"  # 最初は無効
            )
            self.save_button.pack(side=tk.LEFT, padx=5)

            ttk.Button(button_frame, text="Quit", command=self.quit).pack(side=tk.LEFT, padx=5)
    
            # === 結果サマリーエリア ===
            result_frame = ttk.LabelFrame(tab_frame, text="Results", padding="10")
            result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
    
            self.result_label = ttk.Label(result_frame, text="No results yet.", foreground="gray")
            self.result_label.pack(anchor=tk.W)
    
            # === ログエリア ===
            log_frame = ttk.LabelFrame(tab_frame, text="Log", padding="10")
            log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
            tab_frame.rowconfigure(3, weight=1)
    
            self.log_text = scrolledtext.ScrolledText(
                log_frame,
                wrap=tk.WORD,
                width=80,
                height=20,
                font=("Courier", 9)
            )
            self.log_text.pack(fill=tk.BOTH, expand=True)
    
        def log(self, message: str):
            """ログエリアにメッセージを追加"""
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.update_idletasks()
    
        def run_simulation(self):
            """シミュレーション実行"""
            if self.running:
                messagebox.showwarning("Warning", "Simulation is already running!")
                return
    
            # パラメータ取得
            try:
                detector_str = self.detector_var.get()
                detector_type = "yoko" if "Yokogata" in detector_str else "kushi"

                mode_str = self.mode_var.get()
                mode = "ramo_ideal" if "Ideal" in mode_str else "ramo_drift"

                n_events = int(self.events_var.get())
                mu_e = float(self.mu_e_var.get())
                tau_e = float(self.tau_e_var.get())
                mu_h = float(self.mu_h_var.get())
                tau_h = float(self.tau_h_var.get())

                threads_str = self.threads_var.get().strip()
                num_threads = int(threads_str) if threads_str else None

                # Weighting options
                use_fullthick = self.cce_use_fullthick_var.get()
                force_recalc = self.cce_force_recalc_var.get()

                # Field file selection
                field_file_str = self.field_file_var.get()
                if field_file_str == "Auto (from detector type)":
                    # Auto: detector_type から自動決定（simulate_cce 内で処理）
                    field_path = None
                else:
                    # 明示的に指定されたファイル
                    field_path = os.path.join("電界", field_file_str)

                # SRIM file selection
                srim_file_str = self.srim_file_var.get()
                if srim_file_str.startswith("Auto"):
                    # Auto: デフォルトのSRIMファイルを使用（simulate_cce 内で処理）
                    srim_path = None
                else:
                    # 明示的に指定されたファイル
                    srim_path = srim_file_str

            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid parameter: {e}")
                return

            # ログクリア
            self.log_text.delete(1.0, tk.END)
            self.log("Starting simulation...")
            self.log(f"  Detector: {detector_type}")
            self.log(f"  Field file: {field_path if field_path else 'Auto'}")
            self.log(f"  SRIM file: {srim_path if srim_path else 'Auto'}")
            self.log(f"  Mode: {mode}")
            self.log(f"  Events: {n_events}")
            self.log(f"  μ_e: {mu_e} cm²/Vs, τ_e: {tau_e} s")
            self.log(f"  μ_h: {mu_h} cm²/Vs, τ_h: {tau_h} s")
            self.log(f"  Threads: {num_threads if num_threads else 'auto'}")
            self.log(f"  Use full-thickness: {use_fullthick}")
            self.log(f"  Force recalc weighting: {force_recalc}")
            self.log("")

            # 結果をクリア
            self.result_label.config(text="Running...", foreground="blue")
            self.run_button.config(state="disabled")
            self.stop_button.config(state="normal")  # Stopボタンを有効化
            self.running = True
            self.stop_requested = False  # 停止フラグをリセット

            # バックグラウンドで実行（GUIフリーズを防ぐ）
            thread = threading.Thread(
                target=self._run_simulation_thread,
                args=(detector_type, mode, n_events, mu_e, tau_e, mu_h, tau_h, num_threads, use_fullthick, force_recalc, field_path, srim_path),
                daemon=True
            )
            thread.start()

        def stop_simulation(self):
            """シミュレーションの停止をリクエスト"""
            if self.running:
                self.stop_requested = True
                self.log("\n*** Stop requested by user ***")
                self.stop_button.config(state="disabled")
    
        def _run_simulation_thread(
            self,
            detector_type: str,
            mode: str,
            n_events: int,
            mu_e: float,
            tau_e: float,
            mu_h: float,
            tau_h: float,
            num_threads: Optional[int],
            use_fullthick: bool,
            force_recalc: bool,
            field_path: Optional[str],
            srim_path: Optional[str],
        ):
            """バックグラウンドスレッドでシミュレーション実行"""
            try:
                # stdout をキャプチャするため、簡易版として直接 log() に出力
                # （実際の stdout リダイレクトは複雑なので、ここでは省略）

                # シミュレーション実行（停止チェック用コールバック付き）
                def check_stop():
                    return self.stop_requested

                results = simulate_cce(
                    detector_type=detector_type,
                    n_events=n_events,
                    mode=mode,
                    mu_e=mu_e,
                    tau_e=tau_e,
                    mu_h=mu_h,
                    tau_h=tau_h,
                    num_threads=num_threads,
                    seed=None,
                    force_recalc_weighting=force_recalc,
                    use_fullthick=use_fullthick,
                    field_path=field_path,
                    srim_path=srim_path,
                    stop_check=check_stop,
                )
    
                # 結果表示
                mean_cce = results['mean']
                std_cce = results['std']
                min_cce = results['min']
                max_cce = results['max']
                stopped = results.get('stopped', False)
                completed = results['n_events']

                self.log("\n" + "="*70)
                if stopped:
                    self.log("SIMULATION STOPPED BY USER")
                else:
                    self.log("SIMULATION COMPLETED")
                self.log("="*70)
                self.log(f"  Completed events: {completed}/{n_events}")
                if completed > 0:
                    self.log(f"  Mean CCE: {mean_cce:.4f} ± {std_cce:.4f}")
                    self.log(f"  Min CCE: {min_cce:.4f}")
                    self.log(f"  Max CCE: {max_cce:.4f}")
                else:
                    self.log(f"  No events completed.")

                # 結果サマリー更新
                if completed > 0:
                    result_text = (
                        f"Events: {completed}/{n_events}"
                        + (" (stopped)" if stopped else "")
                        + f"  |  Mean: {mean_cce:.4f} ± {std_cce:.4f}  |  "
                        f"Min: {min_cce:.4f}  |  Max: {max_cce:.4f}"
                    )
                    color = "orange" if stopped else "green"
                    self.result_label.config(text=result_text, foreground=color)
                else:
                    self.result_label.config(text="Stopped - no events completed", foreground="orange")
    
                # 結果を保存してヒストグラム・CCEマップ・保存ボタンを有効化
                self.last_results = results
                self.hist_button.config(state="normal")
                self.map_button.config(state="normal")
                self.save_button.config(state="normal")
    
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log("\n" + "="*70)
                self.log("ERROR")
                self.log("="*70)
                self.log(error_msg)
                self.result_label.config(text="Error occurred (see log)", foreground="red")
    
            finally:
                # ボタンを再度有効化/無効化
                self.run_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.running = False
                self.stop_requested = False
    
        def show_histogram(self):
            """CCE ヒストグラムを別ウィンドウに表示"""
            if self.last_results is None:
                messagebox.showwarning("Warning", "No results available. Run simulation first.")
                return
    
            try:
                cce_list = self.last_results['cce_list']
                cce_array = np.array(cce_list)
    
                # matplotlib で別ウィンドウに表示
                plt.figure(figsize=(10, 6))
                plt.hist(cce_array, bins=50, alpha=0.7, edgecolor='black')
                plt.axvline(cce_array.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Mean = {cce_array.mean():.4f}')
                plt.xlabel('CCE', fontsize=12)
                plt.ylabel('Counts', fontsize=12)
                plt.title(f'CCE Distribution (N={len(cce_list)})', fontsize=14)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show histogram: {e}")

        def show_cce_map(self):
            """CCE 空間マップを別ウィンドウに表示"""
            if self.last_results is None:
                messagebox.showwarning("Warning", "No results available. Run simulation first.")
                return

            try:
                x_list = self.last_results['x_list']
                y_list = self.last_results['y_list']
                cce_list = self.last_results['cce_list']

                # ramo_drift モードか確認
                if all(x is None for x in x_list):
                    messagebox.showwarning(
                        "Warning",
                        "CCE map requires ramo_drift mode.\nramo_ideal mode has no spatial variation."
                    )
                    return

                # CCEマップを描画（別ウィンドウ表示、電極形状も重ねる）
                plot_cce_map(
                    x_list, y_list, cce_list,
                    output_file=None,
                    V=self.last_results.get('V_electrode'),
                    X_electrode=self.last_results.get('X_electrode'),
                    Y_electrode=self.last_results.get('Y_electrode'),
                    Z_electrode=self.last_results.get('Z_electrode'),
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to show CCE map: {e}")

        def save_results(self):
            """シミュレーション結果を保存"""
            if self.last_results is None:
                messagebox.showwarning("Warning", "No results available. Run simulation first.")
                return

            try:
                # パラメータ辞書を作成
                # detector_var の値を内部形式に変換
                detector_display = self.detector_var.get()
                if "Yokogata" in detector_display or "横型" in detector_display:
                    detector_type = "yoko"
                elif "Kushigata" in detector_display or "くし形" in detector_display:
                    detector_type = "kushi"
                else:
                    detector_type = detector_display

                # mode_var の値を内部形式に変換
                mode_display = self.mode_var.get()
                if "Ideal" in mode_display:
                    mode = "ramo_ideal"
                elif "Drift" in mode_display:
                    mode = "ramo_drift"
                else:
                    mode = mode_display

                # field_file_var の値を取得
                field_file_str = self.field_file_var.get()
                if field_file_str == "Auto (from detector type)":
                    field_path_display = "Auto"
                else:
                    field_path_display = field_file_str

                params = {
                    'detector_type': detector_type,
                    'field_file': field_path_display,
                    'n_events': int(self.events_var.get()),
                    'mode': mode,
                    'alpha_MeV': 5.486,  # 固定値（Am-241）
                    'mu_e': float(self.mu_e_var.get()),
                    'tau_e': float(self.tau_e_var.get()),
                    'use_fullthick': self.cce_use_fullthick_var.get(),
                    'force_recalc': self.cce_force_recalc_var.get(),
                    'timestamp': datetime.now().isoformat(),
                }

                # 結果を保存
                output_dir = save_simulation_results(
                    results=self.last_results,
                    params=params,
                )

                # 成功メッセージ
                messagebox.showinfo(
                    "Success",
                    f"Results saved to:\n{output_dir}"
                )

                self.log(f"\n{'='*70}")
                self.log(f"Results saved to: {output_dir}")
                self.log('='*70)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
                self.log(f"\nERROR: Failed to save results: {e}")

        # ========== Weighting Potential タブ ==========
    
        def _build_weighting_tab(self):
            """Weighting Potentialタブの構築"""
            tab_frame = self.tab_weighting
            tab_frame.columnconfigure(1, weight=1)
            tab_frame.rowconfigure(0, weight=1)
    
            # 左側：コントロールパネル
            control_frame = ttk.LabelFrame(tab_frame, text="Controls", padding="10")
            control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
            row = 0
    
            # Detectorタイプ選択
            ttk.Label(control_frame, text="Detector:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.weight_detector_var = tk.StringVar(value="yoko")
            detector_combo = ttk.Combobox(
                control_frame,
                textvariable=self.weight_detector_var,
                values=["yoko", "kushi"],
                state="readonly",
                width=15
            )
            detector_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Field File Selection
            ttk.Label(control_frame, text="Field file:").grid(row=row, column=0, sticky=tk.W, pady=2)

            # 利用可能な電界ファイルを探索
            field_dir = "電界"
            available_fields = []
            if os.path.exists(field_dir):
                for f in os.listdir(field_dir):
                    if f.endswith(".npz"):
                        available_fields.append(f)

            # デフォルト値の設定
            default_field = "Auto (from detector type)"
            field_values = [default_field] + sorted(available_fields)

            self.weight_field_file_var = tk.StringVar(value=default_field)
            field_combo = ttk.Combobox(
                control_frame,
                textvariable=self.weight_field_file_var,
                values=field_values,
                state="readonly",
                width=15
            )
            field_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1
    
            # Full-thickness option
            self.use_fullthick_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                control_frame,
                text="Use full-thickness (0-430 μm)",
                variable=self.use_fullthick_var
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

            # Force recalculation option
            self.force_recalc_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                control_frame,
                text="Force recalculation",
                variable=self.force_recalc_var
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

            # Load button
            ttk.Button(control_frame, text="Load Data", command=self.load_weighting_data).grid(
                row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
            )
            row += 1

            # Cross-section axis selection
            ttk.Label(control_frame, text="Cross-section axis:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.slice_axis_var = tk.StringVar(value="z")
            axis_combo = ttk.Combobox(
                control_frame,
                textvariable=self.slice_axis_var,
                values=["x", "y", "z"],
                state="readonly",
                width=10
            )
            axis_combo.bind("<<ComboboxSelected>>", self.on_axis_change)
            axis_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Slice position (numeric input)
            ttk.Label(control_frame, text="Slice position [μm]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.slice_position_var = tk.StringVar(value="430")
            slice_entry = ttk.Entry(control_frame, textvariable=self.slice_position_var, width=15)
            slice_entry.bind("<Return>", lambda e: self.update_weighting_plot())
            slice_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Update button
            ttk.Button(control_frame, text="Update Plot", command=self.update_weighting_plot).grid(
                row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
            )
            row += 1

            # Position range info (will be updated when data is loaded)
            self.slice_range_label = ttk.Label(control_frame, text="Range: --", foreground="gray")
            self.slice_range_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1
    
            # Display type
            ttk.Label(control_frame, text="Display:").grid(row=row, column=0, sticky=tk.W, pady=2)
            row += 1
    
            self.display_var = tk.StringVar(value="phi_w")
            for val, label in [("phi_w", "φ_w (Potential)"), ("|E_w|", "|E_w| (E-field mag)")]:
                ttk.Radiobutton(
                    control_frame,
                    text=label,
                    variable=self.display_var,
                    value=val,
                    command=self.update_weighting_plot
                ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
                row += 1

            # Line profile button
            ttk.Button(
                control_frame,
                text="Show Line Profile",
                command=self.show_line_profile
            ).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            row += 1

            # Statistics button
            ttk.Button(
                control_frame,
                text="Show Statistics",
                command=self.show_weight_statistics
            ).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            row += 1

            # Line profile direction
            ttk.Label(control_frame, text="Profile direction:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.profile_direction_var = tk.StringVar(value="z")
            direction_combo = ttk.Combobox(
                control_frame,
                textvariable=self.profile_direction_var,
                values=["x", "y", "z"],
                state="readonly",
                width=10
            )
            direction_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Line profile coordinates
            ttk.Label(control_frame, text="Fixed coordinates:").grid(row=row, column=0, sticky=tk.W, pady=2)
            row += 1

            profile_frame = ttk.Frame(control_frame)
            profile_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

            ttk.Label(profile_frame, text="x [μm]:").pack(side=tk.LEFT)
            self.profile_x_var = tk.StringVar(value="500")
            ttk.Entry(profile_frame, textvariable=self.profile_x_var, width=6).pack(side=tk.LEFT, padx=2)

            ttk.Label(profile_frame, text="y [μm]:").pack(side=tk.LEFT, padx=(5,0))
            self.profile_y_var = tk.StringVar(value="500")
            ttk.Entry(profile_frame, textvariable=self.profile_y_var, width=6).pack(side=tk.LEFT, padx=2)

            ttk.Label(profile_frame, text="z [μm]:").pack(side=tk.LEFT, padx=(5,0))
            self.profile_z_var = tk.StringVar(value="430")
            ttk.Entry(profile_frame, textvariable=self.profile_z_var, width=6).pack(side=tk.LEFT, padx=2)
            row += 1

            # 右側：プロット領域
            plot_frame = ttk.Frame(tab_frame)
            plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            plot_frame.columnconfigure(0, weight=1)
            plot_frame.rowconfigure(0, weight=1)
    
            # Matplotlib figure
            self.weight_fig = Figure(figsize=(8, 6))
            self.weight_ax = self.weight_fig.add_subplot(111)
            self.weight_canvas = FigureCanvasTkAgg(self.weight_fig, master=plot_frame)
            self.weight_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
            # Toolbar (separate frame to avoid pack/grid conflict)
            toolbar_frame = ttk.Frame(plot_frame)
            toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
            toolbar = NavigationToolbar2Tk(self.weight_canvas, toolbar_frame)
            toolbar.update()
    
            # Status label
            self.weight_status = ttk.Label(plot_frame, text="No data loaded", foreground="gray")
            self.weight_status.grid(row=2, column=0, sticky=tk.W, pady=2)
    
        def load_weighting_data(self):
            """ウェイティングデータを読み込み（全厚or通常、強制再計算対応）"""
            detector_type = self.weight_detector_var.get()
            use_fullthick = self.use_fullthick_var.get()
            force_recalc = self.force_recalc_var.get()

            try:
                self.weight_status.config(text="Loading...", foreground="blue")
                self.update()

                # Field file パスを決定
                field_file_str = self.weight_field_file_var.get()

                # 電界ディレクトリを探す（Windowsパスまたはローカルパス）
                possible_dirs = [
                    r"C:\Users\discu\デスクトップ\python\cce\電界",
                    "電界",
                    "./電界",
                ]
                base_dir = None
                for d in possible_dirs:
                    if os.path.exists(d):
                        base_dir = d
                        break

                if base_dir is None:
                    raise FileNotFoundError("電界 directory not found")

                if field_file_str == "Auto (from detector type)":
                    # Auto: detector_type から決定
                    if detector_type == "yoko":
                        self.field_path = os.path.join(base_dir, "yokogata_field.npz")
                    elif detector_type == "kushi":
                        self.field_path = os.path.join(base_dir, "kushigata_field.npz")
                    else:
                        raise ValueError(f"Unknown detector type: {detector_type}")
                else:
                    # 明示的に指定されたファイル
                    self.field_path = os.path.join(base_dir, field_file_str)

                # 全厚モードか通常モードか
                if use_fullthick:
                    print(f"\n{'='*70}")
                    print(f"GUI: Loading full-thickness weighting potential...")
                    print(f"     Detector: {detector_type}")
                    print(f"     Force recalc: {force_recalc}")
                    print('='*70)

                    # 全厚版を使う
                    phi_w, X, Y, Z = get_weighting_potential_fullthickness(
                        field_path=self.field_path,
                        force_recalc=force_recalc,
                        z_max=430e-6,
                        target_dz=2.5e-6,
                    )
                else:
                    print(f"\n{'='*70}")
                    print(f"GUI: Loading standard weighting potential...")
                    print(f"     Detector: {detector_type}")
                    print(f"     Force recalc: {force_recalc}")
                    print('='*70)

                    # 通常版を使う
                    phi_w, X, Y, Z = get_weighting_potential(
                        field_path=self.field_path,
                        force_recalc=force_recalc,
                    )
    
                dx = X[1] - X[0] if len(X) > 1 else 1e-6
                dy = Y[1] - Y[0] if len(Y) > 1 else 1e-6
                dz = Z[1] - Z[0] if len(Z) > 1 else 1e-6
    
                # E_w = -∇φ_w（中央差分）
                E_wx = np.zeros_like(phi_w)
                E_wy = np.zeros_like(phi_w)
                E_wz = np.zeros_like(phi_w)
    
                E_wx[:, :, 1:-1] = -(phi_w[:, :, 2:] - phi_w[:, :, :-2]) / (2 * dx)
                E_wy[:, 1:-1, :] = -(phi_w[:, 2:, :] - phi_w[:, :-2, :]) / (2 * dy)
                E_wz[1:-1, :, :] = -(phi_w[2:, :, :] - phi_w[:-2, :, :]) / (2 * dz)
    
                self.weight_data = {
                    'phi_w': phi_w,
                    'X': X,
                    'Y': Y,
                    'Z': Z,
                    'E_wx': E_wx,
                    'E_wy': E_wy,
                    'E_wz': E_wz,
                }

                # 範囲情報を更新して初期プロット
                self.weight_status.config(text=f"Loaded: {detector_type}, Grid: {len(X)}x{len(Y)}x{len(Z)}", foreground="green")
                self.on_axis_change()  # 範囲情報を更新してプロットを描画
    
            except FileNotFoundError as e:
                messagebox.showerror("Error", f"File not found: {e}")
                self.weight_status.config(text="Load failed", foreground="red")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
                self.weight_status.config(text="Load failed", foreground="red")
    
        def on_axis_change(self, *args):
            """軸が変更されたときに範囲情報を更新"""
            if self.weight_data is None:
                return

            axis = self.slice_axis_var.get()
            X = self.weight_data['X']
            Y = self.weight_data['Y']
            Z = self.weight_data['Z']

            if axis == 'x':
                min_val = X[0] * 1e6
                max_val = X[-1] * 1e6
                # デフォルト位置を中央に設定
                self.slice_position_var.set(f"{(min_val + max_val) / 2:.1f}")
            elif axis == 'y':
                min_val = Y[0] * 1e6
                max_val = Y[-1] * 1e6
                self.slice_position_var.set(f"{(min_val + max_val) / 2:.1f}")
            else:  # z
                min_val = Z[0] * 1e6
                max_val = Z[-1] * 1e6
                self.slice_position_var.set(f"{Z[-1] * 1e6:.1f}")  # デフォルトは表面

            self.slice_range_label.config(text=f"Range: [{min_val:.1f}, {max_val:.1f}] μm")
            self.update_weighting_plot()

        def update_weighting_plot(self, *args):
            """ウェイティングプロットを更新（x, y, z軸の断面に対応）"""
            if self.weight_data is None:
                return

            try:
                # 設定を取得
                axis = self.slice_axis_var.get()
                position_um = float(self.slice_position_var.get())
                position_m = position_um * 1e-6
                display_type = self.display_var.get()

                phi_w = self.weight_data['phi_w']
                X = self.weight_data['X']
                Y = self.weight_data['Y']
                Z = self.weight_data['Z']

                # 指定位置に最も近いインデックスを見つける
                if axis == 'x':
                    idx = np.argmin(np.abs(X - position_m))
                    actual_pos = X[idx] * 1e6

                    if display_type == "phi_w":
                        data = phi_w[:, :, idx]  # (nz, ny)
                        title = f"φ_w at x={actual_pos:.2f} μm"
                        cmap = 'viridis'
                    else:  # |E_w|
                        E_wx = self.weight_data['E_wx']
                        E_wy = self.weight_data['E_wy']
                        E_wz = self.weight_data['E_wz']
                        data = np.sqrt(E_wx[:, :, idx]**2 + E_wy[:, :, idx]**2 + E_wz[:, :, idx]**2)
                        title = f"|E_w| at x={actual_pos:.2f} μm"
                        cmap = 'hot'

                    extent = [Y[0]*1e6, Y[-1]*1e6, Z[0]*1e6, Z[-1]*1e6]
                    xlabel = 'y [μm]'
                    ylabel = 'z [μm]'

                elif axis == 'y':
                    idx = np.argmin(np.abs(Y - position_m))
                    actual_pos = Y[idx] * 1e6

                    if display_type == "phi_w":
                        data = phi_w[:, idx, :]  # (nz, nx)
                        title = f"φ_w at y={actual_pos:.2f} μm"
                        cmap = 'viridis'
                    else:  # |E_w|
                        E_wx = self.weight_data['E_wx']
                        E_wy = self.weight_data['E_wy']
                        E_wz = self.weight_data['E_wz']
                        data = np.sqrt(E_wx[:, idx, :]**2 + E_wy[:, idx, :]**2 + E_wz[:, idx, :]**2)
                        title = f"|E_w| at y={actual_pos:.2f} μm"
                        cmap = 'hot'

                    extent = [X[0]*1e6, X[-1]*1e6, Z[0]*1e6, Z[-1]*1e6]
                    xlabel = 'x [μm]'
                    ylabel = 'z [μm]'

                else:  # z
                    idx = np.argmin(np.abs(Z - position_m))
                    actual_pos = Z[idx] * 1e6

                    if display_type == "phi_w":
                        data = phi_w[idx, :, :]  # (ny, nx)
                        title = f"φ_w at z={actual_pos:.2f} μm"
                        cmap = 'viridis'
                    else:  # |E_w|
                        E_wx = self.weight_data['E_wx']
                        E_wy = self.weight_data['E_wy']
                        E_wz = self.weight_data['E_wz']
                        data = np.sqrt(E_wx[idx, :, :]**2 + E_wy[idx, :, :]**2 + E_wz[idx, :, :]**2)
                        title = f"|E_w| at z={actual_pos:.2f} μm"
                        cmap = 'hot'

                    extent = [X[0]*1e6, X[-1]*1e6, Y[0]*1e6, Y[-1]*1e6]
                    xlabel = 'x [μm]'
                    ylabel = 'y [μm]'

                # プロット
                self.weight_ax.clear()
                im = self.weight_ax.imshow(
                    data,
                    extent=extent,
                    origin='lower',
                    cmap=cmap,
                    aspect='auto'
                )
                self.weight_ax.set_xlabel(xlabel)
                self.weight_ax.set_ylabel(ylabel)
                self.weight_ax.set_title(title)

                # Colorbarの更新（図が小さくなる問題を修正）
                if hasattr(self, 'weight_colorbar') and self.weight_colorbar is not None:
                    self.weight_colorbar.update_normal(im)
                else:
                    self.weight_colorbar = self.weight_fig.colorbar(im, ax=self.weight_ax)

                self.weight_canvas.draw()

            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid slice position: {e}")
            except Exception as e:
                print(f"Plot update error: {e}")

        def show_line_profile(self):
            """φ_w の line profile を表示（x, y, z 方向に対応）"""
            if self.weight_data is None:
                messagebox.showwarning("Warning", "No weighting data loaded.")
                return

            try:
                # 設定を取得
                direction = self.profile_direction_var.get()
                x_um = float(self.profile_x_var.get())
                y_um = float(self.profile_y_var.get())
                z_um = float(self.profile_z_var.get())

                phi_w = self.weight_data['phi_w']
                X = self.weight_data['X']
                Y = self.weight_data['Y']
                Z = self.weight_data['Z']

                # μm → m に変換
                x_m = x_um * 1e-6
                y_m = y_um * 1e-6
                z_m = z_um * 1e-6

                # 方向に応じてプロファイルを抽出
                if direction == 'x':
                    # x方向のプロファイル: y, z を固定
                    iy = np.argmin(np.abs(Y - y_m))
                    iz = np.argmin(np.abs(Z - z_m))
                    profile_data = phi_w[iz, iy, :]
                    axis_data = X * 1e6
                    xlabel = 'x [μm]'
                    title = f'φ_w(x) at (y={Y[iy]*1e6:.2f} μm, z={Z[iz]*1e6:.2f} μm)'
                    stats_labels = (f"x={X[0]*1e6:.2f}", f"x={X[-1]*1e6:.2f}")

                elif direction == 'y':
                    # y方向のプロファイル: x, z を固定
                    ix = np.argmin(np.abs(X - x_m))
                    iz = np.argmin(np.abs(Z - z_m))
                    profile_data = phi_w[iz, :, ix]
                    axis_data = Y * 1e6
                    xlabel = 'y [μm]'
                    title = f'φ_w(y) at (x={X[ix]*1e6:.2f} μm, z={Z[iz]*1e6:.2f} μm)'
                    stats_labels = (f"y={Y[0]*1e6:.2f}", f"y={Y[-1]*1e6:.2f}")

                else:  # z
                    # z方向のプロファイル: x, y を固定
                    ix = np.argmin(np.abs(X - x_m))
                    iy = np.argmin(np.abs(Y - y_m))
                    profile_data = phi_w[:, iy, ix]
                    axis_data = Z * 1e6
                    xlabel = 'z [μm]'
                    title = f'φ_w(z) at (x={X[ix]*1e6:.2f} μm, y={Y[iy]*1e6:.2f} μm)'
                    stats_labels = (f"z={Z[0]*1e6:.2f}", f"z={Z[-1]*1e6:.2f}")

                # 新しいウィンドウで表示
                profile_window = tk.Toplevel(self)
                profile_window.title(f"φ_w({direction}) Line Profile")
                profile_window.geometry("600x500")

                # Matplotlib figure
                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.plot(axis_data, profile_data, 'b-', linewidth=2)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('φ_w')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)

                # Canvas
                canvas = FigureCanvasTkAgg(fig, master=profile_window)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Toolbar
                toolbar = NavigationToolbar2Tk(canvas, profile_window)
                toolbar.update()

                # 統計情報
                stats_text = f"\nStatistics:\n"
                stats_text += f"  φ_w({stats_labels[0]}):  {profile_data[0]:.6f}\n"
                stats_text += f"  φ_w({stats_labels[1]}): {profile_data[-1]:.6f}\n"
                stats_text += f"  min:        {profile_data.min():.6f}\n"
                stats_text += f"  max:        {profile_data.max():.6f}\n"
                stats_text += f"  mean:       {profile_data.mean():.6f}\n"

                stats_label = ttk.Label(profile_window, text=stats_text, font=('Courier', 9))
                stats_label.pack(pady=5)

                canvas.draw()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid coordinate value: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show line profile: {e}")

        def show_weight_statistics(self):
            """φ_w の統計情報を表示"""
            if self.weight_data is None:
                messagebox.showwarning("Warning", "No weighting data loaded.")
                return

            try:
                phi_w = self.weight_data['phi_w']
                X = self.weight_data['X']
                Y = self.weight_data['Y']
                Z = self.weight_data['Z']
                E_wx = self.weight_data['E_wx']
                E_wy = self.weight_data['E_wy']
                E_wz = self.weight_data['E_wz']

                nz, ny, nx = phi_w.shape

                # 新しいウィンドウ
                stats_window = tk.Toplevel(self)
                stats_window.title("Weighting Potential Statistics")
                stats_window.geometry("800x600")

                # ScrolledText
                text_widget = scrolledtext.ScrolledText(
                    stats_window,
                    width=90,
                    height=35,
                    font=('Courier', 9)
                )
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # ヘッダー
                text_widget.insert(tk.END, "="*80 + "\n")
                text_widget.insert(tk.END, "Weighting Potential Statistics\n")
                text_widget.insert(tk.END, "="*80 + "\n\n")

                text_widget.insert(tk.END, f"Grid dimensions: {nx} x {ny} x {nz}\n")
                text_widget.insert(tk.END, f"X: [{X.min()*1e6:.1f}, {X.max()*1e6:.1f}] μm\n")
                text_widget.insert(tk.END, f"Y: [{Y.min()*1e6:.1f}, {Y.max()*1e6:.1f}] μm\n")
                text_widget.insert(tk.END, f"Z: [{Z.min()*1e6:.1f}, {Z.max()*1e6:.1f}] μm\n\n")

                # Overall statistics
                text_widget.insert(tk.END, "[Overall φ_w statistics]\n")
                text_widget.insert(tk.END, f"  min:    {phi_w.min():.6f}\n")
                text_widget.insert(tk.END, f"  max:    {phi_w.max():.6f}\n")
                text_widget.insert(tk.END, f"  mean:   {phi_w.mean():.6f}\n")
                text_widget.insert(tk.END, f"  median: {np.median(phi_w):.6f}\n\n")

                # Z-dependent statistics（最大30層まで表示）
                text_widget.insert(tk.END, f"[φ_w statistics by z-layer]\n")
                text_widget.insert(tk.END, f"{'z[idx]':>7s} {'z[μm]':>8s} {'min':>10s} {'mean':>10s} {'median':>10s} {'max':>10s}\n")
                text_widget.insert(tk.END, "-"*60 + "\n")

                step = max(1, nz // 30)
                for k in range(0, nz, step):
                    layer = phi_w[k, :, :]
                    text_widget.insert(
                        tk.END,
                        f"{k:7d} {Z[k]*1e6:8.2f} {layer.min():10.6f} {layer.mean():10.6f} "
                        f"{np.median(layer):10.6f} {layer.max():10.6f}\n"
                    )

                # |E_w| statistics
                E_mag = np.sqrt(E_wx**2 + E_wy**2 + E_wz**2)
                text_widget.insert(tk.END, f"\n[|E_w| statistics (overall)]\n")
                text_widget.insert(tk.END, f"  min:    {E_mag.min():.3e} V/m\n")
                text_widget.insert(tk.END, f"  median: {np.median(E_mag):.3e} V/m\n")
                text_widget.insert(tk.END, f"  95%:    {np.percentile(E_mag, 95):.3e} V/m\n")
                text_widget.insert(tk.END, f"  max:    {E_mag.max():.3e} V/m\n\n")

                # Surface vs backside comparison
                text_widget.insert(tk.END, f"[Surface (z={Z[-1]*1e6:.2f} μm) vs Backside (z={Z[0]*1e6:.2f} μm)]\n")
                phi_surface = phi_w[-1, :, :]
                phi_back = phi_w[0, :, :]
                text_widget.insert(tk.END, f"  Surface: mean={phi_surface.mean():.6f}, std={phi_surface.std():.6f}\n")
                text_widget.insert(tk.END, f"  Backside: mean={phi_back.mean():.6f}, std={phi_back.std():.6f}\n")

                text_widget.insert(tk.END, "\n" + "="*80 + "\n")
                text_widget.config(state=tk.DISABLED)  # Read-only

            except Exception as e:
                messagebox.showerror("Error", f"Failed to show statistics: {e}")

        # ========== Electric Field Visualization タブ ==========

        def _build_field_tab(self):
            """Electric Fieldタブの構築"""
            tab_frame = self.tab_field
            tab_frame.columnconfigure(1, weight=1)
            tab_frame.rowconfigure(0, weight=1)

            # 左側：コントロールパネル
            control_frame = ttk.LabelFrame(tab_frame, text="Controls", padding="10")
            control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

            row = 0

            # Field File Selection
            ttk.Label(control_frame, text="Field file:").grid(row=row, column=0, sticky=tk.W, pady=2)

            # 利用可能な電界ファイルを探索
            field_dir = "電界"
            available_fields = []
            if os.path.exists(field_dir):
                for f in os.listdir(field_dir):
                    if f.endswith(".npz"):
                        available_fields.append(os.path.join(field_dir, f))

            # デフォルト値の設定
            default_field = available_fields[0] if available_fields else "No field files found"
            field_values = sorted(available_fields) if available_fields else [default_field]

            self.field_field_file_var = tk.StringVar(value=default_field)
            field_combo = ttk.Combobox(
                control_frame,
                textvariable=self.field_field_file_var,
                values=field_values,
                state="readonly",
                width=15
            )
            field_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Load button
            ttk.Button(control_frame, text="Load Data", command=self.load_field_data).grid(
                row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
            )
            row += 1

            # Field component selection
            ttk.Label(control_frame, text="Field component:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.field_component_var = tk.StringVar(value="|E|")
            component_combo = ttk.Combobox(
                control_frame,
                textvariable=self.field_component_var,
                values=["Ex", "Ey", "Ez", "|E|"],
                state="readonly",
                width=15
            )
            component_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            component_combo.bind("<<ComboboxSelected>>", lambda e: self.update_field_plot())
            row += 1

            # Cross-section axis selection
            ttk.Label(control_frame, text="Cross-section axis:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.field_slice_axis_var = tk.StringVar(value="z")
            axis_combo = ttk.Combobox(
                control_frame,
                textvariable=self.field_slice_axis_var,
                values=["x", "y", "z"],
                state="readonly",
                width=15
            )
            axis_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            axis_combo.bind("<<ComboboxSelected>>", self.on_field_axis_change)
            row += 1

            # Slice position (numeric input)
            ttk.Label(control_frame, text="Slice position [μm]:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.field_slice_position_var = tk.StringVar(value="430")
            slice_entry = ttk.Entry(control_frame, textvariable=self.field_slice_position_var, width=15)
            slice_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            slice_entry.bind("<Return>", lambda e: self.update_field_plot())
            row += 1

            # Position range info
            self.field_slice_range_label = ttk.Label(control_frame, text="Range: --", foreground="gray")
            self.field_slice_range_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

            # Update plot button
            ttk.Button(control_frame, text="Update Plot", command=self.update_field_plot).grid(
                row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
            )
            row += 1

            # Line profile button
            ttk.Button(control_frame, text="Show Line Profile", command=self.show_field_line_profile).grid(
                row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2
            )
            row += 1

            # Line profile direction
            ttk.Label(control_frame, text="Profile direction:").grid(row=row, column=0, sticky=tk.W, pady=2)
            self.field_profile_direction_var = tk.StringVar(value="z")
            direction_combo = ttk.Combobox(
                control_frame,
                textvariable=self.field_profile_direction_var,
                values=["x", "y", "z"],
                state="readonly",
                width=15
            )
            direction_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1

            # Line profile coordinates
            ttk.Label(control_frame, text="Fixed coordinates:").grid(row=row, column=0, sticky=tk.W, pady=2)
            row += 1

            profile_frame = ttk.Frame(control_frame)
            profile_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

            ttk.Label(profile_frame, text="x [μm]:").pack(side=tk.LEFT)
            self.field_profile_x_var = tk.StringVar(value="500")
            ttk.Entry(profile_frame, textvariable=self.field_profile_x_var, width=6).pack(side=tk.LEFT, padx=2)

            ttk.Label(profile_frame, text="y [μm]:").pack(side=tk.LEFT, padx=(5,0))
            self.field_profile_y_var = tk.StringVar(value="500")
            ttk.Entry(profile_frame, textvariable=self.field_profile_y_var, width=6).pack(side=tk.LEFT, padx=2)

            ttk.Label(profile_frame, text="z [μm]:").pack(side=tk.LEFT, padx=(5,0))
            self.field_profile_z_var = tk.StringVar(value="215")
            ttk.Entry(profile_frame, textvariable=self.field_profile_z_var, width=6).pack(side=tk.LEFT, padx=2)
            row += 1

            # 右側：プロット領域
            plot_frame = ttk.Frame(tab_frame)
            plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            plot_frame.columnconfigure(0, weight=1)
            plot_frame.rowconfigure(0, weight=1)

            # Matplotlib figure
            self.field_fig = Figure(figsize=(8, 6))
            self.field_ax = self.field_fig.add_subplot(111)
            self.field_canvas = FigureCanvasTkAgg(self.field_fig, master=plot_frame)
            self.field_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Toolbar
            toolbar_frame = ttk.Frame(plot_frame)
            toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
            toolbar = NavigationToolbar2Tk(self.field_canvas, toolbar_frame)
            toolbar.update()

            # データ保持用
            self.field_data = None

        def load_field_data(self):
            """電場データをロード"""
            try:
                field_path = self.field_field_file_var.get()

                if not os.path.exists(field_path):
                    messagebox.showerror("Error", f"Field file not found: {field_path}")
                    return

                # 表面電場データを取得
                field_data = load_field_npz(field_path)
                Ex = field_data['Ex']
                Ey = field_data['Ey']
                Ez = field_data['Ez']
                X = field_data['X']
                Y = field_data['Y']
                Z = field_data['Z']

                # 電場の大きさを計算
                E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

                # データを保存
                self.field_data = {
                    'Ex': Ex,
                    'Ey': Ey,
                    'Ez': Ez,
                    'E_mag': E_mag,
                    'X': X,
                    'Y': Y,
                    'Z': Z,
                }

                messagebox.showinfo("Success", "Field data loaded successfully!")

                # プロットを更新
                self.on_field_axis_change()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load field data: {e}")
                import traceback
                traceback.print_exc()

        def on_field_axis_change(self, *args):
            """軸変更時にスライス位置の範囲を更新"""
            if self.field_data is None:
                return

            axis = self.field_slice_axis_var.get()
            X = self.field_data['X']
            Y = self.field_data['Y']
            Z = self.field_data['Z']

            if axis == 'x':
                min_val = X[0] * 1e6
                max_val = X[-1] * 1e6
                self.field_slice_position_var.set(f"{(min_val + max_val) / 2:.1f}")
            elif axis == 'y':
                min_val = Y[0] * 1e6
                max_val = Y[-1] * 1e6
                self.field_slice_position_var.set(f"{(min_val + max_val) / 2:.1f}")
            else:  # z
                min_val = Z[0] * 1e6
                max_val = Z[-1] * 1e6
                self.field_slice_position_var.set(f"{Z[-1] * 1e6:.1f}")  # デフォルトは表面

            self.field_slice_range_label.config(text=f"Range: [{min_val:.1f}, {max_val:.1f}] μm")
            self.update_field_plot()

        def update_field_plot(self, *args):
            """電場プロットを更新（x, y, z軸の断面に対応）"""
            if self.field_data is None:
                return

            try:
                # 設定を取得
                axis = self.field_slice_axis_var.get()
                position_um = float(self.field_slice_position_var.get())
                position_m = position_um * 1e-6
                component = self.field_component_var.get()

                Ex = self.field_data['Ex']
                Ey = self.field_data['Ey']
                Ez = self.field_data['Ez']
                E_mag = self.field_data['E_mag']
                X = self.field_data['X']
                Y = self.field_data['Y']
                Z = self.field_data['Z']

                # 電場成分を選択
                if component == "Ex":
                    field_array = Ex
                    label = "Ex [V/m]"
                    cmap = 'RdBu_r'
                elif component == "Ey":
                    field_array = Ey
                    label = "Ey [V/m]"
                    cmap = 'RdBu_r'
                elif component == "Ez":
                    field_array = Ez
                    label = "Ez [V/m]"
                    cmap = 'RdBu_r'
                else:  # |E|
                    field_array = E_mag
                    label = "|E| [V/m]"
                    cmap = 'hot'

                # 指定位置に最も近いインデックスを見つける
                if axis == 'x':
                    idx = np.argmin(np.abs(X - position_m))
                    actual_pos = X[idx] * 1e6
                    data = field_array[:, :, idx]  # (nz, ny)
                    title = f"{component} at x={actual_pos:.2f} μm"
                    extent = [Y[0]*1e6, Y[-1]*1e6, Z[0]*1e6, Z[-1]*1e6]
                    xlabel = 'y [μm]'
                    ylabel = 'z [μm]'

                elif axis == 'y':
                    idx = np.argmin(np.abs(Y - position_m))
                    actual_pos = Y[idx] * 1e6
                    data = field_array[:, idx, :]  # (nz, nx)
                    title = f"{component} at y={actual_pos:.2f} μm"
                    extent = [X[0]*1e6, X[-1]*1e6, Z[0]*1e6, Z[-1]*1e6]
                    xlabel = 'x [μm]'
                    ylabel = 'z [μm]'

                else:  # z
                    idx = np.argmin(np.abs(Z - position_m))
                    actual_pos = Z[idx] * 1e6
                    data = field_array[idx, :, :]  # (ny, nx)
                    title = f"{component} at z={actual_pos:.2f} μm"
                    extent = [X[0]*1e6, X[-1]*1e6, Y[0]*1e6, Y[-1]*1e6]
                    xlabel = 'x [μm]'
                    ylabel = 'y [μm]'

                # プロット
                self.field_ax.clear()
                im = self.field_ax.imshow(
                    data,
                    extent=extent,
                    origin='lower',
                    aspect='auto',
                    cmap=cmap,
                )
                self.field_ax.set_xlabel(xlabel)
                self.field_ax.set_ylabel(ylabel)
                self.field_ax.set_title(title)

                # カラーバー
                if hasattr(self, 'field_colorbar'):
                    self.field_colorbar.remove()
                self.field_colorbar = self.field_fig.colorbar(im, ax=self.field_ax, label=label)

                self.field_canvas.draw()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to update plot: {e}")

        def show_field_line_profile(self):
            """電場の line profile を表示（x, y, z 方向に対応）"""
            if self.field_data is None:
                messagebox.showwarning("Warning", "No field data loaded.")
                return

            try:
                # 設定を取得
                direction = self.field_profile_direction_var.get()
                x_um = float(self.field_profile_x_var.get())
                y_um = float(self.field_profile_y_var.get())
                z_um = float(self.field_profile_z_var.get())
                component = self.field_component_var.get()

                Ex = self.field_data['Ex']
                Ey = self.field_data['Ey']
                Ez = self.field_data['Ez']
                E_mag = self.field_data['E_mag']
                X = self.field_data['X']
                Y = self.field_data['Y']
                Z = self.field_data['Z']

                # 電場成分を選択
                if component == "Ex":
                    field_array = Ex
                    ylabel = "Ex [V/m]"
                elif component == "Ey":
                    field_array = Ey
                    ylabel = "Ey [V/m]"
                elif component == "Ez":
                    field_array = Ez
                    ylabel = "Ez [V/m]"
                else:  # |E|
                    field_array = E_mag
                    ylabel = "|E| [V/m]"

                # μm → m に変換
                x_m = x_um * 1e-6
                y_m = y_um * 1e-6
                z_m = z_um * 1e-6

                # 方向に応じてプロファイルを抽出
                if direction == 'x':
                    # x方向のプロファイル: y, z を固定
                    iy = np.argmin(np.abs(Y - y_m))
                    iz = np.argmin(np.abs(Z - z_m))
                    profile_data = field_array[iz, iy, :]
                    axis_data = X * 1e6
                    xlabel = 'x [μm]'
                    title = f'{component}(x) at (y={Y[iy]*1e6:.2f} μm, z={Z[iz]*1e6:.2f} μm)'

                elif direction == 'y':
                    # y方向のプロファイル: x, z を固定
                    ix = np.argmin(np.abs(X - x_m))
                    iz = np.argmin(np.abs(Z - z_m))
                    profile_data = field_array[iz, :, ix]
                    axis_data = Y * 1e6
                    xlabel = 'y [μm]'
                    title = f'{component}(y) at (x={X[ix]*1e6:.2f} μm, z={Z[iz]*1e6:.2f} μm)'

                else:  # z
                    # z方向のプロファイル: x, y を固定
                    ix = np.argmin(np.abs(X - x_m))
                    iy = np.argmin(np.abs(Y - y_m))
                    profile_data = field_array[:, iy, ix]
                    axis_data = Z * 1e6
                    xlabel = 'z [μm]'
                    title = f'{component}(z) at (x={X[ix]*1e6:.2f} μm, y={Y[iy]*1e6:.2f} μm)'

                # 新しいウィンドウで表示
                profile_window = tk.Toplevel(self)
                profile_window.title(f"{component}({direction}) Line Profile")
                profile_window.geometry("600x500")

                # Matplotlib figure
                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.plot(axis_data, profile_data, 'b-', linewidth=2)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                # Canvas
                canvas = FigureCanvasTkAgg(fig, master=profile_window)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Toolbar
                toolbar = NavigationToolbar2Tk(canvas, profile_window)
                toolbar.update()

                # 統計情報
                stats_text = f"\nStatistics:\n"
                stats_text += f"  min:  {profile_data.min():.3e} V/m\n"
                stats_text += f"  max:  {profile_data.max():.3e} V/m\n"
                stats_text += f"  mean: {profile_data.mean():.3e} V/m\n"

                stats_label = ttk.Label(profile_window, text=stats_text, font=('Courier', 9))
                stats_label.pack(pady=5)

                canvas.draw()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid coordinate value: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show line profile: {e}")

        # ========== Diagnostics タブ ==========

        def _build_diagnostics_tab(self):
            """Diagnosticsタブの構築"""
            tab_frame = self.tab_diagnostics
            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(1, weight=1)
    
            # 上部：ボタン
            button_frame = ttk.Frame(tab_frame)
            button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
    
            ttk.Button(
                button_frame,
                text="Run Diagnostics",
                command=self.run_diagnostics
            ).pack(side=tk.LEFT, padx=5)
    
            # 下部：結果表示
            result_frame = ttk.LabelFrame(tab_frame, text="Diagnostic Results", padding="10")
            result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
            result_frame.columnconfigure(0, weight=1)
            result_frame.rowconfigure(0, weight=1)
    
            self.diag_text = scrolledtext.ScrolledText(
                result_frame,
                wrap=tk.WORD,
                width=100,
                height=30,
                font=("Courier", 9)
            )
            self.diag_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
        def run_diagnostics(self):
            """診断を実行"""
            if self.weight_data is None:
                messagebox.showwarning("Warning", "No weighting data loaded. Go to 'Weighting Potential' tab and load data first.")
                return
    
            self.diag_text.delete(1.0, tk.END)
            self.diag_text.insert(tk.END, "="*70 + "\n")
            self.diag_text.insert(tk.END, "Weighting Potential Diagnostics\n")
            self.diag_text.insert(tk.END, "="*70 + "\n\n")
    
            try:
                phi_w = self.weight_data['phi_w']
                X = self.weight_data['X']
                Y = self.weight_data['Y']
                Z = self.weight_data['Z']
    
                # 電極マスク作成（再利用）
                # field_pathから電位Vを読み込み
                field_data = np.load(self.field_path)
                V = field_data['V']
    
                collect_mask, ground_mask, k_surface = create_electrode_masks(V, Z)
    
                # 1. 電極境界条件チェック
                self.diag_text.insert(tk.END, "[Electrode boundary check]\n")
    
                # Collect electrode
                phi_collect = phi_w[collect_mask]
                if len(phi_collect) > 0:
                    mean_c = phi_collect.mean()
                    min_c = phi_collect.min()
                    max_c = phi_collect.max()
                    max_dev_c = max(abs(1.0 - min_c), abs(max_c - 1.0))
    
                    self.diag_text.insert(tk.END, f"  Collect electrode (φ_w ≈ 1):\n")
                    self.diag_text.insert(tk.END, f"    mean={mean_c:.6f}, min={min_c:.6f}, max={max_c:.6f}\n")
                    self.diag_text.insert(tk.END, f"    max|Δ|={max_dev_c:.6f}\n")
    
                    tol = 1e-2
                    if max_dev_c < tol:
                        self.diag_text.insert(tk.END, f"    → OK (max|Δ| < tol={tol})\n")
                    else:
                        self.diag_text.insert(tk.END, f"    → WARNING: exceeds tol={tol}\n")
    
                # Ground electrode
                phi_ground = phi_w[ground_mask]
                if len(phi_ground) > 0:
                    mean_g = phi_ground.mean()
                    min_g = phi_ground.min()
                    max_g = phi_ground.max()
                    max_dev_g = max(abs(min_g), abs(max_g))
    
                    self.diag_text.insert(tk.END, f"\n  Ground electrode (φ_w ≈ 0):\n")
                    self.diag_text.insert(tk.END, f"    mean={mean_g:.6f}, min={min_g:.6f}, max={max_g:.6f}\n")
                    self.diag_text.insert(tk.END, f"    max|Δ|={max_dev_g:.6f}\n")
    
                    if max_dev_g < tol:
                        self.diag_text.insert(tk.END, f"    → OK (max|Δ| < tol={tol})\n")
                    else:
                        self.diag_text.insert(tk.END, f"    → WARNING: exceeds tol={tol}\n")
    
                # 2. Laplacian residual
                self.diag_text.insert(tk.END, f"\n[Laplacian residual (∇²φ_w ≈ 0)]\n")
    
                nz, ny, nx = phi_w.shape
                dx = X[1] - X[0] if len(X) > 1 else 1e-6
                dy = Y[1] - Y[0] if len(Y) > 1 else 1e-6
                dz = Z[1] - Z[0] if len(Z) > 1 else 1e-6
    
                # 内部セルのみ
                fixed = collect_mask | ground_mask
                residuals = []
    
                for k in range(1, nz-1):
                    for j in range(1, ny-1):
                        for i in range(1, nx-1):
                            if fixed[k, j, i]:
                                continue
    
                            lap = (
                                (phi_w[k, j, i+1] + phi_w[k, j, i-1] - 2*phi_w[k, j, i]) / dx**2 +
                                (phi_w[k, j+1, i] + phi_w[k, j-1, i] - 2*phi_w[k, j, i]) / dy**2 +
                                (phi_w[k+1, j, i] + phi_w[k-1, j, i] - 2*phi_w[k, j, i]) / dz**2
                            )
                            residuals.append(abs(lap))
    
                if residuals:
                    residuals = np.array(residuals)
                    rms = np.sqrt((residuals**2).mean())
                    max_res = residuals.max()
    
                    self.diag_text.insert(tk.END, f"  RMS = {rms:.3e}\n")
                    self.diag_text.insert(tk.END, f"  Max = {max_res:.3e}\n")
    
                    if rms < 1e-4 and max_res < 1e-3:
                        self.diag_text.insert(tk.END, f"  → OK (RMS < 1e-4, Max < 1e-3)\n")
                    else:
                        self.diag_text.insert(tk.END, f"  → WARNING: residuals are relatively large\n")
    
                # 3. |E_w| 統計
                self.diag_text.insert(tk.END, f"\n[|E_w| statistics]\n")
    
                E_wx = self.weight_data['E_wx']
                E_wy = self.weight_data['E_wy']
                E_wz = self.weight_data['E_wz']
                E_mag = np.sqrt(E_wx**2 + E_wy**2 + E_wz**2)
    
                # 表面付近（最後の層）
                iz = len(Z) - 1
                E_surface = E_mag[iz, :, :]
    
                self.diag_text.insert(tk.END, f"  At z = {Z[iz]*1e6:.2f} μm (surface):\n")
                self.diag_text.insert(tk.END, f"    min    = {E_surface.min():.3e} V/m\n")
                self.diag_text.insert(tk.END, f"    median = {np.median(E_surface):.3e} V/m\n")
                self.diag_text.insert(tk.END, f"    95%    = {np.percentile(E_surface, 95):.3e} V/m\n")
                self.diag_text.insert(tk.END, f"    max    = {E_surface.max():.3e} V/m\n")
    
                # Summary
                self.diag_text.insert(tk.END, f"\n{'='*70}\n")
                self.diag_text.insert(tk.END, "Summary:\n")
                self.diag_text.insert(tk.END, "  - Electrode boundary conditions checked.\n")
                self.diag_text.insert(tk.END, "  - Laplacian residual computed in bulk.\n")
                self.diag_text.insert(tk.END, "  → See detailed results above.\n")
                self.diag_text.insert(tk.END, "="*70 + "\n")
    
            except Exception as e:
                import traceback
                self.diag_text.insert(tk.END, f"\nERROR:\n{traceback.format_exc()}\n")


def main_gui():
    """GUI モードのメイン関数"""
    if not TKINTER_AVAILABLE:
        print("ERROR: tkinter is not available. Cannot launch GUI.")
        print("Please install tkinter or use CLI mode instead.")
        return

    app = CCESimulationGUI()
    app.mainloop()


if __name__ == "__main__":
    # --gui オプションで GUI / CLI を切り替え
    import sys

    if "--gui" in sys.argv:
        main_gui()
    else:
        main_cli()
