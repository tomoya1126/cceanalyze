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
        'V': 3D電位 [V], shape (nz, ny, nx)
        'Ex', 'Ey', 'Ez': 3D電界 [V/m]
        'X', 'Y', 'Z': 1D座標軸 [m]

    Notes
    -----
    WARNING: 実際のnpzファイルはα線入射表面領域（z=410〜430 μm付近）のみを
             含んでいます。バルク領域（z=0〜410 μm）は含まれていません。
             ramo_drift モードでは、この制約を考慮した近似が使われます。

    TODO: 実際のnpz内でのキー名（大文字/小文字、x/X等）が違う場合は
          ここを調整すること。
    """
    data = np.load(path)

    # キー名の確認と取得
    # TODO: 実際のキー名に合わせて調整
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

    # 電位と電界
    result['V'] = data['V']
    result['Ex'] = data['Ex']
    result['Ey'] = data['Ey']
    result['Ez'] = data['Ez']

    print(f"Loaded field data from {path}")
    print(f"  Grid: {len(result['X'])} x {len(result['Y'])} x {len(result['Z'])}")
    print(f"  X: [{result['X'].min()*1e6:.2f}, {result['X'].max()*1e6:.2f}] μm")
    print(f"  Y: [{result['Y'].min()*1e6:.2f}, {result['Y'].max()*1e6:.2f}] μm")
    print(f"  Z: [{result['Z'].min()*1e6:.2f}, {result['Z'].max()*1e6:.2f}] μm")
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
    z_seg: np.ndarray,
    dE_seg: np.ndarray,
    x_event: float,
    y_event: float,
    z_surface: float,
    mu_e: float,
    tau_e: float,
) -> float:
    """
    Drift モード: 実電界＋有限寿命を考慮した CCE 計算。

    Parameters
    ----------
    phi_w : np.ndarray
        重み電位, shape (nz, ny, nx)
    X, Y, Z : np.ndarray
        座標軸 [m]
    Ex, Ey, Ez : np.ndarray
        電界成分 [V/m], shape (nz, ny, nx)
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

    Returns
    -------
    float
        CCE (0~1)

    Notes
    -----
    モデルの仮定：
    - 電子のみが信号に寄与（正孔は無視）
    - z 方向の1D近似ドリフト
    - 有限寿命 τ_e による再結合損失を考慮
    - 電界データは z=410〜430 μm のみ含む → バルク領域は近似

    TODO: バルク領域（z < Z.min()）の電界は、表面領域の平均値で近似しています。
          より精密なモデルでは、バルク電界の実測値または理論モデルが必要です。
    """
    # 総e-hペア数と生成電荷
    N_i = dE_seg / W_EH
    N_total = N_i.sum()
    Q_gen = N_total * Q_E  # [C]

    if N_total == 0:
        return 0.0

    # 誘導電荷の計算（N_i個のキャリア分を直接計算）
    Q_induced = 0.0

    # バルク領域の代表電界値（Z範囲内の平均）
    # TODO: より精密な近似が必要な場合はここを改良
    E_mag_bulk = np.sqrt(Ex**2 + Ey**2 + Ez**2).mean()  # [V/m]

    # バルク領域のφ_w近似値（表面に最も近い値を使用）
    # z < Z.min() の場合は Z.min() 位置の φ_w を使う
    phi_w_bulk_cache = {}  # (x_event, y_event) → φ_w at Z.min()

    for i in range(len(z_seg)):
        # セグメント i の生成位置
        # α線は z_surface から入射し、-z 方向（内部）に進む
        z_i = z_surface - z_seg[i]

        # collect 電極までの距離（z方向の1D近似）
        d_i = z_seg[i]  # [m]

        # 電界の取得
        if z_i >= Z[0] and z_i <= Z[-1]:
            # z が電界データの範囲内 → 補間で取得
            Ex_i = trilinear_interpolate(Ex, X, Y, Z, x_event, y_event, z_i)
            Ey_i = trilinear_interpolate(Ey, X, Y, Z, x_event, y_event, z_i)
            Ez_i = trilinear_interpolate(Ez, X, Y, Z, x_event, y_event, z_i)
            E_mag = np.sqrt(Ex_i**2 + Ey_i**2 + Ez_i**2)  # [V/m]
        else:
            # z が電界データの範囲外（バルク領域）→ 近似値を使用
            E_mag = E_mag_bulk  # [V/m]

        # ドリフト速度 [m/s]
        # μ_e [cm²/Vs] * E [V/m] = μ_e * 1e-4 [m²/Vs] * E [V/m] = μ_e * 1e-4 * E [m/s]
        v_drift = mu_e * 1e-4 * E_mag  # [m/s]

        if v_drift == 0:
            # 電界がゼロの場合は電子が動けない → 寄与なし
            continue

        # ドリフト時間 [s]
        t_drift = d_i / v_drift

        # 生存確率（再結合による損失）
        f_survival = np.exp(-t_drift / tau_e)

        # 重み電位
        if z_i >= Z[0] and z_i <= Z[-1]:
            # 範囲内 → 補間
            phi_w_i = trilinear_interpolate(phi_w, X, Y, Z, x_event, y_event, z_i)
        else:
            # 範囲外（バルク領域、z < Z.min()）
            # → Z.min() 位置のφ_wで近似（表面に最も近い値）
            # TODO: より精密なモデルが必要な場合はここを改良
            if (x_event, y_event) not in phi_w_bulk_cache:
                phi_w_bulk_cache[(x_event, y_event)] = trilinear_interpolate(
                    phi_w, X, Y, Z, x_event, y_event, Z[0]
                )
            phi_w_i = phi_w_bulk_cache[(x_event, y_event)]

        # N_i[i] 個のキャリアが誘起する電荷（Shockley-Ramo）
        # Q_e = N_i * e * f_survival * (1 - φ_w(start))
        Q_e_i = N_i[i] * Q_E * f_survival * (1 - phi_w_i)

        # 加算
        Q_induced += Q_e_i

    # CCE = 収集電荷 / 生成電荷
    cce = Q_induced / Q_gen if Q_gen > 0 else 0.0

    return cce


# ========== メインシミュレーション ==========

def simulate_cce(
    detector_type: str = "yoko",
    n_events: int = 1000,
    mode: str = "ramo_ideal",
    mu_e: float = 100.0,
    tau_e: float = 1e-8,
    num_threads: Optional[int] = None,
    seed: Optional[int] = None,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    force_recalc_weighting: bool = False,
    field_path: Optional[str] = None,
    srim_path: Optional[str] = None,
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
    field_path : str | None
        電界npzファイルパス（Noneの場合はdetector_typeから自動決定）
    srim_path : str | None
        SRIM IONIZファイルパス（Noneの場合はデフォルト使用）

    Returns
    -------
    dict
        'cce_list': CCEのリスト
        'mean': 平均CCE
        'std': 標準偏差
        'min': 最小CCE
        'max': 最大CCE
        'n_events': イベント数
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
        srim_path = r"C:\Users\discu\デスクトップ\python\cce\5486keVαinSiCIONIZ.txt"
    print("="*70)
    print("Shockley-Ramo CCE Simulation")
    print("="*70)

    # 乱数生成器
    rng = np.random.default_rng(seed)

    # 1. 重み電位取得（キャッシュあり）
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

    if mode == "ramo_ideal":
        # 理想モード: CCE=1.0（テスト用）
        print(f"\n  Mode: Ideal (CCE=1.0, no recombination)")
        for i in range(n_events):
            cce = compute_cce_ramo_ideal(z_seg, dE_seg)
            cce_list.append(cce)

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

        z_surface = Z[-1]  # 入射面（表面）

        for i in range(n_events):
            # ランダムな (x, y) 位置をサンプリング
            x_event = rng.uniform(X[0], X[-1])
            y_event = rng.uniform(Y[0], Y[-1])

            cce = compute_cce_ramo_drift(
                phi_w, X, Y, Z, Ex, Ey, Ez,
                z_seg, dE_seg,
                x_event, y_event, z_surface,
                mu_e, tau_e
            )
            cce_list.append(cce)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Event {i+1}/{n_events}: CCE = {cce:.4f}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'ramo_ideal' or 'ramo_drift'.")

    cce_array = np.array(cce_list)
    mean_cce = cce_array.mean()
    std_cce = cce_array.std()

    print(f"\n{'='*70}")
    print("Results")
    print('='*70)
    print(f"  Events: {n_events}")
    print(f"  Mean CCE: {mean_cce:.4f} ± {std_cce:.4f}")
    print(f"  Min CCE: {cce_array.min():.4f}")
    print(f"  Max CCE: {cce_array.max():.4f}")

    return {
        'cce_list': cce_list,
        'mean': mean_cce,
        'std': std_cce,
        'min': cce_array.min(),
        'max': cce_array.max(),
        'n_events': n_events,
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
        num_threads=args.num_threads,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        omega=args.omega,
        force_recalc_weighting=args.force_recalc_weighting,
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

        # タブ3: Diagnostics（新規）
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

        self.hist_button = ttk.Button(
            button_frame,
            text="Show Histogram",
            command=self.show_histogram,
            state="disabled"  # 最初は無効
        )
        self.hist_button.pack(side=tk.LEFT, padx=5)

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

            threads_str = self.threads_var.get().strip()
            num_threads = int(threads_str) if threads_str else None

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            return

        # ログクリア
        self.log_text.delete(1.0, tk.END)
        self.log("Starting simulation...")
        self.log(f"  Detector: {detector_type}")
        self.log(f"  Mode: {mode}")
        self.log(f"  Events: {n_events}")
        self.log(f"  μ_e: {mu_e} cm²/Vs")
        self.log(f"  τ_e: {tau_e} s")
        self.log(f"  Threads: {num_threads if num_threads else 'auto'}")
        self.log("")

        # 結果をクリア
        self.result_label.config(text="Running...", foreground="blue")
        self.run_button.config(state="disabled")
        self.running = True

        # バックグラウンドで実行（GUIフリーズを防ぐ）
        thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(detector_type, mode, n_events, mu_e, tau_e, num_threads),
            daemon=True
        )
        thread.start()

    def _run_simulation_thread(
        self,
        detector_type: str,
        mode: str,
        n_events: int,
        mu_e: float,
        tau_e: float,
        num_threads: Optional[int]
    ):
        """バックグラウンドスレッドでシミュレーション実行"""
        try:
            # stdout をキャプチャするため、簡易版として直接 log() に出力
            # （実際の stdout リダイレクトは複雑なので、ここでは省略）

            # シミュレーション実行
            results = simulate_cce(
                detector_type=detector_type,
                n_events=n_events,
                mode=mode,
                mu_e=mu_e,
                tau_e=tau_e,
                num_threads=num_threads,
                seed=None,
            )

            # 結果表示
            mean_cce = results['mean']
            std_cce = results['std']
            min_cce = results['min']
            max_cce = results['max']

            self.log("\n" + "="*70)
            self.log("SIMULATION COMPLETED")
            self.log("="*70)
            self.log(f"  Events: {n_events}")
            self.log(f"  Mean CCE: {mean_cce:.4f} ± {std_cce:.4f}")
            self.log(f"  Min CCE: {min_cce:.4f}")
            self.log(f"  Max CCE: {max_cce:.4f}")

            # 結果サマリー更新
            result_text = (
                f"Events: {n_events}  |  "
                f"Mean: {mean_cce:.4f} ± {std_cce:.4f}  |  "
                f"Min: {min_cce:.4f}  |  "
                f"Max: {max_cce:.4f}"
            )
            self.result_label.config(text=result_text, foreground="green")

            # 結果を保存してヒストグラムボタンを有効化
            self.last_results = results
            self.hist_button.config(state="normal")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log("\n" + "="*70)
            self.log("ERROR")
            self.log("="*70)
            self.log(error_msg)
            self.result_label.config(text="Error occurred (see log)", foreground="red")

        finally:
            # ボタンを再度有効化
            self.run_button.config(state="normal")
            self.running = False

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

        # Load button
        ttk.Button(control_frame, text="Load Data", command=self.load_weighting_data).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Z slice slider
        ttk.Label(control_frame, text="Z Slice:").grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1

        self.z_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=10,
            orient=tk.HORIZONTAL,
            command=self.update_weighting_plot
        )
        self.z_slider.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1

        self.z_label = ttk.Label(control_frame, text="z = ??? μm")
        self.z_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
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
        """ウェイティングデータを読み込み"""
        detector_type = self.weight_detector_var.get()

        try:
            # パスを決定
            base_dir = r"C:\Users\discu\デスクトップ\python\cce\電界"
            if detector_type == "yoko":
                self.field_path = os.path.join(base_dir, "yokogata_field.npz")
                self.weight_path = os.path.join(base_dir, "yokogata_weighting.npz")
            elif detector_type == "kushi":
                self.field_path = os.path.join(base_dir, "kushigata_field.npz")
                self.weight_path = os.path.join(base_dir, "kushigata_weighting.npz")

            # データ読み込み
            field_data = np.load(self.field_path)
            weight_data_raw = np.load(self.weight_path)

            # ウェイティング電界を計算（中央差分）
            phi_w = weight_data_raw['phi_w']
            X = weight_data_raw['X']
            Y = weight_data_raw['Y']
            Z = weight_data_raw['Z']

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

            # Z sliderを更新
            nz = len(Z)
            self.z_slider.configure(from_=0, to=nz-1)
            self.z_slider.set(nz-1)  # デフォルトは表面

            self.weight_status.config(text=f"Loaded: {detector_type}, Grid: {len(X)}x{len(Y)}x{len(Z)}", foreground="green")
            self.update_weighting_plot()

        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File not found: {e}")
            self.weight_status.config(text="Load failed", foreground="red")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.weight_status.config(text="Load failed", foreground="red")

    def update_weighting_plot(self, *args):
        """ウェイティングプロットを更新"""
        if self.weight_data is None:
            return

        try:
            iz = int(self.z_slider.get())
            display_type = self.display_var.get()

            phi_w = self.weight_data['phi_w']
            X = self.weight_data['X']
            Y = self.weight_data['Y']
            Z = self.weight_data['Z']

            # Z label更新
            self.z_label.config(text=f"z = {Z[iz]*1e6:.2f} μm")

            # データ取得
            if display_type == "phi_w":
                data = phi_w[iz, :, :]
                title = f"Weighting Potential φ_w at z={Z[iz]*1e6:.2f} μm"
                cmap = 'viridis'
            elif display_type == "|E_w|":
                E_wx = self.weight_data['E_wx']
                E_wy = self.weight_data['E_wy']
                E_wz = self.weight_data['E_wz']
                data = np.sqrt(E_wx[iz, :, :]**2 + E_wy[iz, :, :]**2 + E_wz[iz, :, :]**2)
                title = f"|E_w| at z={Z[iz]*1e6:.2f} μm"
                cmap = 'hot'

            # プロット
            self.weight_ax.clear()
            im = self.weight_ax.imshow(
                data,
                extent=[X[0]*1e6, X[-1]*1e6, Y[0]*1e6, Y[-1]*1e6],
                origin='lower',
                cmap=cmap,
                aspect='auto'
            )
            self.weight_ax.set_xlabel('x [μm]')
            self.weight_ax.set_ylabel('y [μm]')
            self.weight_ax.set_title(title)

            # Colorbar
            if hasattr(self, 'weight_colorbar'):
                self.weight_colorbar.remove()
            self.weight_colorbar = self.weight_fig.colorbar(im, ax=self.weight_ax)

            self.weight_canvas.draw()

        except Exception as e:
            print(f"Plot update error: {e}")

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
