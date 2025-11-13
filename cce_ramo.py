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


# ========== メインシミュレーション ==========

def simulate_cce(
    field_path: str = r"C:\Users\discu\デスクトップ\python\cce\電界\yokogata_field.npz",
    srim_path: str = r"C:\Users\discu\デスクトップ\python\cce\5486keVαinSiCIONIZ.txt",
    n_events: int = 1000,
    seed: Optional[int] = None,
    max_iter: int = 20000,
    tol: float = 1e-5,
    omega: float = 1.8,
    num_threads: Optional[int] = None,
    force_recalc_weighting: bool = False,
) -> dict:
    """
    n_eventsイベントをシミュレーションしてCCE統計を返す。

    Parameters
    ----------
    field_path : str
        電界npzファイルパス
    srim_path : str
        SRIM IONIZファイルパス
    n_events : int
        シミュレーションするイベント数
    seed : int | None
        乱数シード
    max_iter : int
        重み電位計算の最大反復回数
    tol : float
        重み電位計算の収束判定閾値
    omega : float
        SOR緩和パラメータ (1.0 < omega < 2.0)
    num_threads : int | None
        Numbaスレッド数
    force_recalc_weighting : bool
        Trueの場合、キャッシュを無視して重み電位を再計算

    Returns
    -------
    dict
        'cce_list': CCEのリスト
        'mean': 平均CCE
        'std': 標準偏差
    """
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
    print(f"\nSimulating {n_events} events...")
    cce_list = []

    for i in range(n_events):
        cce = compute_cce_for_one_event(phi_w, X, Y, Z, z_seg, dE_seg, rng)
        cce_list.append(cce)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Event {i+1}/{n_events}: CCE = {cce:.4f}")

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

def main():
    parser = argparse.ArgumentParser(
        description="Shockley-Ramo CCE simulation for SiC detector"
    )
    parser.add_argument(
        "--field",
        type=str,
        default=r"C:\Users\discu\デスクトップ\python\cce\電界\yokogata_field.npz",
        help="Path to field npz file"
    )
    parser.add_argument(
        "--srim",
        type=str,
        default=r"C:\Users\discu\デスクトップ\python\cce\5486keVαinSiCIONIZ.txt",
        help="Path to SRIM IONIZ file"
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=1000,
        help="Number of events to simulate"
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
        "--num-threads",
        type=int,
        default=None,
        help="Number of Numba threads (default: auto)"
    )
    parser.add_argument(
        "--force-recalc-weighting",
        action="store_true",
        help="Force recalculation of weighting potential (ignore cache)"
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
        field_path=args.field,
        srim_path=args.srim,
        n_events=args.n_events,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        omega=args.omega,
        num_threads=args.num_threads,
        force_recalc_weighting=args.force_recalc_weighting,
    )

    # ヒストグラム描画
    plot_cce_histogram(results['cce_list'], args.output)

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
