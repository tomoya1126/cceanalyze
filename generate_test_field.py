#!/usr/bin/env python3
"""
テスト用の簡単な電界データ生成スクリプト

実際のOpenSTFデータがない場合、このスクリプトで簡易的な電界データを生成できます。
平行平板コンデンサのような単純な構造を仮定します。
"""

import numpy as np
import os


def generate_simple_field(bias_voltage=100.0,
                         detector_thickness=500e-6,  # 500 μm
                         nx=50, ny=50, nz=100):
    """
    簡単な電界データを生成

    Parameters:
    -----------
    bias_voltage : float
        バイアス電圧 [V]
    detector_thickness : float
        検出器の厚さ [m]
    nx, ny, nz : int
        グリッドポイント数
    """
    print(f"Generating test field data...")
    print(f"  Bias voltage: {bias_voltage} V")
    print(f"  Detector thickness: {detector_thickness*1e6} μm")
    print(f"  Grid points: {nx} x {ny} x {nz}")

    # 座標軸の生成
    # 横型構造を仮定: x, y方向に電極が配置され、z方向が深さ
    x_max = 1000e-6  # 1 mm
    y_max = 1000e-6  # 1 mm
    z_max = detector_thickness

    # 非一様グリッドを作成（表面と底面で細かく）
    # Z方向（深さ）
    z_fine_top = np.linspace(0, 50e-6, 20)  # 表面50μm
    z_coarse = np.linspace(50e-6, detector_thickness-50e-6, nz-40)
    z_fine_bottom = np.linspace(detector_thickness-50e-6, detector_thickness, 20)
    Z = np.concatenate([z_fine_top, z_coarse[1:-1], z_fine_bottom])
    nz = len(Z)

    # X, Y方向（均一グリッド）
    X = np.linspace(0, x_max, nx)
    Y = np.linspace(0, y_max, ny)

    print(f"  Actual grid points: {nx} x {ny} x {nz}")

    # メッシュグリッド
    ZZ, YY, XX = np.meshgrid(Z, Y, X, indexing='ij')

    # 電位の計算
    # 簡単な横型構造を仮定：
    # - 左側 (x < x_max/3): 0V電極
    # - 右側 (x > 2*x_max/3): 100V電極
    # - 中央部: 線形に変化

    V = np.zeros_like(XX)

    # x座標に基づいて電位を設定
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = XX[k, j, i]

                if x < x_max / 3:
                    # 0V電極領域
                    V[k, j, i] = 0.0
                elif x > 2 * x_max / 3:
                    # 100V電極領域
                    V[k, j, i] = bias_voltage
                else:
                    # 中央部：線形補間
                    x_rel = (x - x_max/3) / (x_max/3)
                    V[k, j, i] = bias_voltage * x_rel

                    # 深さによる電位変化を少し追加（表面効果）
                    z = ZZ[k, j, i]
                    surface_factor = np.exp(-z / (50e-6))  # 50μm decay length
                    V[k, j, i] *= (1.0 - 0.1 * surface_factor)

    # 電界の計算: E = -∇V
    # 非一様グリッドなので中央差分を使用
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)
    Ez = np.zeros_like(V)

    # Ex = -dV/dx
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i == 0:
                    Ex[k, j, i] = -(V[k, j, i+1] - V[k, j, i]) / (X[i+1] - X[i])
                elif i == nx - 1:
                    Ex[k, j, i] = -(V[k, j, i] - V[k, j, i-1]) / (X[i] - X[i-1])
                else:
                    dx_forward = X[i+1] - X[i]
                    dx_backward = X[i] - X[i-1]
                    Ex[k, j, i] = -(dx_backward * (V[k, j, i+1] - V[k, j, i]) +
                                   dx_forward * (V[k, j, i] - V[k, j, i-1])) / \
                                  (dx_forward * dx_backward * (dx_forward + dx_backward))

    # Ey = -dV/dy
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if j == 0:
                    Ey[k, j, i] = -(V[k, j+1, i] - V[k, j, i]) / (Y[j+1] - Y[j])
                elif j == ny - 1:
                    Ey[k, j, i] = -(V[k, j, i] - V[k, j-1, i]) / (Y[j] - Y[j-1])
                else:
                    dy_forward = Y[j+1] - Y[j]
                    dy_backward = Y[j] - Y[j-1]
                    Ey[k, j, i] = -(dy_backward * (V[k, j+1, i] - V[k, j, i]) +
                                   dy_forward * (V[k, j, i] - V[k, j-1, i])) / \
                                  (dy_forward * dy_backward * (dy_forward + dy_backward))

    # Ez = -dV/dz
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if k == 0:
                    Ez[k, j, i] = -(V[k+1, j, i] - V[k, j, i]) / (Z[k+1] - Z[k])
                elif k == nz - 1:
                    Ez[k, j, i] = -(V[k, j, i] - V[k-1, j, i]) / (Z[k] - Z[k-1])
                else:
                    dz_forward = Z[k+1] - Z[k]
                    dz_backward = Z[k] - Z[k-1]
                    Ez[k, j, i] = -(dz_backward * (V[k+1, j, i] - V[k, j, i]) +
                                   dz_forward * (V[k, j, i] - V[k-1, j, i])) / \
                                  (dz_forward * dz_backward * (dz_forward + dz_backward))

    # 統計情報
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    print(f"\n  Potential range: [{V.min():.2f}, {V.max():.2f}] V")
    print(f"  |E| max:  {E_mag.max():.3e} V/m")
    print(f"  |E| mean: {E_mag.mean():.3e} V/m")

    return X, Y, Z, V, Ex, Ey, Ez


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("Test Field Data Generator")
    print("="*60)

    # 出力ディレクトリ
    out_dir = '電界'
    os.makedirs(out_dir, exist_ok=True)

    # 電界データ生成
    X, Y, Z, V, Ex, Ey, Ez = generate_simple_field(
        bias_voltage=100.0,
        detector_thickness=500e-6,  # 500 μm
        nx=50,
        ny=50,
        nz=100
    )

    # 保存
    output_file = os.path.join(out_dir, 'yokogata_field.npz')
    np.savez_compressed(
        output_file,
        X=X, Y=Y, Z=Z,
        V=V, Ex=Ex, Ey=Ey, Ez=Ez
    )

    print(f"\n✅ Test field data saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("\nYou can now run:")
    print("  python test_data_files.py  # to verify the data")
    print("  python cce_simulation.py   # to run the simulation")
    print("\nNote: This is a simplified test field.")
    print("      For production use, replace with actual OpenSTF data.")
    print()


if __name__ == '__main__':
    main()
