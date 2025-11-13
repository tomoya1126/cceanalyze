#!/usr/bin/env python3
"""
データファイルの検証スクリプト

各データファイルが正しく読み込めるか、内容が妥当かをチェックします。
"""

import os
import sys
import numpy as np
import pandas as pd

# NumPy互換性: trapezoidがない場合はtrapzを使用
if hasattr(np, 'trapezoid'):
    trapz_integrate = np.trapezoid
else:
    trapz_integrate = np.trapz


def check_field_file(field_file):
    """電界ファイルのチェック"""
    print(f"\n{'='*60}")
    print(f"Checking field file: {field_file}")
    print('='*60)

    if not os.path.exists(field_file):
        print(f"❌ ERROR: File not found!")
        print(f"   Please run 'python build_and_validate_fields.py' first")
        print(f"   or place your field data at: {field_file}")
        return False

    try:
        data = np.load(field_file)
        print(f"✓ File loaded successfully")

        # 必要なキーをチェック
        required_keys = ['X', 'Y', 'Z', 'V', 'Ex', 'Ey', 'Ez']
        missing_keys = [k for k in required_keys if k not in data.files]

        if missing_keys:
            print(f"❌ ERROR: Missing keys: {missing_keys}")
            print(f"   Available keys: {data.files}")
            return False

        print(f"✓ All required keys present: {required_keys}")

        # データの形状をチェック
        X = data['X']
        Y = data['Y']
        Z = data['Z']
        V = data['V']

        print(f"\n  Grid dimensions:")
        print(f"    Nx = {len(X)}")
        print(f"    Ny = {len(Y)}")
        print(f"    Nz = {len(Z)}")
        print(f"    V.shape = {V.shape}")

        expected_shape = (len(Z), len(Y), len(X))
        if V.shape != expected_shape:
            print(f"❌ ERROR: V shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {V.shape}")
            return False

        print(f"✓ Shape is correct: (Nz, Ny, Nx)")

        # 座標範囲
        print(f"\n  Coordinate ranges:")
        print(f"    X: [{X.min()*1e6:.2f}, {X.max()*1e6:.2f}] μm")
        print(f"    Y: [{Y.min()*1e6:.2f}, {Y.max()*1e6:.2f}] μm")
        print(f"    Z: [{Z.min()*1e6:.2f}, {Z.max()*1e6:.2f}] μm")

        # 電位範囲
        print(f"\n  Potential range:")
        print(f"    V: [{V.min():.2f}, {V.max():.2f}] V")

        if V.max() < 50:
            print(f"⚠ WARNING: Maximum potential seems low for 100V bias")

        # 電界統計
        Ex = data['Ex']
        Ey = data['Ey']
        Ez = data['Ez']
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

        print(f"\n  Electric field statistics:")
        print(f"    |E| max:  {E_mag.max():.3e} V/m")
        print(f"    |E| mean: {E_mag.mean():.3e} V/m")

        print(f"\n✅ Field file is valid!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_srim_file(srim_file):
    """SRIMファイルのチェック"""
    print(f"\n{'='*60}")
    print(f"Checking SRIM file: {srim_file}")
    print('='*60)

    if not os.path.exists(srim_file):
        print(f"❌ ERROR: File not found!")
        print(f"   Please place your SRIM data at: {srim_file}")
        print(f"   Format: depth(Angstrom), ionization(eV/Angstrom)")
        return False

    try:
        # データ読み込み（コメント行をスキップ）
        data_rows = []
        with open(srim_file, 'r', encoding='utf-8') as f:
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
            print(f"❌ ERROR: No valid data found in file")
            return False

        data = np.array(data_rows)
        depth_angstrom = data[:, 0]
        ionization = data[:, 1]

        print(f"✓ File loaded successfully")
        print(f"  Number of data points: {len(data)}")

        # 深さ範囲
        depth_um = depth_angstrom * 1e-4  # Angstrom -> μm
        print(f"\n  Depth range:")
        print(f"    [{depth_um.min():.2f}, {depth_um.max():.2f}] μm")

        if depth_um.max() < 15:
            print(f"⚠ WARNING: Maximum depth seems short for 5.486 MeV alpha")
            print(f"   Expected ~18-19 μm")

        # ブラッグピーク
        bragg_idx = np.argmax(ionization)
        bragg_depth = depth_um[bragg_idx]
        print(f"\n  Bragg peak:")
        print(f"    Depth: {bragg_depth:.2f} μm")
        print(f"    Ionization: {ionization[bragg_idx]:.2e} eV/Angstrom")

        if not (16 < bragg_depth < 19):
            print(f"⚠ WARNING: Bragg peak position unusual for 5.486 MeV alpha")
            print(f"   Expected ~17-18 μm")

        # 総エネルギー
        depth_m = depth_angstrom * 1e-10
        ionization_per_m = ionization * 1e10
        total_energy = trapz_integrate(ionization_per_m, depth_m)
        print(f"\n  Total ionization energy:")
        print(f"    {total_energy/1e6:.3f} MeV")

        if not (4.5 < total_energy/1e6 < 6.0):
            print(f"⚠ WARNING: Total energy unusual for 5.486 MeV alpha")
            print(f"   Expected ~5.0-5.5 MeV")

        # 電子正孔対数
        E_PAIR = 7.8  # eV
        n_pairs = total_energy / E_PAIR
        print(f"\n  Expected e-h pairs:")
        print(f"    {n_pairs:.3e}")

        print(f"\n✅ SRIM file is valid!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_exp_file(exp_file):
    """実験データファイルのチェック（横型とくし形の両方に対応）"""
    print(f"Checking experimental data file: {exp_file}")

    if not os.path.exists(exp_file):
        print(f"⚠ WARNING: File not found!")
        print(f"   Experimental data comparison will be skipped")
        print(f"   If you have experimental data, place it at: {exp_file}")
        return False

    try:
        # ファイルフォーマットの自動判定
        # くし形: 22行ヘッダー + ヒストグラムデータ（ビンセンタ、カウント数）
        # 横型: タブ区切りのイベントデータ（WaveformIndex, ..., PeakHeight）

        # まず最初の数行を読んで判定
        with open(exp_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline() for _ in range(25)]

        # くし形の判定: 22行以上のヘッダーがあり、数値データが続く
        is_kushigata = False
        if len(first_lines) > 22:
            # 23行目（index=22）が数値データかチェック
            try:
                parts = first_lines[22].split()
                if len(parts) >= 2:
                    float(parts[0])
                    float(parts[1])
                    is_kushigata = True
            except:
                pass

        if is_kushigata:
            # くし形: ヒストグラムデータ
            print(f"✓ Detected Kushigata format (histogram data)")
            df = pd.read_csv(
                exp_file,
                skiprows=22,
                sep=r'\s+',  # 空白区切り
                header=None,
                names=['BinCenter', 'Counts']
            )
            print(f"  Number of bins: {len(df)}")

            bin_center = df['BinCenter'].values
            counts = df['Counts'].values

            print(f"\n  Histogram range:")
            print(f"    BinCenter: [{bin_center.min():.3f}, {bin_center.max():.3f}]")
            print(f"    Total counts: {counts.sum():.0f}")
            print(f"    Mean counts per bin: {counts.mean():.1f}")

        else:
            # 横型: イベントデータ（タブまたはカンマ区切り）
            print(f"✓ Detected Yokogata format (event data)")
            try:
                df = pd.read_csv(exp_file, sep='\t')
                if len(df.columns) == 1:
                    df = pd.read_csv(exp_file, sep=',')
            except:
                df = pd.read_csv(exp_file, sep=',')

            print(f"  Number of events: {len(df)}")

            # カラムをチェック
            print(f"\n  Available columns:")
            for col in df.columns:
                print(f"    - {col}")

            if 'PeakHeight' not in df.columns:
                print(f"⚠ WARNING: 'PeakHeight' column not found")
                print(f"   Experimental comparison may not work correctly")
            else:
                peak_height = df['PeakHeight'].values
                print(f"\n  PeakHeight statistics:")
                print(f"    Mean: {peak_height.mean():.3f}")
                print(f"    Std:  {peak_height.std():.3f}")
                print(f"    Min:  {peak_height.min():.3f}")
                print(f"    Max:  {peak_height.max():.3f}")

        print(f"\n✅ Experimental data file is valid!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("Data Files Validation Script")
    print("="*60)

    # ファイルパス（cce_simulation.pyと同じ）
    field_files = [
        ('横型', '電界/yokogata_field.npz'),
        ('くし形', '電界/kushigata_field.npz')
    ]
    srim_file = 'data/5486keVαinSiCIONIZ.txt'

    # チェック実行
    results = {}

    # 電界ファイル
    for name, field_file in field_files:
        print(f"\n{'='*60}")
        print(f"Field file for {name}")
        print('='*60)
        results[f'field_{name}'] = check_field_file(field_file)

    # SRIM
    results['srim'] = check_srim_file(srim_file)

    # 結果サマリー
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    status_icon = lambda x: "✅" if x else "❌"
    print("\nField data:")
    for name, field_file in field_files:
        print(f"  {status_icon(results[f'field_{name}'])} {name}: {field_file}")

    print(f"\nSRIM data:")
    print(f"  {status_icon(results['srim'])} {srim_file}")

    # 必須ファイル（電界とSRIM）がOKかチェック
    field_ok = any(results[f'field_{name}'] for name, _ in field_files)
    if field_ok and results['srim']:
        print("\n✅ Required data files are ready!")
        print("   You can now run: python cce_simulation.py")
        return 0
    else:
        print("\n❌ Some required data files are missing or invalid")
        print("   Please fix the issues above before running the simulation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
