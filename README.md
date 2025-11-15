# SiC検出器 CCE (電荷収集効率) シミュレーション

このプロジェクトは、SiC検出器におけるα線入射時の電子・正孔の軌道を追跡し、電極で収集される電荷量(CCE)を計算するシミュレーションコードです。

## 概要

- OpenSTFで計算した電界データを使用
- SRIMで計算した電子正孔対分布を使用
- Runge-Kutta 4次法でキャリア軌道を追跡
- 実験データとの比較検証機能

## 必要な環境

- Python 3.7以上
- 必要なパッケージ:
  ```bash
  pip install numpy scipy matplotlib pandas
  ```

## ディレクトリ構造

```
cceanalyze/
├── cce_simulation.py          # メインシミュレーションコード
├── build_and_validate_fields.py  # 電界データ生成スクリプト
├── data/                      # データファイル
│   ├── 5486keVαinSiCIONIZ.txt        # SRIMデータ
│   └── SiC2_500_10_clear_α_*.csv     # 実験データ
├── 電界/                      # 電界データ
│   ├── yokogata_field.npz     # 横型電界データ
│   └── kushigata_field.npz    # くし形電界データ
├── くし形電界/                 # くし形電界の元データ
└── 横型電界/                   # 横型電界の元データ
```

## 使用方法

### クイックスタート（テストデータを使用）

実際のOpenSTFデータがない場合、テスト用の簡易電界データで動作確認ができます:

```bash
# 1. テスト用電界データの生成
python generate_test_field.py

# 2. データファイルの検証
python test_data_files.py

# 3. CCEシミュレーションの実行
python cce_simulation.py
```

### 実データを使用する場合

#### 1. 電界データの生成

実際のOpenSTFデータから電界データを生成します（まだ存在しない場合）:

```bash
python build_and_validate_fields.py
```

これにより `電界/` ディレクトリに `yokogata_field.npz` と `kushigata_field.npz` が生成されます。

#### 2. SRIMデータと実験データの配置

- SRIMデータ: `data/5486keVαinSiCIONIZ.txt` に配置
- 実験データ: `data/SiC2_500_10_clear_α_20250124_142116_100.0V.csv` に配置

#### 3. データファイルの検証

```bash
python test_data_files.py
```

#### 4. CCEシミュレーションの実行

```bash
python cce_simulation.py
```

## 入力データ

### 1. 電界データ (`yokogata_field.npz`)

OpenSTFで計算した電界データ。以下を含む:
- `X`, `Y`, `Z`: 座標配列 [m] (1D)
- `V`: 電位 [V] (3D配列: Nz × Ny × Nx)
- `Ex`, `Ey`, `Ez`: 電界成分 [V/m] (3D配列)

非一様格子なので、`scipy.interpolate.RegularGridInterpolator` で補間されます。

### 2. SRIMデータ (`5486keVαinSiCIONIZ.txt`)

α線 (5.486 MeV, Am-241) のSiC中での電離エネルギー損失:
- フォーマット: 深さ(Angstrom), イオン化エネルギー(eV/Angstrom)
- 深さ範囲: 0 - 18.9 μm
- ブラッグピーク: 約17.4 μm付近

### 3. 実験データ (`SiC2_500_10_clear_α_*.csv`)

100Vバイアス時の実験データ:
- `Event`: イベント番号
- `PeakHeight`: パルス波高 [V]
- その他のパラメータ

## 物理パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| α線エネルギー | 5.486 MeV | Am-241 |
| 電子正孔対生成エネルギー | 7.8 eV | SiC |
| 電子移動度 | 800 cm²/Vs | |
| 正孔移動度 | 40 cm²/Vs | |
| バイアス電圧 | 100 V | |
| 電気素量 | 1.602×10⁻¹⁹ C | |

## 実装された機能

### 1. データ読み込み (`FieldInterpolator`, `SRIMDataLoader`)
- 電界データの読み込みと補間
- SRIMデータの読み込みと処理
- 実験データの読み込み

### 2. 電極位置の特定
- 電位分布から自動的に100V電極と0V電極を特定
- 閾値: 100V電極は99V以上、0V電極は1V以下

### 3. 電子正孔対の生成
- SRIMのイオン化分布に基づいて深さごとに生成
- α線入射位置はランダムまたは指定可能
- トラック周辺に小さなランダムオフセット（10 nm std）

### 4. キャリア軌道追跡 (`CarrierTracker`)
- Runge-Kutta 4次法による軌道計算
- 運動方程式: **dr/dt** = μ(**E**) × **E**
  - 電子: 電界と逆方向（高電位へ）
  - 正孔: 電界と同方向（低電位へ）
- 適応的時間ステップ（グリッドサイズの1/10程度）
- 電極到達判定と境界チェック

### 5. CCE計算
- 100V電極に到達した電子数のカウント
- 0V電極に到達した正孔数のカウント
- 総収集電荷 = (N_e + N_h) × e
- CCE = (N_e + N_h) / (2 × 総生成数)

### 6. 可視化 (`Visualizer`)
- 3D軌道プロット（電子・正孔を色分け）
- 収集電荷のヒストグラム
- CCE分布
- 実験データとの比較プロット

## 出力ファイル

シミュレーション実行後、以下のファイルが生成されます:

1. `trajectory_plot.png`: 代表的なキャリア軌道の3Dプロット
2. `charge_histogram.png`: 収集電荷とCCEのヒストグラム
3. `experiment_comparison.png`: 実験データとの比較（実験データがある場合）
4. `cce_statistics.txt`: 統計情報のテキストファイル

## シミュレーションの流れ

### ステップ1: 単一イベントのテスト
1個のα線イベントをシミュレーションし、軌道を可視化します。

```python
simulator = CCESimulator(field_file, srim_file)
result = simulator.simulate_single_alpha(n_sample_trajectories=5)
```

### ステップ2: 複数イベントのシミュレーション
統計を取るために複数イベントを実行します。

```python
results = simulator.simulate_multiple_alphas(n_events=100)
```

### ステップ3: 実験データとの比較
実験データと比較して、シミュレーションの妥当性を検証します。

## 簡略化と仮定

このシミュレーションでは以下を無視しています（初期バージョン）:
- 拡散効果（決定論的ドリフトのみ）
- 再結合
- トラップ効果
- プリアンプ応答（収集電荷量のみ計算）

## カスタマイズ

### 物理パラメータの変更
`cce_simulation.py` の冒頭で定義されている定数を変更してください:

```python
E_PAIR = 7.8  # eV - 電子正孔対生成エネルギー
E_ALPHA = 5.486e6  # eV - α線エネルギー
MU_E = 800e-4  # m²/Vs - 電子移動度
MU_H = 40e-4  # m²/Vs - 正孔移動度
BIAS_VOLTAGE = 100.0  # V
```

### イベント数の変更
`main()` 関数内で変更:

```python
n_events = 100  # イベント数を変更
results_multiple = simulator.simulate_multiple_alphas(n_events=n_events)
```

### 実験データファイルのパス変更
`main()` 関数内で変更:

```python
field_file = '電界/yokogata_field.npz'
srim_file = 'data/5486keVαinSiCIONIZ.txt'
exp_file = 'data/SiC2_500_10_clear_α_20250124_142116_100.0V.csv'
```

## トラブルシューティング

### エラー: Field file not found
`build_and_validate_fields.py` を実行して電界データを生成してください。

### エラー: SRIM file not found
SRIMデータファイルを `data/` ディレクトリに配置してください。

### 軌道が電極に到達しない
- 電界データが正しく読み込まれているか確認
- 電極の閾値が適切か確認（`_identify_electrodes()` メソッド）
- 最大ステップ数や最大時間を増やす（`CarrierTracker` の `max_steps`, `max_time`）

### シミュレーションが遅い
- イベント数を減らす（最初は10イベント程度でテスト）
- `n_sample_trajectories=0` で軌道の保存を無効化
- 時間ステップを大きくする（ただし精度が低下）

## 今後の拡張

- [ ] 拡散効果の実装
- [ ] 再結合効果の実装
- [ ] トラップ効果の実装
- [ ] プリアンプ応答のシミュレーション
- [ ] 並列化による高速化
- [ ] GUI版の作成

## ライセンス

MIT License

## 参考文献

- SRIM: http://www.srim.org/
- OpenSTF: 有限要素法による電界計算

## 問い合わせ

問題や質問がある場合は、Issueを作成してください。
