# A Novel Disaster Risk Assessment Framework Based on Markov Chain Analysis of Evacuation Network Structure and Geometric Coverage Optimization


This project is a Python-based disaster risk assessment simulation framework that generates the distribution of evacuees and shelters on virtual or real-world terrains, and analyzes and visualizes evacuation network structures using a Markov chain model and geometric coverage algorithm.

- Theoretical model based on virtual terrain
- Realistic model using actual data


## 1. Theoretical Model

### Execution File
logical_model/DisasterRiskAssessment_v5.py

### Features
- Generation of terrain, shelters, and evacuees (random generation, reproducibility via seed)
- 3D visualization and map output of the terrain and nodes
- CSV output of analysis results
- Elevation-based model evaluation

### Output Files
- terrain_3d_map.png: 3D visualization of terrain
- evacuation_map.png: Contour map with shelter placement
- locations.csv: Node information (evacuees/shelters + popularity)
- lowland_fraction_matrix_rm*.csv: Lowland passage rate matrix (by elevation threshold)
- transition_matrix_rm*.csv: Transition matrix based on gravity model
- equivalence_assignment_rm*.csv: Node classification by equivalence class
- equivalence_summary_rm*.csv: Statistics on centrality and stationary distribution
- disk_cover_result_rm*.csv: Coverage circles for nodes outside main equivalence classes
- equivalence_with_disks_ristricted*.png: Map with contour, equivalence classes, and disks
- evaluation_over_elevation_std_stationary.csv: Integrated results by elevation
- metrics_vs_elevation_std_stationary.png: Evaluation metrics by elevation
- cumulative_score_vs_elevation_std_stationary.png: Cumulative score variation


## 2. Real Data Model 

### 2.1 Generation of Evacuee and Shelter Data

#### Preparation
1. Create a municipality-named folder inside the `data/` directory and download 10m mesh elevation XML from GSI.
2. Download shelter information file `mergeFromCity_1.csv` from NSDI.

#### Command
python code/11_CreatePoeple.py Mori-machi Shizuoka

#### Output
Evacuee and shelter metadata will be saved in the `results/` directory.


### 2.2 Elevation-Based Network Analysis and Risk Evaluation

#### Execution
Parallel processing using MPI is supported:

- Batch processing (e.g., elev=0 to 5)  
  mpiexec -n 6 python 12_GeoCal_MPI.py Mori-machi Shizuoka 0

- Processing from a specific elevation (e.g., elev ≥ 15)  
  mpiexec -n 6 python 12_GeoCal_MPI.py Mori-machi Shizuoka 15

#### Example Output Files (for elev=10)
- ../results/Mori-machi_load≧10melev.graphml: Filtered network
- ../results/Mori-machi_load≧10melev.png: Visualized map
- ../results/Mori-machi_geodesic_matrix_load≧10melev.csv: Distance matrix
- ../results/Mori-machi_transition_matrix_load≧10melev.csv: Transition probabilities
- ../results/Mori-machi_equivalence_classes_load≧10melev.csv: Equivalence classes
- ../results/Mori-machi_node_load≧10melev.csv: Node + centrality + equivalence class
- ../results/Mori-machi_solve_disk_cover_load≧10melev.csv: Coverage circle list

#### License
Gurobi license must be obtained individually.


## 3. Evaluation Script (Multi-City Support)

This script scores the evacuation network model (equivalence classes, stationary distribution, coverage) for multiple municipalities (e.g., within Shizuoka Prefecture), performs quantitative evaluation per elevation threshold, and generates visualizations.

### Command
python 13_ObjCal.py


---

# マルコフ連鎖による拠点構造解析と幾何的被覆に基づく災害リスク指標の提案

本プロジェクトは、仮想または実世界の地形上において、避難者と避難所の分布を生成し、標高や災害リスクを考慮した避難ネットワークの構造をマルコフ連鎖モデルと幾何的被覆アルゴリズムで解析・可視化するPythonベースの災害リスク評価シミュレーションフレームワークです。

- 仮想地形に基づく理論モデル
- 実データを用いた現実モデル


## 1. 理論モデル

### 実行ファイル
logical_model/DisasterRiskAssessment_v5.py

### 機能概要
- 地形・避難所・避難者の生成（ランダム生成。再現性確保のためseed指定可能）
- 地形・拠点の3D表示と平面地図出力
- 各種解析結果のCSV出力
- 標高別にモデルを評価

### 出力ファイル一覧
- terrain_3d_map.png : 地形の3D可視化
- evacuation_map.png : 等高線＋避難所マップ
- locations.csv : 拠点情報（避難者/避難所 + 人気度）
- lowland_fraction_matrix_rm*.csv : 低地通過率行列（標高閾値別）
- transition_matrix_rm*.csv : 遷移行列（重力モデル）
- equivalence_assignment_rm*.csv : ノードの同値類分類
- equivalence_summary_rm*.csv : 中心性・定常分布などの統計
- disk_cover_result_rm*.csv : 主同値類外ノードへの被覆円情報
- equivalence_with_disks_ristricted*.png : 同値類・被覆円の地図
- evaluation_over_elevation_std_stationary.csv : 標高ごとの統合結果
- metrics_vs_elevation_std_stationary.png : 評価指標の標高依存性グラフ
- cumulative_score_vs_elevation_std_stationary.png : スコアの累積変化


## 2. 実データモデル

### 2.1 避難者・避難所データの生成

#### 事前準備
1. data/ フォルダに自治体名フォルダを作成し、国土地理院より10mメッシュ標高XMLをダウンロードし配置  
2. 国土数値情報から mergeFromCity_1.csv をダウンロード（避難所情報）

#### 実行コマンド
python code/11_CreatePoeple.py 静岡県森町

#### 出力
避難者・避難所メタ情報が results/ に保存されます。


### 2.2 標高別ネットワーク解析・リスク評価

#### 実行方法
MPIによる並列処理が可能です。
- 全体一括処理（例: elev=0〜5）  
  mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県森町 0

- 特定標高以降を処理（例: elev≧15）  
  mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県森町 15

#### 出力ファイル例（elev=10 の場合）
- ../results/静岡県森町_load≧10melev.graphml : フィルタ済みネットワーク
- ../results/静岡県森町_load≧10melev.png : 可視化マップ
- ../results/静岡県森町_geodesic_matrix_load≧10melev.csv : 距離行列
- ../results/静岡県森町_transition_matrix_load≧10melev.csv : 遷移確率
- ../results/静岡県森町_equivalence_classes_load≧10melev.csv : 同値類
- ../results/静岡県森町_node_load≧10melev.csv : ノード + 中心性 + 同値類
- ../results/静岡県森町_solve_disk_cover_load≧10melev.csv : 被覆円一覧

#### ライセンス
Gurobiのライセンスは各自で取得してください。


## 3. 評価スクリプト（複数都市対応）

複数の市町（例：静岡県内）に対して、避難ネットワークモデル（同値類・定常分布・被覆円）のスコアリングを行い、標高ごとの定量評価を実施・グラフ化します。

### 実行コマンド
python 13_ObjCal.py
