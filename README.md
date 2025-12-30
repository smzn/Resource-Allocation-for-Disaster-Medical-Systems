# Resource-Allocation-Optimization-for-Disaster-Medical-Systems-Based-on-Open-BCMP-Queueing-Networks

## English
This project provides code for optimizing resource allocation and operations in disaster medical systems using Open-BCMP (Balanced Complete Multi-Class) Queueing Networks.

During disasters, it constructs Open-BCMP Queueing Networks to determine optimal allocation strategies for medical resources (such as medical units, and equipment). The goal is to reduce service waiting times and maximize resource utilization efficiency.

### Directory Structure
The project includes the following Python scripts and shell scripts:
```
.
├── 01_virtual_medical_area_v7_1.py            # Virtual area generation
├── 02_OpenBCMP_v6_2.py                        # Theoretical calculations based on Open-BCMP
├── 03_visualize_congestion_v4_4.py            # Visualization of computation results
├── 04_visualize_flows_multiscale_v6_1.py      # Visualization with geographical information
├── 05_VirtualOpenBCMPSimulation_v4_1.py       # Open-BCMP simulation
├── 06_OpenBCMPOptimization_v7_11.py           # Optimization algorithm: proposed model in this study
├── 07_AfterOptimization_v7_3.py               # Post-optimization analysis (corresponding to visualizations in 03 and 04)
├── *.sh                                       # Shell scripts for execution
├── bcmp_sa_optimizer_v7_13.py                 # Computational engine for class integration
└── README.md                                  # This file
```

### How to Run
You can also execute the corresponding .sh scripts to run the processes in batch.

#### 1. Prepare the Python Environment
```
git clone https://github.com/smzn/Resource-Allocation-Optimization-for-Disaster-Medical-Systems-Based-on-Open-BCMP-Queueing-Networks.git
cd Resource-Allocation-Optimization-for-Disaster-Medical-Systems-Based-on-Open-BCMP-Queueing-Networks
```

#### 2. Create a Virtual Environment (Virtual Area)
```
python3 01_virtual_medical_area_v7_1.py
```

#### 3. Simulation & Optimization
**Example: running the BCMP model and visualizing results (theoretical value calculation and visualization in a virtual region):**
```
python3 02_OpenBCMP_v6_2.py
python3 03_visualize_congestion_v4_4.py
python3 04_visualize_flows_multiscale_v6_1.py
```

**Optimization and evaluation (optimization after theoretical calculation and visualization):**
```
python3 06_OpenBCMPOptimization_v7_11.py
python3 07_AfterOptimization_v7_3.py
```

### License
This project is licensed under the MIT License.



---
## Japanese
このプロジェクトは、Open-BCMP（Balanced Complete Multi-Class）Queueing Networksを用いて、災害医療システムにおけるリソース配分と運用最適化をするためのコードです。
災害時において医療リソース（診療ユニット、設備など）の最適な配分戦略を求めるために、Open-BCMP Queueing Networksを構築します。これにより、サービス待ち時間の削減や資源利用効率の最大化を目指します。

### ディレクトリ構成
以下のような Python スクリプトやシェルスクリプトが含まれています：
```
.
├── 01_virtual_medical_area_v7_1.py            # 仮想エリア生成
├── 02_OpenBCMP_v6_2.py                        # Open-BCMP理論値算出
├── 03_visualize_congestion_v4_4.py            # 計算結果可視化
├── 04_visualize_flows_multiscale_v6_1.py      # 地理的情報を含めた計算結果可視化
├── 05_VirtualOpenBCMPSimulation_v4_1.py       # Open-BCMPシミュレーション
├── 06_OpenBCMPOptimization_v7_11.py           # 最適化アルゴリズム：本研究のモデル
├── 07_AfterOptimization_v7_3.py               # 最適化後の分析(03,04の可視化に相当）
├── *.sh                                       # 実行用シェルスクリプト
├── bcmp_sa_optimizer_v7_13.py                 # クラス取り込み用計算エンジン
└── README.md                                  # 本ファイル
```

### 実行方法
対応する .sh スクリプトを実行することでも、一括実行できます。
#### 1.Python 環境の準備
```
git clone https://github.com/smzn/Resource-Allocation-Optimization-for-Disaster-Medical-Systems-Based-on-Open-BCMP-Queueing-Networks.git
cd Resource-Allocation-Optimization-for-Disaster-Medical-Systems-Based-on-Open-BCMP-Queueing-Networks
```


#### 2.仮想環境の作成
```
python3 01_virtual_medical_area_v7_1.py
```

#### 3.シミュレーション & 最適化
**BCMPモデルを実行し可視化する例(仮想地域での理論値算出及び可視化)：**
```
python3 02_OpenBCMP_v6_2.py
python3 03_visualize_congestion_v4_4.py
python3 04_visualize_flows_multiscale_v6_1.py
```
**最適化と評価(理論値算出後の最適化及び可視化)：**
```
python3 06_OpenBCMPOptimization_v7_11.py
python3 07_AfterOptimization_v7_3.py
```


### License
このライセンスは、MITです。




