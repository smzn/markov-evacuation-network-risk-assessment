import pandas as pd
import geopandas as gpd
import os
import numpy as np
from geopy.distance import geodesic
import osmnx as ox
import statistics
import matplotlib.pyplot as plt
import networkx as nx
import japanize_matplotlib
import sys
import json
from mpi4py import MPI
import time
import gurobipy as gp
from gurobipy import GRB
from geopy.distance import geodesic
from tqdm import tqdm
pd.set_option('future.no_silent_downcasting', False)

class CreatePeople:
    def __init__(self, municipality_name: str):
        """
        CreatePeople クラスの初期化メソッド
        Constructor for the CreatePeople class.

        :param municipality_name: 市町村名（例："小樽市"）
                                  Name of the municipality (e.g., "Otaru")
        """
        self.municipality_name = municipality_name  # 市町村名を保持 / Store municipality name
        self.gdf = None
        self.center_lat = None
        self.center_lon = None
        self.file_path = None
        self.layer_name = None
        self.elevation_array = None
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.shelters_df = None
        self.df_assigned = None
        self.G = None
        self.distance_matrix = None
        self.P = None
        self.nodeG = None
        self.disk_cover = None

        # 初期化ログを表示 / Display initialization log
        print(f"市町村名「{self.municipality_name}」で初期化しました。")
        print(f"Initialized with municipality: {self.municipality_name}")


    
    def load_state_from_results(self, base_path="../results/meta"):
        """resultsディレクトリからselfの状態を復元する"""
        try:
            # gdfを読み込み
            gdf_path = os.path.join(base_path, f"{self.municipality_name}_gdf.geojson")
            if os.path.exists(gdf_path):
                self.gdf = gpd.read_file(gdf_path)
            # elevation_arrayを読み込み
            elev_path = os.path.join(base_path, f"{self.municipality_name}_elevation.npy")
            if os.path.exists(elev_path):
                self.elevation_array = np.load(elev_path)
            # ノード情報を読み込み
            assigned_path = os.path.join(base_path, f"{self.municipality_name}_assigned_nodes.csv")
            if os.path.exists(assigned_path):
                self.df_assigned = pd.read_csv(assigned_path)
            # メタ情報を読み込み
            meta_path = os.path.join(base_path, f"{self.municipality_name}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    self.lat_min = meta.get("lat_min")
                    self.lat_max = meta.get("lat_max")
                    self.lon_min = meta.get("lon_min")
                    self.lon_max = meta.get("lon_max")
                    self.center_lat = meta.get("center_lat")
                    self.center_lon = meta.get("center_lon")
                    self.municipality_name = meta.get("municipality_name")
            print("状態をresultsフォルダから復元しました。")
        except Exception as e:
            print(f"状態の復元に失敗しました: {e}")


    def get_elevation_from_latlon(self, lat, lon):
        """
        指定された緯度経度から標高をバイリニア補間で取得
        Get interpolated elevation using bilinear interpolation.
        """
        if self.elevation_array is None:
            print("標高データが読み込まれていません。 / Elevation data not loaded.")
            return None
        nrows, ncols = self.elevation_array.shape
        if not (self.lat_min <= lat <= self.lat_max) or not (self.lon_min <= lon <= self.lon_max):
            print("指定した地点はデータの範囲外です。 / Location is outside data bounds.")
            return None
        # 緯度経度 → 浮動小数インデックス（行・列）
        lat_ratio = (self.lat_max - lat) / (self.lat_max - self.lat_min)
        lon_ratio = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        row_f = lat_ratio * (nrows - 1)
        col_f = lon_ratio * (ncols - 1)
        row0 = int(np.floor(row_f))
        row1 = min(row0 + 1, nrows - 1)
        col0 = int(np.floor(col_f))
        col1 = min(col0 + 1, ncols - 1)
        # 比率を計算
        dr = row_f - row0
        dc = col_f - col0
        # 4点補間
        val00 = self.elevation_array[row0, col0]
        val01 = self.elevation_array[row0, col1]
        val10 = self.elevation_array[row1, col0]
        val11 = self.elevation_array[row1, col1]
        # 補間結果（NaNが含まれている場合は None を返す）
        if np.isnan([val00, val01, val10, val11]).any():
            return None
        interpolated = (
            val00 * (1 - dr) * (1 - dc) +
            val01 * (1 - dr) * dc +
            val10 * dr * (1 - dc) +
            val11 * dr * dc
        )
        return interpolated




    def get_filtered_road_network(self, include_types=None, exclude_types=None, output_file="filtered_network.graphml", elev=50, n=10, nrate=0.5):
        if include_types:
            custom_filter = '["highway"~"' + "|".join(include_types) + '"]'
        elif exclude_types:
            custom_filter = '["highway"!~"' + "|".join(exclude_types) + '"]'
        else:
            custom_filter = None

        self.G = ox.graph_from_place(self.municipality_name, network_type="walk", custom_filter=custom_filter)#drive
        
        edges = list(self.G.edges(keys=True, data=True))
        total = len(edges)
        
        edges_to_remove = []
        distances = []
        for idx, (u, v, key, data) in enumerate(edges):
            if "highway" not in data:
                print(f"Edge {idx+1}: highwayなし → 削除予定")
                edges_to_remove.append((u, v, key))
        
        
        for idx, (u, v, key, data) in enumerate(edges):
            try:
                if u == v:
                    # 同一ノード間の自己ループはスキップ / Skip self-loops
                    continue
                lat1, lon1 = self.G.nodes[u]['y'], self.G.nodes[u]['x']
                lat2, lon2 = self.G.nodes[v]['y'], self.G.nodes[v]['x']
                
                #下記のコードの方が正確
                #distance_m = self._getdistance(lat1, lon1, lat2, lon2)
                #こちらは時間短縮（直線距離）バージョン
                distance_m = geodesic((lat1, lon1), (lat2, lon2)).meters
                
                distances.append(distance_m)
                #print(f"Edge {idx+1}/{total}: from {u} to {v} = {distance_m:.2f} m")
                if distance_m == 0:
                    # 距離ゼロエッジをスキップ / Skip zero-length edges
                    continue
                
                
                ####対象edgeをn分割してrate%が対象標高以下ならば削除する
                points = self._interpolate_points(lat1, lon1, lat2, lon2, distance_m, interval_m=round(distance_m/n,1))
                # 各点に標高を付与
                points_with_elev = [(lat, lon, self.get_elevation_from_latlon(lat, lon)) for lat, lon in points]                
                # 条件に合う点の数をカウント
                num_high = sum(1 for _, _, e in points_with_elev if e >= elev)
                ratio = num_high / len(points_with_elev)
                if ratio < nrate:
                    edges_to_remove.append((u, v, key))
                
            except Exception as e:
                print(f"error on edge {u}-{v}: {e}")
                continue

            if idx % max(1, total // 20) == 0:
                percent = (idx + 1) / total * 100
                print(f"{percent:.1f}% 完了（{idx + 1}/{total}）")
                
        
        # 不要なエッジを削除 / Remove unnecessary edges
        self.G.remove_edges_from(edges_to_remove)
        print(f"{len(edges_to_remove)}/{len(edges)} 件のエッジを削除しました。 / Removed {len(edges_to_remove)} out of {len(edges)} edges ({len(edges_to_remove)/len(edges):.2%})")

        # ネットワーク保存 / Save filtered graph
        ox.save_graphml(self.G, filepath=output_file)
        print(f"標高 {elev}m 超の部分のみで構成したネットワークを保存しました: {output_file} \n/ Saved elevation-filtered network to {output_file}")
        
        # 距離統計の表示 / Print edge length statistics
        print("[各エッジの距離統計 / Edge length statistics]")
        print(f"最大値 / Max: {max(distances):.2f} m")
        print(f"最小値 / Min: {min(distances):.2f} m")
        print(f"平均 / Mean: {statistics.mean(distances):.2f} m")
        print(f"中央値 / Median: {statistics.median(distances):.2f} m")
        print(f"標準偏差 / Std Dev: {statistics.stdev(distances):.2f} m")
        
    
    def _interpolate_points(self, lat1, lon1, lat2, lon2, distance, interval_m=5):
        """緯度経度を指定して5m間隔で補間点を作る"""
        start = np.array([lat1, lon1])
        end = np.array([lat2, lon2])

        # 総距離（メートル単位）
        num_points = int(distance // interval_m)

        if num_points <= 1:
            return [(lat1, lon1), (lat2, lon2)]

        lats = np.linspace(lat1, lat2, num_points + 1)
        lons = np.linspace(lon1, lon2, num_points + 1)
        return list(zip(lats, lons))
    
    def _getdistance(self, lat1, lon1, lat2, lon2):
        # 最近傍ノードを取得（X=経度, Y=緯度に注意）
        node1 = ox.distance.nearest_nodes(self.G, X=lon1, Y=lat1)
        node2 = ox.distance.nearest_nodes(self.G, X=lon2, Y=lat2)

        # 最短距離（長さ）を取得（weight="length"ならメートル単位）
        try:
            length = nx.shortest_path_length(self.G, node1, node2, weight="length")  # distance in meters #weight="weight"
            return length
        except nx.NetworkXNoPath:
            return None
        






    def plot_colored_roads(self, output_filepath):
        try:
            
            road_colors = {
                "trunk": "red",
                "primary": "blue",
                "secondary": "green",
                "tertiary": "orange",
            }

            fig, ax = plt.subplots(figsize=(12, 12))

            edge_groups = {}  # color: list of (xs, ys)
            for u, v, k, data in self.G.edges(keys=True, data=True):
                highway_type = data.get("highway")
                if isinstance(highway_type, list):
                    highway_type = highway_type[0]
                highway_type = highway_type or "other"
                color = road_colors.get(highway_type, "gray")

                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                else:
                    # fallback to straight line if no geometry
                    x1, y1 = self.G.nodes[u]["x"], self.G.nodes[u]["y"]
                    x2, y2 = self.G.nodes[v]["x"], self.G.nodes[v]["y"]
                    xs, ys = [x1, x2], [y1, y2]

                edge_groups.setdefault(color, []).append((xs, ys))

            # 描画
            for color, lines in edge_groups.items():
                for xs, ys in lines:
                    ax.plot(xs, ys, color=color, linewidth=2)

            # 凡例
            legend_labels = {
                "trunk": "Trunk",
                "primary": "Primary",
                "secondary": "Secondary",
                "tertiary": "Tertiary",
                "gray": "Other"
            }
            used_colors = set(edge_groups.keys())
            handles = [plt.Line2D([0], [0], color=c, lw=2, label=legend_labels.get(k, "Other"))
                    for k, c in road_colors.items() if c in used_colors]
            if "gray" in used_colors:
                handles.append(plt.Line2D([0], [0], color="gray", lw=2, label="Other"))

            ax.legend(handles=handles, title="Road Types", loc="upper left")
            plt.title(f"Color-coded Road Types in {self.municipality_name}", fontsize=16)
            plt.axis("off")
            plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
            print(f"地図を保存しました: {output_filepath}")

        except Exception as e:
            print(f"エラー: 道路地図の生成に失敗しました - {e}")
        

    
    def client_to_shelter_matrix(self, outputfile):
        # クライアントとシェルターの抽出
        clients = self.df_assigned.copy()
        shelters = self.df_assigned.copy()

        # ノード番号を一括取得（高速）
        print("最近傍ノードを一括取得中...")
        clients["node"] = ox.distance.nearest_nodes(self.G, X=clients["x"], Y=clients["y"], return_dist=False)
        shelters["node"] = ox.distance.nearest_nodes(self.G, X=shelters["x"], Y=shelters["y"], return_dist=False)

        # 出力距離行列（避難者→避難所）
        matrix = pd.DataFrame(index=clients["id"], columns=shelters["id"])

        print("距離計算を開始します...")
        for _, c_row in tqdm(clients.iterrows(), total=len(clients)):
            source = c_row["node"]
            try:
                lengths = nx.single_source_dijkstra_path_length(self.G, source, weight="length")
            except nx.NetworkXNoPath:
                lengths = {}

            for _, s_row in shelters.iterrows():
                target = s_row["node"]
                dist = lengths.get(target, 99999)  # 到達不能なら99999
                matrix.at[c_row["id"], s_row["id"]] = dist

        # 保存
        matrix.to_csv(outputfile, encoding="utf-8-sig")
        print(f"距離行列（避難者→避難所）を {outputfile} に保存しました。")

        self.distance_matrix = matrix
    """
    def client_to_shelter_matrix(self, outputfile):
        self.distance_matrix = pd.DataFrame(index=self.df_assigned["id"], columns=self.df_assigned["id"])
        total_clients = len(self.df_assigned)
        last_percent = -1  # 最後に表示した進捗率（％）


        for idx, (i, client_row) in enumerate(self.df_assigned.iterrows()):
            for j, shelter_row in self.df_assigned.iterrows():
                dist = self._getdistance(client_row["y"], client_row["x"], shelter_row["y"], shelter_row["x"])#これにあとで戻すけれども
                #dist = geodesic((client_row["y"], client_row["x"]), (shelter_row["y"], shelter_row["x"])).meters#直線距離
                self.distance_matrix.at[client_row["id"], shelter_row["id"]] = dist
            
            # 進捗率の計算
            percent = int((idx + 1) / total_clients * 100)
            if percent % 5 == 0 and percent != last_percent:
                print(f"{percent}% 完了（{client_row['id']} : {client_row['name']}）")
                last_percent = percent
                
        # 距離行列をCSVで保存（任意）
        self.distance_matrix.to_csv(outputfile, encoding="utf-8-sig")
        print("直線距離による避難者→避難所の距離行列を保存しました。")
    """




    def gravity_transition_matrix(self, outputfile, beta=0.0001, ninkido=1000):
        """
        重力モデルによって避難者から避難所への遷移確率行列を算出し、CSVに保存する
        :param outputfile: 出力先CSVファイル名
        :param beta: 距離減衰係数（大きいほど近くを優先）
        :return: DataFrame形式の遷移確率行列（避難者 × 避難所）
        """
        # 確実にDataFrameから距離行列を取得（再代入）
        D_df = self.distance_matrix.copy()
        D_df = D_df.replace([None], np.nan).fillna(1e6).infer_objects(copy=False)#Noneの部分は距離を大きくして計算
        if not isinstance(D_df, pd.DataFrame):
            raise TypeError("self.distance_matrix must be a pandas DataFrame")

        D = D_df.values.astype(float)

        # 魅力度ベクトルA
        A = np.where(self.df_assigned["type"].isin(["shelter", "city_hall"]), ninkido, 1)

        # 重力モデル：指数関数 × 魅力度
        W = np.exp(-beta * D) * A[np.newaxis, :] 
        P = W / W.sum(axis=1, keepdims=True)

        # DataFrame化（index=避難者id, columns=避難所id）
        evac_ids = self.distance_matrix.index
        shelter_ids = self.distance_matrix.columns
        P_df = pd.DataFrame(P, index=evac_ids, columns=shelter_ids)

        # 保存
        P_df.to_csv(outputfile, encoding="utf-8-sig")
        print(f"重力モデル遷移行列を保存しました: {outputfile}")

        self.P = P_df
        return P_df
    

    def analyze_equivalence_classes(self, output_csv_path="equivalence_classes.csv", penalty_threshold=0.00001):#, penalty_threshold=0.00001
        """
        対称距離行列からグラフを構築し、同値類（連結成分）を判定・出力する。
        
        :param matrix_csv_path: 対称距離行列（CSV形式）のパス
        :param output_csv_path: 出力する同値類リストのCSVパス
        :param penalty_threshold: パスがないとみなす距離のしきい値
        """
        node_ids = self.df_assigned.index.tolist()

        self.nodeG = nx.DiGraph()  # 有向グラフに変更
        self.nodeG.add_nodes_from(node_ids)

        for i, source in enumerate(node_ids):
            for j, target in enumerate(node_ids):
                if i == j:
                    continue
                #p = self.P.iloc[i, j]
                #if p > penalty_threshold:
                if not pd.isna(self.distance_matrix.iloc[i, j]):
                    if self.P.values[i, j] >= penalty_threshold:
                        self.nodeG.add_edge(source, target)

        #components = list(nx.connected_components(self.nodeG))#無向グラフ用
        # 強連結成分（方向を無視した連結性）で同値類を構築
        components = list(nx.strongly_connected_components(self.nodeG))
        num_classes = len(components)
        # ノード数が多い順にソート
        #components.sort(key=lambda x: len(x), reverse=True)

        # 構造化して保存
        output_rows = []
        for idx, comp in enumerate(components, start=1):
            for node in sorted(comp):
                output_rows.append({"equivalence_class": idx, "node_id": node})

        df_output = pd.DataFrame(output_rows)
        df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        # 簡易出力
        print(f"同値類（連結成分）の数: {num_classes}")
        for idx, comp in enumerate(components, start=1):
            print(f"  - 同値類 {idx}: {len(comp)} ノード")

        print(f"結果を {output_csv_path} に保存しました。")
        
        # equivalence_class 列を追加（DataFrameのインデックスと照合）
        # ノードID → 同値類ID の辞書を作成
        node_to_class = {
            node: idx
            for idx, comp in enumerate(components, start=1)
            for node in comp
        }
        # index（node_id）を使って map で対応づける
        self.df_assigned["equivalence_class"] = self.df_assigned.index.map(node_to_class)
        self.df_assigned.to_csv(f"../results/{municipality}_node_load≧{elev}melev.csv")

    


    def calculate_centralities(self, output_csv_path= f"centralities.csv"):
        centrality_keys = [
            'in_degree', 'out_degree',
            'degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
            'closeness_centrality', 'betweenness_centrality',
            'pagerank', 'eigenvector_centrality',
            'hits_hub', 'hits_authority'
        ]

        # 各列を初期化（NaN）
        for key in centrality_keys:
            self.df_assigned[key] = float('nan')

        for eq_class, group in self.df_assigned.groupby("equivalence_class"):
            nodes = group.index.tolist()

            # サブグラフを作成
            subgraph = self.nodeG.subgraph(nodes)

            centralities = {
                'in_degree': dict(subgraph.in_degree()),
                'out_degree': dict(subgraph.out_degree()),
                'degree_centrality': nx.degree_centrality(subgraph),
                'in_degree_centrality': nx.in_degree_centrality(subgraph),
                'out_degree_centrality': nx.out_degree_centrality(subgraph),
                'closeness_centrality': nx.closeness_centrality(subgraph),
                'betweenness_centrality': nx.betweenness_centrality(subgraph),
                'pagerank': nx.pagerank(subgraph),
            }

            try:
                centralities['eigenvector_centrality'] = nx.eigenvector_centrality(subgraph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                centralities['eigenvector_centrality'] = {node: float('nan') for node in nodes}

            try:
                if len(subgraph) >= 3:  # ノード数が3以上あるときだけ実行（min(A.shape) > 1）
                    hits_hub, hits_auth = nx.hits(subgraph, max_iter=1000)
                    centralities['hits_hub'] = hits_hub
                    centralities['hits_authority'] = hits_auth
                else:
                    raise ValueError("Too small for HITS")
            except Exception:
                centralities['hits_hub'] = {node: float('nan') for node in nodes}
                centralities['hits_authority'] = {node: float('nan') for node in nodes}

            # 各指標を self.df_assigned に追加
            for key, values in centralities.items():
                self.df_assigned.loc[nodes, key] = self.df_assigned.loc[nodes].index.map(values)

            # サブグラフの中央性を CSV に保存
            #df_sub = pd.DataFrame(centralities)
            #df_sub.index.name = "node_id"
            #df_sub.to_csv(f"../results/{municipality}_centralities_class{eq_class}_load≧{elev}melev.csv")
        self.df_assigned.to_csv(f"../results/{municipality}_node_load≧{elev}melev.csv")



    def stationary_distribution(self, outputfile='stationary_distribution.csv'):
        #results = []

        for eq_class, group in self.df_assigned.groupby("equivalence_class"):
            nodes = group.index.tolist()
            if len(nodes) > 1: 
                P_sub = self.P.loc[nodes, nodes].values.T  # サブグラフの遷移行列（転置）
                n = P_sub.shape[0]
                A = P_sub - np.eye(n)
                A = np.vstack([A, np.ones(n)])  # 合計1の制約
                b = np.zeros(n + 1)
                b[-1] = 1
                # 最小二乗法で解く
                try:
                    pi = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    print("SVD failed, applying fallback method")
                    # 疑似逆行列を使った解法
                    A_pinv = np.linalg.pinv(A)
                    pi = A_pinv @ b
                # 結果を元のノードIDに戻す
                node_to_class_pi = dict(zip(nodes, pi))
                # 元データフレームに定常分布をマッピング
                self.df_assigned.loc[nodes, "stationary_distribution"] = self.df_assigned.loc[nodes].index.map(node_to_class_pi)
                # 個別ファイルに保存（eq_class名を含めたファイル名に）
                #filename = f"../results/{municipality}_stationary_distribution_class{eq_class}_load≧{elev}melev.csv"
                #pd.Series(pi, index=nodes).to_csv(filename)
                #results.append((eq_class, pi))

        # 全体も保存
        self.df_assigned.to_csv(f"../results/{municipality}_node_load≧{elev}melev.csv")



    def solve_disk_cover(self, radius_km=1.0):
        # 1. 元データのコピーとフィルタリング 
        self.df_assigned["circle_id"] = np.nan  # 初期化
        # 同値類ごとのノード数をカウント
        class_counts = self.df_assigned["equivalence_class"].value_counts()
        # 最大ノード数
        max_size = class_counts.max()
        # ノード数が最大の同値類（複数あってもOK）
        largest_classes = class_counts[class_counts == max_size].index.tolist()
        # 最大の同値類を除外
        df_filtered = self.df_assigned[~self.df_assigned["equivalence_class"].isin(largest_classes)].copy()
        if df_filtered.empty:
            print("対象となる点がありません。")
            return
        locations = list(zip(df_filtered["y"], df_filtered["x"]))  # (lat, lon)
        n_points = len(locations)
        #  2. カバー関係作成 
        cover = [[False for _ in range(n_points)] for _ in range(n_points)]
        for i in range(n_points):
            for j in range(n_points):
                if geodesic(locations[i], locations[j]).km <= radius_km:
                    cover[i][j] = True
        #  3. Gurobi モデル構築 
        model = gp.Model("DiskCover")
        x = model.addVars(n_points, vtype=GRB.BINARY, name="x")
        for j in range(n_points):
            model.addConstr(
                gp.quicksum(x[i] for i in range(n_points) if cover[i][j]) >= 1,
                name=f"cover_{j}"
            )
        model.setObjective(gp.quicksum(x[i] for i in range(n_points)), GRB.MINIMIZE)
        model.setParam('OutputFlag', 0)  # ログ非表示
        model.optimize()
        if model.status != GRB.OPTIMAL:
            print("最適解が得られませんでした。")
            return
        #  4. 中心点インデックスを取得 
        center_indices = [i for i in range(n_points) if x[i].x > 0.5]
        print(f"最小の被覆円数: {len(center_indices)}")
        #  5. 各点に circle_id を割り当て 
        assigned_ids = [-1] * n_points  # 中心点のインデックスで管理
        for circle_id, center_idx in enumerate(center_indices):
            center_point = locations[center_idx]
            for j in range(n_points):
                if geodesic(center_point, locations[j]).km <= radius_km:
                    assigned_ids[j] = circle_id
        #  6. self.df_assigned に反映 
        df_filtered = df_filtered.reset_index()  # 元の index を取り出す
        for idx, circle_id in zip(df_filtered["index"], assigned_ids):
            if circle_id != -1:
                self.df_assigned.at[idx, "circle_id"] = circle_id
        # 7. 円の中心点情報を DataFrame にまとめて保持
        center_data = []
        for circle_id, center_idx in enumerate(center_indices):
            row = df_filtered.loc[center_idx]  # reset_index() 後なので loc OK
            center_data.append({
                "circle_id": circle_id,
                "y": row["y"],
                "x": row["x"]
            })
        self.disk_cover = pd.DataFrame(center_data)
        self.df_assigned.to_csv(f"../results/{municipality}_node_load≧{elev}melev.csv")
        self.disk_cover.to_csv(f"../results/{municipality}_solve_disk_cover_load≧{elev}melev.csv")


        

# このスクリプトが直接実行されたときに main() を呼び出す
# Call main() only if this script is run directly
if __name__ == "__main__":
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[Rank {rank}/{size}]")

    # 都道府県+市町村名を指定（例："静岡県御前崎市"）
    # Specify municipality name (e.g., "静岡県御前崎市") in Japanese
    municipality = sys.argv[1]
    startelev = int(sys.argv[2])
    elev = rank + startelev
    print(f"＊＊＊＊{municipality} 高度制限{elev}m＊＊＊＊")

    start = time.time()

    # CreatePeople クラスのインスタンスを作成 / Create an instance of CreatePeople
    geo = CreatePeople(municipality)
    geo.load_state_from_results()
    print()


    #道路情報のインポート
    #対象高度(m)以下を削除する
    geo.get_filtered_road_network(output_file=f"../results/{municipality}_load≧{elev}melev.graphml", elev=elev, nrate=0.5)#exclude_types=["trunk"], 
    print('ネットワーク再作成完了')
    print()

    # 可視化
    geo.plot_colored_roads(f"../results/{municipality}_load≧{elev}melev.png")
    print('可視化完了')
    print()

    #距離行列を計算
    geo.client_to_shelter_matrix(outputfile = f"../results/{municipality}_geodesic_matrix_load≧{elev}melev.csv")
    print('避難者->避難所の距離を計算完了')
    print()

    #重力モデルによる遷移確率の計算
    geo.gravity_transition_matrix(outputfile = f"../results/{municipality}_transition_matrix_load≧{elev}melev.csv", beta=0.001, ninkido=1000)
    print("重力モデルによる遷移確率の計算完了")
    print()

    #同値類
    geo.analyze_equivalence_classes(output_csv_path= f"../results/{municipality}_equivalence_classes_load≧{elev}melev.csv", penalty_threshold=0.00001)#
    print("同値類計算完了")
    print()

    #定常分布
    geo.stationary_distribution()
    print("定常分布計算完了")
    print()
    
    #中心性
    geo.calculate_centralities()
    print("中心性計算完了")
    print()


    #最大被覆問題
    geo.solve_disk_cover(radius_km=1.0)

    

    print(f"[rank={rank}, elev={elev}, {time.time()-start}sec]")

    MPI.Finalize()









#避難所データ
#指定緊急避難場所・ 指定避難所データ　都道府県・市町村別　ダウンロード一覧
# 指定避難所データで実施する
#https://hinanmap.gsi.go.jp/hinanjocp/hinanbasho/koukaidate.html
    

#標準地域コード 一覧 | 国勢調査町丁・字等別境界データセット
#Geoshapeリポジトリ > 国勢調査町丁・字等別境界データセット > 標準地域コード 一覧
#https://geoshape.ex.nii.ac.jp/ka/resource/
#https://geoshape.ex.nii.ac.jp/ka/topojson/2020/22/r2ka22223.topojson


#高度のデータ
#基盤地図情報ダウンロードサービス
#10mメッシュ10B
#https://service.gsi.go.jp/kiban/app/map/?search=dem#5/34.999999999999986/135


# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県掛川市 0 > ../results/12_GeoCal_MPI_静岡県掛川市_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県掛川市 6 > ../results/12_GeoCal_MPI_静岡県掛川市_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県掛川市 12 > ../results/12_GeoCal_MPI_静岡県掛川市_n6_12.txt

# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県菊川市 0 > ../results/12_GeoCal_MPI_静岡県菊川市_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県菊川市 6 > ../results/12_GeoCal_MPI_静岡県菊川市_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県菊川市 12 > ../results/12_GeoCal_MPI_静岡県菊川市_n6_12.txt

# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県御前崎市 0 > ../results/12_GeoCal_MPI_静岡県御前崎市_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県御前崎市 6 > ../results/12_GeoCal_MPI_静岡県御前崎市_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県御前崎市 12 > ../results/12_GeoCal_MPI_静岡県御前崎市_n6_12.txt

# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県森町 0 > ../results/12_GeoCal_MPI_静岡県森町_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県森町 6 > ../results/12_GeoCal_MPI_静岡県森町_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県森町 12 > ../results/12_GeoCal_MPI_静岡県森町_n6_12.txt

# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県袋井市 0 > ../results/12_GeoCal_MPI_静岡県袋井市_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県袋井市 6 > ../results/12_GeoCal_MPI_静岡県袋井市_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県袋井市 12 > ../results/12_GeoCal_MPI_静岡県袋井市_n6_12.txt

# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県磐田市 0 > ../results/12_GeoCal_MPI_静岡県磐田市_n6_0.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県磐田市 6 > ../results/12_GeoCal_MPI_静岡県磐田市_n6_6.txt
# caffeinate -di mpiexec -n 6 python 12_GeoCal_MPI.py 静岡県磐田市 12 > ../results/12_GeoCal_MPI_静岡県磐田市_n6_12.txt

#pyenv deactivate
#pyenv activate py311omaezaki

