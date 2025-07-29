import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
import networkx as nx
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances
import matplotlib.patches as patches
import os

@dataclass
class Location:
    id: int
    type: str  # 'person' or 'shelter'
    x: float
    y: float
    z: float

class DisasterRiskAssessment:
    def __init__(self, width=10000, height=10000,
                 num_people=300, num_shelters=30,
                 mean_z=15, std_z=5, max_z=100):
        """
        災害リスク指標計算クラス
        """
        self.width = width
        self.height = height
        self.num_people = num_people
        self.num_shelters = num_shelters
        self.mean_z = mean_z
        self.std_z = std_z
        self.max_z = max_z
        self.people = None
        self.shelters = None

    def generate_terrain(self, resolution=300,
                        num_peaks=6, mean_height=80, std_height=30,
                        min_spread=400, max_spread=1200, seed=None):
        """
        多重ガウス分布による標高マップを生成（terrain_xx, yy, zzを保存）
        """
        if seed is not None:
            np.random.seed(seed)

        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        for _ in range(num_peaks):
            cx = np.random.uniform(0, self.width)
            cy = np.random.uniform(0, self.height)
            h = max(0, np.random.normal(mean_height, std_height))
            s = np.random.uniform(min_spread, max_spread)
            zz += h * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * s**2))

        self.terrain_xx = xx
        self.terrain_yy = yy
        self.terrain_zz = zz

    def get_elevation_at(self, x, y):
        """
        地形マップから任意座標 (x, y) の標高を補間取得
        """
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形データが生成されていません。generate_terrain() を先に呼び出してください。")

        xi = np.linspace(0, self.width, self.terrain_xx.shape[1])
        yi = np.linspace(0, self.height, self.terrain_yy.shape[0])
        interpolator = RegularGridInterpolator((yi, xi), self.terrain_zz)

        point = np.array([[y, x]])  # 注意：scipyは (y, x) 順
        return float(interpolator(point))
    
    def generate_locations_from_terrain(self):
        """
        地形に基づいて避難対象者と避難所の拠点を生成
        """
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形が生成されていません。先に generate_terrain() を呼び出してください。")

        self.people = self._generate_locations(self.num_people, "person", start_id=0)
        self.shelters = self._generate_locations(self.num_shelters, "shelter", start_id=self.num_people)


    def save_terrain_3d_plot(self, filename="terrain_3d_map.png"):
        """
        terrain_zz を 3DサーフェスでPNG保存
        """
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形が生成されていません。先に generate_terrain() を呼び出してください。")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.terrain_xx, self.terrain_yy, self.terrain_zz,
                        cmap='terrain', linewidth=0, antialiased=False)
        ax.set_title("Generated Terrain (3D View)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Elevation [m]")

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved terrain 3D map to {filename}")


    def _generate_locations(self, n, loc_type, start_id=0):
        """
        ランダムな位置と標高を持つLocationリストを生成
        """
        locations = []
        for i in range(n):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            z = self.get_elevation_at(x, y)  # 地形から取得
            #z = np.clip(np.random.normal(self.mean_z, self.std_z), 0, self.max_z)
            locations.append(Location(id=start_id + i, type=loc_type, x=x, y=y, z=z))
        return locations

    def _generate_elevation_grid(self, resolution=100):
        """
        グリッド状の標高（正規分布）を生成して等高線用に使用
        """
        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)
        xx, yy = np.meshgrid(x, y)
        zz = np.random.normal(self.mean_z, self.std_z, size=xx.shape)
        zz = np.clip(zz, 0, 300)
        return xx, yy, zz

    def show_area(self, filename="evacuation_map.png", figsize=(5, 5), dpi=200):
        """
        拠点と等高線を表示
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        print(f"Figure size (inch): {fig.get_size_inches()}")
        print(f"DPI: {fig.get_dpi()}")
        print(f"Figure size (px): {fig.get_size_inches() * fig.get_dpi()}")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(f"Evacuation Map ({self.num_people} people, {self.num_shelters} shelters)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)

        # 等高線描画
        xx, yy, zz = self._generate_elevation_grid()
        contour = ax.contourf(xx, yy, zz, levels=20, cmap="terrain", alpha=0.4)

        # 拠点描画
        px = [p.x for p in self.people]
        py = [p.y for p in self.people]
        ax.scatter(px, py, color='red', s=10, label="People")

        sx = [s.x for s in self.shelters]
        sy = [s.y for s in self.shelters]
        ax.scatter(sx, sy, color='blue', s=30, marker='^', label="Shelters")

        ax.legend(loc='center left', bbox_to_anchor=(-0.1, -0.1), borderaxespad=0.)
        plt.colorbar(contour, ax=ax, label="Elevation [m]")
        #plt.show()
        # 保存（表示しない）
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved area plot to {filename}")

    def save_to_csv(self, filename="locations.csv", person_popularity=1, shelter_popularity=30):
        """
        拠点情報をCSVに保存し、人気度列を追加
        """
        all_locations = self.people + self.shelters
        df = pd.DataFrame([vars(loc) for loc in all_locations])

        # popularity列の追加
        df["popularity"] = df["type"].apply(lambda t: person_popularity if t == "person" else shelter_popularity)

        df.to_csv(filename, index=False)
        print(f"Saved with popularity: {filename}")


    def compute_distance_matrix(self, locations):
        """
        拠点間の直線距離行列（2次元）を返す
        """
        n = len(locations)
        data = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dx = locations[i].x - locations[j].x
                dy = locations[i].y - locations[j].y
                distance = np.sqrt(dx**2 + dy**2)
                data[i, j] = distance

        index = [loc.id for loc in locations]
        df = pd.DataFrame(data, index=index, columns=index)
        return df
    
    def load_locations_from_csv(self, filename="locations.csv"):
        df = pd.read_csv(filename)
        locations = []
        for _, row in df.iterrows():
            loc = Location(id=int(row["id"]),
                        type=row["type"],
                        x=row["x"],
                        y=row["y"],
                        z=row["z"])
            locations.append(loc)
        return locations, df.set_index("id")

    def compute_lowland_fraction_matrix(self,
                                        rm=10.0,
                                        n_samples=50,
                                        output_csv="lowland_fraction_matrix.csv"):
        """
        すべての拠点間で、直線上における標高rm以下の割合を算出し、行列としてCSVに保存する
        """
        # 地形が存在するか確認
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形が生成されていません。generate_terrain() を先に呼んでください。")

        # 拠点データ結合
        all_locations = self.people + self.shelters
        df = pd.DataFrame([vars(loc) for loc in all_locations])
        df = df.sort_values("id").reset_index(drop=True)
        ids = df["id"].tolist()
        coords = df[["x", "y"]].to_numpy()
        n = len(ids)

        # 標高補間器の準備
        xi = np.linspace(0, self.width, self.terrain_xx.shape[1])
        yi = np.linspace(0, self.height, self.terrain_yy.shape[0])
        interpolator = RegularGridInterpolator((yi, xi), self.terrain_zz)

        result = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # 自己遷移はスキップ
                x0, y0 = coords[i]
                x1, y1 = coords[j]
                xs = np.linspace(x0, x1, n_samples)
                ys = np.linspace(y0, y1, n_samples)
                points = np.vstack((ys, xs)).T  # (y, x) 順

                try:
                    zs = interpolator(points)
                except ValueError:
                    zs = np.full(n_samples, 9999)  # 範囲外なら高地扱い

                ratio = np.mean(zs <= rm)
                result[i, j] = ratio

        df_result = pd.DataFrame(result, index=ids, columns=ids)
        df_result.to_csv(output_csv)
        print(f"Saved: {output_csv}")
        return df_result
    
    def compute_transition_matrix_from_csv(self, location_csv="locations.csv",
                                        alpha=1.0, beta=1.0, gamma=1.0,
                                        epsilon=1e-3,
                                        output_csv="transition_matrix.csv",
                                        lowland_matrix_csv=None, rp=None):
        """
        popularity列を含むlocations.csvを読み込み、重力モデルに基づく推移確率行列を計算・保存。
        任意で、低地割合行列を用いて危険経路を除外。
        """
        # 拠点情報読み込み
        df = pd.read_csv(location_csv)
        if "popularity" not in df.columns:
            raise ValueError("locations.csv に 'popularity' 列が含まれていません。")
        df = df.sort_values("id").reset_index(drop=True)
        ids = df["id"].tolist()
        coords = df[["x", "y"]].to_numpy()

        # 距離行列
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) + epsilon
        np.fill_diagonal(dist_matrix, np.inf)

        # 人気度ベクトル
        p = df["popularity"].to_numpy()

        # 重力モデルの基本重み行列
        weight_matrix = (p[:, None] ** alpha) * (p[None, :] ** beta) / (dist_matrix ** gamma)

        # 低地割合行列のフィルタ処理（オプション）
        if lowland_matrix_csv and rp is not None:
            lowland_df = pd.read_csv(lowland_matrix_csv, index_col=0)
            lowland_df.index = lowland_df.index.astype(int)
            lowland_df.columns = lowland_df.columns.astype(int)
            lowland_arr = lowland_df.loc[ids, ids].to_numpy()
            # rp以上の経路は使えない＝weightを0にする
            weight_matrix[lowland_arr >= rp] = 0.0

        # 正規化（行ごとに確率化）
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(weight_matrix, row_sums, where=row_sums != 0)

        # 保存
        df_result = pd.DataFrame(transition_matrix, index=ids, columns=ids)
        df_result.to_csv(output_csv)
        print(f"Saved: {output_csv}")
        return df_result

    def analyze_equivalence_classes(self,
                                    transition_matrix_csv="transition_matrix.csv",
                                    location_csv="locations.csv",
                                    output_node_csv="equivalence_assignment.csv",
                                    output_summary_csv="equivalence_summary.csv",
                                    threshold=1e-4):
        """
        推移確率行列に基づいて同値類ごとに定常分布・中心性を計算し、正規化した中心性も含めてCSV出力
        """
        # 入力読み込み
        P = pd.read_csv(transition_matrix_csv, index_col=0)
        P.index = P.index.astype(int)
        P.columns = P.columns.astype(int)
        df_loc = pd.read_csv(location_csv).set_index("id")

        # 有向グラフ G
        G = nx.DiGraph()
        for i in P.index:
            for j in P.columns:
                if P.at[i, j] >= threshold:
                    G.add_edge(i, j, weight=P.at[i, j])

        # 無向グラフ（中心性用）
        P_undir = (P + P.T) / 2
        G_undir = nx.Graph()
        for i in P.index:
            for j in P.columns:
                if P_undir.at[i, j] >= threshold:
                    G_undir.add_edge(i, j, weight=P_undir.at[i, j])

        # 同値類（強連結成分）
        components = list(nx.strongly_connected_components(G))
        assignment_rows = []
        summary_rows = []

        for idx, comp in enumerate(components):
            eq_class = f"E{idx}"
            nodes = sorted(comp)
            subG = G.subgraph(nodes)
            subG_undir = G_undir.subgraph(nodes)

            # 定常分布（PageRankで近似）
            try:
                pi = nx.pagerank(subG, weight='weight')
            except:
                pi = {n: 1 / len(nodes) for n in nodes}

            # 中心性（非正規化）
            in_deg = dict(subG.in_degree(weight=None))
            out_deg = dict(subG.out_degree(weight=None))
            undeg = dict(subG_undir.degree(weight=None))

            # 正規化用の最大値取得
            max_in = max(in_deg.values()) if in_deg else 0
            max_out = max(out_deg.values()) if out_deg else 0
            max_undeg = max(undeg.values()) if undeg else 0

            for n in nodes:
                row = {
                    "id": n,
                    "type": df_loc.at[n, "type"],
                    "x": df_loc.at[n, "x"],
                    "y": df_loc.at[n, "y"],
                    "z": df_loc.at[n, "z"],
                    "equivalence_class": eq_class,
                    "stationary": pi.get(n, 0),
                    "in_degree": in_deg.get(n, 0),
                    "out_degree": out_deg.get(n, 0),
                    "undirected_degree": undeg.get(n, 0),
                    "in_degree_norm": in_deg[n] / max_in if max_in > 0 else 0,
                    "out_degree_norm": out_deg[n] / max_out if max_out > 0 else 0,
                    "undirected_degree_norm": undeg[n] / max_undeg if max_undeg > 0 else 0
                }
                assignment_rows.append(row)

            summary_rows.append({
                "equivalence_class": eq_class,
                "num_total": len(nodes),
                "num_person": sum(df_loc.loc[n, "type"] == "person" for n in nodes),
                "num_shelter": sum(df_loc.loc[n, "type"] == "shelter" for n in nodes),
                "member_ids": str(nodes),
            })

        # 保存
        pd.DataFrame(assignment_rows).to_csv(output_node_csv, index=False)
        pd.DataFrame(summary_rows).to_csv(output_summary_csv, index=False)
        print(f"Saved: {output_node_csv}, {output_summary_csv}")


    def visualize_equivalence_classes(self,
                                    eq_csv_path="equivalence_assignment.csv",
                                    output_path="equivalence_map.png",
                                    figsize=(10, 10), dpi=300):
        """
        同値類の分類を地形上に可視化し、PNGで保存
        - 同値類を色分け、最大クラス以外を薄く
        - personとshelterをアイコンで区別
        - 等高線を表示（ヒートマップではない）
        - IDを拠点にラベル表示
        """
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形が生成されていません。generate_terrain() を先に呼んでください。")

        eq_df = pd.read_csv(eq_csv_path)
        unique_classes = sorted(eq_df["equivalence_class"].unique())
        class_colors = cm.get_cmap('tab20', len(unique_classes))
        color_map = {cls: class_colors(i) for i, cls in enumerate(unique_classes)}

        # 最大同値類を特定
        class_sizes = eq_df["equivalence_class"].value_counts()
        largest_class = class_sizes.idxmax()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # 等高線（ヒートマップでなく線）
        ax.contour(self.terrain_xx, self.terrain_yy, self.terrain_zz,
                levels=20, cmap="terrain", linewidths=0.7)

        # 拠点プロット
        for _, row in eq_df.iterrows():
            x, y, node_id = row["x"], row["y"], int(row["id"])
            cls, t = row["equivalence_class"], row["type"]
            color = color_map[cls]
            in_main_class = (cls == largest_class)
            alpha = 1.0 if in_main_class else 0.6

            # マーカー形状を分類
            if t == "person":
                marker = 'o' if in_main_class else '*'
                size = 30 if in_main_class else 60
            else:  # shelter
                marker = '^' if in_main_class else 'D'
                size = 70 if in_main_class else 60

            ax.scatter(x, y, marker=marker, s=size, color=color,
                       edgecolors='black', alpha=alpha)
            ax.text(x + 20, y + 20, str(node_id), fontsize=6, color='black', alpha=alpha)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Person (Main Class)',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', color='w', label='Person (Isolated)',
                    markerfacecolor='gray', markersize=8, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='^', color='w', label='Shelter (Main Class)',
                    markerfacecolor='gray', markersize=8, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='D', color='w', label='Shelter (Isolated)',
                    markerfacecolor='gray', markersize=7, markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.set_title("Equivalence Class Map with Terrain", fontsize=12)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.tight_layout()

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
        

    def compute_disk_cover(self, R=1000, grid_step=100,
                        input_node_csv="equivalence_assignment.csv",
                        output_csv="disk_cover_result.csv"):
        """
        最大半径Rの円で、主同値類以外のノードをグリッド上の円中心で貪欲に被覆する
        - 円の中心候補は grid_step[m] 間隔のグリッド点
        - 被覆成功ごとに covered セットを更新
        - 最終的に被覆円の中心・半径・カバー対象IDを CSV に保存
        """

        # 1. 拠点読み込み
        df = pd.read_csv(input_node_csv)
        main_class = df["equivalence_class"].value_counts().idxmax()
        df_target = df[df["equivalence_class"] != main_class].copy()
        target_coords = df_target[["x", "y"]].to_numpy()
        target_ids = df_target["id"].tolist()

        if len(target_coords) == 0:
            print("対象ノードがありません。全て主同値類に属しています。")
            return None

        print(f"対象ノード数: {len(target_coords)}")
        print(f"対象ID一覧: {target_ids}")

        # 2. グリッド中心候補作成
        grid_x = np.arange(0, self.width + grid_step, grid_step)
        grid_y = np.arange(0, self.height + grid_step, grid_step)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])

        # 3. 各中心からR以内にあるノードを探索
        dists = pairwise_distances(grid_points, target_coords)
        cover_matrix = dists <= R  # [grid_point_index, target_index]

        covered = set()
        result_rows = []
        iteration = 0
        total = len(target_ids)

        print(f"Starting disk cover computation (R={R}, grid_step={grid_step})...")

        # 4. 貪欲に最大カバーする中心を選び続ける
        while len(covered) < total:
            # 未カバー対象の列インデックス
            uncovered_mask = np.array([tid not in covered for tid in target_ids])
            effective_covers = (cover_matrix[:, uncovered_mask]).sum(axis=1)

            best_idx = np.argmax(effective_covers)
            max_cover = effective_covers[best_idx]

            if max_cover == 0:
                print("すでにカバーされたノードしか見つかりません。終了します。")
                break

            # 実際にこの中心でカバーできる ID を取得
            cover_idxs = np.where(cover_matrix[best_idx])[0]
            cover_ids = [target_ids[i] for i in cover_idxs if target_ids[i] not in covered]
            for cid in cover_ids:
                covered.add(cid)

            result_rows.append({
                "center_x": grid_points[best_idx, 0],
                "center_y": grid_points[best_idx, 1],
                "radius": R,
                "covered_ids": str(cover_ids),
                "num_covered": len(cover_ids)
            })

            iteration += 1
            print(f"[Iteration {iteration}] Covered: {len(covered)} / {total} ({len(covered)/total:.1%})")

        # 結果保存
        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(output_csv, index=False)
        print(f"✅ Disk cover completed: {iteration} disks used.")
        print(f"Saved result to: {output_csv}")

        return output_csv

    def visualize_equivalence_and_disks(self,
                                        eq_csv_path="equivalence_assignment.csv",
                                        disk_csv_path="disk_cover_result.csv",
                                        output_path="equivalence_with_disks.png",
                                        rm=None,
                                        figsize=(10, 10), dpi=300):
        """
        同値類の分類と被覆円を地形上に可視化して保存
        - 同値類を色分け、最大クラス以外を区別
        - person/shelterを形状で区別
        - 最大同値類外: person→星、shelter→ひし形
        - 被覆円を赤の破線で表示
        - 等高線を背景に表示
        """
        if not hasattr(self, "terrain_zz"):
            raise RuntimeError("地形が生成されていません。generate_terrain() を先に呼んでください。")

        eq_df = pd.read_csv(eq_csv_path)
        disk_df = pd.read_csv(disk_csv_path)
        unique_classes = sorted(eq_df["equivalence_class"].unique())
        class_colors = cm.get_cmap('tab20', len(unique_classes))
        color_map = {cls: class_colors(i) for i, cls in enumerate(unique_classes)}

        class_sizes = eq_df["equivalence_class"].value_counts()
        largest_class = class_sizes.idxmax()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if rm is not None:
            rm_str = f"_ristricted{int(rm)}"
            title_suffix = f" (Restricted Elevation ≤ {int(rm)}m)"
        else:
            rm_str = ""
            title_suffix = ""

        output_path = f"{output_path}{rm_str}.png"


        # 等高線表示
        ax.contour(self.terrain_xx, self.terrain_yy, self.terrain_zz,
                levels=20, cmap="terrain", linewidths=0.7)

        # 拠点プロット
        for _, row in eq_df.iterrows():
            x, y, node_id = row["x"], row["y"], int(row["id"])
            cls, t = row["equivalence_class"], row["type"]
            color = color_map[cls]
            in_main_class = (cls == largest_class)
            alpha = 1.0 if in_main_class else 0.7

            # マーカー形状
            if t == "person":
                marker = 'o' if in_main_class else '*'
                size = 30 if in_main_class else 60
            else:
                marker = '^' if in_main_class else 'D'
                size = 70 if in_main_class else 60

            ax.scatter(x, y, marker=marker, s=size, color=color,
                    edgecolors='black', alpha=alpha)
            ax.text(x + 20, y + 20, str(node_id), fontsize=6, color='black', alpha=alpha)

        # 被覆円の描画
        for _, row in disk_df.iterrows():
            cx, cy, radius = row["center_x"], row["center_y"], row["radius"]
            circle = plt.Circle((cx, cy), radius, color='red', fill=False,
                                linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_patch(circle)

        # 凡例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Person (Main Class)',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', color='w', label='Person (Isolated)',
                    markerfacecolor='gray', markersize=8, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='^', color='w', label='Shelter (Main Class)',
                    markerfacecolor='gray', markersize=8, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='D', color='w', label='Shelter (Isolated)',
                    markerfacecolor='gray', markersize=7, markeredgecolor='black'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Cover Disk')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        #ax.set_title("Equivalence Classes and Disk Covers", fontsize=12)
        ax.set_title(f"Equivalence Classes and Disk Covers{title_suffix}", fontsize=12)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    def summarize_equivalence_analysis(self,
                                        eq_csv_path="equivalence_assignment.csv",
                                        disk_csv_path="disk_cover_result.csv",
                                        output_summary_csv="equivalence_summary.csv"):
        """
        同値類の統計・被覆円の統計・主同値類内の shelter の中心性・定常分布指標を集約して保存
        """

        # 同値類データ読み込み
        df = pd.read_csv(eq_csv_path)

        # 1. 同値類の個数
        num_equivalence_classes = df["equivalence_class"].nunique()

        # 2. 主同値類の特定
        main_class = df["equivalence_class"].value_counts().idxmax()

        # 3. 主同値類以外の要素数
        non_main_class_elements = df[df["equivalence_class"] != main_class].shape[0]

        # 4. 被覆円の個数（ファイルが存在する場合のみ）
        try:
            disk_df = pd.read_csv(disk_csv_path)
            num_cover_disks = 0 if disk_df.empty else disk_df.shape[0]
        except Exception:
            num_cover_disks = 0

        # 5. 主同値類に属する shelter の抽出
        shelters_main = df[(df["equivalence_class"] == main_class) & (df["type"] == "shelter")]

        eps = 1e-10
        max_in_all = df["in_degree"].max() + eps
        max_out_all = df["out_degree"].max() + eps
        max_undeg_all = df["undirected_degree"].max() + eps

        shelters_main = shelters_main.copy()
        shelters_main["in_degree_norm"] = shelters_main["in_degree"] / max_in_all
        shelters_main["out_degree_norm"] = shelters_main["out_degree"] / max_out_all
        shelters_main["undirected_degree_norm"] = shelters_main["undirected_degree"] / max_undeg_all

        # 中心性・定常分布の統計
        mean_in_deg = shelters_main["in_degree"].mean()
        mean_out_deg = shelters_main["out_degree"].mean()
        mean_undeg = shelters_main["undirected_degree"].mean()

        max_in_deg = shelters_main["in_degree"].max()
        max_in_deg_id = shelters_main.loc[shelters_main["in_degree"].idxmax(), "id"]

        mean_stat = shelters_main["stationary"].mean()
        std_stat = shelters_main["stationary"].std()  # ✅ 標準偏差を追加
        max_stat = shelters_main["stationary"].max()
        max_stat_id = shelters_main.loc[shelters_main["stationary"].idxmax(), "id"]

        mean_in_deg_norm = shelters_main["in_degree_norm"].mean()
        mean_out_deg_norm = shelters_main["out_degree_norm"].mean()
        mean_undeg_norm = shelters_main["undirected_degree_norm"].mean()

        max_in_deg_norm = shelters_main["in_degree_norm"].max()
        max_out_deg_norm = shelters_main["out_degree_norm"].max()
        max_undeg_norm = shelters_main["undirected_degree_norm"].max()

        max_in_deg_norm_id = shelters_main.loc[shelters_main["in_degree_norm"].idxmax(), "id"]
        max_out_deg_norm_id = shelters_main.loc[shelters_main["out_degree_norm"].idxmax(), "id"]
        max_undeg_norm_id = shelters_main.loc[shelters_main["undirected_degree_norm"].idxmax(), "id"]

        summary = {
            "num_equivalence_classes": num_equivalence_classes,
            "num_non_main_class_elements": non_main_class_elements,
            "num_cover_disks": num_cover_disks,

            "mean_in_degree_shelter_main": mean_in_deg,
            "mean_out_degree_shelter_main": mean_out_deg,
            "mean_undirected_degree_shelter_main": mean_undeg,
            "max_in_degree_shelter_main": max_in_deg,
            "max_in_degree_shelter_main_id": int(max_in_deg_id),

            "mean_in_degree_norm_shelter_main": mean_in_deg_norm,
            "mean_out_degree_norm_shelter_main": mean_out_deg_norm,
            "mean_undirected_degree_norm_shelter_main": mean_undeg_norm,
            "max_in_degree_norm_shelter_main": max_in_deg_norm,
            "max_in_degree_norm_shelter_main_id": int(max_in_deg_norm_id),
            "max_out_degree_norm_shelter_main": max_out_deg_norm,
            "max_out_degree_norm_shelter_main_id": int(max_out_deg_norm_id),
            "max_undirected_degree_norm_shelter_main": max_undeg_norm,
            "max_undirected_degree_norm_shelter_main_id": int(max_undeg_norm_id),

            "mean_stationary_shelter_main": mean_stat,
            "std_stationary_shelter_main": std_stat,  # ✅ 追加
            "max_stationary_shelter_main": max_stat,
            "max_stationary_shelter_main_id": int(max_stat_id)
        }

        pd.DataFrame([summary]).to_csv(output_summary_csv, index=False)
        print(f"✅ Summary saved to: {output_summary_csv}")

        print("===== Equivalence Analysis Summary =====")
        print(f"🔢 同値類の個数: {num_equivalence_classes}")
        print(f"🚷 主同値類以外の要素数: {non_main_class_elements}")
        print(f"🟢 被覆円の個数: {num_cover_disks}")
        print("--- 主同値類内 shelter の中心性指標（非正規化） ---")
        print(f"  ・平均 in-degree : {mean_in_deg:.2f}")
        print(f"  ・平均 out-degree: {mean_out_deg:.2f}")
        print(f"  ・平均 undirected: {mean_undeg:.2f}")
        print(f"  ・最大 in-degree : {max_in_deg} (ID={int(max_in_deg_id)})")
        print("--- 主同値類内 shelter の中心性指標（正規化） ---")
        print(f"  ・平均 in-degree : {mean_in_deg_norm:.4f}")
        print(f"  ・平均 out-degree: {mean_out_deg_norm:.4f}")
        print(f"  ・平均 undirected: {mean_undeg_norm:.4f}")
        print(f"  ・最大 in-degree : {max_in_deg_norm:.4f} (ID={int(max_in_deg_norm_id)})")
        print(f"  ・最大 out-degree: {max_out_deg_norm:.4f} (ID={int(max_out_deg_norm_id)})")
        print(f"  ・最大 undirected: {max_undeg_norm:.4f} (ID={int(max_undeg_norm_id)})")
        print("--- 主同値類内 shelter の定常分布 ---")
        print(f"  ・平均 stationary : {mean_stat:.6f}")
        print(f"  ・標準偏差 stationary: {std_stat:.6f}")  # ✅ 出力に追加
        print(f"  ・最大 stationary : {max_stat:.6f} (ID={int(max_stat_id)})")
        print("========================================")

        return summary


    '''
    def summarize_equivalence_analysis(self,
                                    eq_csv_path="equivalence_assignment.csv",
                                    disk_csv_path="disk_cover_result.csv",
                                    output_summary_csv="equivalence_summary.csv"):
        """
        同値類の統計・被覆円の統計・主同値類内の shelter の中心性・定常分布指標を集約して保存
        """

        # 同値類データ読み込み
        df = pd.read_csv(eq_csv_path)

        # 1. 同値類の個数
        num_equivalence_classes = df["equivalence_class"].nunique()

        # 2. 主同値類の特定
        main_class = df["equivalence_class"].value_counts().idxmax()

        # 3. 主同値類以外の要素数
        non_main_class_elements = df[df["equivalence_class"] != main_class].shape[0]

        # 4. 被覆円の個数（ファイルが存在する場合のみ）
        try:
            disk_df = pd.read_csv(disk_csv_path)
            if disk_df.empty:
                num_cover_disks = 0
            else:
                num_cover_disks = disk_df.shape[0]
        except Exception:
            num_cover_disks = 0

        # 5. shelter（避難所）で主同値類に属するもののみ抽出
        shelters_main = df[(df["equivalence_class"] == main_class) & (df["type"] == "shelter")]

        # 正規化のために全体の最大値取得（0割防止のためepsを加える）
        eps = 1e-10
        max_in_all = df["in_degree"].max() + eps
        max_out_all = df["out_degree"].max() + eps
        max_undeg_all = df["undirected_degree"].max() + eps

        # 正規化済み列の追加（主同値類 shelter のみに対して）
        shelters_main = shelters_main.copy()
        shelters_main["in_degree_norm"] = shelters_main["in_degree"] / max_in_all
        shelters_main["out_degree_norm"] = shelters_main["out_degree"] / max_out_all
        shelters_main["undirected_degree_norm"] = shelters_main["undirected_degree"] / max_undeg_all

        # 各種指標の平均・最大・ID
        mean_in_deg = shelters_main["in_degree"].mean()
        mean_out_deg = shelters_main["out_degree"].mean()
        mean_undeg = shelters_main["undirected_degree"].mean()

        max_in_deg = shelters_main["in_degree"].max()
        max_in_deg_id = shelters_main.loc[shelters_main["in_degree"].idxmax(), "id"]

        mean_stat = shelters_main["stationary"].mean()
        max_stat = shelters_main["stationary"].max()
        max_stat_id = shelters_main.loc[shelters_main["stationary"].idxmax(), "id"]

        # 正規化中心性指標（平均・最大）
        mean_in_deg_norm = shelters_main["in_degree_norm"].mean()
        mean_out_deg_norm = shelters_main["out_degree_norm"].mean()
        mean_undeg_norm = shelters_main["undirected_degree_norm"].mean()

        max_in_deg_norm = shelters_main["in_degree_norm"].max()
        max_out_deg_norm = shelters_main["out_degree_norm"].max()
        max_undeg_norm = shelters_main["undirected_degree_norm"].max()

        max_in_deg_norm_id = shelters_main.loc[shelters_main["in_degree_norm"].idxmax(), "id"]
        max_out_deg_norm_id = shelters_main.loc[shelters_main["out_degree_norm"].idxmax(), "id"]
        max_undeg_norm_id = shelters_main.loc[shelters_main["undirected_degree_norm"].idxmax(), "id"]

        # 結果まとめ
        summary = {
            "num_equivalence_classes": num_equivalence_classes,
            "num_non_main_class_elements": non_main_class_elements,
            "num_cover_disks": num_cover_disks,

            "mean_in_degree_shelter_main": mean_in_deg,
            "mean_out_degree_shelter_main": mean_out_deg,
            "mean_undirected_degree_shelter_main": mean_undeg,
            "max_in_degree_shelter_main": max_in_deg,
            "max_in_degree_shelter_main_id": int(max_in_deg_id),

            "mean_in_degree_norm_shelter_main": mean_in_deg_norm,
            "mean_out_degree_norm_shelter_main": mean_out_deg_norm,
            "mean_undirected_degree_norm_shelter_main": mean_undeg_norm,
            "max_in_degree_norm_shelter_main": max_in_deg_norm,
            "max_in_degree_norm_shelter_main_id": int(max_in_deg_norm_id),
            "max_out_degree_norm_shelter_main": max_out_deg_norm,
            "max_out_degree_norm_shelter_main_id": int(max_out_deg_norm_id),
            "max_undirected_degree_norm_shelter_main": max_undeg_norm,
            "max_undirected_degree_norm_shelter_main_id": int(max_undeg_norm_id),

            "mean_stationary_shelter_main": mean_stat,
            "max_stationary_shelter_main": max_stat,
            "max_stationary_shelter_main_id": int(max_stat_id)
        }

        # CSV 出力
        pd.DataFrame([summary]).to_csv(output_summary_csv, index=False)
        print(f"✅ Summary saved to: {output_summary_csv}")

        # 画面出力（見やすく整形）
        print("===== Equivalence Analysis Summary =====")
        print(f"🔢 同値類の個数: {num_equivalence_classes}")
        print(f"🚷 主同値類以外の要素数: {non_main_class_elements}")
        print(f"🟢 被覆円の個数: {num_cover_disks}")
        print("--- 主同値類内 shelter の中心性指標（非正規化） ---")
        print(f"  ・平均 in-degree : {mean_in_deg:.2f}")
        print(f"  ・平均 out-degree: {mean_out_deg:.2f}")
        print(f"  ・平均 undirected: {mean_undeg:.2f}")
        print(f"  ・最大 in-degree : {max_in_deg} (ID={int(max_in_deg_id)})")
        print("--- 主同値類内 shelter の中心性指標（正規化） ---")
        print(f"  ・平均 in-degree : {mean_in_deg_norm:.4f}")
        print(f"  ・平均 out-degree: {mean_out_deg_norm:.4f}")
        print(f"  ・平均 undirected: {mean_undeg_norm:.4f}")
        print(f"  ・最大 in-degree : {max_in_deg_norm:.4f} (ID={int(max_in_deg_norm_id)})")
        print(f"  ・最大 out-degree: {max_out_deg_norm:.4f} (ID={int(max_out_deg_norm_id)})")
        print(f"  ・最大 undirected: {max_undeg_norm:.4f} (ID={int(max_undeg_norm_id)})")
        print("--- 主同値類内 shelter の定常分布 ---")
        print(f"  ・平均 stationary : {mean_stat:.6f}")
        print(f"  ・最大 stationary : {max_stat:.6f} (ID={int(max_stat_id)})")
        print("========================================")

        return summary
    '''

    def evaluate_model_score(self, summary_csv="equivalence_summary.csv",
                            lambda_=1.0, mu=1.0, nu=100.0, rm=0):
        """
        指標に基づいてモデルスコアを計算し、画面表示＋返却

        評価式: λ×同値類の個数 + μ×被覆円の個数 − ν×正規化in-degree平均
        """
        df = pd.read_csv(summary_csv)

        # 値取得
        n_eq = df.at[0, "num_equivalence_classes"]
        n_disks = df.at[0, "num_cover_disks"]
        mean_in_deg_norm = df.at[0, "mean_in_degree_norm_shelter_main"]

        # 評価値計算
        score = lambda_ * n_eq + mu * n_disks - nu * mean_in_deg_norm

        # 表示
        print("===== Model Evaluation Score =====")
        print(f"🗻 制限標高 rm = {rm} [m]")
        print(f"🔢 同値類の個数: {n_eq}")
        print(f"🟢 被覆円の個数: {n_disks}")
        print(f"📈 shelterの正規化in-degree平均: {mean_in_deg_norm:.4f}")
        print(f"🧮 評価式: λ×{n_eq} + μ×{n_disks} − ν×{mean_in_deg_norm:.4f}")
        print(f"➡️ 評価値: {score:.4f}")
        print("===================================")

        return score
    
    def evaluate_model_score_std_stationary(self, summary_csv="equivalence_summary.csv",
                                            lambda_=1.0, mu=1.0, nu=100.0, rm=0):
        """
        指標に基づいてモデルスコアを計算（スコアは小さいほど良いとする）
        評価式: λ×同値類の個数 + μ×被覆円の個数 + ν×定常分布の標準偏差
        """
        df = pd.read_csv(summary_csv)

        n_eq = df.at[0, "num_equivalence_classes"]
        n_disks = df.at[0, "num_cover_disks"]
        std_stat = df.at[0, "std_stationary_shelter_main"]

        score = lambda_ * n_eq + mu * n_disks + nu * std_stat

        print("===== Model Evaluation Score (Smaller is Better) =====")
        print(f"🗻 制限標高 rm = {rm} [m]")
        print(f"🔢 同値類の個数: {n_eq}")
        print(f"🟢 被覆円の個数: {n_disks}")
        print(f"📊 shelterの定常分布の標準偏差: {std_stat:.6f}")
        print(f"🧮 評価式: λ×{n_eq} + μ×{n_disks} + ν×{std_stat:.6f}")
        print(f"➡️ 評価スコア（小さいほど良い）: {score:.4f}")
        print("===================================")

        return score

    def run_elevation_sweep_analysis_std_stationary(self, rm_min=0, rm_max=15, lambda_=1.0, mu=1.0, nu=10.0):
        """
        標高制限 rm を rm_min から rm_max まで1m刻みで変化させながら、
        各 rm に対応する災害リスクを評価する。
        
        実行内容：
            ① rm 以下の地形割合マトリクスを計算
            ② 危険な経路を除いた重力モデルにより推移確率行列を構築
            ③ 同値類を分類し、定常分布と中心性を含めた分析を実施
            ④ 主同値類に属さない拠点を貪欲法で被覆（disk cover）
            ⑤ 統計をまとめた summary CSV を保存
            ⑥ 定常分布の標準偏差に基づいてスコアを算出（小さいほど良い）
            ⑦ 結果をCSVに保存
            ⑧ 各評価指標のグラフを描画・保存
            ⑨ スコアの累積和をグラフ描画・保存
        """
        results = []

        for rm in range(rm_min, rm_max + 1):
            print(f"\n=== 標高 {rm}m の解析を開始 ===")
            try:
                # ファイル名準備
                lowland_csv = f"lowland_fraction_matrix_rm{rm}.csv"
                transition_csv = f"transition_matrix_rm{rm}.csv"
                eq_assign_csv = f"equivalence_assignment_rm{rm}.csv"
                eq_summary_csv = f"equivalence_summary_rm{rm}.csv"
                disk_csv = f"disk_cover_result_rm{rm}.csv"

                # ① 標高rm以下の地形割合マトリクスを作成
                self.compute_lowland_fraction_matrix(rm=rm, output_csv=lowland_csv)

                # ② 危険経路除去済みの推移確率行列を構築
                self.compute_transition_matrix_from_csv(
                    lowland_matrix_csv=lowland_csv,
                    rp=0.5,  # 50%以上が低地なら除外
                    output_csv=transition_csv
                )

                # ③ 同値類と中心性、定常分布を解析・CSV出力
                self.analyze_equivalence_classes(
                    transition_matrix_csv=transition_csv,
                    location_csv="locations.csv",
                    output_node_csv=eq_assign_csv,
                    output_summary_csv=eq_summary_csv,
                    threshold=1e-3
                )

                # 同値類の確認表示
                df_eq = pd.read_csv(eq_assign_csv)
                print(f"📊 [rm={rm}] Unique equivalence classes:", df_eq["equivalence_class"].unique())
                print(f"📊 [rm={rm}] Class counts:\n", df_eq["equivalence_class"].value_counts())

                # ④ 主同値類外ノードを被覆するdisk cover実行
                self.compute_disk_cover(
                    input_node_csv=eq_assign_csv,
                    output_csv=disk_csv
                )

                # 被覆円がなければ、等高線付きマップを保存
                if not os.path.exists(disk_csv):
                    eq_map_output = f"equivalence_map_rm{rm}.png"
                    self.visualize_equivalence_classes(
                        eq_csv_path=eq_assign_csv,
                        output_path=eq_map_output,
                        figsize=(10, 10),
                        dpi=300
                    )
                    print(f"✅ 被覆円なし: Saved {eq_map_output}")

                # ⑤ 統計 summary を保存（定常分布標準偏差含む）
                summary = self.summarize_equivalence_analysis(
                    eq_csv_path=eq_assign_csv,
                    disk_csv_path=disk_csv,
                    output_summary_csv=eq_summary_csv
                )

                # ⑥ 評価関数（定常分布の標準偏差を使用）
                score = self.evaluate_model_score_std_stationary(
                    summary_csv=eq_summary_csv,
                    lambda_=lambda_,
                    mu=mu,
                    nu=nu,
                    rm=rm
                )

                # 結果に rm とスコアを追加
                summary["elevation"] = rm
                summary["score"] = score
                results.append(summary)

                # ⑦ 可視化：diskと同値類を重ねたマップ出力
                self.visualize_equivalence_and_disks(
                    eq_csv_path=eq_assign_csv,
                    disk_csv_path=disk_csv,
                    output_path="equivalence_with_disks.png",
                    rm=rm,
                    figsize=(10, 10),
                    dpi=300
                )

            except Exception as e:
                print(f"⚠️ 標高 {rm}m でエラー: {e}")
                continue

        # ⑧ 全結果をCSV保存
        df_results = pd.DataFrame(results).sort_values("elevation")
        df_results["score_cumsum"] = df_results["score"].cumsum()
        df_results.to_csv("evaluation_over_elevation_std_stationary.csv", index=False)
        print("✅ 全ての結果を evaluation_over_elevation_std_stationary.csv に保存しました。")

        # ⑨ グラフ描画：各指標とスコアの推移
        plt.figure(figsize=(10, 6))
        score_equiv = df_results["num_equivalence_classes"] * lambda_
        score_cover = df_results["num_cover_disks"] * mu
        score_std_stat = df_results["std_stationary_shelter_main"] * nu

        plt.plot(df_results["elevation"], score_equiv, marker='o', label=f"Equivalence Classes × λ ({lambda_})")
        plt.plot(df_results["elevation"], score_cover, marker='s', label=f"Cover Disks × μ ({mu})")
        plt.plot(df_results["elevation"], score_std_stat, marker='^', label=f"Std(Stationary) × ν ({nu})")
        plt.plot(df_results["elevation"], df_results["score"], marker='x', linestyle='--', color='black', label="Total Score")

        plt.xlabel("Elevation Threshold [m]")
        plt.ylabel("Weighted Metric Value")
        plt.title(f"Evaluation Metrics vs Elevation (using Std of Stationary, λ={lambda_}, μ={mu}, ν={nu})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("metrics_vs_elevation_std_stationary.png", dpi=300)
        plt.close()

        # 累積スコアのプロット
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["elevation"], df_results["score_cumsum"], marker='o', color='black')
        plt.xlabel("Elevation Threshold [m]")
        plt.ylabel("Cumulative Score")
        plt.title("Cumulative Evaluation Score (Std Stationary Version)")
        plt.grid(True)
        plt.savefig("cumulative_score_vs_elevation_std_stationary.png", dpi=300)
        plt.close()

        return df_results

    def run_elevation_sweep_analysis(self, rm_min=0, rm_max=15, lambda_=1.0, mu=1.0, nu=10.0):
        """
        標高制限 rm を rm_min から rm_max まで1m刻みで変化させながら、
        各 rm に対応する災害リスクを評価し、推移確率の構築 → 同値類解析 → 被覆円評価 →
        指標サマリとスコアを計算し、CSVとグラフで結果を保存する。
        """
        results = []

        for rm in range(rm_min, rm_max + 1):
            print(f"\n=== 標高 {rm}m の解析を開始 ===")
            try:
                # 各 rm に応じたファイル名を用意
                lowland_csv = f"lowland_fraction_matrix_rm{rm}.csv"
                transition_csv = f"transition_matrix_rm{rm}.csv"
                eq_assign_csv = f"equivalence_assignment_rm{rm}.csv"
                eq_summary_csv = f"equivalence_summary_rm{rm}.csv"
                disk_csv = f"disk_cover_result_rm{rm}.csv"

                # ① 各拠点ペア間の「rm以下の標高を通る割合」行列を作成・保存
                self.compute_lowland_fraction_matrix(rm=rm, output_csv=lowland_csv)

                # ② 低地割合に基づいて「危険な経路を除外」し、重力モデルで推移確率行列を作成・保存
                self.compute_transition_matrix_from_csv(
                    lowland_matrix_csv=lowland_csv,
                    rp=0.5,  # 危険な経路（低地割合 ≥ 50%）は除外
                    output_csv=transition_csv
                )

                # ③ 作成した推移確率行列から同値類や中心性を解析し、ノード分類とサマリをCSV出力
                self.analyze_equivalence_classes(
                    transition_matrix_csv=transition_csv,
                    location_csv="locations.csv",
                    output_node_csv=eq_assign_csv,
                    output_summary_csv=eq_summary_csv,
                    threshold=1e-3
                )

                # ✅ 同値類CSVを読み込んでクラスの状況を確認
                df_eq = pd.read_csv(eq_assign_csv)
                print(f"📊 [rm={rm}] Unique equivalence classes:", df_eq["equivalence_class"].unique())
                print(f"📊 [rm={rm}] Class counts:\n", df_eq["equivalence_class"].value_counts())

                # ④ 主同値類に属さない拠点を貪欲法で被覆し、被覆円の情報を保存
                self.compute_disk_cover(
                    input_node_csv=eq_assign_csv,
                    output_csv=disk_csv
                )
                # ✅ 被覆円ファイルが存在しない場合 → 被覆円なしの等高線付き可視化図を出力
                if not os.path.exists(disk_csv):
                    eq_map_output = f"equivalence_map_rm{rm}.png"
                    self.visualize_equivalence_classes(
                        eq_csv_path=eq_assign_csv,
                        output_path=eq_map_output,
                        figsize=(10, 10),
                        dpi=300
                    )
                    print(f"✅ 被覆円なし: Saved {eq_map_output}")

                # ⑤ 各種統計（同値類数・被覆円数・shelterの中心性・定常分布）をまとめたサマリをCSV出力
                summary = self.summarize_equivalence_analysis(
                    eq_csv_path=eq_assign_csv,
                    disk_csv_path=disk_csv,
                    output_summary_csv=eq_summary_csv
                )

                # ✅ 被覆円を重ねた可視化図を保存（rmごとにファイル名が変わる）
                self.visualize_equivalence_and_disks(
                    eq_csv_path=eq_assign_csv,
                    disk_csv_path=disk_csv,
                    output_path="equivalence_with_disks.png",  # rmに応じて自動で suffix が付く
                    rm=rm,
                    figsize=(10, 10),
                    dpi=300
                )

                # ⑥ 評価式に基づき、総合スコアを計算
                score = self.evaluate_model_score(
                    summary_csv=eq_summary_csv,
                    lambda_=lambda_,
                    mu=mu,
                    nu=nu,
                    rm=rm
                )

                # 結果を記録（標高とスコアを追加）
                summary["elevation"] = rm
                summary["score"] = score
                results.append(summary)

            except Exception as e:
                print(f"⚠️ 標高 {rm}m でエラー: {e}")
                continue

        # ⑦ すべての rm に対する結果をDataFrameにまとめてCSV保存
        df_results = pd.DataFrame(results).sort_values("elevation")
        df_results["score_cumsum"] = df_results["score"].cumsum()
        df_results.to_csv("evaluation_over_elevation.csv", index=False)
        print("✅ 全ての結果を evaluation_over_elevation.csv に保存しました。")

        # ⑧ 標高ごとの指標の推移を折れ線グラフにして保存（重み付きで表示）

        plt.figure(figsize=(10, 6))

        # 重みをかけた指標
        score_equiv = df_results["num_equivalence_classes"] * lambda_
        score_cover = df_results["num_cover_disks"] * mu
        score_in_deg = df_results["mean_in_degree_norm_shelter_main"] * nu

        plt.plot(df_results["elevation"], score_equiv, marker='o', label=f"Equivalence Classes × λ ({lambda_})")
        plt.plot(df_results["elevation"], score_cover, marker='s', label=f"Cover Disks × μ ({mu})")
        plt.plot(df_results["elevation"], score_in_deg, marker='^', label=f"Mean In-degree × ν ({nu})")
        plt.plot(df_results["elevation"], df_results["score"], marker='x', linestyle='--', color='black', label="Total Score")

        plt.xlabel("Elevation Threshold [m]")
        plt.ylabel("Weighted Metric Value")
        plt.title(f"Evaluation Metrics vs Elevation (λ={lambda_}, μ={mu}, ν={nu})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("metrics_vs_elevation.png", dpi=300)
        plt.close()

        # ⑨ 評価スコアの累積和をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["elevation"], df_results["score_cumsum"], marker='o', color='black')
        plt.xlabel("Elevation Threshold [m]")
        plt.ylabel("Cumulative Score")
        plt.title("Cumulative Evaluation Score")
        plt.grid(True)
        plt.savefig("cumulative_score_vs_elevation.png", dpi=300)
        plt.close()

        return df_results
    

if __name__ == "__main__":

    sim = DisasterRiskAssessment(num_people=300, num_shelters=30, mean_z=15, std_z=5, max_z=100)

    # 地形と拠点の初期生成（1回だけでOK）
    sim.generate_terrain(
        resolution=300,
        num_peaks=5,
        mean_height=15,
        std_height=5,
        min_spread=1500,
        max_spread=3000
    )
    sim.save_terrain_3d_plot("terrain_3d_map.png") # ✅ 地形の3Dサーフェス図（1回だけでOK）
    sim.generate_locations_from_terrain() # 拠点の生成（地形に依存）
    sim.show_area(filename="evacuation_map.png", figsize=(5, 5), dpi=200) # ✅ 拠点の平面配置図（1回だけでOK）
    sim.save_to_csv("locations.csv", person_popularity=1, shelter_popularity=30) # 拠点情報の保存（popularity付き）

    # elevation sweep の実行（評価と可視化を含む）
    #sim.run_elevation_sweep_analysis(rm_min=0, rm_max=15, lambda_=1.0, mu=1.0, nu=10.0)
    #sim.run_elevation_sweep_analysis_std_stationary(rm_min=0, rm_max=15, lambda_=1.0, mu=1.0, nu=1000.0)
    
    # ✅ λ の自動計算（100 / 全拠点数）
    total_nodes = sim.num_people + sim.num_shelters
    lambda_auto = 100.0 / total_nodes

    # elevation sweep 実行（評価関数に λ を自動設定）(2025/07/23)
    sim.run_elevation_sweep_analysis_std_stationary(
        rm_min=0, rm_max=15,
        lambda_=lambda_auto,
        mu=1.0,
        nu=1000.0
    )





