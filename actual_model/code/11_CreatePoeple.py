import pandas as pd
import geopandas as gpd
import os
import numpy as np
import scipy.ndimage
import xml.etree.ElementTree as ET
import folium
import random
from shapely.geometry import Point
from geopy.distance import geodesic
import japanize_matplotlib
import sys
import json

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

        # 初期化ログを表示 / Display initialization log
        print(f"市町村名「{self.municipality_name}」で初期化しました。")
        print(f"Initialized with municipality: {self.municipality_name}")


    def save_state_to_results(self, base_path="../results/meta"):
        """selfの状態をresultsディレクトリに保存する"""
        os.makedirs(base_path, exist_ok=True)

        # gdfをGeoJSONとして保存
        if self.gdf is not None:
            self.gdf.to_file(os.path.join(base_path, f"{self.municipality_name}_gdf.geojson"), driver="GeoJSON")

        # elevation_arrayを保存
        if self.elevation_array is not None:
            np.save(os.path.join(base_path, f"{self.municipality_name}_elevation.npy"), self.elevation_array)

        # df_assignedをCSVで保存
        if self.df_assigned is not None:
            self.df_assigned.to_csv(os.path.join(base_path, f"{self.municipality_name}_assigned_nodes.csv"), index=False, encoding="utf-8-sig")

        # メタデータ保存（緯度経度範囲など）
        meta = {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "municipality_name": self.municipality_name
        }
        with open(os.path.join(base_path, f"{self.municipality_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("状態をresultsフォルダに保存しました。")


    def load_data(self, csv_path="../data/geoshapeCode.csv", base_url="https://geoshape.ex.nii.ac.jp/ka/topojson/2020"):
        """
        TopoJSONファイルを市町村名から動的に取得して読み込む
        Load TopoJSON data based on municipality name using its code.

        :param csv_path: CSVファイルのパス / Path to CSV with municipality and code
        :param base_url: TopoJSONファイルのベースURL
        """
        try:
            # CSVを読み込む / Load CSV with municipality-code mapping
            df = pd.read_csv(csv_path)

            # 市町村名に一致する行を取得 / Find matching row
            match = df[df["municipality"] == self.municipality_name]

            if match.empty:
                print(f"エラー: 市町村名「{self.municipality_name}」がCSVに見つかりません。")
                return

            # code を取得してゼロパディング（5桁）に変換 / Get code and zero-pad to 5 digits
            code_raw = match["code"].values[0]
            code = str(code_raw).zfill(5)

            print(code)
            pref_code = code[:2]  # 都道府県コード（例："22"）

            # ファイル名とローカルパス
            local_dir = '../data'
            filename = f"r2ka{code}.topojson"
            local_path = os.path.join(local_dir, filename)

            # ファイルの存在確認
            if os.path.exists(local_path):
                print(f"ローカルファイルを使用します: {local_path}")
                self.gdf = gpd.read_file(local_path)
            else:
                url = f"{base_url}/{pref_code}/{filename}"
                print(f"TopoJSON URL: {url}")
                print("TopoJSONをダウンロード中...")
                self.gdf = gpd.read_file(url, layer="town")

                # ディレクトリがなければ作成し保存（キャッシュ用）
                os.makedirs(local_dir, exist_ok=True)
                self.gdf.to_file(local_path, driver="GeoJSON")  # 保存形式はGeoJSON
                print(f"TopoJSONをローカルに保存しました: {local_path}")

            # 座標系の設定 / Set CRS (WGS84)
            if self.gdf.crs is None:
                self.gdf.set_crs(epsg=4326, inplace=True)

            # その後、投影変換などを安全に実行できる
            projected = self.gdf.to_crs(epsg=6677)

            # 中心座標を計算（その後WGS84に戻してもOK）
            center = projected.geometry.centroid.union_all().centroid
            center_wgs84 = gpd.GeoSeries([center], crs=6677).to_crs(epsg=4326).geometry[0]

            self.center_lat = center_wgs84.y
            self.center_lon = center_wgs84.x


            print(f"成功: データの読み込みが完了しました。（{self.municipality_name}）")
            print(f"Success: Data loading completed. ({self.municipality_name})")

            print(f"中心座標（Center）: ({self.center_lat:.5f}, {self.center_lon:.5f})")
            print(f"Centroid (latitude, longitude): ({self.center_lat:.5f}, {self.center_lon:.5f})")


        except Exception as e:
            print(f"エラー: データの読み込みに失敗しました - {e}")
            print(f"Error: Failed to load data - {e}")


    def create_grayscale_map(self, save_path="map.html"):
        """
        グレースケール背景に境界線を描いたFolium地図を作成してHTML保存

        :param save_path: 保存するHTMLファイルのパス
        """
        if self.gdf is None:
            print("GeoDataFrameが存在しません。先にload_data()を呼んでください。")
            return

        # グレースケールタイルの地図作成（CartoDB Positronがグレースケール）
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=11,
            tiles="CartoDB positron"
        )

        # 境界線（行政界）をGeoJSONで追加
        folium.GeoJson(
            self.gdf,
            name="Boundary",
            style_function=lambda x: {
                "fillColor": "gray",  # グレースケール
                "color": "black",     # 境界線色
                "weight": 1,
                "fillOpacity": 0.2
            }
        ).add_to(m)

        # 凡例追加など必要ならここに

        # HTMLファイルに保存
        m.save(save_path)
        print(f"地図を {save_path} に保存しました。")








    
    def merge_gml_elevation_tiles_10m(self, input_dir, output_prefix):
        """
        GMLタイルを結合し、補間と保存を行う統合関数
        Merge GML elevation tiles, interpolate missing values, and save outputs
        """
        results = self._load_all_tiles(input_dir)
        if not results:
            print("有効なデータが見つかりませんでした。 / No valid data found.")
            return None

        merged_array, bounds = self._merge_tiles(results)
        merged_array = self._interpolate_missing(merged_array)

        self._save_outputs(merged_array, bounds, output_prefix)

        return merged_array, *bounds

    def _load_all_tiles(self, input_dir):
        """
        GMLタイルをすべて読み込む / Load all GML tiles from directory
        """
        results = []
        for filename in os.listdir(input_dir):
            if filename.endswith(".xml"):
                filepath = os.path.join(input_dir, filename)
                result = self._parse_gml_dem_10m(filepath)
                if result:
                    results.append(result)

        print(f"タイル読み込み完了: {len(results)} タイル / Loaded {len(results)} tiles")
        return results

    def _merge_tiles(self, results):
        """
        GMLタイルを1つの配列に統合 / Merge tiles into one elevation array
        """
        sample = results[0]
        tile_rows, tile_cols = sample["elevations"].shape
        lat_res = (sample["lat_range"][1] - sample["lat_range"][0]) / tile_rows
        lon_res = (sample["lon_range"][1] - sample["lon_range"][0]) / tile_cols

        lat_min = min(r["lat_range"][0] for r in results)
        lat_max = max(r["lat_range"][1] for r in results)
        lon_min = min(r["lon_range"][0] for r in results)
        lon_max = max(r["lon_range"][1] for r in results)

        total_rows = int(round((lat_max - lat_min) / lat_res))
        total_cols = int(round((lon_max - lon_min) / lon_res))
        merged_array = np.full((total_rows, total_cols), np.nan)

        for r in results:
            elev = r["elevations"]
            row_start = int(round((lat_max - r["lat_range"][1]) / lat_res))
            col_start = int(round((r["lon_range"][0] - lon_min) / lon_res))
            row_end = row_start + elev.shape[0]
            col_end = col_start + elev.shape[1]
            merged_array[row_start:row_end, col_start:col_end] = elev

        print(f"統合完了！ shape: {merged_array.shape} / Merge complete!")
        return merged_array, (lat_min, lat_max, lon_min, lon_max)

    def _interpolate_missing(self, array):
        """
        NaN を最小値 + 平滑化で補間 / Interpolate NaNs using min-fill + smoothing
        """
        if np.any(np.isnan(array)):
            print("NaN を一括補間中... / Interpolating NaN values...")
            filled = np.nan_to_num(array, nan=np.nanmin(array))
            smoothed = scipy.ndimage.gaussian_filter(filled, sigma=1)
            array[np.isnan(array)] = smoothed[np.isnan(array)]
            print("補間完了 / Interpolation complete")
        return array

    def _save_outputs(self, array, bounds, prefix):
        """
        出力ファイルを保存 / Save elevation array and bounds
        """
        lat_min, lat_max, lon_min, lon_max = bounds

        np.save(f"{prefix}.npy", array)
        np.savetxt(f"{prefix}.csv", array, delimiter=",", fmt="%.2f")
        self.elevation_array = array
        print(f"統合配列を保存しました: {prefix}.npy / .csv")

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        print("緯度経度範囲をインスタンス変数として保存しました。 / Saved coordinate range to instance attributes")


    def _parse_gml_dem_10m(self, xml_file):
        """
        GML標高データ（10mメッシュ）を読み込んで numpy 配列として返す
        Parse GML elevation data (10m mesh) into a structured dictionary (no interpolation).

        :param xml_file: XMLファイルのパス / Path to the GML (XML) file
        :return: dict with elevation array and lat/lon bounds, or None on failure
        """
        ns = {'gml': "http://www.opengis.net/gml/3.2"}

        try:
            # XMLツリーの読み込み / Parse XML tree
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 緯度経度の範囲を取得 / Extract coordinate bounds
            lower_corner = root.find('.//gml:Envelope/gml:lowerCorner', ns).text.split()
            upper_corner = root.find('.//gml:Envelope/gml:upperCorner', ns).text.split()
            lat_min, lon_min = map(float, lower_corner)
            lat_max, lon_max = map(float, upper_corner)

            # グリッドのサイズ（X, Y）を取得 / Get grid size
            grid_size = root.find('.//gml:GridEnvelope/gml:high', ns).text.split()
            grid_x, grid_y = map(int, grid_size)  # Note: +1 due to grid indexing
            expected_size = (grid_x + 1) * (grid_y + 1)

            # 標高値の取得 / Extract elevation values
            tuple_list = root.find('.//gml:tupleList', ns)
            if tuple_list is None or not tuple_list.text:
                raise ValueError("標高データが見つかりません / Elevation data missing")

            elevation_values = tuple_list.text.strip().split("\n")
            elevations = np.array([float(e.split(",")[1]) for e in elevation_values])
            elevations[elevations == -9999.0] = np.nan  # -9999 を NaN に変換 / Replace no-data with NaN

            # データ数が想定と異なる場合は補正 / Adjust if size mismatch
            actual_size = elevations.size
            if actual_size < expected_size:
                elevations = np.pad(elevations, (0, expected_size - actual_size), mode='edge')
            elif actual_size > expected_size:
                elevations = elevations[:expected_size]

            # 緯度 × 経度の2D配列に変換 / Reshape to 2D array (lat, lon)
            elevations = elevations.reshape((grid_y + 1, grid_x + 1))

            # 結果を辞書で返す / Return as dictionary
            return {
                "filename": os.path.basename(xml_file),
                "elevations": elevations,
                "lat_range": (lat_min, lat_max),
                "lon_range": (lon_min, lon_max)
            }

        except Exception as e:
            print(f"エラー: {xml_file} の読み込みに失敗しました - {e}")
            print(f"Error: Failed to parse {xml_file} - {e}")
            return None





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




    def assign_support_needs(self, total_support_needs=0.01):
        """
        総要支援者人数を地区の人口比率に基づいて割り振る  
        Distribute total number of support-needing individuals proportionally based on district population.

        :param total_support_needs: 全体の要支援者の総数  
                                    Total number of people who require support.
        """
        # 「JINKO」列が存在するか確認  
        # Check if "JINKO" (population) column exists
        if "JINKO" not in self.gdf.columns:
            print("エラー: 人口（JINKO）データが存在しません。 / Error: 'JINKO' (population) column not found.")
            return

        # 総人口を計算 / Calculate total population
        total_population = self.gdf["JINKO"].sum()

        # 各地区の人口に基づいて要支援者数を割り当て  
        # Assign support-needing individuals proportionally based on population
        self.gdf["SUPPORT_NEEDS"] = (
            self.gdf["JINKO"] * total_support_needs#割合で今回生成する人数を決める
        ).round().astype(int)

        print("要支援者人数の割り当てが完了しました。 / Support needs assignment complete.")




    def load_shelters(self, csv_file, city_office_info=None):
        """
        避難所データのCSVファイルを読み込み、データフレームとして保存
        :param csv_file: 避難所情報を含むCSVファイルのパス
        """
        try:
            self.shelters_df = pd.read_csv(csv_file)

            #かき、政令指定都市に対応できるように修正2025/07/23
            filtered_df = self.shelters_df[self.shelters_df['都道府県名及び市町村名'] == self.municipality_name]
            # 完全一致がない場合は、住所の始めで一致する行を取得
            if filtered_df.empty:
                filtered_df = self.shelters_df[
                    self.shelters_df['住所'].str.startswith(self.municipality_name)
                ]
            self.shelters_df = filtered_df.copy()
            
            #self.shelters_df = self.shelters_df[self.shelters_df['都道府県名及び市町村名'] == self.municipality_name]

            # 市役所の情報が提供された場合に追加
            if city_office_info:
                city_office_df = pd.DataFrame([city_office_info])
                self.shelters_df = pd.concat([self.shelters_df, city_office_df], ignore_index=True)

            print(f"避難所データが正常にロードされました。対象避難所数: {len(self.shelters_df)}")
        except Exception as e:
            print(f"エラー: 避難所データの読み込みに失敗しました - {e}")


    def assign_random_support_needs(self, output_csv_path, min_distance_m=50, min_elevation_m=5.0):
        """ 各地区の要支援者をランダムに割り当て、位置情報と標高データを設定して保存 """
        if self.gdf is None:
            print("エラー: データをロードしてください。")
            return

        assigned_data = []

        # 1. 市役所データを先に追加
        id_counter = 0  # idのカウンタを0から開始
        for _, row in self.shelters_df.iterrows():
            if row['備考'] == '市役所':
                entry_type = 'city_hall'
                elevation = self.get_elevation_from_latlon(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': (0 if pd.isna(row['指定緊急避難場所との住所同一']) else row['指定緊急避難場所との住所同一']) + 1,
                    'name': row['施設・場所名'],
                })
                id_counter += 1  # idをインクリメント

        # 2. 一般の避難所データを追加
        for _, row in self.shelters_df.iterrows():
            if row['備考'] != '市役所':
                entry_type = 'shelter'
                elevation = self.get_elevation_from_latlon(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': (0 if pd.isna(row['指定緊急避難場所との住所同一']) else row['指定緊急避難場所との住所同一']) + 1,
                    'name': row['施設・場所名'],
                })
                id_counter += 1  # idをインクリメント


        # 3. 要配慮者データを追加（制約緩和付き）
        for _, row in self.gdf.iterrows():
            support_needs = row['SUPPORT_NEEDS']
            polygon = row['geometry']
            for i in range(support_needs):
                current_min_distance = min_distance_m
                current_min_elevation = min_elevation_m

                while True:
                    success = False
                    for attempt in range(50):
                        minx, miny, maxx, maxy = polygon.bounds
                        lon = random.uniform(minx, maxx)
                        lat = random.uniform(miny, maxy)
                        random_point = Point(lon, lat)

                        if not polygon.contains(random_point):
                            continue

                        too_close = False
                        for shelter_row in self.shelters_df.itertuples():
                            dist = geodesic((lat, lon), (shelter_row.緯度, shelter_row.経度)).meters
                            if dist < current_min_distance:
                                too_close = True
                                break
                        if too_close:
                            continue

                        elevation = self.get_elevation_from_latlon(lat, lon)
                        if elevation is None or elevation < current_min_elevation:
                            continue

                        # 成功したら保存へ
                        if (current_min_elevation != min_elevation_m) or (current_min_distance != min_distance_m):
                            print(f"選定成功: lat={lat:.6f}, lon={lon:.6f}, elevation={elevation:.2f}, "
                                f"距離制約={current_min_distance:.2f}m, 高度制約={current_min_elevation:.2f}m")
                        success = True
                        break

                    if success:
                        break  # 条件を満たす点が見つかった

                    # 条件を少し緩める
                    current_min_distance = max(0, current_min_distance - 0.1)
                    current_min_elevation = max(0, current_min_elevation - 0.1)

                assigned_data.append({
                    'id': id_counter,
                    'type': 'client',
                    'x': lon,
                    'y': lat,
                    'z': elevation,
                    'demand': random.choice([1, 2]),
                    'priority': random.randint(1, 5),
                    'name': f"{row['S_NAME']}_{i+1}",
                })
                id_counter += 1
        """
        # 3. 要配慮者データを追加（修正版）
        for _, row in self.gdf.iterrows():
            support_needs = row['SUPPORT_NEEDS']
            polygon = row['geometry']  # シェープ情報
            for i in range(support_needs):
                while True:
                    # ポリゴン内にランダムポイントを生成
                    minx, miny, maxx, maxy = polygon.bounds
                    lon = random.uniform(minx, maxx)
                    lat = random.uniform(miny, maxy)
                    random_point = Point(lon, lat)

                    # ポイントがポリゴン内にあるかを確認
                    if not polygon.contains(random_point):
                        continue

                    # 追加制約ここから
                    too_close = False
                    for shelter_row in self.shelters_df.itertuples():
                        dist = geodesic((lat, lon), (shelter_row.緯度, shelter_row.経度)).meters
                        if dist < min_distance_m:  # xmメートル未満の近接
                            too_close = True
                            break
                    if too_close:
                        continue

                    elevation = self.get_elevation_from_latlon(lat, lon)
                    if elevation is None or elevation < min_elevation_m:#高度が一定以上であればOK
                        continue
                    # 追加制約ここまで
                    break

                # 有効な座標と標高データで追加
                assigned_data.append({
                    'id': id_counter,
                    'type': 'client',
                    'x': lon,
                    'y': lat,
                    'z': elevation,
                    'demand': random.choice([1, 2]),
                    'priority': random.randint(1, 5),
                    'name': f"{row['S_NAME']}_{i+1}",
                })
                id_counter += 1  # idをインクリメント
        """
        # データをDataFrameに変換してCSV保存
        self.df_assigned = pd.DataFrame(assigned_data)
        self.df_assigned.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"要支援者データと避難所情報が {output_csv_path} に保存されました。")




        

# このスクリプトが直接実行されたときに main() を呼び出す
# Call main() only if this script is run directly
if __name__ == "__main__":
    # 都道府県+市町村名を指定（例："静岡県御前崎市"）
    # Specify municipality name (e.g., "静岡県御前崎市") in Japanese
    municipality = sys.argv[1]
    print(f"＊＊＊＊{municipality}＊＊＊＊")

    # CreatePeople クラスのインスタンスを作成 / Create an instance of CreatePeople
    geo = CreatePeople(municipality)
    print()

    # TopoJSON を読み込み（geoshapeCode.csv に対応コードが必要）
    # Load the TopoJSON data using municipality-to-code mapping from CSV
    geo.load_data()
    geo.create_grayscale_map(save_path=f"../results/{municipality}.html")
    print()

    # 高度タイルの統合・補間・保存（1回だけでOK）
    input_dir = f"../data/{municipality}"
    elevation_array, lat_min, lat_max, lon_min, lon_max = geo.merge_gml_elevation_tiles_10m(
        input_dir,
        output_prefix=f"../results/{municipality}elev"
    )
    print('高度情報のcsvおよびnpy保存完了 / Elevation data saved.')
    print()

    #要支援者ノード作成
    geo.assign_support_needs(total_support_needs=0.01)#(1=100%)
    print()

    #避難場所割り当て
    geo.load_shelters("../data/mergeFromCity_1.csv", city_office_info={
        "施設・場所名": sys.argv[2], "緯度": float(sys.argv[3]), "経度": float(sys.argv[4]), "備考": "市役所"
    })
    geo.assign_random_support_needs(f"../results/{municipality}_node.csv", min_distance_m=50, min_elevation_m=2.0)#ここのパラメタはいじる必要あり
    print('要支援者と避難所のノード生成完了')
    print()

    #メタデータの保存
    geo.save_state_to_results( base_path="../results/meta")
    print("メタデータ保存")







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


# python 11_CreatePoeple.py 静岡県御前崎市 御前崎市役所 34.63816733144997 138.12803209464337 > ../results/11_CreatePoeple_静岡県御前崎市.txt
# python 11_CreatePoeple.py 静岡県森町 森町役場 34.83575831770259 137.9271716983627 > ../results/11_CreatePoeple_静岡県森町.txt
# python 11_CreatePoeple.py 静岡県掛川市 掛川市役所 34.76905473830715 137.9988511216999 > ../results/11_CreatePoeple_静岡県掛川市.txt
# python 11_CreatePoeple.py 静岡県菊川市 菊川市役所 34.75799906370183 138.08473578358598 > ../results/11_CreatePoeple_静岡県菊川市.txt
# python 11_CreatePoeple.py 静岡県袋井市 袋井市役所 34.75036426230184 137.92444669572538 > ../results/11_CreatePoeple_静岡県袋井市.txt
# python 11_CreatePoeple.py 静岡県磐田市 磐田市役所 34.718228612494194 137.85148963940244 > ../results/11_CreatePoeple_静岡県磐田市.txt

#pyenv deactivate
#pyenv activate py311omaezaki

