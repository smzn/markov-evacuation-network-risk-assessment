import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 都市名（日本語）とその英語名の対応
    citynames = [
        '静岡県掛川市',
        '静岡県菊川市',
        '静岡県吉田町',
        '静岡県湖西市',
        '静岡県御前崎市',
        '静岡県牧之原市',
        '静岡県焼津市',
        '静岡県森町',
        '静岡県静岡市葵区',
        '静岡県静岡市駿河区',
        '静岡県静岡市清水区',
        '静岡県川根本町',
        '静岡県袋井市',
        '静岡県島田市',
        '静岡県藤枝市',
        '静岡県磐田市'
    ]

    citynames_eng = [
        'Kakegawa, Shizuoka',
        'Kikugawa, Shizuoka',
        'Yoshida, Shizuoka',
        'Kosai, Shizuoka',
        'Omaezaki, Shizuoka',
        'Makinohara, Shizuoka',
        'Yaizu, Shizuoka',
        'Morimachi, Shizuoka',
        'Aoi Ward, Shizuoka',
        'Suruga Ward, Shizuoka',
        'Shimizu Ward, Shizuoka',
        'Kawanehon, Shizuoka',
        'Fukuroi, Shizuoka',
        'Shimada, Shizuoka',
        'Fujieda, Shizuoka',
        'Iwata, Shizuoka'
    ]


    # 定数
    ALPHA_CONST = 100
    BETA = 1
    GAMMA = 1000
    EXCLUDE_COLS = ['elev', 'client_equivalence_rate']


    def analyze_city(cityname):
        result_dict = {}

        for elev in range(0, 16):
            df_assigned = pd.read_csv(f'../results/{cityname}_node_load≧{elev}melev.csv', index_col=0)

            circle_id = int(df_assigned['circle_id'].max()) + 1 if 'circle_id' in df_assigned.columns and pd.notna(df_assigned['circle_id'].max()) else 0

            non_client_df = df_assigned[df_assigned['type'] != 'client']
            if len(non_client_df) == 0:
                continue

            most_common_class = non_client_df['equivalence_class'].value_counts().idxmax()
            filtered_df = non_client_df[non_client_df['equivalence_class'] == most_common_class]
            std_val = float(filtered_df['stationary_distribution'].std())

            client_eq_rate = len(filtered_df) / len(non_client_df)
            alpha_score = ALPHA_CONST / len(df_assigned) if len(df_assigned) > 0 else 0

            score = (
                df_assigned['equivalence_class'].max() * alpha_score +
                circle_id * BETA +
                std_val * GAMMA
            )

            # グラフに使うスケーリング済の値も保存
            result_dict[elev] = {
                'elev': elev,
                'equivalence_class_scaled': df_assigned['equivalence_class'].max() * alpha_score,
                'circle_id_scaled': circle_id * BETA,
                'stationary_distribution_std_scaled': std_val * GAMMA,
                'score': score,
                'client_equivalence_rate': client_eq_rate,
                'label': f'×λ({alpha_score:.2f}) + ×μ({BETA}) + ×ν({GAMMA})',
                'Evacuees per Shelter' : len(df_assigned[df_assigned['type'] == 'client'])/ len(non_client_df)
            }

        df_result = pd.DataFrame(result_dict).T
        df_result.to_csv(f'../results/{cityname}_results.csv')

        return df_result


    def plot_city_results(city_eng_name, df_result):
        # 描画用カラム（スケーリング済）
        plot_cols = ['equivalence_class_scaled', 'circle_id_scaled', 'stationary_distribution_std_scaled', 'score']

        ax = df_result[plot_cols].plot(figsize=(6, 4))

        handles, _ = ax.get_legend_handles_labels()
        label_info = df_result['label'].iloc[0]
        alpha_label, beta_label, gamma_label = [x.strip() for x in label_info.split('+')]

        # 凡例書き換え
        updated_labels = [
            f'Equivalence classes {alpha_label}',
            f'Covering circles {beta_label}',
            f'Stationary std {gamma_label}',
            'score'
        ]

        ax.legend(handles, updated_labels, loc='upper left')
        ax.set_title(f'Results for {city_eng_name}')
        ax.set_xlabel('Elevation threshold (m)')
        ax.set_ylabel('Value')


    # 実行ループ
    for city_ja, city_en in zip(citynames, citynames_eng):
        try:
            result_df = analyze_city(city_ja)
            plot_city_results(city_en, result_df)

            plt.tight_layout()
            #plt.show()
            plt.savefig(f'../results/{city_ja}_results.png')
        except:
            print(city_ja,'なし')


#pyenv activate py311omaezaki
#python 13_ObjCal.py