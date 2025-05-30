import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 結果ファイルのパスパターン
result_files = glob.glob('locust_results/r_0.8/locust_*_stats.csv')

# データを格納するリスト
all_data = []

# 各ファイルを読み込む
for file in result_files:
    df = pd.read_csv(file)
    # ファイル名から実行番号を取得
    run_number = int(os.path.basename(file).split('_')[1])
    df['run_number'] = run_number
    all_data.append(df)

# すべてのデータを結合
combined_data = pd.concat(all_data)

# エンドポイントごとの平均応答時間の推移
plt.figure(figsize=(15, 8))
for endpoint in combined_data['Name'].unique():
    endpoint_data = combined_data[combined_data['Name'] == endpoint]
    plt.plot(endpoint_data['run_number'], 
             endpoint_data['Average Response Time'], 
             marker='o', 
             label=endpoint)

plt.title('Average Response Time by Endpoint Across Runs')
plt.xlabel('Run Number')
plt.ylabel('Average Response Time (ms)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('response_time_trends.png')

# 統計サマリー
summary = combined_data.groupby('Name').agg({
    'Average Response Time': ['mean', 'std', 'min', 'max'],
    'Median Response Time': ['mean', 'std'],
    '95%': ['mean', 'std']
}).round(2)

print("\nResponse Time Statistics (ms):")
print(summary)

# ヒートマップで95パーセンタイルの応答時間を可視化
pivot_data = combined_data.pivot_table(
    values='95%',
    index='Name',
    columns='run_number'
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('95th Percentile Response Time by Endpoint and Run')
plt.tight_layout()
plt.savefig('response_time_heatmap.png')

# エンドポイントごとの応答時間分布
plt.figure(figsize=(15, 8))
sns.boxplot(data=combined_data, x='Name', y='Average Response Time')
plt.xticks(rotation=45)
plt.title('Response Time Distribution by Endpoint')
plt.tight_layout()
plt.savefig('response_time_distribution.png') 