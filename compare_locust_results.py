import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# 現在のディレクトリを確認
print("Current working directory:", os.getcwd())

# アーキテクチャと実行回数の設定
architectures = ['mono', 'hybrid', 'micro','original']
runs = range(1, 6)  # 1から5までの実行

# 結果を格納する辞書
results = {}

# 各アーキテクチャの各実行結果を読み込む
for arch in architectures:
    arch_data = []
    for run in runs:
        file_path = os.path.join(os.getcwd(), 'locust_results', arch, f'result_{run}_stats.csv')
        print(f"Reading file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['run_number'] = run
            arch_data.append(df)
            print(f"Successfully loaded {arch} run_{run}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    if arch_data:
        results[arch] = pd.concat(arch_data)

# データが存在する場合のみプロットを作成
if results:
    # 1. 平均応答時間の比較（分離した棒グラフ）
    plt.figure(figsize=(20, 8))
    x = np.arange(len(results[architectures[0]]['Name'].unique()))
    width = 0.25
    
    for i, arch in enumerate(architectures):
        data = results[arch]
        avg_response = data.groupby('Name')['Average Response Time'].mean()
        plt.bar(x + i*width, avg_response.values, width, label=arch)
    
    plt.title('Average Response Time Comparison Across Different Architectures')
    plt.xlabel('Endpoint')
    plt.ylabel('Average Response Time (ms)')
    plt.xticks(x + width, avg_response.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('response_time_comparison.png')

    # 2. 95パーセンタイルの比較（分離した棒グラフ）
    plt.figure(figsize=(20, 8))
    for i, arch in enumerate(architectures):
        data = results[arch]
        p95_response = data.groupby('Name')['95%'].mean()
        plt.bar(x + i*width, p95_response.values, width, label=arch)
    
    plt.title('95th Percentile Response Time Comparison')
    plt.xlabel('Endpoint')
    plt.ylabel('95th Percentile Response Time (ms)')
    plt.xticks(x + width, p95_response.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('p95_comparison.png')

    # 3. 統計サマリーの作成
    summary = {}
    for arch, data in results.items():
        summary[arch] = data.groupby('Name').agg({
            'Average Response Time': ['mean', 'std', 'min', 'max'],
            'Median Response Time': ['mean', 'std'],
            '95%': ['mean', 'std']
        }).round(2)

    # 4. ヒートマップで比較
    for metric in ['Average Response Time', '95%']:
        plt.figure(figsize=(12, 8))
        comparison_data = pd.DataFrame()
        
        for arch, data in results.items():
            comparison_data[arch] = data.groupby('Name')[metric].mean()
        
        if not comparison_data.empty:
            sns.heatmap(comparison_data, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title(f'{metric} Comparison Across Architectures')
            plt.tight_layout()
            plt.savefig(f'{metric.lower().replace(" ", "_")}_heatmap.png')

    # 5. 統計サマリーの出力
    print("\nResponse Time Statistics (ms):")
    for arch, stats in summary.items():
        print(f"\nArchitecture = {arch}:")
        print(stats)

    # 6. エンドポイントごとの応答時間分布（分離したボックスプロット）
    plt.figure(figsize=(20, 8))
    for i, arch in enumerate(architectures):
        data = results[arch]
        # Create boxplot with offset for each architecture
        sns.boxplot(data=data, x='Name', y='Average Response Time', 
                   label=arch, width=0.3)
    
    plt.title('Response Time Distribution by Endpoint and Architecture')
    plt.xlabel('Endpoint')
    plt.ylabel('Average Response Time (ms)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('response_time_distribution_comparison.png')

    # 7. 各アーキテクチャの実行ごとの応答時間比較
    for arch in architectures:
        plt.figure(figsize=(20, 8))
        data = results[arch]
        
        # 実行ごとのデータを分離
        for run in runs:
            run_data = data[data['run_number'] == run]
            avg_response = run_data.groupby('Name')['Average Response Time'].mean()
            
            if run == 1:  # 最初の実行でx軸の位置を設定
                x = np.arange(len(avg_response))
                width = 0.15  # 5つの実行を収めるために幅を調整
            
            plt.bar(x + (run-1)*width, avg_response.values, width, 
                   label=f'Run {run}', alpha=0.7)
        
        plt.title(f'Average Response Time by Run - {arch.capitalize()} Architecture')
        plt.xlabel('Endpoint')
        plt.ylabel('Average Response Time (ms)')
        plt.xticks(x + width*2, avg_response.index, rotation=45)  # 中央に合わせてラベルを配置
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'response_time_by_run_{arch}.png')

    # 8. 各アーキテクチャの実行ごとの95パーセンタイル比較
    for arch in architectures:
        plt.figure(figsize=(20, 8))
        data = results[arch]
        
        for run in runs:
            run_data = data[data['run_number'] == run]
            p95_response = run_data.groupby('Name')['95%'].mean()
            
            if run == 1:
                x = np.arange(len(p95_response))
                width = 0.15
            
            plt.bar(x + (run-1)*width, p95_response.values, width, 
                   label=f'Run {run}', alpha=0.7)
        
        plt.title(f'95th Percentile Response Time by Run - {arch.capitalize()} Architecture')
        plt.xlabel('Endpoint')
        plt.ylabel('95th Percentile Response Time (ms)')
        plt.xticks(x + width*2, p95_response.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'p95_by_run_{arch}.png')

    # 9. 各アーキテクチャの実行ごとの統計情報
    for arch in architectures:
        print(f"\nDetailed Statistics for {arch.capitalize()} Architecture:")
        data = results[arch]
        
        for run in runs:
            run_data = data[data['run_number'] == run]
            print(f"\nRun {run}:")
            stats = run_data.groupby('Name').agg({
                'Average Response Time': ['mean', 'std', 'min', 'max'],
                'Median Response Time': ['mean', 'std'],
                '95%': ['mean', 'std']
            }).round(2)
            print(stats)
else:
    print("Error: No data found for any architecture") 