#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ファイルパス設定
BASE_DIR = "locust_results/users"
APPROACHES = ["propose", "random"]
OUT_DIR = "locust_results/figures/experiment_trends"

def read_stats_csv(file_path):
    """
    CSVファイルを読み込み、エンドポイント別の統計を抽出
    """
    try:
        df = pd.read_csv(file_path)
        # Aggregated行を除外してエンドポイント別データのみ取得
        endpoint_df = df[df['Name'] != 'Aggregated'].copy()
        aggregated_row = df[df['Name'] == 'Aggregated'].iloc[0] if len(df[df['Name'] == 'Aggregated']) > 0 else None
        
        return endpoint_df, aggregated_row
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def calculate_success_rate(row):
    """
    成功率を計算
    """
    request_count = row['Request Count']
    failure_count = row['Failure Count']
    if request_count > 0:
        return ((request_count - failure_count) / request_count) * 100
    return 0

def find_experiment_data():
    """
    全Kill率とユーザー数の実験データを収集
    """
    all_data = []
    
    for approach in APPROACHES:
        approach_dir = os.path.join(BASE_DIR, approach)
        if not os.path.exists(approach_dir):
            continue
            
        # Kill率ディレクトリを検索
        for kill_dir in glob.glob(os.path.join(approach_dir, "kill_*")):
            kill_rate = os.path.basename(kill_dir)
            
            # ユーザー数ディレクトリを検索
            for user_dir in glob.glob(os.path.join(kill_dir, "*")):
                if not os.path.isdir(user_dir):
                    continue
                user_name = os.path.basename(user_dir)
                if not user_name.isdigit():
                    continue
                user_count = int(user_name)
                
                # Hybridディレクトリを検索
                hybrid_dir = os.path.join(user_dir, "Hybrid")
                if not os.path.exists(hybrid_dir):
                    continue
                
                # 3回の実験ファイルを検索
                for run_num in [1, 2, 3]:
                    stats_file = os.path.join(hybrid_dir, f"locust_Hybrid_run_{run_num}_stats.csv")
                    if os.path.exists(stats_file):
                        endpoint_df, aggregated_row = read_stats_csv(stats_file)
                        if aggregated_row is not None:
                            aggregated_row['approach'] = approach
                            aggregated_row['kill_rate'] = kill_rate
                            aggregated_row['user_count'] = user_count
                            aggregated_row['run_number'] = run_num
                            aggregated_row['success_rate'] = calculate_success_rate(aggregated_row)
                            all_data.append(aggregated_row)
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def plot_trends_by_kill_rate(df):
    """
    Kill率ごとに推移をプロット
    """
    if df.empty:
        print("No data found")
        return
    
    # Kill率ごとにグループ化
    for kill_rate in sorted(df['kill_rate'].unique()):
        kill_df = df[df['kill_rate'] == kill_rate]
        if kill_df.empty:
            continue
            
        print(f"Processing kill rate: {kill_rate}")
        
        # ユーザー数ごとにサブプロットを作成
        user_counts = sorted(kill_df['user_count'].unique())
        n_users = len(user_counts)
        
        if n_users == 0:
            continue
            
        # 応答時間の推移
        fig, axes = plt.subplots(2, n_users, figsize=(5*n_users, 10))
        if n_users == 1:
            axes = axes.reshape(2, 1)
        
        for i, user_count in enumerate(user_counts):
            user_df = kill_df[kill_df['user_count'] == user_count].sort_values('run_number')
            
            # 応答時間
            ax1 = axes[0, i]
            for approach in APPROACHES:
                approach_data = user_df[user_df['approach'] == approach]
                if not approach_data.empty:
                    ax1.plot(approach_data['run_number'], approach_data['Average Response Time'], 
                            marker='o', linewidth=2, markersize=8, label=approach.capitalize())
            
            ax1.set_title(f'Response Time\n{kill_rate}, {user_count} users', fontsize=12)
            ax1.set_xlabel('Experiment Run')
            ax1.set_ylabel('Response Time (ms)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xticks([1, 2, 3])
            
            # 成功率
            ax2 = axes[1, i]
            for approach in APPROACHES:
                approach_data = user_df[user_df['approach'] == approach]
                if not approach_data.empty:
                    ax2.plot(approach_data['run_number'], approach_data['success_rate'], 
                            marker='s', linewidth=2, markersize=8, label=approach.capitalize())
            
            ax2.set_title(f'Success Rate\n{kill_rate}, {user_count} users', fontsize=12)
            ax2.set_xlabel('Experiment Run')
            ax2.set_ylabel('Success Rate (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_xticks([1, 2, 3])
            ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'trends_{kill_rate}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

def plot_comparison_summary(df):
    """
    全データの比較サマリーをプロット
    """
    if df.empty:
        return
    
    # Kill率ごとの平均値を計算
    summary_data = []
    for kill_rate in df['kill_rate'].unique():
        for user_count in df[df['kill_rate'] == kill_rate]['user_count'].unique():
            for approach in APPROACHES:
                data = df[(df['kill_rate'] == kill_rate) & 
                         (df['user_count'] == user_count) & 
                         (df['approach'] == approach)]
                if not data.empty:
                    summary_data.append({
                        'kill_rate': kill_rate,
                        'user_count': user_count,
                        'approach': approach,
                        'avg_response_time': data['Average Response Time'].mean(),
                        'avg_success_rate': data['success_rate'].mean(),
                        'avg_rps': data['Requests/s'].mean()
                    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 比較グラフ
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 応答時間比較
    ax1 = axes[0, 0]
    for approach in APPROACHES:
        approach_data = summary_df[summary_df['approach'] == approach]
        if not approach_data.empty:
            for kill_rate in sorted(approach_data['kill_rate'].unique()):
                kill_data = approach_data[approach_data['kill_rate'] == kill_rate].sort_values('user_count')
                ax1.plot(kill_data['user_count'], kill_data['avg_response_time'], 
                        marker='o', linewidth=2, markersize=8, 
                        label=f'{approach.capitalize()} ({kill_rate})')
    
    ax1.set_title('Average Response Time by Kill Rate and Users', fontsize=14)
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Average Response Time (ms)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 成功率比較
    ax2 = axes[0, 1]
    for approach in APPROACHES:
        approach_data = summary_df[summary_df['approach'] == approach]
        if not approach_data.empty:
            for kill_rate in sorted(approach_data['kill_rate'].unique()):
                kill_data = approach_data[approach_data['kill_rate'] == kill_rate].sort_values('user_count')
                ax2.plot(kill_data['user_count'], kill_data['avg_success_rate'], 
                        marker='s', linewidth=2, markersize=8, 
                        label=f'{approach.capitalize()} ({kill_rate})')
    
    ax2.set_title('Average Success Rate by Kill Rate and Users', fontsize=14)
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Average Success Rate (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    
    # RPS比較
    ax3 = axes[1, 0]
    for approach in APPROACHES:
        approach_data = summary_df[summary_df['approach'] == approach]
        if not approach_data.empty:
            for kill_rate in sorted(approach_data['kill_rate'].unique()):
                kill_data = approach_data[approach_data['kill_rate'] == kill_rate].sort_values('user_count')
                ax3.plot(kill_data['user_count'], kill_data['avg_rps'], 
                        marker='^', linewidth=2, markersize=8, 
                        label=f'{approach.capitalize()} ({kill_rate})')
    
    ax3.set_title('Average RPS by Kill Rate and Users', fontsize=14)
    ax3.set_xlabel('Number of Users')
    ax3.set_ylabel('Average RPS')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # サマリーテーブル
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # テーブルデータを作成
    table_data = []
    for kill_rate in sorted(summary_df['kill_rate'].unique()):
        for user_count in sorted(summary_df[summary_df['kill_rate'] == kill_rate]['user_count'].unique()):
            row_data = [f"{kill_rate}\n{user_count} users"]
            for approach in APPROACHES:
                data = summary_df[(summary_df['kill_rate'] == kill_rate) & 
                                 (summary_df['user_count'] == user_count) & 
                                 (summary_df['approach'] == approach)]
                if not data.empty:
                    row_data.append(f"{data['avg_response_time'].iloc[0]:.1f}ms\n{data['avg_success_rate'].iloc[0]:.1f}%")
                else:
                    row_data.append("N/A")
            table_data.append(row_data)
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Config', 'Propose', 'Random'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Summary Table', fontsize=14)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'comparison_summary.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def print_summary_table(df):
    """
    サマリーテーブルを表示
    """
    if df.empty:
        print("No data to summarize")
        return
    
    print("\n" + "="*100)
    print("EXPERIMENT TRENDS SUMMARY")
    print("="*100)
    
    for kill_rate in sorted(df['kill_rate'].unique()):
        print(f"\nKill Rate: {kill_rate}")
        print("-" * 80)
        
        for user_count in sorted(df[df['kill_rate'] == kill_rate]['user_count'].unique()):
            print(f"\nUsers: {user_count}")
            user_data = df[(df['kill_rate'] == kill_rate) & (df['user_count'] == user_count)]
            
            for approach in APPROACHES:
                approach_data = user_data[user_data['approach'] == approach].sort_values('run_number')
                if not approach_data.empty:
                    print(f"\n  {approach.capitalize()}:")
                    for _, row in approach_data.iterrows():
                        print(f"    Run {row['run_number']}: RT={row['Average Response Time']:.1f}ms, "
                              f"Success={row['success_rate']:.2f}%, RPS={row['Requests/s']:.2f}")

def main():
    """
    メイン処理
    """
    print("Experiment Trends Analysis (All Kill Rates and Users)")
    print("=" * 60)
    print(f"Input directory: {BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print("-" * 60)
    
    # 出力ディレクトリ作成
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # データ収集
    print("Collecting experiment data...")
    df = find_experiment_data()
    
    if df.empty:
        print("No experiment data found")
        return
    
    print(f"Found data for {len(df)} experiments")
    print(f"Kill rates: {sorted(df['kill_rate'].unique())}")
    print(f"User counts: {sorted(df['user_count'].unique())}")
    
    # 推移グラフ作成
    print("\nCreating trend plots...")
    plot_trends_by_kill_rate(df)
    
    # 比較サマリー作成
    print("\nCreating comparison summary...")
    plot_comparison_summary(df)
    
    # サマリーテーブル表示
    print_summary_table(df)
    
    print(f"\nAnalysis complete! Check the '{OUT_DIR}' directory for output images.")

if __name__ == "__main__":
    main()
