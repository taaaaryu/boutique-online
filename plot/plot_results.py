#!/usr/bin/env python3
"""
Locust結果の可視化スクリプト
各ユーザー数、実装形態ごとの平均成功率と遅延時間をプロット
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
LOAD_PATH = "locust_results/users/diffrent_svc_avail" #csvファイルのパス
PATH = "locust_results/users/result_figures/diffrent_svc_avail" #結果の保存先
USER_COUNTS = [2000, 2500, 3000]

def load_request_logs(base_dir=LOAD_PATH):
    """
    各ユーザー数、実装形態のrequest_log.csvファイルを読み込んでレスポンス時間分布を分析
    """
    success_distribution_data = []
    failure_distribution_data = []
    
    # ユーザー数のリスト
    user_counts = USER_COUNTS
    # 実装形態のリスト
    architectures = ['Mono', 'Hybrid', 'Micro']
    
    for user_count in user_counts:
        for arch in architectures:
            # 各実装形態のディレクトリパス
            arch_dir = Path(base_dir) / str(user_count) / arch
            
            # request_log.csvファイルを検索
            log_files = list(arch_dir.glob("request_log_*.csv"))
            
            if log_files:
                success_response_times = []
                failure_response_times = []
                
                for log_file in log_files:
                    try:
                        df = pd.read_csv(log_file)
                        # 成功したリクエストを抽出
                        success_requests = df[df['status'] == 'SUCCESS']
                        if not success_requests.empty:
                            success_response_times.extend(success_requests['response_time'].tolist())
                        
                        # 失敗したリクエストを抽出
                        failure_requests = df[df['status'] == 'FAILURE']
                        if not failure_requests.empty:
                            failure_response_times.extend(failure_requests['response_time'].tolist())
                            
                    except Exception as e:
                        print(f"Error reading {log_file}: {e}")
                
                # 成功データの処理
                if success_response_times:
                    response_times = np.array(success_response_times)
                    success_distribution_data.append({
                        'user_count': user_count,
                        'architecture': arch,
                        'response_times': response_times,
                        'mean_response_time': np.mean(response_times),
                        'median_response_time': np.median(response_times),
                        'std_response_time': np.std(response_times),
                        'p95_response_time': np.percentile(response_times, 95),
                        'p99_response_time': np.percentile(response_times, 99),
                        'total_requests': len(response_times),
                        'status': 'SUCCESS'
                    })
                
                # 失敗データの処理
                if failure_response_times:
                    response_times = np.array(failure_response_times)
                    failure_distribution_data.append({
                        'user_count': user_count,
                        'architecture': arch,
                        'response_times': response_times,
                        'mean_response_time': np.mean(response_times),
                        'median_response_time': np.median(response_times),
                        'std_response_time': np.std(response_times),
                        'p95_response_time': np.percentile(response_times, 95),
                        'p99_response_time': np.percentile(response_times, 99),
                        'total_requests': len(response_times),
                        'status': 'FAILURE'
                    })
    
    return success_distribution_data, failure_distribution_data

def plot_response_time_distributions(distribution_data):
    """
    レスポンス時間分布をプロット
    """
    if not distribution_data:
        print("No distribution data found.")
        return
    
    # アーキテクチャのリストを定義
    architectures = ['Mono', 'Hybrid', 'Micro']
    
    # 全データからX軸とY軸の範囲を決定
    all_response_times = []
    all_densities = []
    
    # まず全データを収集してX軸範囲を決定
    for data in distribution_data:
        all_response_times.extend(data['response_times'])
    
    x_min = min(all_response_times)
    x_max = max(all_response_times)
    
    # 各アーキテクチャでヒストグラムの密度を計算してY軸範囲を決定
    for arch in architectures:
        arch_data = [d for d in distribution_data if d['architecture'] == arch]
        for data in arch_data:
            response_times = data['response_times']
            # ヒストグラムの密度を計算
            hist, bin_edges = np.histogram(response_times, bins=50, density=True)
            all_densities.extend(hist)
    
    y_min = 0
    y_max = max(all_densities) * 1.1  # 最大値より少し余裕を持たせる
    
    # アーキテクチャごとにサブプロットを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    
    for i, arch in enumerate(architectures):
        ax = axes[i]
        arch_data = [d for d in distribution_data if d['architecture'] == arch]
        
        for data in arch_data:
            user_count = data['user_count']
            response_times = data['response_times']
            
            # ヒストグラムをプロット
            ax.hist(response_times, bins=50, alpha=0.6, 
                   label=f'{user_count} users', density=True)
        
        # X軸とY軸の範囲を統一
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title(f'{arch} - Response Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/response_time_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_percentile_comparison(distribution_data):
    """
    パーセンタイル比較をプロット
    """
    if not distribution_data:
        return
    
    # ユーザー数ごとにサブプロットを作成
    user_counts = sorted(list(set([d['user_count'] for d in distribution_data])))
    fig, axes = plt.subplots(1, len(user_counts), figsize=(6*len(user_counts), 6))
    
    if len(user_counts) == 1:
        axes = [axes]
    
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    markers = {'Mono': 'o', 'Hybrid': 's', 'Micro': '^'}
    
    for i, user_count in enumerate(user_counts):
        ax = axes[i]
        user_data = [d for d in distribution_data if d['user_count'] == user_count]
        
        architectures = []
        p95_values = []
        p99_values = []
        
        for data in user_data:
            architectures.append(data['architecture'])
            p95_values.append(data['p95_response_time'])
            p99_values.append(data['p99_response_time'])
        
        x = np.arange(len(architectures))
        width = 0.35
        
        ax.bar(x - width/2, p95_values, width, label='95th Percentile', alpha=0.8)
        ax.bar(x + width/2, p99_values, width, label='99th Percentile', alpha=0.8)
        
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title(f'Response Time Percentiles - {user_count} Users')
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/response_time_percentiles.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_architecture_comparison_by_user_count(distribution_data):
    """
    同じユーザー数で各実装形態のレスポンス時間分布を比較
    """
    if not distribution_data:
        return
    
    # ユーザー数ごとにサブプロットを作成
    user_counts = sorted(list(set([d['user_count'] for d in distribution_data])))
    fig, axes = plt.subplots(1, len(user_counts), figsize=(6*len(user_counts), 6))
    
    if len(user_counts) == 1:
        axes = [axes]
    
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    
    for i, user_count in enumerate(user_counts):
        ax = axes[i]
        user_data = [d for d in distribution_data if d['user_count'] == user_count]
        
        # 各アーキテクチャのデータをプロット
        for data in user_data:
            arch = data['architecture']
            response_times = data['response_times']
            
            # ヒストグラムをプロット
            ax.hist(response_times, bins=50, alpha=0.6, 
                   color=colors[arch], label=arch, density=True)
        
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title(f'Response Time Distribution - {user_count} Users')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/architecture_comparison_by_user_count.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_failure_response_time_distributions(failure_distribution_data):
    """
    失敗リクエストのレスポンス時間分布をプロット
    """
    if not failure_distribution_data:
        print("No failure distribution data found.")
        return
    
    # アーキテクチャのリストを定義
    architectures = ['Mono', 'Hybrid', 'Micro']
    
    # 全データからX軸とY軸の範囲を決定
    all_response_times = []
    all_densities = []
    
    # まず全データを収集してX軸範囲を決定
    for data in failure_distribution_data:
        all_response_times.extend(data['response_times'])
    
    if not all_response_times:
        print("No failure response time data found.")
        return
    
    x_min = min(all_response_times)
    x_max = max(all_response_times)
    
    # 各アーキテクチャでヒストグラムの密度を計算してY軸範囲を決定
    for arch in architectures:
        arch_data = [d for d in failure_distribution_data if d['architecture'] == arch]
        for data in arch_data:
            response_times = data['response_times']
            # ヒストグラムの密度を計算
            hist, bin_edges = np.histogram(response_times, bins=50, density=True)
            all_densities.extend(hist)
    
    y_min = 0
    y_max = max(all_densities) * 1.1 if all_densities else 1.0
    
    # アーキテクチャごとにサブプロットを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    
    for i, arch in enumerate(architectures):
        ax = axes[i]
        arch_data = [d for d in failure_distribution_data if d['architecture'] == arch]
        
        for data in arch_data:
            user_count = data['user_count']
            response_times = data['response_times']
            
            # ヒストグラムをプロット
            ax.hist(response_times, bins=50, alpha=0.6, 
                   label=f'{user_count} users', density=True)
        
        # X軸とY軸の範囲を統一
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title(f'{arch} - FAILURE Response Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/failure_response_time_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_failure_architecture_comparison_by_user_count(failure_distribution_data):
    """
    同じユーザー数で各実装形態の失敗レスポンス時間分布を比較
    """
    if not failure_distribution_data:
        return
    
    # ユーザー数ごとにサブプロットを作成
    user_counts = sorted(list(set([d['user_count'] for d in failure_distribution_data])))
    fig, axes = plt.subplots(1, len(user_counts), figsize=(6*len(user_counts), 6))
    
    if len(user_counts) == 1:
        axes = [axes]
    
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    
    for i, user_count in enumerate(user_counts):
        ax = axes[i]
        user_data = [d for d in failure_distribution_data if d['user_count'] == user_count]
        
        # 各アーキテクチャのデータをプロット
        for data in user_data:
            arch = data['architecture']
            response_times = data['response_times']
            
            # ヒストグラムをプロット
            ax.hist(response_times, bins=50, alpha=0.6, 
                   color=colors[arch], label=arch, density=True)
        
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title(f'FAILURE Response Time Distribution - {user_count} Users')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/failure_architecture_comparison_by_user_count.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_distribution_summary(distribution_data):
    """
    分布分析のサマリーを表示
    """
    print("=== Response Time Distribution Summary ===")
    print()
    
    for user_count in sorted(list(set([d['user_count'] for d in distribution_data]))):
        print(f"User Count: {user_count}")
        print("-" * 60)
        
        user_data = [d for d in distribution_data if d['user_count'] == user_count]
        for data in user_data:
            print(f"{data['architecture']:8} | "
                  f"Mean: {data['mean_response_time']:6.1f}ms | "
                  f"Median: {data['median_response_time']:6.1f}ms | "
                  f"P95: {data['p95_response_time']:6.1f}ms | "
                  f"P99: {data['p99_response_time']:6.1f}ms | "
                  f"Total: {data['total_requests']:6d} requests")
        print()

def print_failure_distribution_summary(failure_distribution_data):
    """
    失敗分布分析のサマリーを表示
    """
    print("=== FAILURE Response Time Distribution Summary ===")
    print()
    
    for user_count in sorted(list(set([d['user_count'] for d in failure_distribution_data]))):
        print(f"User Count: {user_count}")
        print("-" * 60)
        
        user_data = [d for d in failure_distribution_data if d['user_count'] == user_count]
        for data in user_data:
            print(f"{data['architecture']:8} | "
                  f"Mean: {data['mean_response_time']:6.1f}ms | "
                  f"Median: {data['median_response_time']:6.1f}ms | "
                  f"P95: {data['p95_response_time']:6.1f}ms | "
                  f"P99: {data['p99_response_time']:6.1f}ms | "
                  f"Total: {data['total_requests']:6d} requests")
        print()

def load_stats_data(base_dir="locust_results/change_users"):
    """
    各ユーザー数、実装形態のstats.csvファイルを読み込んでデータを整理
    """
    data = []
    
    # ユーザー数のリスト
    user_counts = USER_COUNTS
    # 実装形態のリスト
    architectures = ['Mono', 'Hybrid', 'Micro']
    
    for user_count in user_counts:
        for arch in architectures:
            # 各実装形態のディレクトリパス
            arch_dir = Path(LOAD_PATH) / str(user_count) / arch
            
            # stats.csvファイルを検索
            stats_files = list(arch_dir.glob("*stats.csv"))
            temp_data = [[],[],[]]
            
            for stats_file in stats_files:
                try:
                    df = pd.read_csv(stats_file)
                    
                    # Aggregated行のみを抽出
                    aggregated_row = df[df['Name'] == 'Aggregated']
                    
                    if not aggregated_row.empty:
                        # Aggregated行からデータを取得
                        request_count = aggregated_row['Request Count'].iloc[0]
                        failure_count = aggregated_row['Failure Count'].iloc[0]
                        avg_response_time = aggregated_row['Average Response Time'].iloc[0]
                        avg_rps = aggregated_row['Requests/s'].iloc[0]
                        
                        # 成功率を計算（成功リクエスト数 / 総リクエスト数）
                        success_rate = ((request_count - failure_count) / request_count * 100) if request_count > 0 else 0
                        
                        temp_data[0].append(avg_response_time)
                        temp_data[1].append(success_rate)
                        temp_data[2].append(avg_rps)
                        
                except Exception as e:
                    print(f"Error reading {stats_file}: {e}")
            
            if temp_data[0]:  # データが存在する場合のみ追加
                data.append({
                    'user_count': user_count,
                    'architecture': arch,
                    'avg_response_time': np.average(temp_data[0]),
                    'success_rate': np.average(temp_data[1]),
                    'avg_rps': np.average(temp_data[2]),
                    'file': stats_file.name
                })
    
    return pd.DataFrame(data)

def plot_results(df):
    """
    結果をプロット
    """
    # プロット設定
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 色の設定
    colors = {'Mono': 'red', 'Hybrid': 'blue', 'Micro': 'green'}
    markers = {'Mono': 'o', 'Hybrid': 's', 'Micro': '^'}
    
    # 1. 平均応答時間のプロット
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        ax1.plot(arch_data['user_count'], arch_data['avg_response_time'], 
                marker=markers[arch], color=colors[arch], linewidth=2, 
                markersize=8, label=arch)
    
    ax1.set_xlabel('User Count')
    ax1.set_ylabel('Average Response Time (ms)')
    ax1.set_title('Average Response Time by User Count and Architecture')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 成功率のプロット
    for arch in df['architecture'].unique():
        arch_data = df[df['architecture'] == arch]
        ax2.plot(arch_data['user_count'], arch_data['success_rate'], 
                marker=markers[arch], color=colors[arch], linewidth=2, 
                markersize=8, label=arch)
    
    ax2.set_xlabel('User Count')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by User Count and Architecture')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)  # 成功率は0-100%の範囲
    
    plt.tight_layout()
    plt.savefig(f"{PATH}/locust_results_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(df):
    """
    結果のサマリーを表示
    """
    print("=== Locust Results Summary ===")
    print()
    
    for user_count in sorted(df['user_count'].unique()):
        print(f"User Count: {user_count}")
        print("-" * 40)
        
        user_data = df[df['user_count'] == user_count]
        for _, row in user_data.iterrows():
            print(f"{row['architecture']:8} | "
                  f"Response Time: {row['avg_response_time']:6.1f}ms | "
                  f"Success Rate: {row['success_rate']:5.1f}% | "
                  f"RPS: {row['avg_rps']:5.1f}")
        print()

def main():
    """
    メイン処理
    """
    print("Loading Locust results...")
    
    # データ読み込み
    df = load_stats_data()
    
    if df.empty:
        print("No data found. Please check the directory structure and CSV files.")
        return
    
    print(f"Loaded data for {len(df)} test runs")
    print()
    
    # サマリー表示
    print_summary(df)
    
    # プロット作成
    print("Creating plots...")
    plot_results(df)
    
    print("Analysis complete! Check 'locust_results_analysis.png' for the plots.")
    
    # レスポンス時間分布分析
    print("\n" + "="*50)
    print("Analyzing response time distributions from request logs...")
    
    # 分布データ読み込み
    success_distribution_data, failure_distribution_data = load_request_logs()
    
    # 成功データの分析
    if success_distribution_data:
        print(f"Loaded SUCCESS distribution data for {len(success_distribution_data)} configurations")
        print()
        
        # 分布サマリー表示
        print_distribution_summary(success_distribution_data)
        
        # 分布プロット作成
        print("Creating SUCCESS distribution plots...")
        plot_response_time_distributions(success_distribution_data)
        plot_percentile_comparison(success_distribution_data)
        plot_architecture_comparison_by_user_count(success_distribution_data)
        
        print("SUCCESS distribution analysis complete! Check:")
        print("- 'response_time_distributions.png' for histogram plots")
        print("- 'response_time_percentiles.png' for percentile comparisons")
        print("- 'architecture_comparison_by_user_count.png' for architecture comparison by user count")
    else:
        print("No SUCCESS request log data found for distribution analysis.")
    
    # 失敗データの分析
    if failure_distribution_data:
        print(f"\nLoaded FAILURE distribution data for {len(failure_distribution_data)} configurations")
        print()
        
        # 失敗分布サマリー表示
        print_failure_distribution_summary(failure_distribution_data)
        
        # 失敗分布プロット作成
        print("Creating FAILURE distribution plots...")
        plot_failure_response_time_distributions(failure_distribution_data)
        plot_failure_architecture_comparison_by_user_count(failure_distribution_data)
        
        print("FAILURE distribution analysis complete! Check:")
        print("- 'failure_response_time_distributions.png' for failure histogram plots")
        print("- 'failure_architecture_comparison_by_user_count.png' for failure architecture comparison")
    else:
        print("No FAILURE request log data found for distribution analysis.")

if __name__ == "__main__":
    main() 