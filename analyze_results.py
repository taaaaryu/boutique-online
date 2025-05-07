#!/usr/bin/env python3

import os
import re
import pandas as pd
import numpy as np
from glob import glob

def parse_k6_results(result_file):
    """k6の結果ファイルから重要なメトリクスを抽出"""
    with open(result_file, 'r') as f:
        content = f.read()
    
    # 重要なメトリクスを抽出
    metrics = {}
    for metric in ['http_req_duration', 'http_reqs', 'iterations']:
        pattern = f'{metric}.*?avg=([\d.]+)'
        match = re.search(pattern, content)
        if match:
            metrics[metric] = float(match.group(1))
    
    return metrics

def analyze_experiments():
    """実験結果を解析して平均値を計算"""
    k6_dir = "experiment_results/k6_results"
    csv_dir = "experiment_results/csv_results"
    
    # 結果を格納するデータフレーム
    results = []
    
    # 各pause_intervalの結果を処理
    for pause_interval in [40, 60, 80]:  # スクリプトで設定した値と一致させる
        k6_files = glob(f"{k6_dir}/k6_result_{pause_interval}_*.txt")
        csv_files = glob(f"{csv_dir}/csv_result_{pause_interval}_*.csv")
        
        # k6の結果を集計
        k6_metrics = []
        for file in k6_files:
            metrics = parse_k6_results(file)
            if metrics:
                k6_metrics.append(metrics)
        
        # CSVの結果を集計
        csv_metrics = []
        for file in csv_files:
            df = pd.read_csv(file)
            # 最後の行のデータを使用
            last_row = df.iloc[-1]
            csv_metrics.append({
                'running_pods': sum(last_row[f'{dep}_running'] for dep in ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice', 'paymentservice', 'frontend', 'productcatalogservice', 'recommendationservice', 'shippingservice']),
                'paused_pods': sum(last_row[f'{dep}_paused'] for dep in ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice', 'paymentservice', 'frontend', 'productcatalogservice', 'recommendationservice', 'shippingservice'])
            })
        
        # 平均値を計算
        if k6_metrics:
            avg_k6 = {
                metric: np.mean([m[metric] for m in k6_metrics])
                for metric in k6_metrics[0].keys()
            }
        else:
            avg_k6 = {}
            
        if csv_metrics:
            avg_csv = {
                metric: np.mean([m[metric] for m in csv_metrics])
                for metric in csv_metrics[0].keys()
            }
        else:
            avg_csv = {}
        
        # 結果を保存
        results.append({
            'pause_interval': pause_interval,
            **avg_k6,
            **avg_csv
        })
    
    # 結果をデータフレームに変換
    df_results = pd.DataFrame(results)
    
    # 結果を表示
    print("\n実験結果の平均値:")
    print(df_results.to_string(index=False))
    
    # 結果をCSVに保存
    df_results.to_csv("experiment_results/summary.csv", index=False)
    print("\n詳細な結果は experiment_results/summary.csv に保存されました。")

if __name__ == "__main__":
    analyze_experiments() 