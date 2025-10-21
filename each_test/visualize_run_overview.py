#!/usr/bin/env python3
"""
Visualize a single test run across all services.

Input directory structure (run-root):
  each_test/individual_service_results/<MMDD-TargetService>/
    <service>/replica<N>/<users>/<service>/system_metrics_full_*.csv

Output:
  <run-root>/overview/summary_all_services.csv
  <run-root>/overview/replica<N>/all_services_<metric>_vs_users.png
  <run-root>/overview/<target-service>_replicas_<metric>_vs_users.png

Behavior:
  - Aggregates ALL_PODS rows per (service, users, replicas)
  - If multiple run files exist, computes mean and std across files
  - Plots, for each replicas and metric: x=users, line=service, error bars=std
  - Additional plots for target service: x=users, y=metrics, lines=replicas
"""

import os
import re
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RUN_USERS_DIR_RE = re.compile(r"^(\d+)$")
RUN_REPL_DIR_RE = re.compile(r"^replica(\d+)$")


def find_system_metric_files(run_root: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (csv_path, service, users, replicas).
    
    Supports the current directory structure:
    - CURRENT: <service>/replica<N>/<users>/<service>/system_metrics_full_*.csv
    """
    results: List[Tuple[str, str, int, int]] = []
    if not os.path.isdir(run_root):
        return results
    
    for service in sorted(os.listdir(run_root)):
        svc_dir = os.path.join(run_root, service)
        if not os.path.isdir(svc_dir):
            continue
            
        # Current structure: <service>/replica<N>/<users>/<service>/
        for repl_name in sorted(os.listdir(svc_dir)):
            m = RUN_REPL_DIR_RE.match(repl_name)
            if not m:
                continue
            replicas = int(m.group(1))
            repl_dir = os.path.join(svc_dir, repl_name)
            if not os.path.isdir(repl_dir):
                continue
            for users_name in sorted(os.listdir(repl_dir)):
                if not RUN_USERS_DIR_RE.match(users_name):
                    continue
                users = int(users_name)
                users_dir = os.path.join(repl_dir, users_name)
                if not os.path.isdir(users_dir):
                    continue
                for service_subdir in sorted(os.listdir(users_dir)):
                    service_subdir_path = os.path.join(users_dir, service_subdir)
                    if not os.path.isdir(service_subdir_path):
                        continue
                    for fn in os.listdir(service_subdir_path):
                        if not fn.endswith('.csv'):
                            continue
                        # system_metrics_full と latency_metrics の両方をサポート
                        if not any(pattern in fn for pattern in ['system_metrics_full', 'latency_metrics']):
                            continue
                        results.append((os.path.join(service_subdir_path, fn), service_subdir, users, replicas))
    
    return results


def aggregate_by_service_users_replicas(files: List[Tuple[str, str, int, int]], metric_cols: List[str]) -> pd.DataFrame:
    """Load ALL_PODS rows and compute mean/std across multiple files for same (service, users, replicas).
    Returns a DataFrame with columns: service, users, replicas, <metric>_mean, <metric>_std
    """
    from collections import defaultdict
    groups: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
    for path, service, users, replicas in files:
        groups[(service, users, replicas)].append(path)

    rows: List[Dict] = []
    for (service, users, replicas), paths in sorted(groups.items()):
        # Load each file, filter ALL_PODS, compute per-file mean for each metric
        per_file_means: Dict[str, List[float]] = {m: [] for m in metric_cols}
        for path in paths:
            try:
                # Read CSV with flexible column handling
                expected_cols = ['timestamp', 'pod_name', 'cpu_usage_seconds_total', 'cpu_usage_percent', 
                               'cpu_throttled_seconds_total', 'memory_working_set_bytes', 'memory_limit_bytes', 
                               'memory_usage_percent', 'network_receive_bytes_total', 'network_transmit_bytes_total']
                
                # First, try to read normally
                df = pd.read_csv(path, engine='python', on_bad_lines='skip')
                
                # If column count doesn't match, try with explicit column names
                if len(df.columns) != len(expected_cols):
                    # Create column names for extra columns
                    extra_cols = [f'extra_col_{i}' for i in range(max(0, len(df.columns) - len(expected_cols)))]
                    df = pd.read_csv(path, names=expected_cols + extra_cols, header=0, engine='python', on_bad_lines='skip')
                    # Keep only expected columns
                    df = df[expected_cols]
                
                # Normalize column names (remove whitespace)
                df.columns = [str(c).strip() for c in df.columns]
                
                if 'pod_name' not in df.columns:
                    continue
                
                # Convert pod_name to string and strip whitespace
                df['pod_name'] = df['pod_name'].astype(str).str.strip()
                df_ap = df[df['pod_name'] == 'ALL_PODS']
                if df_ap.empty:
                    continue
                
                for m in metric_cols:
                    if m in df_ap.columns:
                        # Convert to numeric, handling any non-numeric values
                        series = pd.to_numeric(df_ap[m], errors='coerce')
                        if series.notna().any():
                            per_file_means[m].append(float(series.mean()))
            except Exception as e:
                print(f"Warning: Error processing {path}: {e}")
                continue
        if all(len(vals) == 0 for vals in per_file_means.values()):
            continue
        row: Dict = {'service': service, 'users': users, 'replicas': replicas, 'run_count': len(paths)}
        for m in metric_cols:
            vals = [v for v in per_file_means[m] if pd.notna(v)]
            if vals:
                row[f'{m}_mean'] = float(np.mean(vals))
                row[f'{m}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                row[f'{m}_mean'] = np.nan
                row[f'{m}_std'] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all_services_vs_users(df: pd.DataFrame, run_root: str, metrics: List[str]):
    if df.empty:
        return []
    saved: List[str] = []
    out_base = os.path.join(run_root, 'overview')
    os.makedirs(out_base, exist_ok=True)
    for replicas in sorted(df['replicas'].unique()):
        sub = df[df['replicas'] == replicas]
        if sub.empty:
            continue
        rep_dir = os.path.join(out_base, f'replica{replicas}')
        os.makedirs(rep_dir, exist_ok=True)
        users_sorted = sorted(sub['users'].unique())
        services_sorted = sorted(sub['service'].unique())
        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col not in sub.columns:
                continue
            plt.figure(figsize=(12, 6))
            for svc in services_sorted:
                ssub = sub[sub['service'] == svc]
                y = [float(ssub[ssub['users'] == u][mean_col].mean()) if not ssub[ssub['users'] == u].empty else np.nan for u in users_sorted]
                yerr = [float(ssub[ssub['users'] == u][std_col].mean()) if not ssub[ssub['users'] == u].empty else 0.0 for u in users_sorted]
                plt.errorbar(users_sorted, y, yerr=yerr, marker='o', linewidth=1.5, capsize=3, label=svc)
            plt.title(f'All Services - {metric.replace("_", " ").title()} vs Users (replicas={replicas})')
            plt.xlabel('users')
            plt.ylabel(metric.replace('_', ' '))
            plt.grid(alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
            out_path = os.path.join(rep_dir, f'all_services_{metric}_vs_users.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            saved.append(out_path)
    return saved


def plot_target_service_replicas_vs_users(df: pd.DataFrame, run_root: str, target_service: str, metrics: List[str]):
    """Plot target service with replicas as lines (x=users, y=metrics, lines=replicas)."""
    if df.empty:
        return []
    
    # Filter for target service only
    target_df = df[df['service'] == target_service]
    if target_df.empty:
        print(f"Warning: No data found for target service '{target_service}'")
        return []
    
    saved: List[str] = []
    out_base = os.path.join(run_root, 'overview')
    os.makedirs(out_base, exist_ok=True)
    
    users_sorted = sorted(target_df['users'].unique())
    replicas_sorted = sorted(target_df['replicas'].unique())
    
    # Colors for different replicas
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col not in target_df.columns:
            continue
            
        plt.figure(figsize=(12, 6))
        
        for i, replicas in enumerate(replicas_sorted):
            rep_data = target_df[target_df['replicas'] == replicas]
            if rep_data.empty:
                continue
                
            y = []
            yerr = []
            for users in users_sorted:
                user_data = rep_data[rep_data['users'] == users]
                if not user_data.empty:
                    y.append(float(user_data[mean_col].mean()))
                    yerr.append(float(user_data[std_col].mean()) if std_col in user_data.columns else 0.0)
                else:
                    y.append(np.nan)
                    yerr.append(0.0)
            
            color = colors[i % len(colors)]
            plt.errorbar(users_sorted, y, yerr=yerr, marker='o', linewidth=2, 
                        capsize=3, label=f'replicas={replicas}', color=color)
        
        plt.title(f'{target_service} - {metric.replace("_", " ").title()} vs Users (by Replicas)')
        plt.xlabel('users')
        plt.ylabel(metric.replace('_', ' '))
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = os.path.join(out_base, f'{target_service}_replicas_{metric}_vs_users.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        saved.append(out_path)
    
    return saved


def main():
    parser = argparse.ArgumentParser(description='Visualize run overview across all services')
    parser.add_argument('--run-root', required=True, help='Run root dir (e.g., each_test/individual_service_results/1008-frontend)')
    parser.add_argument('--metrics', nargs='+', default=['cpu_usage_percent', 'memory_usage_percent', 'network_receive_bytes_total', 'network_transmit_bytes_total'],
                        help='Metrics to aggregate and plot')
    parser.add_argument('--out-summary', default=None, help='Optional path to write summary CSV (defaults to <run-root>/overview/summary_all_services.csv)')
    parser.add_argument('--target-service', default=None, help='Target service for replica comparison plots (e.g., frontend)')
    args = parser.parse_args()

    files = find_system_metric_files(args.run_root)
    if not files:
        print('No system_metrics_full files found under run-root.')
        return

    df = aggregate_by_service_users_replicas(files, args.metrics)
    if df.empty:
        print('No aggregations produced.')
        return

    # Save summary
    out_dir = os.path.join(args.run_root, 'overview')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = args.out_summary or os.path.join(out_dir, 'summary_all_services.csv')
    df.sort_values(by=['service', 'replicas', 'users'], inplace=True)
    df.to_csv(summary_path, index=False)
    print(f'Wrote summary: {summary_path}')

    # Plots per replicas and metric (all services)
    saved = plot_all_services_vs_users(df, args.run_root, args.metrics)
    if saved:
        print('Generated all-services plots:')
        for p in saved:
            print(f'  {p}')
    else:
        print('No all-services plots generated.')

    # Target service replica comparison plots
    if args.target_service:
        target_saved = plot_target_service_replicas_vs_users(df, args.run_root, args.target_service, args.metrics)
        if target_saved:
            print(f'Generated target service ({args.target_service}) replica plots:')
            for p in target_saved:
                print(f'  {p}')
        else:
            print(f'No target service ({args.target_service}) replica plots generated.')


if __name__ == '__main__':
    main()


