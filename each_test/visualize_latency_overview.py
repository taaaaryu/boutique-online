#!/usr/bin/env python3
"""
Visualize latency metrics from test runs.

Input directory structure (run-root):
  each_test/individual_service_results/<MMDD-TargetService>/
    <service>/replica<N>/<users>/<service>/latency_metrics_*.csv

Output:
  <run-root>/overview/latency_summary.csv
  <run-root>/overview/replica<N>/latency_<metric>_vs_users.png
  <run-root>/overview/<target-service>_replicas_latency_<metric>_vs_users.png
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


def find_latency_metric_files(run_root: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (csv_path, service, users, replicas) for latency metrics."""
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
                        if 'latency_metrics' not in fn:
                            continue
                        results.append((os.path.join(service_subdir_path, fn), service_subdir, users, replicas))
    
    return results


def aggregate_latency_by_service_users_replicas(files: List[Tuple[str, str, int, int]], metric_cols: List[str]) -> pd.DataFrame:
    """Load latency metrics and compute mean/std across multiple files for same (service, users, replicas)."""
    from collections import defaultdict
    groups: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
    for path, service, users, replicas in files:
        groups[(service, users, replicas)].append(path)

    rows: List[Dict] = []
    for (service, users, replicas), paths in sorted(groups.items()):
        # Load each file, compute per-file mean for each metric
        per_file_means: Dict[str, List[float]] = {m: [] for m in metric_cols}
        for path in paths:
            try:
                # Read latency CSV with flexible column handling
                expected_latency_cols = ['timestamp', 'service_name', 'request_duration_avg', 
                                       'request_rate_total', 'response_size_bytes']
                
                # First, try to read normally
                df = pd.read_csv(path, engine='python', on_bad_lines='skip')
                
                # Skip empty files (header only)
                if len(df) == 0:
                    print(f"Warning: Empty latency file (header only): {path}")
                    continue
                
                # If column count doesn't match, try with explicit column names
                if len(df.columns) != len(expected_latency_cols):
                    extra_cols = [f'extra_col_{i}' for i in range(max(0, len(df.columns) - len(expected_latency_cols)))]
                    df = pd.read_csv(path, names=expected_latency_cols + extra_cols, header=0, engine='python', on_bad_lines='skip')
                    # Keep only expected columns
                    df = df[expected_latency_cols]
                
                # Normalize column names (remove whitespace)
                df.columns = [str(c).strip() for c in df.columns]
                
                for m in metric_cols:
                    if m in df.columns:
                        # Convert to numeric, handling any non-numeric values
                        series = pd.to_numeric(df[m], errors='coerce')
                        if series.notna().any():
                            per_file_means[m].append(float(series.mean()))
            except Exception as e:
                print(f"Warning: Error processing latency file {path}: {e}")
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


def plot_latency_all_services_vs_users(df: pd.DataFrame, run_root: str, metrics: List[str]):
    """Plot latency metrics for all services vs users."""
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
            # Apply log scale for duration metrics
            if 'duration' in metric:
                plt.yscale('log')
            plt.title(f'All Services - {metric.replace("_", " ").title()} vs Users (replicas={replicas})')
            plt.xlabel('users')
            # add units for duration metrics
            y_label = metric.replace('_', ' ')
            if 'duration' in metric:
                y_label = f"{y_label} (ms)"
            plt.ylabel(y_label)
            plt.grid(alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
            out_path = os.path.join(rep_dir, f'latency_{metric}_vs_users.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            saved.append(out_path)
    return saved


def plot_target_service_latency_replicas_vs_users(df: pd.DataFrame, run_root: str, target_service: str, metrics: List[str]):
    """Plot target service latency with replicas as lines (x=users, y=metrics, lines=replicas)."""
    if df.empty:
        return []
    
    # Filter for target service only
    target_df = df[df['service'] == target_service]
    if target_df.empty:
        print(f"Warning: No latency data found for target service '{target_service}'")
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
        
        # Apply log scale for duration metrics
        if 'duration' in metric:
            plt.yscale('log')
        plt.title(f'{target_service} - {metric.replace("_", " ").title()} vs Users (by Replicas)')
        plt.xlabel('users')
        # add units for duration metrics
        y_label = metric.replace('_', ' ')
        if 'duration' in metric:
            y_label = f"{y_label} (ms)"
        plt.ylabel(y_label)
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = os.path.join(out_base, f'{target_service}_replicas_latency_{metric}_vs_users.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        saved.append(out_path)
    
    return saved


def main():
    parser = argparse.ArgumentParser(description='Visualize latency metrics across all services')
    parser.add_argument('--run-root', required=True, help='Run root dir (e.g., each_test/individual_service_results/1008-frontend)')
    parser.add_argument('--metrics', nargs='+', default=['request_duration_avg', 'request_rate_total', 'response_size_bytes'],
                        help='Latency metrics to aggregate and plot')
    parser.add_argument('--out-summary', default=None, help='Optional path to write summary CSV (defaults to <run-root>/overview/latency_summary.csv)')
    parser.add_argument('--target-service', default=None, help='Target service for replica comparison plots (e.g., frontend)')
    args = parser.parse_args()

    files = find_latency_metric_files(args.run_root)
    if not files:
        print('No latency_metrics files found under run-root.')
        return

    df = aggregate_latency_by_service_users_replicas(files, args.metrics)
    if df.empty:
        print('No latency aggregations produced.')
        return

    # Save summary
    out_dir = os.path.join(args.run_root, 'overview')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = args.out_summary or os.path.join(out_dir, 'latency_summary.csv')
    df.sort_values(by=['service', 'replicas', 'users'], inplace=True)
    df.to_csv(summary_path, index=False)
    print(f'Wrote latency summary: {summary_path}')

    # Plots per replicas and metric (all services)
    saved = plot_latency_all_services_vs_users(df, args.run_root, args.metrics)
    if saved:
        print('Generated all-services latency plots:')
        for p in saved:
            print(f'  {p}')
    else:
        print('No all-services latency plots generated.')

    # Target service replica comparison plots
    if args.target_service:
        target_saved = plot_target_service_latency_replicas_vs_users(df, args.run_root, args.target_service, args.metrics)
        if target_saved:
            print(f'Generated target service ({args.target_service}) latency replica plots:')
            for p in target_saved:
                print(f'  {p}')
        else:
            print(f'No target service ({args.target_service}) latency replica plots generated.')


if __name__ == '__main__':
    main()
