#!/usr/bin/env python3
"""
Compare system metrics across different user counts and replica counts for services.

Scans a results root like:
  individual_service_results/replica<replicas>/<users>/<service>/<service>_system_metrics_<timestamp>.csv

Produces a summary CSV with one row per (service, replicas, users, file),
including aggregate statistics (avg/max) for key metrics.

Usage examples:
  python3 compare_system_metrics.py --root ./individual_service_results --service cartservice --output cartservice_comparison.csv
  python3 compare_system_metrics.py --root ./individual_service_results --output all_services_comparison.csv
"""

import os
import re
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
from matplotlib import rcParams
rcParams.update({
    'legend.fontsize': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


RESULTS_PATTERN = re.compile(r"replica(\d+)" + re.escape(os.sep) + r"(\d+)" + re.escape(os.sep) + r"([^/\\]+)")
RUN_PATTERN = re.compile(r"_run(\d+)\.csv$")


def find_system_metric_files(root_dir: str, service_filter: Optional[str] = None) -> List[Tuple[str, str, int, int, Optional[int]]]:
    """Find all system metrics CSV files under root.

    Returns list of tuples: (file_path, service, replicas, users, run_number)
    run_number is None if not found in filename
    """
    matches: List[Tuple[str, str, int, int, Optional[int]]] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        m = RESULTS_PATTERN.search(rel)
        if not m:
            continue
        replicas = int(m.group(1))
        users = int(m.group(2))
        service = m.group(3)
        if service_filter and service != service_filter:
            continue
        for fn in filenames:
            if fn.endswith('.csv') and ('_system_metrics_full' in fn or fn.startswith('system_metrics_full')):
                # Extract run number if present
                run_match = RUN_PATTERN.search(fn)
                run_num = int(run_match.group(1)) if run_match else None
                matches.append((os.path.join(dirpath, fn), service, replicas, users, run_num))
    return matches


def summarize_file(csv_path: str) -> Dict[str, float]:
    """Compute aggregate stats from a system metrics CSV file."""
    df = pd.read_csv(csv_path)
    # Defensive: ensure expected columns exist
    required = [
        'cpu_usage_percent', 'cpu_throttled_seconds_total',
        'memory_usage_percent', 'network_receive_bytes_total', 'network_transmit_bytes_total'
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0

    summary = {
        'rows': len(df),
        'cpu_usage_percent_avg': float(df['cpu_usage_percent'].mean()) if len(df) else 0.0,
        'cpu_usage_percent_max': float(df['cpu_usage_percent'].max()) if len(df) else 0.0,
        'cpu_throttle_avg': float(df['cpu_throttled_seconds_total'].mean()) if len(df) else 0.0,
        'cpu_throttle_max': float(df['cpu_throttled_seconds_total'].max()) if len(df) else 0.0,
        'mem_usage_percent_avg': float(df['memory_usage_percent'].mean()) if len(df) else 0.0,
        'mem_usage_percent_max': float(df['memory_usage_percent'].max()) if len(df) else 0.0,
        'net_rx_avg': float(df['network_receive_bytes_total'].mean()) if len(df) else 0.0,
        'net_rx_max': float(df['network_receive_bytes_total'].max()) if len(df) else 0.0,
        'net_tx_avg': float(df['network_transmit_bytes_total'].mean()) if len(df) else 0.0,
        'net_tx_max': float(df['network_transmit_bytes_total'].max()) if len(df) else 0.0,
    }
    # Optional: time span
    if 'timestamp' in df.columns and len(df):
        try:
            ts = pd.to_datetime(df['timestamp'], format='ISO8601')
            summary['timespan_seconds'] = float((ts.max() - ts.min()).total_seconds())
        except Exception:
            summary['timespan_seconds'] = 0.0
    else:
        summary['timespan_seconds'] = 0.0
    return summary


def summarize_multiple_files(csv_paths: List[str]) -> Dict[str, float]:
    """Compute aggregate stats from multiple run files (e.g., run1~run5).
    
    Returns mean and std for each metric across runs.
    """
    all_stats = []
    for path in csv_paths:
        stats = summarize_file(path)
        all_stats.append(stats)
    
    if not all_stats:
        return {}
    
    # Compute mean and std across runs
    result = {}
    metric_keys = [
        'cpu_usage_percent_avg', 'cpu_usage_percent_max',
        'cpu_throttle_avg', 'cpu_throttle_max',
        'mem_usage_percent_avg', 'mem_usage_percent_max',
        'net_rx_avg', 'net_rx_max',
        'net_tx_avg', 'net_tx_max',
        'timespan_seconds'
    ]
    
    for key in metric_keys:
        values = [s.get(key, 0.0) for s in all_stats]
        result[key] = float(np.mean(values))
        result[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    
    result['run_count'] = len(all_stats)
    result['rows'] = int(np.mean([s.get('rows', 0) for s in all_stats]))
    
    return result


def build_comparison(root_dir: str, service_filter: Optional[str] = None) -> pd.DataFrame:
    rows: List[Dict] = []
    files = find_system_metric_files(root_dir, service_filter)
    
    # Group files by (service, replicas, users)
    from collections import defaultdict
    grouped = defaultdict(list)
    for csv_path, service, replicas, users, run_num in files:
        key = (service, replicas, users)
        grouped[key].append((csv_path, run_num))
    
    # Process each group
    for (service, replicas, users), file_list in sorted(grouped.items()):
        # Sort by run number (None comes first)
        file_list.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
        csv_paths = [path for path, _ in file_list]
        
        # Compute stats across multiple runs
        if len(csv_paths) > 1:
            stats = summarize_multiple_files(csv_paths)
            file_str = f'{len(csv_paths)} runs'
        else:
            stats = summarize_file(csv_paths[0])
            stats['run_count'] = 1
            # Add _std columns with 0 for single run
            for key in list(stats.keys()):
                if key not in ['rows', 'run_count']:
                    stats[f'{key}_std'] = 0.0
            file_str = os.path.basename(csv_paths[0])
        
        # Try read success rate from sibling *_results_stats.csv (Aggregated row)
        # Use first file's directory
        req_count = None
        fail_count = None
        success_rate = None
        try:
            base_dir = os.path.dirname(csv_paths[0])
            cand = [fn for fn in os.listdir(base_dir) if fn.endswith('_results_stats.csv')]
            if cand:
                # use latest
                cand.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
                stats_path = os.path.join(base_dir, cand[0])
                sdf = pd.read_csv(stats_path)
                # Aggregated row usually has empty Type and Name==Aggregated or empty
                # Safer: pick last row
                agg = sdf.tail(1).iloc[0]
                req_count = int(agg['Request Count']) if 'Request Count' in sdf.columns else None
                fail_count = int(agg['Failure Count']) if 'Failure Count' in sdf.columns else None
                if req_count is not None and req_count > 0 and fail_count is not None:
                    success_rate = (req_count - fail_count) / req_count * 100.0
        except Exception:
            pass
        
        row = {
            'service': service,
            'replicas': replicas,
            'users': users,
            'file': file_str,
        }
        row.update(stats)
        if req_count is not None:
            row['requests'] = req_count
        if fail_count is not None:
            row['failures'] = fail_count
        if success_rate is not None:
            row['success_rate_percent'] = success_rate
        rows.append(row)
    return pd.DataFrame(rows)


def _list_service_matrix(root_dir: str, service: str) -> Tuple[List[int], List[int]]:
    files = find_system_metric_files(root_dir, service)
    replicas = sorted({rep for _p, _s, rep, _u, _r in files})
    users = sorted({u for _p, _s, _r, u, _rn in files})
    return replicas, users


def _load_timeseries_avg(root_dir: str, service: str, replicas: int, users: int, metric: str) -> Optional[pd.DataFrame]:
    """Load latest matching CSV and return per-timestamp average for metric across pods."""
    base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
    if not os.path.isdir(base_dir):
        return None
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and ('_system_metrics_full' in fn or fn.startswith('system_metrics_full_'))]
    if not candidates:
        return None
    # Prioritize system_metrics_full_*.csv files (have all columns)
    full_files = [fn for fn in candidates if fn.startswith('system_metrics_full_')]
    if full_files:
        candidates = full_files
    # pick latest by timestamp in filename or mtime
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
    path = os.path.join(base_dir, candidates[0])
    try:
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or metric not in df.columns:
            return None
        # Exclude ALL_PODS from per-pod averages
        if 'pod_name' in df.columns:
            df = df[df['pod_name'] != 'ALL_PODS']
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        # average across pods for each timestamp
        df_avg = df.groupby('timestamp', as_index=False)[metric].mean()
        return df_avg
    except Exception:
        return None


def _load_timeseries_by_pods(root_dir: str, service: str, replicas: int, users: int, metric: str) -> Dict[str, pd.DataFrame]:
    """Load latest matching CSV and return per-pod timeseries data."""
    base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
    if not os.path.isdir(base_dir):
        return {}
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and ('_system_metrics_full' in fn or fn.startswith('system_metrics_full_'))]
    if not candidates:
        return {}
    # Prioritize system_metrics_full_*.csv files (have all columns)
    full_files = [fn for fn in candidates if fn.startswith('system_metrics_full_')]
    if full_files:
        candidates = full_files
    # pick latest by timestamp in filename or mtime
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
    path = os.path.join(base_dir, candidates[0])
    try:
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or metric not in df.columns or 'pod_name' not in df.columns:
            return {}
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        
        # Group by pod_name and return individual pod data (exclude ALL_PODS)
        pod_data = {}
        for pod_name in df['pod_name'].unique():
            if pod_name == 'ALL_PODS':
                continue
            pod_df = df[df['pod_name'] == pod_name].copy()
            pod_df = pod_df[['timestamp', metric]].sort_values('timestamp')
            pod_data[pod_name] = pod_df
        return pod_data
    except Exception:
        return {}


def _load_timeseries_total(root_dir: str, service: str, replicas: int, users: int, metric: str) -> Optional[pd.DataFrame]:
    """Load latest matching CSV and return per-timestamp total for metric across pods."""
    base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
    if not os.path.isdir(base_dir):
        return None
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and ('_system_metrics_full' in fn or fn.startswith('system_metrics_full_') or fn.startswith('system_metrics_'))]
    if not candidates:
        return None
    # Prioritize system_metrics_full_*.csv files (have all columns)
    full_files = [fn for fn in candidates if fn.startswith('system_metrics_full_')]
    if full_files:
        candidates = full_files
    # pick latest by timestamp in filename or mtime
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
    path = os.path.join(base_dir, candidates[0])
    try:
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or metric not in df.columns:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        # sum across pods for each timestamp
        df_total = df.groupby('timestamp', as_index=False)[metric].sum()
        return df_total
    except Exception:
        return None








def _linestyle_for_index(i: int) -> str:
    styles = ['-', '--', '-.', ':']
    return styles[i % len(styles)]

def _linestyle_for_replicas(replicas: int) -> str:
    """Map replica count to a consistent line style."""
    if replicas == 1:
        return '-'
    if replicas == 2:
        return '--'
    if replicas == 3:
        return '-.'
    return ':'




def generate_metric_specific_overview(root_dir: str, df: pd.DataFrame, service: str, outdir: str, metric_col: str) -> List[str]:
    """Create metric-specific overview with 2 subplots: metric boxplot (top), success rate line chart (bottom).
    Top subplot shows boxplot for each (replicas, users) combination using all data points from all runs.
    Bottom subplot shows success rate as line chart grouped by replicas.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []
    sdf = df[df['service'] == service].copy()
    if sdf.empty:
        return saved

    have_sr = 'success_rate_percent' in sdf.columns
    
    # Extract the base metric name (remove _avg suffix if present)
    base_metric = metric_col.replace('_avg', '').replace('_max', '')
    
    # Get all files for this service to collect raw data
    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved
    
    # Group files by (replicas, users)
    from collections import defaultdict
    grouped = defaultdict(list)
    for csv_path, svc, replicas, users, run_num in files:
        if svc == service:
            key = (replicas, users)
            grouped[key].append(csv_path)
    
    if not grouped:
        return saved
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(grouped.keys())
    
    # Collect all data points for boxplot
    boxplot_data = []
    labels = []
    replica_for_each_box = []
    
    for replicas, users in sorted_keys:
        csv_paths = grouped[(replicas, users)]
        all_values = []
        
        # Read all CSVs for this combination and collect all data points
        for csv_path in csv_paths:
            try:
                df_csv = pd.read_csv(csv_path)
                if base_metric not in df_csv.columns:
                    continue
                
                # Exclude ALL_PODS if pod_name column exists
                if 'pod_name' in df_csv.columns:
                    df_csv = df_csv[df_csv['pod_name'] != 'ALL_PODS']
                
                # Collect all values from this CSV
                values = df_csv[base_metric].dropna().tolist()
                all_values.extend(values)
            except Exception:
                continue
        
        if all_values:
            boxplot_data.append(all_values)
            labels.append(f'R{replicas}\nU{users}')
            replica_for_each_box.append(replicas)
        else:
            boxplot_data.append([])
            labels.append(f'R{replicas}\nU{users}')
            replica_for_each_box.append(replicas)
    
    if not boxplot_data or all(len(d) == 0 for d in boxplot_data):
        return saved
    
    # Create 2-subplot figure: boxplot (top) and success rate (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 1.2), 10))
    
    # Top subplot: boxplot
    ax = axes[0]
    bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
    
    # Color boxes by replica count
    replica_colors = {1: 'lightblue', 2: 'lightgreen', 3: 'lightyellow', 4: 'lightcoral'}
    for i, replicas in enumerate(replica_for_each_box):
        color = replica_colors.get(replicas, 'lightgray')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)
    
    ylabel = base_metric.replace('_', ' ')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{base_metric.replace("_", " ").title()} Distribution (All Runs)')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate labels if too many
    if len(labels) > 10:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bottom subplot: success rate vs users (line chart)
    if have_sr:
        ax = axes[1]
        replicas_list = sorted(sdf['replicas'].unique())
        users_list = sorted(sdf['users'].unique())
        
        for idx, rep in enumerate(replicas_list):
            sub = sdf[sdf['replicas'] == rep]
            y = [sub[sub['users'] == u]['success_rate_percent'].mean() if not sub[sub['users'] == u].empty else np.nan for u in users_list]
            ax.plot(users_list, y, label=f'replicas={rep}', linestyle=_linestyle_for_index(idx), marker='o', alpha=0.9)
        ax.set_xlabel('Users')
        ax.set_ylabel('Success rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    else:
        # If no success rate data, hide bottom subplot
        axes[1].axis('off')
    
    fig.suptitle(f'{service} - {base_metric.replace("_", " ").title()} Overview')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Generate filename based on metric
    metric_name = base_metric.replace('_', '_')
    path = os.path.join(outdir, f'{service}_{metric_name}_overview.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)

    return saved


def generate_individual_distribution_plots(root_dir: str, service: str, outdir: str) -> List[str]:
    """Generate distribution analysis for each user count: timeseries + histogram in one plot.
    Creates one plot per user count showing both timeseries and distribution.
    Shows individual pod lines and excludes ALL_PODS.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []
    
    # Get all user counts and replicas for this service
    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved
    
    user_counts = sorted({users for _p, _s, _r, users in files})
    replicas = sorted({replicas for _p, _s, replicas, _u in files})
    
    metrics = ['cpu_usage_percent', 'memory_usage_percent', 'network_receive_bytes_total', 'network_transmit_bytes_total']
    
    for user_count in user_counts:
        for metric in metrics:
            # Create 2x2 subplot: timeseries (top) and histogram (bottom) for each replica
            fig, axes = plt.subplots(2, len(replicas), figsize=(6*len(replicas), 8))
            if len(replicas) == 1:
                axes = axes.reshape(2, 1)
            
            for rep_idx, replica in enumerate(replicas):
                # Load timeseries data (avg excludes ALL_PODS) and per-pod data
                ts_data = _load_timeseries_avg(root_dir, service, replica, user_count, metric)
                pod_data = _load_timeseries_by_pods(root_dir, service, replica, user_count, metric)
                
                # Top row: timeseries
                ax_ts = axes[0, rep_idx]
                if ts_data is not None and not ts_data.empty:
                    # Plot individual pods if available
                    if pod_data:
                        for pod_name, pod_df in pod_data.items():
                            if not pod_df.empty:
                                ax_ts.plot(pod_df['timestamp'], pod_df[metric], 
                                         alpha=0.7, linewidth=1)
                    else:
                        # Fallback to average if no pod data
                        ax_ts.plot(ts_data['timestamp'], ts_data[metric], 
                                 alpha=0.8, linewidth=1.5, label='Average')
                    
                    ax_ts.set_title(f'Replica {replica} - Timeseries')
                    ax_ts.set_ylabel(metric.replace('_', ' '))
                    ax_ts.grid(alpha=0.3)
                    handles, labels = ax_ts.get_legend_handles_labels()
                    if labels:
                        ax_ts.legend(fontsize=8, handlelength=1)
                else:
                    ax_ts.set_title(f'Replica {replica} - No Data')
                
                # Bottom row: histogram using average (pods-only)
                ax_hist = axes[1, rep_idx]
                if ts_data is not None and not ts_data.empty:
                    ax_hist.hist(ts_data[metric], bins=20, alpha=0.7, edgecolor='black')
                    ax_hist.set_title(f'Replica {replica} - Distribution')
                    ax_hist.set_xlabel(metric.replace('_', ' '))
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.grid(alpha=0.3)
                else:
                    ax_hist.set_title(f'Replica {replica} - No Data')
            
            fig.suptitle(f'{service} - {metric.replace("_", " ").title()} Analysis (Users: {user_count})')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            metric_name = metric.replace('_', '_')
            path = os.path.join(outdir, f'{service}_{metric_name}_users_{user_count}_distribution.png')
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
    
    return saved


def generate_overall_trend_plots(root_dir: str, service: str, outdir: str) -> List[str]:
    """Generate overall trend analysis: how metrics change with user count increase.
    Shows both replica 1 and 2 on the same plot for comparison.
    Shows individual pod lines only (excludes ALL_PODS).
    """
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []
    
    # Get all user counts and replicas for this service
    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved
    
    user_counts = sorted({users for _p, _s, _r, users in files})
    replicas = sorted({replicas for _p, _s, replicas, _u in files})
    
    metrics = ['cpu_usage_seconds_total', 'cpu_usage_percent', 'memory_usage_percent', 'network_receive_bytes_total', 'network_transmit_bytes_total']
    
    for metric in metrics:
        for rep_idx, replica in enumerate(replicas):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            labelled_users = set()
            # Plot timeseries for each user count
            for user_idx, user_count in enumerate(user_counts):
                # Load data for this user count and replica
                pod_data = _load_timeseries_by_pods(root_dir, service, replica, user_count, metric)
                avg_data = _load_timeseries_avg(root_dir, service, replica, user_count, metric)
                
                # Plot individual pods if available
                if pod_data:
                    used_label = False
                    for pod_name, pod_df in pod_data.items():
                        if not pod_df.empty:
                            # Normalize timestamps to start from 0
                            pod_df = pod_df.copy()
                            pod_df['timestamp'] = pd.to_datetime(pod_df['timestamp'])
                            start_time = pod_df['timestamp'].min()
                            pod_df['time_seconds'] = (pod_df['timestamp'] - start_time).dt.total_seconds()
                            label_text = None
                            if user_count not in labelled_users and not used_label:
                                label_text = f'Users: {user_count}'
                                labelled_users.add(user_count)
                                used_label = True
                            ax.plot(pod_df['time_seconds'], pod_df[metric], 
                                   alpha=0.7, linewidth=1, 
                                   label=label_text)
                
                # Fallback to average if no pod data
                if not pod_data and avg_data is not None and not avg_data.empty:
                    avg_data = avg_data.copy()
                    avg_data['timestamp'] = pd.to_datetime(avg_data['timestamp'])
                    start_time = avg_data['timestamp'].min()
                    avg_data['time_seconds'] = (avg_data['timestamp'] - start_time).dt.total_seconds()
                    label_text = None
                    if user_count not in labelled_users:
                        label_text = f'Users: {user_count}'
                        labelled_users.add(user_count)
                    ax.plot(avg_data['time_seconds'], avg_data[metric], 
                           alpha=0.85, linewidth=1.5,
                           label=label_text)
            
            ax.set_title(f'Replica {replica} - {metric.replace("_", " ").title()} Trend')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, handlelength=1)
        
            fig.suptitle(f'{service} - {metric.replace("_", " ").title()} Trends by User Count (Replica {replica})')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            metric_name = metric.replace('_', '_')
            path = os.path.join(outdir, f'{service}_{metric_name}_trends_{replica}.png')
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
    
    return saved


def generate_boxplot_overview(root_dir: str, service: str, outdir: str, metric: str) -> List[str]:
    """Create boxplot overview: one box per (replicas, users) combination showing distribution across all data points.
    
    Each box represents all timestamp data from all runs for that specific (replicas, users) combination.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []
    
    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved
    
    # Group files by (replicas, users)
    from collections import defaultdict
    grouped = defaultdict(list)
    for csv_path, svc, replicas, users, run_num in files:
        if svc == service:
            key = (replicas, users)
            grouped[key].append(csv_path)
    
    if not grouped:
        return saved
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(grouped.keys())
    
    # Collect all data points for each (replicas, users) combination
    boxplot_data = []
    labels = []
    
    for replicas, users in sorted_keys:
        csv_paths = grouped[(replicas, users)]
        all_values = []
        
        # Read all CSVs for this combination and collect all data points
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
                if metric not in df.columns:
                    continue
                
                # Exclude ALL_PODS if pod_name column exists
                if 'pod_name' in df.columns:
                    df = df[df['pod_name'] != 'ALL_PODS']
                
                # Collect all values from this CSV
                values = df[metric].dropna().tolist()
                all_values.extend(values)
            except Exception as e:
                print(f"Warning: Failed to read {csv_path}: {e}")
                continue
        
        if all_values:
            boxplot_data.append(all_values)
            labels.append(f'R{replicas}\nU{users}')
        else:
            # Add empty list to maintain alignment
            boxplot_data.append([])
            labels.append(f'R{replicas}\nU{users}')
    
    if not boxplot_data or all(len(d) == 0 for d in boxplot_data):
        print(f"No data found for {service} - {metric}")
        return saved
    
    # Create boxplot
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(labels) * 1.5), 6))
    
    bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
    
    # Color boxes by replica count
    replica_colors = {1: 'lightblue', 2: 'lightgreen', 3: 'lightyellow', 4: 'lightcoral'}
    for i, (replicas, users) in enumerate(sorted_keys):
        color = replica_colors.get(replicas, 'lightgray')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)
    
    ax.set_xlabel('(Replicas, Users)')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'{service} - {metric.replace("_", " ").title()} Distribution (All Runs)')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate labels if too many
    if len(labels) > 10:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    
    path = os.path.join(outdir, f'{service}_{metric}_boxplot.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)
    
    print(f"Generated boxplot: {path}")
    return saved


def generate_allpods_overview(root_dir: str, service: str, outdir: str, metric: str) -> List[str]:
    """Create ALL_PODS overview: metric vs users, one line per replica count with error bars (styles by replicas)."""
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []

    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved

    replicas_list = sorted({replicas for _p, _s, replicas, _u, _rn in files})
    users_list = sorted({users for _p, _s, _r, users, _rn in files})
    
    print(f"DEBUG: replicas_list = {replicas_list}")
    print(f"DEBUG: users_list = {users_list}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for replicas in replicas_list:
        y_values: List[float] = []
        y_errors: List[float] = []
        print(f"DEBUG: Processing replicas={replicas}")
        for users in users_list:
            base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
            print(f"DEBUG: Checking base_dir={base_dir}, exists={os.path.isdir(base_dir)}")
            if not os.path.isdir(base_dir):
                y_values.append(np.nan)
                y_errors.append(0.0)
                continue
            candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and ('_system_metrics_' in fn or fn.startswith('system_metrics_full_') or fn.startswith('system_metrics_'))]
            print(f"DEBUG: Found {len(candidates)} candidates: {candidates[:3] if candidates else []}")
            if not candidates:
                y_values.append(np.nan)
                y_errors.append(0.0)
                continue
            
            # Group candidates by run number
            run_files = []
            for fn in candidates:
                run_match = RUN_PATTERN.search(fn)
                if run_match:
                    run_files.append(fn)
            
            # If we have multiple run files, compute mean and std across runs
            if len(run_files) > 1:
                full_files = [fn for fn in run_files if fn.startswith('system_metrics_full_')]
                if full_files:
                    run_files = full_files
                
                run_means = []
                for fn in run_files:
                    path = os.path.join(base_dir, fn)
                    try:
                        df = pd.read_csv(path)
                        if 'pod_name' not in df.columns or metric not in df.columns:
                            continue
                        df_ap = df[df['pod_name'] == 'ALL_PODS']
                        if df_ap.empty:
                            continue
                        mean_val = float(df_ap[metric].mean())
                        run_means.append(mean_val)
                    except Exception:
                        continue
                
                if run_means:
                    y_values.append(float(np.mean(run_means)))
                    y_errors.append(float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0)
                    print(f"DEBUG: Mean={np.mean(run_means):.4f}, Std={np.std(run_means, ddof=1) if len(run_means) > 1 else 0.0:.4f} from {len(run_means)} runs")
                else:
                    y_values.append(np.nan)
                    y_errors.append(0.0)
            else:
                # Single file or no run files
                full_files = [fn for fn in candidates if fn.startswith('system_metrics_full_')]
                if full_files:
                    candidates = full_files
                candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
                path = os.path.join(base_dir, candidates[0])
                print(f"DEBUG: Reading file: {path}")
                try:
                    df = pd.read_csv(path)
                    print(f"DEBUG: CSV loaded, shape={df.shape}, columns={df.columns.tolist()}")
                    if 'pod_name' not in df.columns or metric not in df.columns:
                        print(f"DEBUG: Missing columns - pod_name={'pod_name' in df.columns}, metric={metric in df.columns}")
                        y_values.append(np.nan)
                        y_errors.append(0.0)
                        continue
                    df_ap = df[df['pod_name'] == 'ALL_PODS']
                    print(f"DEBUG: ALL_PODS rows found: {len(df_ap)}")
                    if df_ap.empty:
                        print(f"DEBUG: No ALL_PODS data found")
                        y_values.append(np.nan)
                        y_errors.append(0.0)
                        continue
                    # average across timestamps for ALL_PODS
                    mean_val = float(df_ap[metric].mean())
                    print(f"DEBUG: Mean value for {metric}: {mean_val}")
                    y_values.append(mean_val)
                    y_errors.append(0.0)
                except Exception as e:
                    print(f"DEBUG: Exception occurred: {e}")
                    y_values.append(np.nan)
                    y_errors.append(0.0)
        
        print(f"DEBUG: y_values for replicas={replicas}: {y_values}")
        print(f"DEBUG: y_errors for replicas={replicas}: {y_errors}")
        ax.errorbar(users_list, y_values, yerr=y_errors, label=f'replicas={replicas}', 
                   linestyle=_linestyle_for_replicas(replicas), marker='o', alpha=0.9, capsize=3)

    ax.set_xlabel('users')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, handlelength=1)

    fig.suptitle(f'{service} - ALL_PODS {metric.replace("_", " ").title()} vs Users')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(outdir, f'{service}_{metric}_allpods_overview.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(description='Compare system metrics across user/replica settings')
    parser.add_argument('--root', default='./individual_service_results', help='Results root directory')
    parser.add_argument('--service', default=None, help='Filter by service name (optional)')
    parser.add_argument('--output', default=None, help='Output CSV path (optional)')
    parser.add_argument('--plot', action='store_true', help='Generate metric-specific overview plots')
    parser.add_argument('--overview-metric', default='cpu_usage_percent_avg', help='Aggregated metric column to plot in overview (from comparison CSV)')
    parser.add_argument('--outdir', default=None, help='Directory to save plots (defaults to root)')
    parser.add_argument('--distribution', action='store_true', help='Generate individual distribution analysis (timeseries + histogram per user count)')
    parser.add_argument('--trends', action='store_true', help='Generate overall trend analysis (how metrics change with user count)')
    parser.add_argument('--allpods-overview', action='store_true', help='Generate ALL_PODS overview (metric vs users, lines per replicas)')
    parser.add_argument('--allpods-metric', default='cpu_usage_percent', help='Metric column to use for ALL_PODS overview')
    parser.add_argument('--boxplot', action='store_true', help='Generate boxplot showing distribution of all data points per (replicas, users)')
    parser.add_argument('--boxplot-metric', default='cpu_usage_percent', help='Metric column to use for boxplot')
    args = parser.parse_args()

    df = build_comparison(args.root, args.service)
    if df.empty:
        print('No system metrics found for the specified criteria.')
        return

    # Sort for readability
    df.sort_values(by=['service', 'replicas', 'users'], inplace=True)

    # Default output
    if not args.output:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = args.service if args.service else 'all_services'
        #args.output = os.path.join(args.root, f'{base}_system_metrics_comparison_{ts}.csv')

    df.to_csv(args.output, index=False)
    print(f'Comparison written to: {args.output}')

    # Plots (metric-specific overview with 2 subplots: boxplot + success rate)
    if args.plot:
        if not args.service:
            print('Plotting requires --service to be specified. Skipping plots.')
            return
        outdir = args.outdir or args.root
        metric_imgs = generate_metric_specific_overview(args.root, df, args.service, outdir, metric_col=args.overview_metric)
        if metric_imgs:
            print('Generated plots:')
            for p in metric_imgs:
                print(f'  {p}')
        else:
            print('No plots generated (no matching data).')

    # Distribution analysis (individual user count analysis)
    if args.distribution:
        if not args.service:
            print('Distribution analysis requires --service to be specified. Skipping.')
            return
        outdir = args.outdir or args.root
        dist_imgs = generate_individual_distribution_plots(args.root, args.service, outdir)
        if dist_imgs:
            print('Generated distribution plots:')
            for p in dist_imgs:
                print(f'  {p}')
        else:
            print('No distribution plots generated (no matching data).')

    # Trend analysis (overall user count trends)
    if args.trends:
        if not args.service:
            print('Trend analysis requires --service to be specified. Skipping.')
            return
        outdir = args.outdir or args.root
        trend_imgs = generate_overall_trend_plots(args.root, args.service, outdir)
        if trend_imgs:
            print('Generated trend plots:')
            for p in trend_imgs:
                print(f'  {p}')
        else:
            print('No trend plots generated (no matching data).')

    # ALL_PODS overview
    if args.allpods_overview:
        if not args.service:
            print('ALL_PODS overview requires --service to be specified. Skipping.')
            return
        outdir = args.outdir or args.root
        ap_imgs = generate_allpods_overview(args.root, args.service, outdir, metric=args.allpods_metric)
        if ap_imgs:
            print('Generated ALL_PODS overview:')
            for p in ap_imgs:
                print(f'  {p}')
        else:
            print('No ALL_PODS overview generated (no matching data).')

    # Boxplot
    if args.boxplot:
        if not args.service:
            print('Boxplot requires --service to be specified. Skipping.')
            return
        outdir = args.outdir or args.root
        bp_imgs = generate_boxplot_overview(args.root, args.service, outdir, metric=args.boxplot_metric)
        if bp_imgs:
            print('Generated boxplot:')
            for p in bp_imgs:
                print(f'  {p}')
        else:
            print('No boxplot generated (no matching data).')


if __name__ == '__main__':
    main()


