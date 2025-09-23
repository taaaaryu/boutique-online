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


def find_system_metric_files(root_dir: str, service_filter: Optional[str] = None) -> List[Tuple[str, str, int, int]]:
    """Find all system metrics CSV files under root.

    Returns list of tuples: (file_path, service, replicas, users)
    """
    matches: List[Tuple[str, str, int, int]] = []
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
            if fn.endswith('_system_metrics_'):
                # unlikely, guard
                continue
            if fn.endswith('.csv') and fn.endswith('_system_metrics_' + fn.split('_system_metrics_')[-1]):
                # generic guard
                pass
            if fn.endswith('.csv') and '_system_metrics_' in fn:
                matches.append((os.path.join(dirpath, fn), service, replicas, users))
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
            ts = pd.to_datetime(df['timestamp'])
            summary['timespan_seconds'] = float((ts.max() - ts.min()).total_seconds())
        except Exception:
            summary['timespan_seconds'] = 0.0
    else:
        summary['timespan_seconds'] = 0.0
    return summary


def build_comparison(root_dir: str, service_filter: Optional[str] = None) -> pd.DataFrame:
    rows: List[Dict] = []
    files = find_system_metric_files(root_dir, service_filter)
    for csv_path, service, replicas, users in sorted(files):
        stats = summarize_file(csv_path)
        # Try read success rate from sibling *_results_stats.csv (Aggregated row)
        req_count = None
        fail_count = None
        success_rate = None
        try:
            base_dir = os.path.dirname(csv_path)
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
            'file': os.path.basename(csv_path),
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
    replicas = sorted({rep for _p, _s, rep, _u in files})
    users = sorted({u for _p, _s, _r, u in files})
    return replicas, users


def _load_timeseries_avg(root_dir: str, service: str, replicas: int, users: int, metric: str) -> Optional[pd.DataFrame]:
    """Load latest matching CSV and return per-timestamp average for metric across pods."""
    base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
    if not os.path.isdir(base_dir):
        return None
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and '_system_metrics_' in fn]
    if not candidates:
        return None
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
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
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and '_system_metrics_' in fn]
    if not candidates:
        return {}
    # pick latest by timestamp in filename or mtime
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
    path = os.path.join(base_dir, candidates[0])
    try:
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or metric not in df.columns or 'pod_name' not in df.columns:
            return {}
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
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
    candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and '_system_metrics_' in fn]
    if not candidates:
        return None
    # pick latest by timestamp in filename or mtime
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
    path = os.path.join(base_dir, candidates[0])
    try:
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or metric not in df.columns:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'])
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




def generate_metric_specific_overview(df: pd.DataFrame, service: str, outdir: str, metric_col: str) -> List[str]:
    """Create metric-specific overview with 2 subplots: metric vs users (top), success rate vs users (bottom).
    Each subplot shows multiple replica lines.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []
    sdf = df[df['service'] == service].copy()
    if sdf.empty:
        return saved

    have_metric = metric_col in sdf.columns
    have_sr = 'success_rate_percent' in sdf.columns
    if not (have_metric or have_sr):
        return saved

    replicas = sorted(sdf['replicas'].unique())
    users = sorted(sdf['users'].unique())

    # Create 2-subplot figure: metric (top) and success rate (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top subplot: metric vs users
    if have_metric:
        ax = axes[0]
        for idx, rep in enumerate(replicas):
            sub = sdf[sdf['replicas'] == rep]
            y = [sub[sub['users'] == u][metric_col].mean() if not sub[sub['users'] == u].empty else np.nan for u in users]
            ax.plot(users, y, label=f'replicas={rep}', linestyle=_linestyle_for_index(idx), marker='o', alpha=0.9)
        ylabel = metric_col.replace('_', ' ')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    
    # Bottom subplot: success rate vs users
    if have_sr:
        ax = axes[1]
        for idx, rep in enumerate(replicas):
            sub = sdf[sdf['replicas'] == rep]
            y = [sub[sub['users'] == u]['success_rate_percent'].mean() if not sub[sub['users'] == u].empty else np.nan for u in users]
            ax.plot(users, y, label=f'replicas={rep}', linestyle=_linestyle_for_index(idx), marker='o', alpha=0.9)
        ax.set_xlabel('users')
        ax.set_ylabel('Success rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    
    fig.suptitle(f'{service} - {metric_col.replace("_", " ").title()} vs Users')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Generate filename based on metric
    metric_name = metric_col.replace('_avg', '').replace('_', '_')
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


def generate_allpods_overview(root_dir: str, service: str, outdir: str, metric: str) -> List[str]:
    """Create ALL_PODS overview: metric vs users, one line per replica count (styles by replicas)."""
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []

    files = find_system_metric_files(root_dir, service)
    if not files:
        return saved

    replicas_list = sorted({replicas for _p, _s, replicas, _u in files})
    users_list = sorted({users for _p, _s, _r, users in files})

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for replicas in replicas_list:
        y_values: List[float] = []
        for users in users_list:
            base_dir = os.path.join(root_dir, f'replica{replicas}', str(users), service)
            if not os.path.isdir(base_dir):
                y_values.append(np.nan)
                continue
            candidates = [fn for fn in os.listdir(base_dir) if fn.endswith('.csv') and '_system_metrics_' in fn]
            if not candidates:
                y_values.append(np.nan)
                continue
            candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
            path = os.path.join(base_dir, candidates[0])
            try:
                df = pd.read_csv(path)
                if 'pod_name' not in df.columns or metric not in df.columns:
                    y_values.append(np.nan)
                    continue
                df_ap = df[df['pod_name'] == 'ALL_PODS']
                if df_ap.empty:
                    y_values.append(np.nan)
                    continue
                # average across timestamps for ALL_PODS
                y_values.append(float(df_ap[metric].mean()))
            except Exception:
                y_values.append(np.nan)
        ax.plot(users_list, y_values, label=f'replicas={replicas}', linestyle=_linestyle_for_replicas(replicas), marker='o', alpha=0.9)

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

    # Plots (metric-specific overview with 2 subplots)
    if args.plot:
        if not args.service:
            print('Plotting requires --service to be specified. Skipping plots.')
            return
        outdir = args.outdir or args.root
        metric_imgs = generate_metric_specific_overview(df, args.service, outdir, metric_col=args.overview_metric)
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


if __name__ == '__main__':
    main()


