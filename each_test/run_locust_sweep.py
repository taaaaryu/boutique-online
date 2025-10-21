#!/usr/bin/env python3
"""
Run multiple headless Locust experiments for a service and collect per-request latency stats.
Saves per-run summary CSV and an overall sweep CSV and a users-vs-latency plot under each run's overview dir.

Usage: python3 each_test/run_locust_sweep.py --service frontend --replica 1 --users 5,10,20 --duration 30 --output-root ./each_test/individual_service_results

This script expects Locust to be installed and available in PATH.
"""
import argparse
import os
import subprocess
import time
import csv
from pathlib import Path
import statistics
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make sure repo root is on sys.path so `each_test` package imports work when
# this script is run directly from the repository root.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from each_test.integrated_test_system import ServiceLoadTester


def run_locust_for(service, replica, users, duration, output_root, spawn_rate=10):
    # Prepare output dir: individual_service_results/service/replica{replica}/{users}
    target_dir = Path(output_root) / service / f"replica{replica}" / str(users)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create locustfile
    tester = ServiceLoadTester(service, users, duration, str(target_dir))
    locustfile = tester.create_locustfile()

    # Ensure LOCUST_OUTPUT_DIR env so the generated locustfile uses same path
    env = os.environ.copy()
    env['LOCUST_OUTPUT_DIR'] = str(target_dir)
    # If user provided TARGET_HOST via env, keep it; otherwise locustfile will use its default.
    if 'TARGET_HOST' in os.environ:
        env['TARGET_HOST'] = os.environ['TARGET_HOST']

    # Prepare locust command
    cmd = [
        'locust',
        '-f', locustfile,
        '--headless',
        '--users', str(users),
        '--spawn-rate', str(spawn_rate),
        '--run-time', f'{duration}s',
        '--csv', f'{target_dir}/{service}_results',
        '--logfile', f'{target_dir}/{service}_locust.log'
    ]

    print(f"Running Locust: service={service} users={users} duration={duration}s -> {target_dir}")
    try:
        start = time.time()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=duration + 120)
        end = time.time()
        print(f"Locust finished (rc={result.returncode}) in {end-start:.1f}s")
        if result.returncode != 0:
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"Locust timed out for users={users}")
        return None

    # After run, look for per-request CSV
    per_request_path = target_dir / f"{service}_per_request.csv"
    if not per_request_path.exists():
        print(f"Per-request CSV not found at {per_request_path}")
        # create an empty downstream_success entry to record the failed run
        downstream_dir = target_dir
        downstream_path = downstream_dir / f"downstream_success_{int(time.time())}_run.csv"
        with open(downstream_path, 'w', newline='') as df:
            w = csv.writer(df)
            w.writerow(['timestamp','source_service','destination_service','request_rate','error_rate','success_rate_percent'])
        print(f"Wrote empty downstream success placeholder to {downstream_path}")
        return None

    # Read CSV and compute stats
    try:
        df = pd.read_csv(per_request_path)
    except pd.errors.EmptyDataError:
        print(f"Per-request CSV {per_request_path} is empty")
        df = pd.DataFrame(columns=['timestamp','request_type','name','response_time_ms','response_length','status','success'])
    # Expect columns: timestamp,request_type,name,response_time_ms,response_length,status,success
    if 'response_time_ms' not in df.columns:
        print(f"Unexpected CSV format in {per_request_path}")
        return None

    # Convert types
    df['response_time_ms'] = pd.to_numeric(df['response_time_ms'], errors='coerce')
    df['success'] = pd.to_numeric(df['success'], errors='coerce').fillna(0).astype(int)

    # Compute stats grouped by request name and overall
    overall = {}
    overall['service'] = service
    overall['replica'] = replica
    overall['users'] = users
    overall['count'] = int(df.shape[0])
    overall['mean_ms'] = float(df['response_time_ms'].mean()) if overall['count']>0 else None
    overall['median_ms'] = float(df['response_time_ms'].median()) if overall['count']>0 else None
    overall['p95_ms'] = float(df['response_time_ms'].quantile(0.95)) if overall['count']>0 else None
    overall['success_rate'] = float(df['success'].sum())/overall['count']*100.0 if overall['count']>0 else None

    # Save per-run summary CSV in target_dir/overview/sweep_summary.csv (append)
    overview_dir = target_dir / 'overview'
    overview_dir.mkdir(exist_ok=True)
    run_summary_path = overview_dir / 'sweep_summary.csv'
    header = ['service','replica','users','count','mean_ms','median_ms','p95_ms','success_rate']
    write_header = not run_summary_path.exists()
    with open(run_summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([overall[h] for h in header])

    # Also write a downstream-style CSV for this run so it can replace downstream_success files if needed.
    # We'll write a minimal single-row CSV with source_service=service, destination_service='client_observed'
    downstream_dir = target_dir
    downstream_path = downstream_dir / f"downstream_success_{int(time.time())}_run.csv"
    with open(downstream_path, 'w', newline='') as df:
        w = csv.writer(df)
        w.writerow(['timestamp','source_service','destination_service','request_rate','error_rate','success_rate_percent'])
        req_rate = overall['count'] / float(duration) if overall['count']>0 else 0.0
        error_rate = (100.0 - overall['success_rate']) if overall['success_rate'] is not None else None
        success_percent = overall['success_rate'] if overall['success_rate'] is not None else None
        w.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), service, 'client_observed', f"{req_rate:.6f}", f"{error_rate:.6f}" if error_rate is not None else '', f"{success_percent:.6f}" if success_percent is not None else ''])
    print(f"Wrote downstream-style summary to {downstream_path}")

    return overall


def _evaluate_sla(results, metric_key, sla_ms):
    """Return (max_users_within_sla, first_violation_users)."""
    if not results:
        return None, None
    within = [r['users'] for r in results if r.get(metric_key) is not None and r[metric_key] <= sla_ms]
    max_within = max(within) if within else None
    violations = [r['users'] for r in results if r.get(metric_key) is not None and r[metric_key] > sla_ms]
    first_violation = min(violations) if violations else None
    return max_within, first_violation


def run_sweep(service, replica, user_list, duration, output_root, spawn_rate=10,
              sla_ms=None, sla_metric='p95_ms', stop_on_sla_breach=False):
    results = []
    for u in user_list:
        res = run_locust_for(service, replica, u, duration, output_root, spawn_rate=spawn_rate)
        if res is not None:
            results.append(res)
            if sla_ms is not None and sla_metric in res and res[sla_metric] is not None:
                if res[sla_metric] > sla_ms:
                    print(f"SLA breach detected: users={u} {sla_metric}={res[sla_metric]:.2f}ms > {sla_ms}ms")
                    if stop_on_sla_breach:
                        print("stop_on_sla_breach enabled => aborting remaining runs")
                        break
        # Small cooldown between runs
        time.sleep(2)

    if not results:
        print('No successful runs')
        return None

    # Save overall sweep CSV
    sweep_df = pd.DataFrame(results)
    sweep_root = Path(output_root) / service / f'replica{replica}'
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_csv = sweep_root / f'{service}_locust_sweep_summary.csv'
    sweep_df.to_csv(sweep_csv, index=False)

    if sla_ms is not None:
        max_within, first_violation = _evaluate_sla(results, sla_metric, sla_ms)
        print(f"SLA evaluation ({sla_metric} <= {sla_ms}ms):")
        if max_within is not None:
            print(f"  - 最大許容ユーザー数: {max_within}")
        else:
            print("  - SLAを満たすユーザー数はありませんでした")
        if first_violation is not None:
            print(f"  - 最初にSLAを超えたユーザー数: {first_violation}")
        else:
            print("  - SLA違反は検出されませんでした")

    # Plot users vs median/p95
    plt.figure()
    plt.plot(sweep_df['users'], sweep_df['median_ms'], marker='o', label='median (ms)')
    plt.plot(sweep_df['users'], sweep_df['p95_ms'], marker='x', label='p95 (ms)')
    plt.xlabel('users')
    plt.ylabel('response time (ms)')
    plt.title(f'{service} client-observed latency vs users')
    plt.legend()
    plot_path = sweep_root / 'overview' / f'{service}_users_vs_latency.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    print(f'Wrote sweep summary to {sweep_csv} and plot to {plot_path}')
    return {'summary_csv': str(sweep_csv), 'plot': str(plot_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--service', required=True)
    parser.add_argument('--replica', type=int, default=1)
    parser.add_argument('--users', required=True, help='comma-separated user counts, e.g. 5,10,20')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--output-root', default='./each_test/individual_service_results')
    parser.add_argument('--spawn-rate', type=int, default=10)
    parser.add_argument('--sla-ms', type=float, default=None,
                        help='SLA閾値(ミリ秒)。指定すると判定を出力し、stop-on-sla-breachと組み合わせ可')
    parser.add_argument('--sla-metric', choices=['median_ms', 'p95_ms'], default='p95_ms',
                        help='SLA判定に使うメトリクス')
    parser.add_argument('--stop-on-sla-breach', action='store_true',
                        help='SLAを超えた時点で残りのユーザー数をスキップ')
    args = parser.parse_args()

    user_list = [int(x) for x in args.users.split(',') if x.strip()]
    run_sweep(
        args.service,
        args.replica,
        user_list,
        args.duration,
        args.output_root,
        spawn_rate=args.spawn_rate,
        sla_ms=args.sla_ms,
        sla_metric=args.sla_metric,
        stop_on_sla_breach=args.stop_on_sla_breach,
    )
