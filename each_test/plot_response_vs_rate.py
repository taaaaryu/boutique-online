#!/usr/bin/env python3
"""Quick script: plot response time vs request rate from a latency_metrics CSV.
Usage: python3 each_test/plot_response_vs_rate.py <latency_csv_path>
Produces: <run-dir>/overview/response_time_series.png and response_vs_rate_scatter.png
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 each_test/plot_response_vs_rate.py <latency_csv_path>')
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print('File not found:', csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Expect columns: timestamp,service_name,request_duration_avg,request_rate_total,...
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'timestamp' not in df.columns:
        print('timestamp column not found')
        sys.exit(1)
    # parse timestamp
    try:
        df['ts'] = pd.to_datetime(df['timestamp'])
    except Exception:
        df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # choose duration and rate columns
    dur_col = None
    rate_col = None
    for c in df.columns:
        if 'duration' in c and dur_col is None:
            dur_col = c
        if ('request_rate' in c or 'request_rate_total' in c) and rate_col is None:
            rate_col = c
    if dur_col is None or rate_col is None:
        print('Could not find duration or rate columns. Found:', df.columns)
        sys.exit(1)
    # prepare output dir
    run_dir = os.path.dirname(os.path.dirname(csv_path))  # up one to service dir
    out_dir = os.path.join(run_dir, 'overview')
    os.makedirs(out_dir, exist_ok=True)
    # time series plot
    plt.figure(figsize=(10,4))
    plt.plot(df['ts'], df[rate_col], label='request_rate (req/s)')
    ax2 = plt.gca().twinx()
    ax2.plot(df['ts'], df[dur_col], color='orange', label='request_duration_avg (ms)')
    plt.gca().set_xlabel('time')
    plt.gca().set_ylabel('req/s')
    ax2.set_ylabel('duration (ms)')
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.gca().legend(lines+lines2, labels+labels2, loc='upper left')
    plt.title('Request rate and average duration over time')
    plt.tight_layout()
    out_ts = os.path.join(out_dir, 'response_time_series.png')
    plt.savefig(out_ts)
    plt.close()
    # scatter plot
    plt.figure(figsize=(6,4))
    plt.scatter(df[rate_col], df[dur_col], alpha=0.6)
    plt.xlabel('request_rate (req/s)')
    plt.ylabel('request_duration_avg (ms)')
    plt.title('Response time vs request rate')
    plt.grid(True)
    plt.tight_layout()
    out_scatter = os.path.join(out_dir, 'response_vs_rate_scatter.png')
    plt.savefig(out_scatter)
    plt.close()
    print('Wrote:', out_ts, out_scatter)

if __name__ == '__main__':
    main()
