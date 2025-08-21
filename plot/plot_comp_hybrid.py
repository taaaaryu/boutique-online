#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = "locust_results/users"
APPROACHES = ["propose", "random"]
TARGET_ARCHITECTURE = "Hybrid"
TARGET_KILL_RATES = ["kill_0.001", "kill_0.0005", "kill_0.0001"]
TARGET_USER_COUNTS = ["1000", "1250"]
OUT_DIR = "locust_results/users/result_figures/compare_hybrid"

def _read_aggregated_row(stats_csv):
    df = pd.read_csv(stats_csv)
    row = df[df["Name"] == "Aggregated"]
    if row.empty:
        return None
    return {
        "request_count": float(row["Request Count"].iloc[0]),
        "failure_count": float(row["Failure Count"].iloc[0]),
        "avg_response_time": float(row["Average Response Time"].iloc[0]),
        "rps": float(row["Requests/s"].iloc[0]),
    }

def _parse_kill_value(kill_tag: str) -> float:
    # kill_0.1 → 0.1
    try:
        return float(kill_tag.split("_", 1)[1])
    except Exception:
        return float("inf")

def collect_hybrid_stats():
    """
    Hybridアーキテクチャのみの統計を収集
    走査: locust_results/users/{propose,random}/kill_*/<users>/Hybrid/*stats.csv
    """
    rows = []
    for approach in APPROACHES:
        approach_root = os.path.join(BASE_DIR, approach)
        if not os.path.isdir(approach_root):
            continue

        for kill_tag in TARGET_KILL_RATES:
            kill_dir = os.path.join(approach_root, kill_tag)
            if not os.path.isdir(kill_dir):
                continue
            kill_value = _parse_kill_value(kill_tag)

            for user_dir in glob.glob(os.path.join(kill_dir, "*")):
                if not os.path.isdir(user_dir):
                    continue
                user_name = os.path.basename(user_dir)
                if user_name not in TARGET_USER_COUNTS:
                    continue
                user_count = int(user_name)

                # Hybridアーキテクチャのみ
                arch_dir = os.path.join(user_dir, TARGET_ARCHITECTURE)
                if not os.path.isdir(arch_dir):
                    continue

                stats_files = glob.glob(os.path.join(arch_dir, "*stats.csv"))
                if not stats_files:
                    continue

                total_req = 0.0
                total_fail = 0.0
                wt_sum_resp = 0.0
                rps_values = []

                for sf in stats_files:
                    try:
                        agg = _read_aggregated_row(sf)
                        if not agg or agg["request_count"] <= 0:
                            continue
                        req = agg["request_count"]
                        fail = agg["failure_count"]
                        avg_rt = agg["avg_response_time"]
                        rps = agg["rps"]

                        total_req += req
                        total_fail += fail
                        wt_sum_resp += avg_rt * req
                        rps_values.append(rps)
                    except Exception as e:
                        print(f"Warn: failed to read {sf}: {e}")

                if total_req <= 0:
                    continue

                rows.append({
                    "approach": approach,
                    "kill_tag": kill_tag,
                    "kill_value": kill_value,
                    "user_count": user_count,
                    "avg_response_time": wt_sum_resp / total_req,
                    "success_rate": (total_req - total_fail) / total_req * 100.0,
                    "avg_rps": float(np.mean(rps_values)) if rps_values else 0.0,
                    "total_requests": total_req,
                    "total_failures": total_fail,
                })

    cols = ["approach","kill_tag","kill_value","user_count",
            "avg_response_time","success_rate","avg_rps","total_requests","total_failures"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_hybrid_comparison(df, metric, ylabel, out_dir, filename):
    """
    Hybridアーキテクチャの比較グラフを作成
    """
    if df.empty:
        print(f"No data for plotting {metric} under {out_dir}.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = {"propose": "#1f77b4", "random": "#ff7f0e"}
    
    # kill率ごとに異なるマーカーと線のスタイルを使用
    kill_markers = {
        "kill_0.001": "o",   # 円
        "kill_0.0005": "s",  # 四角
        "kill_0.0001": "D",  # ダイヤ
    }
    kill_line_styles = {
        "kill_0.001": "-",   # 実線
        "kill_0.0005": "--", # 破線
        "kill_0.0001": ":",  # 点線
    }

    for approach in APPROACHES:
        approach_df = df[df["approach"] == approach]
        if approach_df.empty:
            continue

        for kill_tag in TARGET_KILL_RATES:
            kill_df = approach_df[approach_df["kill_tag"] == kill_tag].sort_values("user_count")
            if kill_df.empty:
                continue

            xs = kill_df["user_count"].tolist()
            ys = kill_df[metric].tolist()
            
            # ラベル
            label = f"{approach.capitalize()} (kill={float(kill_tag.split('_')[1])*100}%)"
            
            # プロット
            ax.plot(xs, ys, marker=kill_markers[kill_tag], color=colors[approach],
                    linestyle=kill_line_styles[kill_tag], linewidth=2, markersize=8, 
                    label=label, alpha=0.8)

    ax.set_title(f"Hybrid Architecture: {ylabel} Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Users", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    ensure_outdir(out_dir)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def print_hybrid_summary(df):
    """
    Hybridアーキテクチャの比較サマリーを表示
    """
    if df.empty:
        print("No data to summarize.")
        return
    
    print("=== Hybrid Architecture Comparison Summary ===")
    print(f"Target kill rates: {TARGET_KILL_RATES}")
    print("-" * 60)
    
    for kill_tag in TARGET_KILL_RATES:
        sub_k = df[df["kill_tag"] == kill_tag]
        if sub_k.empty:
            print(f"\nKill: {kill_tag} - No data found")
            continue
            
        print(f"\nKill: {kill_tag}")
        for uc in sorted(sub_k["user_count"].unique().tolist()):
            print(f"  Users: {uc}")
            for approach in APPROACHES:
                row = sub_k[sub_k["approach"] == approach]
                if row.empty:
                    print(f"    {approach:8} | No data")
                    continue
                r = row.iloc[0]
                print(f"    {approach:8} | Resp: {r['avg_response_time']:.1f} ms | "
                      f"Success: {r['success_rate']:.2f}% | RPS: {r['avg_rps']:.2f} | "
                      f"Req: {int(r['total_requests'])} | Fail: {int(r['total_failures'])}")

def main():
    """
    メイン処理
    """
    print("Hybrid Architecture Comparison (Propose vs Random)")
    print("=" * 60)
    print(f"Target architecture: {TARGET_ARCHITECTURE}")
    print(f"Target kill rates: {TARGET_KILL_RATES}")
    print(f"Input directory: {BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print("-" * 60)
    
    # データ収集
    df = collect_hybrid_stats()
    if df.empty:
        print("No stats found for Hybrid architecture.")
        return

    # サマリー表示
    print_hybrid_summary(df)

    # 比較グラフ作成
    ensure_outdir(OUT_DIR)
    
    # 成功率の比較
    plot_hybrid_comparison(df, "success_rate", "Success Rate (%)", 
                          OUT_DIR, "hybrid_success_rate_comparison.png")
    
    # 応答時間の比較
    plot_hybrid_comparison(df, "avg_response_time", "Average Response Time (ms)", 
                          OUT_DIR, "hybrid_response_time_comparison.png")
    
    # RPSの比較
    plot_hybrid_comparison(df, "avg_rps", "Requests/s", 
                          OUT_DIR, "hybrid_rps_comparison.png")

    print(f"\nOutputs saved under: {OUT_DIR}/")

if __name__ == "__main__":
    main()