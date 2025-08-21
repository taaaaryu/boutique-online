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
ARCHITECTURES = ["Mono", "Hybrid", "Micro"]
OUT_DIR = "locust_results/users/result_figures/compare_propose_random"

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

def collect_aggregated_stats_by_kill():
    """
    走査:
      locust_results/users/{propose,random}/kill_*/<users>/{Mono,Hybrid,Micro}/*stats.csv
    集計:
      - 平均応答時間: リクエスト数重み付け平均
      - 成功率: (総リクエスト-総失敗)/総リクエスト*100
      - RPS: 単純平均
    粒度:
      (approach, kill_tag, user_count, architecture)
    """
    rows = []
    for approach in APPROACHES:
        approach_root = os.path.join(BASE_DIR, approach)
        if not os.path.isdir(approach_root):
            continue

        for kill_dir in glob.glob(os.path.join(approach_root, "kill_*")):
            kill_tag = os.path.basename(kill_dir)
            kill_value = _parse_kill_value(kill_tag)

            for user_dir in glob.glob(os.path.join(kill_dir, "*")):
                if not os.path.isdir(user_dir):
                    continue
                user_name = os.path.basename(user_dir)
                if not user_name.isdigit():
                    continue
                user_count = int(user_name)

                for arch in ARCHITECTURES:
                    arch_dir = os.path.join(user_dir, arch)
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
                        "architecture": arch,
                        "avg_response_time": wt_sum_resp / total_req,
                        "success_rate": (total_req - total_fail) / total_req * 100.0,
                        "avg_rps": float(np.mean(rps_values)) if rps_values else 0.0,
                        "total_requests": total_req,
                        "total_failures": total_fail,
                    })

    cols = ["approach","kill_tag","kill_value","user_count","architecture",
            "avg_response_time","success_rate","avg_rps","total_requests","total_failures"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_by_metric(df, metric, ylabel, out_dir, filename):
    if df.empty:
        print(f"No data for plotting {metric} under {out_dir}.")
        return

    fig, axes = plt.subplots(1, len(ARCHITECTURES), figsize=(6*len(ARCHITECTURES), 5), sharey=False)
    if len(ARCHITECTURES) == 1:
        axes = [axes]

    colors = {"propose": "#1f77b4", "random": "#ff7f0e"}
    markers = {"propose": "o", "random": "s"}

    for i, arch in enumerate(ARCHITECTURES):
        ax = axes[i]
        arch_df = df[df["architecture"] == arch]

        for approach in APPROACHES:
            sub = arch_df[arch_df["approach"] == approach].sort_values("user_count")
            if sub.empty:
                continue
            xs = sub["user_count"].tolist()
            ys = sub[metric].tolist()
            ax.plot(xs, ys, marker=markers[approach], color=colors[approach],
                    linewidth=2, markersize=7, label=approach.capitalize())

        ax.set_title(arch)
        ax.set_xlabel("Users")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    ensure_outdir(out_dir)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def print_summary_by_kill(df):
    if df.empty:
        print("No data to summarize.")
        return
    print("=== Comparison Summary by Kill Rate (Propose vs Random) ===")
    for kill_tag in [k for _,k in sorted(zip(df["kill_value"], df["kill_tag"]))]:
        sub_k = df[df["kill_tag"] == kill_tag]
        if sub_k.empty:
            continue
        print(f"\nKill: {kill_tag}")
        for uc in sorted(sub_k["user_count"].unique().tolist()):
            print(f"  Users: {uc}")
            for arch in ARCHITECTURES:
                sub = sub_k[(sub_k["user_count"] == uc) & (sub_k["architecture"] == arch)]
                if sub.empty:
                    continue
                print(f"  - {arch}")
                for approach in APPROACHES:
                    row = sub[sub["approach"] == approach]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    print(f"    {approach:8} | Resp: {r['avg_response_time']:.1f} ms | "
                          f"Success: {r['success_rate']:.2f}% | RPS: {r['avg_rps']:.2f} | "
                          f"Req: {int(r['total_requests'])} | Fail: {int(r['total_failures'])}")

def main():
    df = collect_aggregated_stats_by_kill()
    if df.empty:
        print("No stats found under locust_results/users/{propose,random}.")
        return

    print_summary_by_kill(df)

    # kill率ごとにフォルダを分けて保存
    for kill_tag, df_k in df.groupby("kill_tag"):
        out_dir = os.path.join(OUT_DIR, kill_tag)
        plot_by_metric(df_k, "avg_response_time", "Average Response Time (ms)", out_dir, "resp_time_compare.png")
        plot_by_metric(df_k, "success_rate", "Success Rate (%)", out_dir, "success_rate_compare.png")
        plot_by_metric(df_k, "avg_rps", "Requests/s", out_dir, "rps_compare.png")

    print(f"\nOutputs saved under: {OUT_DIR}/kill_*/")

if __name__ == "__main__":
    main()