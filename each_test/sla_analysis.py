#!/usr/bin/env python3
"""SLA判定と結果整理ツール。

指定したテスト実行ディレクトリ（例: each_test/individual_service_results/1021-frontend）を走査し、
サービス×レプリカ×ユーザー数ごとのクライアント観測レイテンシとPodリソース指標を集計する。

出力:
  * 標準出力へテーブル形式のサマリー
  * --save-csv を指定した場合、集計結果CSVを保存

SLA閾値（既定: 1000ms）以下で収まる最大ユーザー数や、初めて違反したユーザー数を算出して
マイクロサービス配置アルゴリズム用の入力に利用できる。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


LATENCY_GLOB = "latency_metrics_*.csv"
SYSTEM_GLOB = "system_metrics_full_*.csv"


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_metadata(path: Path) -> Tuple[str, Optional[int], Optional[int]]:
    """Return (service_name, users, replicas) inferred from path segments."""
    service = path.parent.name
    users = None
    replicas = None
    for part in path.parents:
        name = part.name
        if name.startswith("replica"):
            replicas = _to_int(name.replace("replica", ""))
        elif name.isdigit() and users is None:
            users = _to_int(name)
    return service, users, replicas


def _collect_latency(run_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for csv_path in run_root.rglob(LATENCY_GLOB):
        service, users, replicas = _extract_metadata(csv_path)
        if users is None or replicas is None:
            continue
        try:
            df = pd.read_csv(csv_path, engine="python")
        except pd.errors.EmptyDataError:
            continue
        if "request_duration_avg" not in df.columns:
            continue
        df = df.apply(pd.to_numeric, errors="coerce")
        duration = df["request_duration_avg"].dropna()
        if duration.empty:
            continue
        rate = df.get("request_rate_total")
        error = df.get("request_error_rate")
        success = df.get("request_success_rate_percent")
        rows.append({
            "service": service,
            "users": users,
            "replicas": replicas,
            "run_file": str(csv_path),
            "count": int(duration.size),
            "duration_mean_ms": float(duration.mean()),
            "duration_p50_ms": float(duration.quantile(0.5)),
            "duration_p95_ms": float(duration.quantile(0.95)),
            "rate_mean": float(rate.dropna().mean()) if isinstance(rate, pd.Series) else math.nan,
            "error_rate_mean": float(error.dropna().mean()) if isinstance(error, pd.Series) and not error.dropna().empty else 0.0,
            "success_rate_mean": float(success.dropna().mean()) if isinstance(success, pd.Series) and not success.dropna().empty else math.nan,
        })
    return rows


def _collect_system(run_root: Path) -> Dict[Tuple[str, int, int], List[float]]:
    cpu_stats: Dict[Tuple[str, int, int], List[float]] = {}
    for csv_path in run_root.rglob(SYSTEM_GLOB):
        service, users, replicas = _extract_metadata(csv_path)
        if users is None or replicas is None:
            continue
        try:
            df = pd.read_csv(csv_path, engine="python")
        except pd.errors.EmptyDataError:
            continue
        if "cpu_usage_percent" not in df.columns:
            continue
        cpu = pd.to_numeric(df["cpu_usage_percent"], errors="coerce").dropna()
        if cpu.empty:
            continue
        key = (service, users, replicas)
        cpu_stats.setdefault(key, []).append(float(cpu.mean()))
    return cpu_stats


def _aggregate(latency_rows: Iterable[Dict], cpu_stats: Dict[Tuple[str, int, int], List[float]]) -> pd.DataFrame:
    if not latency_rows:
        return pd.DataFrame()
    df = pd.DataFrame(latency_rows)
    grouped = df.groupby(["service", "replicas", "users"], as_index=False).agg({
        "duration_mean_ms": ["mean", "std"],
        "duration_p50_ms": ["mean", "std"],
        "duration_p95_ms": ["mean", "std"],
        "rate_mean": ["mean"],
        "error_rate_mean": ["mean"],
        "success_rate_mean": ["mean"],
        "count": ["sum"],
    })
    grouped.columns = ["_".join(filter(None, col)).rstrip("_") for col in grouped.columns]
    grouped.rename(columns={
        "duration_mean_ms_mean": "duration_mean_ms",
        "duration_mean_ms_std": "duration_mean_ms_std",
        "duration_p50_ms_mean": "duration_p50_ms",
        "duration_p50_ms_std": "duration_p50_ms_std",
        "duration_p95_ms_mean": "duration_p95_ms",
        "duration_p95_ms_std": "duration_p95_ms_std",
        "rate_mean_mean": "rate_mean",
        "error_rate_mean_mean": "error_rate_mean",
        "success_rate_mean_mean": "success_rate_mean",
        "count_sum": "sample_count",
    }, inplace=True)

    # Attach CPU stats if available
    cpu_avg = []
    cpu_std = []
    for row in grouped.itertuples(index=False):
        key = (row.service, row.users, row.replicas)
        samples = cpu_stats.get(key, [])
        if samples:
            cpu_avg.append(float(pd.Series(samples).mean()))
            cpu_std.append(float(pd.Series(samples).std(ddof=1)) if len(samples) > 1 else 0.0)
        else:
            cpu_avg.append(math.nan)
            cpu_std.append(math.nan)
    grouped["cpu_usage_percent"] = cpu_avg
    grouped["cpu_usage_percent_std"] = cpu_std
    return grouped.sort_values(["service", "replicas", "users"]).reset_index(drop=True)


def _find_threshold(df: pd.DataFrame, target_service: str, replicas: Optional[int], sla_ms: float, metric: str) -> Dict[str, Optional[int]]:
    subset = df[df["service"] == target_service]
    if replicas is not None:
        subset = subset[subset["replicas"] == replicas]
    subset = subset.sort_values("users")
    metric_col = {
        "p95": "duration_p95_ms",
        "p50": "duration_p50_ms",
        "mean": "duration_mean_ms",
    }[metric]
    within = subset[subset[metric_col] <= sla_ms]
    violations = subset[subset[metric_col] > sla_ms]
    return {
        "max_within_sla": int(within["users"].max()) if not within.empty else None,
        "first_violation": int(violations["users"].min()) if not violations.empty else None,
    }


def format_summary(df: pd.DataFrame, target_service: str, metric: str) -> str:
    metric_col = {
        "p95": "duration_p95_ms",
        "p50": "duration_p50_ms",
        "mean": "duration_mean_ms",
    }[metric]
    lines = ["service,replicas,users,{}(ms),cpu%(avg),sample_count".format(metric_col)]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.service},{row.replicas},{row.users},{getattr(row, metric_col):.2f},{row.cpu_usage_percent if not math.isnan(row.cpu_usage_percent) else ''},{row.sample_count}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SLAに基づくテスト結果集計ツール")
    parser.add_argument("run_root", help="集計対象ディレクトリ (例: each_test/individual_service_results/1021-frontend)")
    parser.add_argument("--service", default="frontend", help="SLA評価対象サービス")
    parser.add_argument("--replicas", type=int, default=None, help="SLA評価対象レプリカ数。未指定なら全て")
    parser.add_argument("--metric", choices=["p95", "p50", "mean"], default="p95", help="SLA判定に使うレイテンシ指標")
    parser.add_argument("--sla-ms", type=float, default=1000.0, help="SLA閾値(ミリ秒)")
    parser.add_argument("--save-csv", default=None, help="集計結果を書き出すパス")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise SystemExit(f"指定ディレクトリが見つかりません: {run_root}")

    latency_rows = _collect_latency(run_root)
    if not latency_rows:
        raise SystemExit("latency_metrics CSV が見つからないため終了します")
    cpu_stats = _collect_system(run_root)
    aggregated = _aggregate(latency_rows, cpu_stats)

    metric_col = {"p95": "duration_p95_ms", "p50": "duration_p50_ms", "mean": "duration_mean_ms"}[args.metric]
    print("=== 集計結果 (主要列のみ) ===")
    print(aggregated[["service", "replicas", "users", metric_col, "cpu_usage_percent", "sample_count"]].to_string(index=False))

    threshold = _find_threshold(aggregated, args.service, args.replicas, args.sla_ms, args.metric)
    print("\n=== SLA評価 ===")
    print(f"対象サービス: {args.service}")
    print(f"評価指標: {args.metric} (閾値 {args.sla_ms:.0f} ms)")
    if args.replicas is not None:
        print(f"評価レプリカ数: {args.replicas}")
    max_within = threshold["max_within_sla"]
    first_violation = threshold["first_violation"]
    if max_within is not None:
        print(f"  - 最大許容ユーザー数: {max_within}")
    else:
        print("  - SLAを満たすユーザー数は見つかりませんでした")
    if first_violation is not None:
        print(f"  - 最初にSLAを超えたユーザー数: {first_violation}")
    else:
        print("  - SLA違反は検出されませんでした")

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)
        print(f"\n集計結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
