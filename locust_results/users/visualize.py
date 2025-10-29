#!/usr/bin/env python3
"""Generate summary plots from Locust test results.

This script expects the following directory layout:
locust_results/users/test/{user_count}/replicas_{replica_count}/run_{n}/

Within each run directory the script looks for:
* locust_stats.csv - to collect average response time statistics.
* request_log_unknown_run_*.csv - to build latency histograms and CDFs.

The script outputs plots and summary tables into an output directory
(`--output-dir`, defaults to `locust_results/users/figures`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend.
import matplotlib.pyplot as plt  # noqa: E402  (deferred import after backend selection)


def parse_user_and_replica(path: Path) -> Tuple[int, int]:
    """Extract user count and replica count by scanning parent directories."""
    for parent in path.parents:
        name = parent.name
        if name.startswith("replicas_"):
            try:
                replica_count = int(name.split("_", 1)[1])
                user_count = int(parent.parent.name)
                return user_count, replica_count
            except (ValueError, IndexError):
                break
    raise ValueError(f"Failed to parse user/replica from path: {path}")


def parse_run_number(path: Path) -> int | None:
    """Extract run number from the path if available."""
    for parent in path.parents:
        name = parent.name
        if name.startswith("run_"):
            try:
                return int(name.split("_", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def load_average_response_times(base_dir: Path) -> pd.DataFrame:
    """Read locust_stats.csv files and return run-level average response times."""
    records = []
    for stats_path in base_dir.glob("*/replicas_*/run_*/locust_stats.csv"):
        try:
            users, replicas = parse_user_and_replica(stats_path)
        except ValueError:
            continue

        run_number = parse_run_number(stats_path)

        df = pd.read_csv(stats_path)
        aggregated_row = df.loc[df["Name"] == "Aggregated"]
        if aggregated_row.empty:
            aggregated_row = df.loc[df["Type"].astype(str).str.strip() == ""]
        if aggregated_row.empty:
            continue

        row = aggregated_row.iloc[0]
        avg_response_time = float(row["Average Response Time"])

        p99_response_time = float("nan")
        for col in ("99%", "99% ", "p99", "P99"):
            if col in row and pd.notna(row[col]):
                try:
                    p99_response_time = float(row[col])
                    break
                except (TypeError, ValueError):
                    continue

        records.append(
            {
                "users": users,
                "replicas": replicas,
                "run": run_number,
                "avg_response_time": avg_response_time,
                "p99_response_time": p99_response_time,
                "source": str(stats_path),
            }
        )

    return pd.DataFrame.from_records(records)


def plot_mean_response_times(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot mean average response time per replica count."""
    mean_df = (
        df.groupby(["users", "replicas"], as_index=False)["avg_response_time"].mean()
    )
    if mean_df.empty:
        raise ValueError("No data available to plot mean response times.")

    figure_path = output_dir / "avg_response_time_by_users.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for replicas in sorted(mean_df["replicas"].unique()):
        subset = mean_df.loc[mean_df["replicas"] == replicas].sort_values("users")
        ax.plot(
            subset["users"],
            subset["avg_response_time"],
            marker="o",
            label=f"replicas={replicas}",
        )

    ax.set_xlabel("Users")
    ax.set_ylabel("Average response time (ms)")
    ax.set_title("Mean average response time by user load")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Replica count")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    summary_csv = output_dir / "avg_response_time_summary.csv"
    mean_df.sort_values(["replicas", "users"]).to_csv(summary_csv, index=False)

    return figure_path


def collect_request_times(
    base_dir: Path,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Aggregate response times grouped by user count then replica count."""
    grouped: Dict[int, Dict[int, Iterable[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for log_path in base_dir.glob("*/replicas_*/run_*/request_log_unknown_run_*.csv"):
        try:
            users, replicas = parse_user_and_replica(log_path)
        except ValueError:
            continue

        try:
            df = pd.read_csv(log_path)
        except pd.errors.EmptyDataError:
            continue

        if "response_time" not in df.columns:
            continue

        values = (
            df["response_time"]
            .dropna()
            .astype(float)
            .to_numpy(copy=False)
        )
        if values.size == 0:
            continue
        grouped[users][replicas].append(values)

    aggregated: Dict[int, Dict[int, np.ndarray]] = {}
    for users, replicas_map in grouped.items():
        aggregated[users] = {}
        for replicas, arrays in replicas_map.items():
            aggregated[users][replicas] = np.concatenate(list(arrays))  # type: ignore[arg-type]
    return aggregated


def plot_response_time_distributions(
    distributions: Dict[int, Dict[int, np.ndarray]], output_dir: Path
) -> None:
    """Create histogram and CDF plots overlaid by replicas for each user count."""
    dist_dir = output_dir / "response_time_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    for users in sorted(distributions):
        replica_map = distributions[users]
        if not replica_map:
            continue

        all_values_list = [values for values in replica_map.values() if values.size]
        if not all_values_list:
            continue

        combined_values = np.concatenate(all_values_list)
        bins = np.histogram_bin_edges(combined_values, bins=50)

        fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        axes[0].set_title("Histogram (density)")
        axes[0].set_xlabel("Response time (ms)")
        axes[0].set_ylabel("Density")

        axes[1].set_title("CDF")
        axes[1].set_xlabel("Response time (ms)")
        axes[1].set_ylabel("Probability")
        axes[1].set_ylim(0, 1)

        for replicas in sorted(replica_map):
            values = replica_map[replicas]
            if values.size == 0:
                continue

            axes[0].hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=f"replicas={replicas}",
            )

            sorted_values = np.sort(values)
            cdf = np.linspace(0, 1, sorted_values.size, endpoint=True)
            axes[1].plot(sorted_values, cdf, label=f"replicas={replicas}")

        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].legend(title="Replica count", loc="lower right")
        fig.suptitle(f"response time users {users}")
        fig.tight_layout()

        output_path = dist_dir / f"response_time_users_{users}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def collect_service_p99(
    base_dir: Path,
) -> Dict[int, Dict[int, Dict[int, Dict[str, float]]]]:
    """Collect per-service mean p99 response times grouped by users, replicas, and run."""
    data: Dict[int, Dict[int, Dict[int, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    pattern = "*/replicas_*/run_*/services/*/prom_latency_*.csv"
    for csv_path in base_dir.glob(pattern):
        try:
            users, replicas = parse_user_and_replica(csv_path)
        except ValueError:
            continue

        run_number = parse_run_number(csv_path)
        if run_number is None:
            continue

        service_name = csv_path.parent.name
        if service_name == "frontend":
            continue
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue

        if "p99_ms" not in df.columns:
            continue

        values = pd.to_numeric(df["p99_ms"], errors="coerce").dropna()
        if values.empty:
            continue

        mean_p99 = float(values.mean())
        data[users][replicas][run_number][service_name] = mean_p99

    return data


def plot_locust_vs_service_p99(
    stats_df: pd.DataFrame,
    service_p99: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Create side-by-side plots comparing Locust stats to service p99 response times."""
    figure_dir = output_dir / "service_latency"
    figure_dir.mkdir(parents=True, exist_ok=True)

    available_keys = {
        (row["users"], row["replicas"])
        for _, row in stats_df[["users", "replicas"]].drop_duplicates().iterrows()
    }

    for users, replicas_map in service_p99.items():
        all_runs = sorted(
            {run for runs_data in replicas_map.values() for run in runs_data.keys()}
        )
        if not all_runs:
            continue

        for replicas, runs_data in replicas_map.items():
            if (users, replicas) not in available_keys:
                continue
            subset = stats_df[
                (stats_df["users"] == users) & (stats_df["replicas"] == replicas)
            ].copy()
            if subset.empty or not runs_data:
                continue

            subset = subset.dropna(subset=["run", "p99_response_time"])

            run_means = subset.groupby("run")["p99_response_time"].mean()
            if run_means.empty and not runs_data:
                continue

            run_numbers = all_runs
            run_labels = [str(int(run)) for run in run_numbers]
            locust_series = run_means.reindex(run_numbers)
            locust_array = locust_series.to_numpy(dtype=float)
            left_values = np.nan_to_num(locust_array, nan=0.0)

            fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

            # Left: Locust average response times per run
            axes[0].bar(run_labels, left_values, color="#1f77b4", alpha=0.8)
            axes[0].set_xlabel("Run")
            axes[0].set_ylabel("Locust 99th percentile (ms)")
            axes[0].set_title("Locust measured 99th percentile")
            axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)
            if not np.all(np.isnan(locust_array)):
                mean_val = float(np.nanmean(locust_series.to_numpy()))
                axes[0].axhline(
                    mean_val,
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Mean p99 {mean_val:.1f} ms",
                )
                axes[0].legend()

            # Right: Stacked bar of service p99 averages
            service_names = {
                service
                for run in run_numbers
                for service in runs_data.get(run, {})
            }
            if not service_names:
                plt.close(fig)
                continue
            service_names = sorted(
                service_names,
                key=lambda svc: sum(
                    runs_data.get(run, {}).get(svc, 0.0) for run in run_numbers
                ),
                reverse=True,
            )

            totals = np.zeros(len(run_numbers))
            for run_idx, run in enumerate(run_numbers):
                totals[run_idx] = sum(runs_data.get(run, {}).values())

            if locust_series.size and not np.all(np.isnan(locust_array)):
                max_locust = float(np.nanmax(locust_array))
            else:
                max_locust = 0.0
            max_services = totals.max() if totals.size else 0.0
            max_val = max(max_locust, max_services)
            if max_val <= 0:
                max_val = 1.0

            bottoms = np.zeros(len(run_numbers))
            for service in service_names:
                heights = np.array(
                    [runs_data.get(run, {}).get(service, 0.0) for run in run_numbers]
                )
                axes[1].bar(
                    run_labels,
                    heights,
                    bottom=bottoms,
                    label=service,
                )
                bottoms += heights

            axes[1].set_ylabel("Response time (ms)")
            axes[1].set_title("Service p99 mean response time (stacked)")
            axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
            axes[1].legend(loc="upper right", bbox_to_anchor=(1.02, 1))

            axes[0].set_ylim(0, max_val * 1.1)
            axes[1].set_ylim(0, max_val * 1.1)

            fig.suptitle(f"Users {users} / replicas {replicas}")
            fig.tight_layout()

            figure_path = (
                figure_dir
                / f"users_{users}_replicas_{replicas}_locust_vs_service_p99.png"
            )
            fig.savefig(figure_path, dpi=200, bbox_inches="tight")
            plt.close(fig)


def plot_service_p99_by_replica(
    service_p99: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    output_dir: Path,
    target_users: Optional[Set[int]] = None,
) -> None:
    """Plot stacked bars comparing service p99 means across replicas for each user count."""
    figure_dir = output_dir / "service_latency_by_replica"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for users, replicas_map in service_p99.items():
        if target_users is not None and users not in target_users:
            continue

        aggregated: Dict[int, Dict[str, float]] = {}
        for replicas, runs_data in replicas_map.items():
            service_values: Dict[str, List[float]] = defaultdict(list)
            for run_data in runs_data.values():
                for service, value in run_data.items():
                    service_values[service].append(value)
            service_means = {
                service: float(np.mean(values))
                for service, values in service_values.items()
                if values
            }
            if service_means:
                aggregated[replicas] = service_means

        if not aggregated:
            continue

        replica_ids = sorted(aggregated.keys())
        all_services = {
            service for data in aggregated.values() for service in data.keys()
        }
        if not all_services:
            continue

        service_order = sorted(
            all_services,
            key=lambda svc: sum(
                aggregated.get(rep, {}).get(svc, 0.0) for rep in replica_ids
            ),
            reverse=True,
        )

        x = np.arange(len(replica_ids))
        bottoms = np.zeros(len(replica_ids))

        fig, ax = plt.subplots(figsize=(8, 5))
        for service in service_order:
            heights = np.array(
                [aggregated.get(rep, {}).get(service, 0.0) for rep in replica_ids]
            )
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                label=service,
                width=0.6,
            )
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels([str(rep) for rep in replica_ids])
        ax.set_xlabel("Replica count")
        ax.set_ylabel("Response time (ms)")
        ax.set_title(f"Service p99 mean response time by replicas (users {users})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1), fontsize='small')

        fig.tight_layout()
        figure_path = figure_dir / f"users_{users}_service_p99_by_replica.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)


def collect_frontend_p99(
    base_dir: Path,
) -> Dict[int, Dict[int, Dict[int, float]]]:
    """Collect frontend mean p99 response times grouped by users, replicas, and run."""
    data: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

    pattern = "*/replicas_*/run_*/services/frontend/prom_latency_frontend.csv"
    for csv_path in base_dir.glob(pattern):
        try:
            users, replicas = parse_user_and_replica(csv_path)
        except ValueError:
            continue

        run_number = parse_run_number(csv_path)
        if run_number is None:
            continue

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue

        if "p99_ms" not in df.columns:
            continue

        values = pd.to_numeric(df["p99_ms"], errors="coerce").dropna()
        if values.empty:
            continue

        data[users][replicas][run_number] = float(values.mean())

    return data


def plot_frontend_vs_locust(
    stats_df: pd.DataFrame,
    frontend_p99: Dict[int, Dict[int, Dict[int, float]]],
    output_dir: Path,
) -> None:
    """Plot side-by-side comparison of Locust and frontend p99 response times."""
    figure_dir = output_dir / "frontend_latency"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for users, replicas_map in frontend_p99.items():
        all_runs = sorted(
            {run for runs_map in replicas_map.values() for run in runs_map.keys()}
        )
        if not all_runs:
            continue

        for replicas, runs_map in replicas_map.items():
            subset = stats_df[
                (stats_df["users"] == users) & (stats_df["replicas"] == replicas)
            ].copy()
            if subset.empty and not runs_map:
                continue

            subset = subset.dropna(subset=["run", "p99_response_time"])
            run_means = subset.groupby("run")["p99_response_time"].mean()

            if run_means.empty and not runs_map:
                continue

            run_numbers = all_runs
            locust_series = run_means.reindex(run_numbers)
            locust_array = locust_series.to_numpy(dtype=float)
            locust_values = np.nan_to_num(locust_array, nan=0.0)

            frontend_array = np.array([runs_map.get(run, np.nan) for run in run_numbers], dtype=float)
            frontend_values = np.nan_to_num(frontend_array, nan=0.0)

            if np.all(locust_values == 0.0) and np.all(frontend_values == 0.0):
                continue

            run_labels = [str(int(run)) for run in run_numbers]

            fig, ax = plt.subplots(figsize=(8, 4))
            indices = np.arange(len(run_numbers))
            width = 0.35

            ax.bar(indices - width / 2, locust_values, width=width, label="Locust p99", color="#1f77b4")
            ax.bar(indices + width / 2, frontend_values, width=width, label="Frontend p99", color="#ff7f0e")

            ax.set_xticks(indices)
            ax.set_xticklabels(run_labels)
            ax.set_xlabel("Run")
            ax.set_ylabel("Response time (ms)")
            ax.set_title(f"Locust p99 vs Frontend p99 (users {users}, replicas {replicas})")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax.legend()

            max_val = max(locust_values.max(), frontend_values.max()) if run_numbers else 0.0
            if max_val <= 0:
                max_val = 1.0
            ax.set_ylim(0, max_val * 1.1)

            fig.tight_layout()
            figure_path = (
                figure_dir
                / f"users_{users}_replicas_{replicas}_locust_vs_frontend.png"
            )
            fig.savefig(figure_path, dpi=200)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test",
        help="Root directory containing the user test results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to store generated figures and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_df = load_average_response_times(base_dir)
    if stats_df.empty:
        print(f"[WARN] No locust_stats.csv files found under {base_dir}")
    else:
        figure_path = plot_mean_response_times(stats_df, output_dir)
        print(f"[INFO] Saved mean response time plot to {figure_path}")

    service_p99 = collect_service_p99(base_dir)
    if service_p99 and not stats_df.empty:
        plot_locust_vs_service_p99(stats_df, service_p99, output_dir)
        print(f"[INFO] Saved service latency comparison figures to {output_dir}")
        plot_service_p99_by_replica(service_p99, output_dir, target_users={250})
        print(f"[INFO] Saved replica comparison figures to {output_dir}")
    elif service_p99:
        print("[WARN] Service latency data found, but Locust stats missing for comparison.")

    frontend_p99 = collect_frontend_p99(base_dir)
    if frontend_p99 and not stats_df.empty:
        plot_frontend_vs_locust(stats_df, frontend_p99, output_dir)
        print(f"[INFO] Saved frontend latency comparison figures to {output_dir}")
    elif frontend_p99:
        print("[WARN] Frontend latency data found, but Locust stats missing for comparison.")

    distributions = collect_request_times(base_dir)
    if not distributions:
        print(f"[WARN] No request logs found under {base_dir}")
    else:
        plot_response_time_distributions(distributions, output_dir)
        print(f"[INFO] Saved response time distributions to {output_dir}")


if __name__ == "__main__":
    main()
