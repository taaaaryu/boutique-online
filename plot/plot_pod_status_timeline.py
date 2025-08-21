#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import glob
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 固定設定
BASE_DIR = "locust_results/users"  # CSVファイルの場所
OUT_BASE_DIR = "locust_results/figures"  # 出力ディレクトリの場所
SERVICES = ["frontend","adservice","cartservice","checkoutservice","currencyservice",
            "emailservice","paymentservice","productcatalogservice","recommendationservice","shippingservice"]

def parse_groups(row, num_services=10, max_groups=9):
    """
    service_group_*列から各サービスの所属グループIDを抽出
    """
    group_ids = [-1]*num_services
    for gi in range(max_groups):
        col = f"service_group_{gi}"
        if col not in row or pd.isna(row[col]) or row[col]=="":
            continue
        try:
            vec = ast.literal_eval(row[col])
        except Exception:
            continue
        for s in range(min(num_services, len(vec))):
            if vec[s] == 1 and group_ids[s] == -1:
                group_ids[s] = gi
    return group_ids

def build_group_matrix(df, services, max_groups=9, use_opt_steps_only=True, forward_fill=True):
    """
    時系列の group_matrix を構築（shape: time x service）
    """
    if use_opt_steps_only and "optimize_flag" in df.columns:
        df_iter = df[df["optimize_flag"].fillna(0).astype(int) == 1].copy()
        if df_iter.empty:
            df_iter = df.copy()
    else:
        df_iter = df.copy()

    matrix = []
    for _, row in df_iter.iterrows():
        matrix.append(parse_groups(row, num_services=len(services), max_groups=max_groups))
    matrix = np.array(matrix, dtype=int)  # (time, service)

    if forward_fill and matrix.size > 0:
        for s in range(matrix.shape[1]):
            last = -1
            for i in range(matrix.shape[0]):
                if matrix[i, s] != -1:
                    last = matrix[i, s]
                else:
                    matrix[i, s] = last
    return matrix, df_iter

def stabilize_group_labels(group_matrix):
    """
    ステップ間でグループIDが入れ替わる問題に対し、
    前ステップとの最大重なりでラベルを安定化する（貪欲マッチング）。
    """
    if group_matrix.size == 0:
        return group_matrix
    gm = group_matrix.copy().astype(int)

    # 先頭はそのまま
    next_label = int(np.max(gm[0])) + 1 if np.max(gm[0]) >= 0 else 0

    for t in range(1, gm.shape[0]):
        prev = gm[t - 1]
        curr = gm[t].copy()
        prev_groups = sorted(set(prev[prev >= 0]))
        curr_groups = sorted(set(curr[curr >= 0]))

        if not prev_groups or not curr_groups:
            gm[t] = curr
            continue

        # 重なり行列
        overlap = np.zeros((len(prev_groups), len(curr_groups)), dtype=int)
        for i, pg in enumerate(prev_groups):
            for j, cg in enumerate(curr_groups):
                overlap[i, j] = int(np.sum((prev == pg) & (curr == cg)))

        assigned_prev = set()
        assigned_curr = set()
        mapping = {}

        # 最大の重なりから順に確定
        while overlap.size and len(assigned_prev) < len(prev_groups) and len(assigned_curr) < len(curr_groups):
            i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
            if overlap[i, j] == 0:
                break
            if i not in assigned_prev and j not in assigned_curr:
                mapping[curr_groups[j]] = prev_groups[i]
                assigned_prev.add(i)
                assigned_curr.add(j)
            overlap[i, :] = -1
            overlap[:, j] = -1

        # 未割当の現ステップグループには新ラベルを付与
        for cg in curr_groups:
            if cg not in mapping:
                mapping[cg] = next_label
                next_label += 1

        # マッピング適用
        for s in range(gm.shape[1]):
            g = curr[s]
            if g >= 0:
                curr[s] = mapping[g]
        gm[t] = curr

    return gm

def plot_availability_heatmap(df, services, ax, title="Redundancy Heatmap"):
    """
    冗長化ヒートマップ: サービス×時間、色 = running + paused
    """
    # 時刻
    t = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(range(len(df)))

    # 冗長化数計算
    mat_rows = []
    for svc in services:
        run_col = f"{svc}_running"
        pa_col = f"{svc}_paused"
        run = df[run_col].fillna(0.0).values if run_col in df.columns else np.zeros(len(df))
        pau = df[pa_col].fillna(0.0).values if pa_col in df.columns else np.zeros(len(df))
        mat_rows.append(run + pau)
    mat = np.array(mat_rows)  # (service, time)

    # 描画（数値範囲を最大5に設定）
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=5, cmap="RdYlGn")
    ax.set_yticks(range(len(services)))
    ax.set_yticklabels(services)

    # x軸ラベル（等間引き）
    if len(t) > 1:
        idx = np.linspace(0, len(t) - 1, num=min(10, len(t))).astype(int)
        labels = [pd.to_datetime(t.iloc[i]).strftime("%H:%M") for i in idx] if "timestamp" in df.columns else [str(int(i)) for i in idx]
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Time")
    ax.set_ylabel("Service")
    ax.set_title(title)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Redundancy (running + paused)")
    
    return im

def plot_group_membership_discrete(df, services, ax, title="Service Group Membership"):
    """
    サービスグループ可視化: 縦=サービス、横=最適化ステップ、色=安定化後のグループID
    """
    group_matrix, df_iter = build_group_matrix(df, services=services, max_groups=10,
                                              use_opt_steps_only=True, forward_fill=True)
    if group_matrix.size == 0:
        ax.text(0.5, 0.5, "No group data found", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return None
    
    group_matrix = stabilize_group_labels(group_matrix)  # (time x service)

    # 行列を (service x time) へ転置
    mat = group_matrix.T.astype(float)
    mat[mat < 0] = np.nan  # 未所属は NaN（グレー塗り）

    # 離散 colormap 準備（最大10個のグループに制限）
    unique_ids = sorted({int(v) for v in np.unique(mat[~np.isnan(mat)])})
    # 最大10個のグループに制限
    if len(unique_ids) > 10:
        unique_ids = unique_ids[:10]
        # 制限されたグループID以外はNaNに設定
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]) and int(mat[i, j]) not in unique_ids:
                    mat[i, j] = np.nan
    
    K = len(unique_ids)
    print(f"Number of groups: {K}")
    cmap = plt.get_cmap("tab20", K)

    # 0..K-1 に詰め直す
    id_to_idx = {gid: k for k, gid in enumerate(unique_ids)}
    mapped = np.full_like(mat, np.nan)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                mapped[i, j] = id_to_idx[int(mat[i, j])]

    # 描画
    im = ax.imshow(mapped, aspect="auto", interpolation="nearest", cmap=cmap)

    # y軸
    ax.set_yticks(range(len(services)))
    ax.set_yticklabels(services)

    # x軸（最適化ステップ）
    if "timestamp" in df_iter.columns:
        t = pd.to_datetime(df_iter["timestamp"])
        labels = [pd.to_datetime(x).strftime("%H:%M") for x in t]
    else:
        labels = [f"step {i}" for i in range(mat.shape[1])]

    if len(labels) > 10:
        keep = np.linspace(0, len(labels) - 1, num=10).astype(int)
        ax.set_xticks(keep)
        ax.set_xticklabels([labels[i] for i in keep], rotation=45, ha="right")
    else:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Optimize steps")
    ax.set_ylabel("Service")
    ax.set_title(title)

    # 凡例（安定化後のグループID、最大10個）
    if K > 0:
        colors = [cmap(i) for i in range(K)]
        handles = [Patch(facecolor=colors[id_to_idx[gid]], label=f"G{gid}") for gid in unique_ids]
        ax.legend(handles=handles, title="Group (stabilized)", loc="upper right", fontsize=8)

    return im

def plot_comparison(propose_csv, random_csv, out_path):
    """
    提案手法とランダム手法を比較する可視化
    """
    # データ読み込み
    df_propose = pd.read_csv(propose_csv)
    df_random = pd.read_csv(random_csv)
    
    # 図作成（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 上段: 冗長化ヒートマップ
    plot_availability_heatmap(df_propose, SERVICES, axes[0, 0], "Propose - Redundancy Heatmap")
    plot_availability_heatmap(df_random, SERVICES, axes[0, 1], "Random - Redundancy Heatmap")
    
    # 下段: サービスグループ可視化
    plot_group_membership_discrete(df_propose, SERVICES, axes[1, 0], "Propose - Service Group Membership")
    plot_group_membership_discrete(df_random, SERVICES, axes[1, 1], "Random - Service Group Membership")
    
    # タイトル
    propose_name = os.path.basename(propose_csv)
    random_name = os.path.basename(random_csv)
    fig.suptitle(f"Comparison: Propose vs Random\n{propose_name} vs {random_name}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def find_matching_csvs():
    """
    kill_0.001とkill_0.0005で、提案手法とランダム手法のマッチするCSVファイルを見つける（改善版）
    """
    comparisons = []
    target_kills = ["kill_0.001", "kill_0.0005"]
    
    # 各kill率とユーザー数の組み合わせでファイルを収集
    for kill_rate in target_kills:
        propose_files = {}
        random_files = {}
        
        # proposeファイルを収集
        propose_pattern = os.path.join(BASE_DIR, "propose", kill_rate, "*", "Hybrid", "pod_status-*.csv")
        for csv_file in glob.glob(propose_pattern, recursive=True):
            rel_path = os.path.relpath(csv_file, BASE_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) >= 4:
                user_count = parts[2]  # ユーザー数
                key = f"{kill_rate}_{user_count}"
                if key not in propose_files:
                    propose_files[key] = []
                propose_files[key].append(csv_file)
        
        # randomファイルを収集
        random_pattern = os.path.join(BASE_DIR, "random", kill_rate, "*", "Hybrid", "pod_status-*.csv")
        for csv_file in glob.glob(random_pattern, recursive=True):
            rel_path = os.path.relpath(csv_file, BASE_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) >= 4:
                user_count = parts[2]  # ユーザー数
                key = f"{kill_rate}_{user_count}"
                if key not in random_files:
                    random_files[key] = []
                random_files[key].append(csv_file)
        
        # マッチする組み合わせを見つける
        for key in propose_files:
            if key in random_files:
                # 最新のファイルを選択（タイムスタンプが新しいもの）
                propose_csv = max(propose_files[key], key=os.path.getctime)
                random_csv = max(random_files[key], key=os.path.getctime)
                
                kill_rate, user_count = key.split('_', 1)
                comparisons.append({
                    'kill_rate': kill_rate,
                    'user_count': user_count,
                    'propose_csv': propose_csv,
                    'random_csv': random_csv,
                    'propose_name': os.path.basename(propose_csv),
                    'random_name': os.path.basename(random_csv)
                })
    
    return comparisons

def main():
    """
    メイン処理
    """
    print("Pod Status Timeline Comparison (Propose vs Random)")
    print("=" * 60)
    print(f"Input directory: {BASE_DIR}")
    print(f"Output directory: {OUT_BASE_DIR}")
    print("-" * 60)
    
    # マッチするCSVファイルを見つける
    comparisons = find_matching_csvs()
    
    if not comparisons:
        print("No matching CSV files found for comparison.")
        return
    
    print(f"Found {len(comparisons)} comparison pairs")
    
    # 各比較ペアを処理
    for comp in comparisons:
        print(f"Processing: {comp['kill_rate']} - {comp['user_count']} users")
        print(f"  Propose: {comp['propose_name']}")
        print(f"  Random:  {comp['random_name']}")
        
        # 出力ディレクトリを作成
        out_dir = os.path.join(OUT_BASE_DIR, "comparison", comp['kill_rate'], comp['user_count'])
        os.makedirs(out_dir, exist_ok=True)
        
        # 比較図を作成
        out_name = f"comparison_{comp['kill_rate']}_{comp['user_count']}.png"
        out_path = os.path.join(out_dir, out_name)
        
        try:
            plot_comparison(comp['propose_csv'], comp['random_csv'], out_path)
        except Exception as e:
            print(f"Error processing comparison: {e}")
    
    print("\nComparison complete!")
    print(f"Check the '{OUT_BASE_DIR}/comparison' directory for output images.")

if __name__ == "__main__":
    main()