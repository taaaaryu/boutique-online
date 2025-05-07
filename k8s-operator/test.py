import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# CSVの読み込み（ファイルが存在する必要あり）
df = pd.read_csv("k8s-operator/pod_status.csv", parse_dates=['timestamp'])

# サービス一覧取得
services = [col.replace('_running', '') for col in df.columns if col.endswith('_running')]

# イベントタイム取得
optimize_times = df[df['optimize_flag'] == 1]['timestamp']
pause_times = df[df['pause_flag'] == 1]['timestamp']

# サブプロット
fig, axes = plt.subplots(len(services), 1, figsize=(14, 2.5 * len(services)), sharex=True)

for i, service in enumerate(services):
    ax = axes[i]
    ax.plot(df['timestamp'], df[f"{service}_running"], label=f"{service}", color='tab:blue')

    # イベントマーカー描画
    for t in optimize_times:
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1, label='optimize' if t == optimize_times.iloc[0] else "")
    for t in pause_times:
        ax.axvline(x=t, color='purple', linestyle=':', linewidth=1, label='pause' if t == pause_times.iloc[0] else "")

    ax.set_ylabel("Running Pods")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xlabel("Time")
plt.suptitle("Running Pods per Service with Event Markers", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# 保存
os.makedirs("k8s-operator/graphs", exist_ok=True)
plt.savefig("k8s-operator/graphs/running_pods_subplots.png", dpi=150)
plt.show()
