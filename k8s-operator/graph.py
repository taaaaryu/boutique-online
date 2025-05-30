import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV読み込みと前処理
df = pd.read_csv("k8s-operator/pod_status-20250530-135855.csv", parse_dates=['timestamp']).set_index('timestamp')
running_cols = [c for c in df.columns if c.endswith('_running')]

# 出力先ディレクトリ作成
output_dir = "k8s-operator/graphs/services"
os.makedirs(output_dir, exist_ok=True)

# サービスごとに1つずつグラフを作成・保存
for col in running_cols:
    service_name = col.replace('_running', '')
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[col], label=f"{service_name} running pods", color='tab:blue')
    plt.title(f'{service_name} – Running Pods Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Running Pods')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{service_name}_running.png", dpi=150)
    plt.close()
