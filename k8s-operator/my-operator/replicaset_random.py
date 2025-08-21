import kopf
from kubernetes import client, config
from kubernetes.stream import stream
import kubernetes
import numpy as np
import pandas as pd
import random
import logging
import threading
from datetime import datetime, timedelta
import time
from itertools import combinations, chain
import csv
import os
import sys  # これも忘れずに！
from datetime import datetime, timedelta
import collect_pod_logs
import importlib.util


GENERATION = 30
NUM_START = 50
NUM_NEXT = 30
all_deployments = ["frontend", "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice","productcatalogservice", "recommendationservice", "shippingservice"]
NAMESPACE = "default"
paused_pods = {}
service_groups = []  # グローバルなサービスグループ
pause_counts = {dep: 0 for dep in all_deployments}  # グローバルなpause回数辞書
# RM（Resilience Margin）サンプルを蓄積
rm_records = {dep: [] for dep in all_deployments}
r_adds=[0.8,1,1.2]
r_add=1.2
SERVICE_AVAILABILITY=0.99

KILL_PROBABILITY = 0.0001
SERVER_AVAILABILITY = 1
algo_interval = 300
kill_interval = 40
pause_interval = 80
log_interval = 20
PROGRAM_START_TIME = datetime.now()
last_optimize_time = None  # 前回のOptimize時刻を記録
# pause_intervalごとにファイルを分けるjp:
# 実行ごとに一意なCSVファイル名を生成し、すべてのログをこのファイルに記録します
LOG_DIR = os.environ.get("LOG_DIR", ".")
RUN_NUM = os.environ.get("RUN_NUM", "0")
ARCH_TYPE = os.environ.get("ARCH_TYPE", "unknown")
CSV_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
csv_filename = f"{LOG_DIR}/pod_status-{ARCH_TYPE}-{pause_interval}-{CSV_TIMESTAMP}.csv"
pod_log_csv = f"{LOG_DIR}/pod_http_log_{ARCH_TYPE}_run_{RUN_NUM}.csv"
REPLICA=2.5 # サービスあたりのレプリカ数,リソースに影響あり
# ---------------------------

# ---------------------------
# ランダムアルゴリズムの実装
# ---------------------------
def random_service_grouping(num_services, max_groups=None):
    """
    サービス実装形態をランダムに決定
    どのサービスがどのPodにデプロイされるかをランダムに決定
    """
    if max_groups is None:
        max_groups = num_services
    
    # グループ数をランダムに決定（1からmax_groupsまで）
    num_groups = random.randint(1, min(max_groups, num_services))
    
    # 各サービスをランダムにグループに割り当て
    service_groups = []
    for i in range(num_groups):
        group = [0] * num_services
        service_groups.append(group)
    
    # 各サービスをランダムなグループに割り当て
    for service_idx in range(num_services):
        group_idx = random.randint(0, num_groups - 1)
        service_groups[group_idx][service_idx] = 1
    
    # 空のグループがあれば削除
    service_groups = [group for group in service_groups if sum(group) > 0]
    
    return service_groups

def random_redundancy_allocation(num_services, max_redundancy=5, total_resource_limit=None):
    """
    冗長化度合いをランダムに決定
    各Podがどれほど冗長化されるかをランダムに決定
    """
    if total_resource_limit is None:
        total_resource_limit = num_services * 3  # デフォルトのリソース制限
    
    # 各サービスにランダムな冗長化度合いを割り当て
    redundancy_list = []
    remaining_resource = total_resource_limit
    
    for i in range(num_services):
        if remaining_resource <= 0:
            redundancy_list.append(1)  # 最小値
        else:
            # 残りリソース内でランダムに冗長化度合いを決定
            max_possible = min(max_redundancy, remaining_resource)
            redundancy = random.randint(1, max_possible)
            redundancy_list.append(redundancy)
            remaining_resource -= redundancy
    
    return redundancy_list

def random_optimization_algorithm(num_services, max_redundancy=5, total_resource_limit=None):
    """
    ランダム最適化アルゴリズム
    サービス実装形態と冗長化度合いをランダムに決定
    """
    print("=== Random Optimization Algorithm ===")
    
    # 1. サービス実装形態をランダムに決定
    service_groups = random_service_grouping(num_services)
    print(f"Random service grouping: {service_groups}")
    
    # 2. 冗長化度合いをランダムに決定
    redundancy_list = random_redundancy_allocation(num_services, max_redundancy, total_resource_limit)
    print(f"Random redundancy allocation: {redundancy_list}")
    
    # 3. 結果を返す（既存の最適化アルゴリズムと同じ形式）
    best_solution_list = service_groups
    best_software_count = len(service_groups)
    best_RUE = random.uniform(0.1, 1.0)  # ランダムな評価値
    
    return best_solution_list, best_software_count, best_RUE, redundancy_list

# ---------------------------
# Greedy_Redundancyアルゴリズムの実装（シンプル版）
# ---------------------------
def greedy_redundancy(sw_avail, sw_resource, H, max_redundancy):
    num_sw = len(sw_avail)
    redundancy_list = [1] * num_sw
    sum_resource = np.sum(sw_resource)
    effective_avail = list(sw_avail)
    
    while sum_resource <= H:
        sorted_indices = np.argsort(effective_avail)
        updated = False
        for idx in sorted_indices:
            if redundancy_list[idx] >= max_redundancy:
                continue
            plus_resource = sw_resource[idx]
            if (sum_resource + plus_resource) <= H:
                redundancy_list[idx] += 1
                sum_resource += plus_resource
                effective_avail[idx] = 1 - (1 - sw_avail[idx]) ** redundancy_list[idx]
                updated = True
                break
        if not updated:
            break
    return redundancy_list

def parse_resource_limit(resource_limit_str, num_services):
    if resource_limit_str.endswith("n"):
        factor = int(resource_limit_str[:-1])
        return factor * num_services
    return int(resource_limit_str)

# === 追加: 可用性計算用関数 ===
def calculate_service_availability(csv_filename, all_deployments):
    if not os.path.exists(csv_filename):
        return [1.0] * len(all_deployments)
    df = pd.read_csv(csv_filename)
    optimize_rows = df.index[df["optimize_flag"] == 1].tolist()
    if optimize_rows:
        last_optimize_idx = optimize_rows[-1]
        logs_since_last_optimize = df.iloc[last_optimize_idx+1:] if last_optimize_idx+1 < len(df) else pd.DataFrame()
    else:
        logs_since_last_optimize = df
    service_availability = []
    for dep in all_deployments:
        running_col = f"{dep}_running"
        paused_col = f"{dep}_paused"
        total_running = logs_since_last_optimize[running_col].sum() if running_col in logs_since_last_optimize else 0
        total_paused = logs_since_last_optimize[paused_col].sum() if paused_col in logs_since_last_optimize else 0
        total = total_running + total_paused
        avail = total_running / total if total > 0 else 1.0
        service_availability.append(avail)
    return service_availability

def collect_pod_logs_timer(spec, logger, **kwargs):
    """
    定期的にPodログを収集するタイマー
    """
    logger.info("Collecting pod logs...")
    
    # ログディレクトリを作成
    os.makedirs(LOG_DIR, exist_ok=True)

    # 環境変数を設定
    os.environ['LOG_DIR'] = LOG_DIR
    os.environ['ARCH_TYPE'] = ARCH_TYPE
    os.environ['RUN_NUM'] = str(RUN_NUM)
    os.environ['NAMESPACE'] = NAMESPACE
    
    # collect_pod_logs.pyのmain()関数を呼び出し
    collect_pod_logs.main()
    logger.info(f"Pod logs collection completed. Files saved to {LOG_DIR}")


# ---------------------------
# Operator本体：CRD を監視して、最適化アルゴリズムを実行し、結果をCRD statusに反映
# ---------------------------

@kopf.on.create('myapp.example.com', 'v1alpha1', 'AppConfig')
def first_optimize(spec, meta, status, logger, **kwargs):
    logger.info("AppConfig created")
    optimize_appconfig(spec, meta, status, logger, **kwargs)

@kopf.timer('myapp.example.com', 'v1alpha1', 'AppConfig', interval=algo_interval)
def optimize_appconfig(spec, meta, status, logger, **kwargs):
    global service_groups, pause_counts, csv_filename, all_redundancy_list, last_optimize_time
    
    collect_pod_logs_timer(spec, logger, **kwargs)

    namespace = meta.get('namespace', 'default')
    
    max_redundancy = 5

    server_avail = SERVER_AVAILABILITY
    service_resource = 1
    num_services = len(all_deployments) - 1
    H = (num_services + 1) * REPLICA

    if not os.path.exists(csv_filename):
        service_avail = [SERVICE_AVAILABILITY] * len(all_deployments)
    else:
        service_avail = calculate_service_availability(csv_filename, all_deployments)
        logger.info(f"Calculated service availabilities: {service_avail}")
    
    last_optimize_time = datetime.now()

    # === ランダムアルゴリズムを実行 ===
    logger.info("Using RANDOM algorithm for service grouping and redundancy allocation")
    
    # ランダム最適化アルゴリズムを実行
    best_solution_list, best_software_count, best_RUE, all_redundancy_list = random_optimization_algorithm(
        num_services=len(all_deployments), 
        max_redundancy=max_redundancy, 
        total_resource_limit=H
    )
    
    logger.info(f"Random optimization result: service groups: {best_solution_list}, software count: {best_software_count}, redundancy list: {all_redundancy_list}")
    service_groups = best_solution_list
    
    config.load_kube_config()

    apps = client.AppsV1Api()
    for i, deployment in enumerate(all_deployments):
        replicas = all_redundancy_list[i]
        ns = "default"
        body = {"spec": {"replicas": replicas}}
        try:
            apps.patch_namespaced_deployment(deployment, ns, body)
            logger.info(f"Updated deployment: {deployment} with replicas: {replicas}")
        except kubernetes.client.exceptions.ApiException as e:
            logger.error(f"Failed to update deployment {deployment}: {e}")
    
    # === optimize時に拡張CSVログを出力 ===
    log_pod_status(spec, optimize_flag=1, service_groups=best_solution_list, service_availabilities=service_avail)



# ---- Helper ----
def get_deployment_name(pod):
    for owner in pod.metadata.owner_references or []:
        if owner.kind == "ReplicaSet":
            return owner.name.rsplit("-", 1)[0]
    return None




def scale_deployment(v1_apps, deployment_name, namespace, duration, logger):
    try:
        # 1) 現在の replicas を取得
        dep = v1_apps.read_namespaced_deployment(deployment_name, namespace)
        original = dep.spec.replicas or 0
        if original == 0:
            logger.warning(f"{deployment_name} の replicas が 0 なのでスケールダウンをスキップ")
            return

        # 2) スケールダウン
        new_replicas = original - 1
        patch = {"spec": {"replicas": new_replicas}}
        v1_apps.patch_namespaced_deployment(deployment_name, namespace, patch)
        logger.info(f"Scaled down {deployment_name}: {original} → {new_replicas}")

        # カウントも増やしておく
        pause_counts[deployment_name] += 1

        # 3) 停止時間待ち
        time.sleep(duration)

        # 4) 元に戻す
        patch = {"spec": {"replicas": original}}
        v1_apps.patch_namespaced_deployment(deployment_name, namespace, patch)
        logger.info(f"Restored {deployment_name}: {new_replicas} → {original}")

    except Exception as e:
        logger.error(f"Failed to scale deployment {deployment_name}: {e}")


@kopf.timer('myapp.example.com', 'v1alpha1', 'AppConfig', interval=kill_interval)
def kill_sidecar_timer(spec, logger, **kwargs):
    global service_groups, pause_counts

    if not service_groups:
        logger.warning("service_groups not yet initialized. Skipping this round.")
        return

    # in-cluster / kube-config の読み込み
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    apps_v1 = client.AppsV1Api()

    for svc_idx, deployment in enumerate(all_deployments):
        if random.random() >= KILL_PROBABILITY:
            continue

        # 所属グループが既に処理済みならスキップ
        grp = get_group_id(svc_idx)
        if grp == -1:
            continue

        # まずこのサービスをスケールダウン
        threading.Thread(
            target=scale_deployment,
            args=(apps_v1, deployment, NAMESPACE, pause_interval, logger),
        ).start()
        logger.info(f"Triggered scale-down for deployment: {deployment}")

        # 同じグループに属する他のサービスも１つずつダウンさせる
        for other_idx, belongs in enumerate(service_groups[grp]):
            if belongs != 1 or other_idx == svc_idx:
                continue
            dep2 = all_deployments[other_idx]
            threading.Thread(
                target=scale_deployment,
                args=(apps_v1, dep2, NAMESPACE, pause_interval, logger),
            ).start()
            logger.info(f"Triggered scale-down for sibling deployment: {dep2}")


def get_group_id(service_index):
    for idx, group in enumerate(service_groups):
        if group[service_index] == 1:
            return idx
    return -1

@kopf.timer('myapp.example.com', 'v1alpha1', 'AppConfig', interval=log_interval)
def log_pod_status_timer(spec, logger, **kwargs):
    log_pod_status(spec, optimize_flag=0, service_groups=None, service_availabilities=None, **kwargs)

def log_pod_status(spec, optimize_flag, service_groups, service_availabilities, **kwargs):
    global paused_pods, csv_filename, all_redundancy_list
    now = datetime.now()
    now_iso = now.isoformat()
    if datetime.now() - PROGRAM_START_TIME > timedelta(hours=10):
        print("3時間が経過したためプログラムを終了します。")
        sys.exit(0)

    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    v1 = client.CoreV1Api()

    deployments = apps_v1.list_namespaced_deployment(namespace=NAMESPACE).items
    desired_replicas = {dep.metadata.name: dep.spec.replicas for dep in deployments if dep.metadata.name in all_deployments}

    pods = v1.list_namespaced_pod(namespace=NAMESPACE).items
    status_counts = {dep: {"running": 0, "paused": 0} for dep in all_deployments}

    currently_paused_pods = set()
    now_epoch = time.time()
    for pod_name, resume_time in paused_pods.items():
        if now_epoch < resume_time:
            currently_paused_pods.add(pod_name)

    for pod in pods:
        deployment = get_deployment_name(pod)
        if deployment not in all_deployments:
            continue
        if not deployment:
            continue
        pod_name = pod.metadata.name
        if pod.status.phase == "Running":
            status_counts[deployment]["running"] += 1

    for dep in range(len(all_deployments)):
        total_expected = all_redundancy_list[dep]
        running_now = status_counts[all_deployments[dep]]["running"]
        paused_now = total_expected - running_now
        # pausedの値を0以上に制限し、累積ではなく現在の状態を反映
        status_counts[all_deployments[dep]]["paused"] = max(0, paused_now)

    # === 拡張CSVヘッダー ===
    max_groups = 9  # サービス数分
    header = ["timestamp"]
    for dep in all_deployments:
        header += [f"{dep}_running", f"{dep}_paused"]
    for i in range(max_groups):
        header.append(f"service_group_{i}")
    for i in range(max_groups):
        header.append(f"service_avail_{i}")
    header += ["optimize_flag", "pause_flag"]

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # === データ行 ===
    row = [now_iso]
    for dep in all_deployments:
        running = status_counts[dep]["running"]
        paused = status_counts[dep]["paused"]
        row += [running, paused]
    # サービスグループ
    if service_groups is None:
        service_groups = []
    for i in range(max_groups):
        if i < len(service_groups):
            row.append(str(service_groups[i]))
        else:
            row.append("")
    # 可用性
    if service_availabilities is None:
        service_availabilities = []
    for i in range(max_groups):
        if i < len(service_availabilities):
            row.append(str(service_availabilities[i]))
        else:
            row.append("")
    row += [optimize_flag, 0]
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
