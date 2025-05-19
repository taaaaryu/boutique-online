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

# ---------------------------
# ユーザー提供のアルゴリズムコード（グローバル定数はCRDから渡すので、ここではデフォルト値）
# ---------------------------
GENERATION = 10
NUM_NEXT = 10
all_deployments = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice","productcatalogservice", "recommendationservice", "shippingservice"]
NAMESPACE = "boutique"
KILL_PROBABILITY = 0.05  # 各サービスがkillされる確率
paused_pods = {}
service_groups = []  # グローバルなサービスグループ
pause_counts = {dep: 0 for dep in all_deployments}  # グローバルなpause回数辞書
# RM（Resilience Margin）サンプルを蓄積
rm_records = {dep: [] for dep in all_deployments}
r_add=1.05
algo_interval = 120
kill_interval = 40
pause_interval = 40*kill_interval
log_interval = 20
PROGRAM_START_TIME = datetime.now()
# pause_intervalごとにファイルを分けるjp:
CSV_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
csv_filename = f"pod_status-{pause_interval}-{CSV_TIMESTAMP}.csv"
REPLICA=3
# ---------------------------


def calc_software_av(services_group, service_avail, services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

def calc_software_av_matrix(services_in_sw, service_avail, server_avail):
    services_array = np.array(services_in_sw, dtype=int)
    #print("Services array:", services_array, "Service avail:", service_avail)   
    sw_avail_list = []
    count = 0
    for k in services_array:
        sw_avail = 1
        for i in range(k):
            sw_avail *= service_avail[count]
            count += 1
        sw_avail_list.append(sw_avail * server_avail)
    return sw_avail_list

def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    # 各グループの可用性の平均を評価指標として計算（簡易版）
    sum_matrix = np.sum(matrix, axis=1)
    software_availability = calc_software_av_matrix(sum_matrix, service_avail, server_avail)
    system_avail = np.prod(software_availability)
    matrix_resource = (r_add ** (sum_matrix - 1)) * sum_matrix * 1  # service_resource=1と仮定
    total_servers = np.sum(matrix_resource)
    return system_avail / total_servers if total_servers > 0 else 0

def make_matrix(service, software_count):
    # service: numpy array, software_count: int
    matrix = np.zeros((software_count, len(service) + 1), dtype=int)
    service_list = service.tolist()
    a = random.sample(service_list, software_count - 1)
    a.append(len(service) + 1)
    a.sort()
    idx = 0
    for i in range(software_count):
        for k in range(idx, a[i]):
            matrix[i][k] = 1
            idx += 1
    return matrix

def divide_sw(matrix, one_list):
    flag = 0
    cp_list = one_list.copy()
    while flag == 0:
        idx = random.randint(0, len(cp_list) - 2)
        start = cp_list[idx]
        end = cp_list[idx + 1]
        if end - start > 1:
            a = random.randint(start + 1, end - 1)
            div_matrix = np.insert(matrix, idx + 1, 0, axis=0)
            for i in range(a, cp_list[idx + 1]):
                div_matrix[idx][i] = 0
                div_matrix[idx + 1][i] = 1
            flag = 1
        else:
            continue
    return div_matrix

def integrate_sw(matrix, one_list):
    cp_list = one_list.copy()
    idx = random.randint(1, len(cp_list) - 2)
    start = cp_list[idx - 1]
    end = cp_list[idx + 1]
    for i in range(start, end):
        matrix[idx - 1][i] = 1
    new_matrix = np.delete(matrix, idx, 0)
    return new_matrix

def find_ones(matrix):
    arr = np.array(matrix)
    rows, cols = np.nonzero(arr)
    positions = [[col + 1 for col in cols[rows == row]] for row in np.unique(rows)]
    return positions

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H):
    best_RUEs = [-np.inf] * NUM_NEXT
    best_matrices = [None] * NUM_NEXT
    best_counts = [0] * NUM_NEXT

    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    for k in range(GENERATION):
        RUE_list = [best_RUE]
        matrix = best_matrix.copy()
        one_list = []
        col = 0
        for i in range(len(matrix[0])):
            if matrix[col][i] == 0:
                one_list.append(i)
                col += 1

        mini_RUE_list = [0]
        matrix_list = [[0]]
        for j in range(len(one_list)):
            a = one_list[j]
            one = matrix.copy()
            one[j][a - 1] = 0
            one[j][a] = 1

            one_new_RUE = calc_RUE(one, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(one_new_RUE)
            matrix_list.append(one)
            two = matrix.copy()
            two[j][a - 1] = 1
            two[j][a] = 0
            two_new_RUE = calc_RUE(two, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(two_new_RUE)
            matrix_list.append(two)

        new_RUE = max(mini_RUE_list)
        idx = mini_RUE_list.index(new_RUE)
        new_matrix = matrix_list[idx]
        RUE_list.append(new_RUE)

        one_list.append(len(matrix[0]))
        one_list.insert(0, 0)

        if software_count <= len(matrix[0]) - 1:
            new_sw_p_matrix = divide_sw(matrix, one_list)
            new_RUE_p = calc_RUE(new_sw_p_matrix, len(new_sw_p_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_p)
        else:
            new_RUE_p = 0

        if software_count >= 2:
            new_sw_n_matrix = integrate_sw(matrix, one_list)
            new_RUE_n = calc_RUE(new_sw_n_matrix, len(new_sw_n_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_n)
        else:
            new_RUE_n = 0

        max_RUE = max(RUE_list)

        if max_RUE > best_RUE:
            if max_RUE == new_RUE:
                best_RUE = new_RUE
                best_matrix = new_matrix
            elif max_RUE == new_RUE_p:
                best_RUE = max_RUE
                best_matrix = new_sw_p_matrix
                software_count += 1
            elif max_RUE == new_RUE_n:
                best_RUE = max_RUE
                best_matrix = new_sw_n_matrix
                software_count -= 1
        else:
            best_RUE = max_RUE

        if best_RUE > best_RUEs[0]:
            for i in range(NUM_NEXT - 1, 0, -1):
                best_RUEs[i] = best_RUEs[i - 1]
                best_matrices[i] = best_matrices[i - 1]
                best_counts[i] = best_counts[i - 1]
            best_RUEs[0] = best_RUE
            best_matrices[0] = best_matrix
            best_counts[0] = software_count
        else:
            for i in range(1, NUM_NEXT):
                if best_RUE > best_RUEs[i]:
                    for j in range(NUM_NEXT - 1, i, -1):
                        best_RUEs[j] = best_RUEs[j - 1]
                        best_matrices[j] = best_matrices[j - 1]
                        best_counts[j] = best_counts[j - 1]
                    best_RUEs[i] = best_RUE
                    best_matrices[i] = best_matrix
                    best_counts[i] = software_count
                    break
    return best_matrices, best_counts, best_RUEs, RUE_list

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, NUM_START):
    best_global_matrices = [None] * NUM_NEXT
    best_global_RUEs = [-np.inf] * NUM_NEXT
    best_global_counts = [0] * NUM_NEXT
    RUE_list = []
    x_gene = np.arange(1, GENERATION + 1)
    service = np.arange(1, num_service + 1)
    n = num_service  # n をサービス数とする

    software_count_float = np.random.normal(num_service / 2, 2, NUM_START)
    software_counts = np.clip(software_count_float.astype(int), 1, n)

    for software_count in software_counts:
        matrix = make_matrix(service, software_count)
        best_matrices, best_counts, best_RUEs_local, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H)
        RUE_list.append(RUE_each_list)
        for i in range(NUM_NEXT):
            if best_RUEs_local[i] > best_global_RUEs[i]:
                if best_matrices[i] is not None and (best_global_matrices[i] is None or not np.array_equal(best_matrices[i], best_global_matrices[i])):
                    best_global_matrices[i] = best_matrices[i]
                    best_global_counts[i] = best_counts[i]
                    best_global_RUEs[i] = best_RUEs_local[i]
    print("Best global RUEs:", best_global_RUEs)
    return best_global_matrices, best_global_counts, best_global_RUEs

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

# ---------------------------
# Operator本体：CRD を監視して、最適化アルゴリズムを実行し、結果をCRD statusに反映
# ---------------------------
@kopf.on.create('myapp.example.com', 'v1alpha1', 'AppConfig')
def init_pod_status(spec, logger, **kwargs):
    logger.info("初期化: log_pod_status を一度実行して CSV を作成します")
    log_pod_status(spec)

@kopf.on.create('myapp.example.com', 'v1alpha1', 'AppConfig')
@kopf.timer('myapp.example.com', 'v1alpha1', 'AppConfig', interval=algo_interval)
def optimize_appconfig(spec, meta, status, logger, **kwargs):
    global service_groups, pause_counts, csv_filename

    namespace = meta.get('namespace', 'boutique')
    preferences = spec.get('preferences', {})
    generation = int(preferences.get('generation', GENERATION))
    NUM_START = int(preferences.get('numStart', 50))
    max_redundancy = int(preferences.get('maxReplicas', 3))

    server_avail = 0.95
    service_resource = 1
    num_services = len(all_deployments) - 1
    H = (num_services + 1) * REPLICA

    if not os.path.exists(csv_filename):
        # CSVが存在しない = 初回実行
        service_avail = [0.99] * len(all_deployments)
        logger.warning(f"{csv_filename} not found. Using default 0.99 availability.")
    else:
        df = pd.read_csv(csv_filename, parse_dates=["timestamp"])

        # optimize_flag == 1 の行インデックスを探す
        optimize_rows = df.index[df["optimize_flag"] == 1].tolist()

        if len(optimize_rows) >= 3:
            start_idx = optimize_rows[-3]
            df_filtered = df.loc[start_idx+1:]
        else:
            df_filtered = df

        service_avail = []
        for dep in all_deployments:
            run_col = f"{dep}_running"
            pause_col = f"{dep}_paused"

            total_running = df_filtered[run_col].sum()
            total_paused = df_filtered[pause_col].sum()
            total = total_running + total_paused

            avail = total_running / total if total > 0 else 1.0
            service_avail.append(avail)

        logger.info(f"Calculated service availabilities: {service_avail}")

    pause_counts = {dep: 0 for dep in all_deployments}

    best_matrices, best_counts, best_RUEs = multi_start_greedy(r_add, service_avail, server_avail, H, num_services, NUM_START)
    best_solution = best_matrices[0]
    best_solution_list = best_solution.tolist() if isinstance(best_solution, np.ndarray) else best_solution
    best_software_count = int(best_counts[0])
    best_RUE = float(best_RUEs[0])

    groups = find_ones(best_solution)
    group_sizes = [sum(row) for row in best_solution]
    group_avail = []
    size_start = 0
    size_end = 0
    for size in group_sizes:
        size_end += size
        prod = np.prod(service_avail[size_start:size_end])
        group_avail.append(prod * server_avail)
        size_start += size
    sw_resource = [size * service_resource for size in group_sizes]

    redundancy_list = greedy_redundancy(group_avail, sw_resource, H, max_redundancy)
    redundancy_list = [int(r) for r in redundancy_list]
    group_sizes = [int(size) for size in group_sizes]
    all_redundancy_list = []
    for i in range(len(group_sizes)):
        all_redundancy_list += [redundancy_list[i]] * group_sizes[i]
    all_redundancy_list = [int(r) for r in all_redundancy_list]

    logger.info(f"Optimization result (grouping): best solution matrix: {best_solution_list}, software count: {best_software_count}, RUE: {best_RUE}")
    service_groups = best_solution_list
    config.load_kube_config()

    apps = client.AppsV1Api()
    for i, deployment in enumerate(all_deployments):
        replicas = all_redundancy_list[i]
        ns = "boutique"
        body = {"spec": {"replicas": replicas}}
        try:
            apps.patch_namespaced_deployment(deployment, ns, body)
            logger.info(f"Updated deployment: {deployment} with replicas: {replicas}")
        except kubernetes.client.exceptions.ApiException as e:
            logger.error(f"Failed to update deployment {deployment}: {e}")
    if os.path.exists(csv_filename := f"pod_status-{CSV_TIMESTAMP}.csv"):
        df = pd.read_csv(csv_filename)
        if not df.empty:
            df.loc[df.index[-1], "optimize_flag"] = 1
            df.to_csv(csv_filename, index=False)
        else:
            logger.warning("optimize_flag を立てる前に log_pod_status を一度実行してください。")
    else:
        logger.warning("optimize_flag を立てるための CSV がまだ存在しません。")



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
def log_pod_status(spec, **kwargs):
    global paused_pods, csv_filename
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
        if pod_name in currently_paused_pods:
            status_counts[deployment]["paused"] += 1
        elif pod.status.phase == "Running":
            status_counts[deployment]["running"] += 1

    for dep in all_deployments:
        if dep not in desired_replicas:
            continue
        total_expected = desired_replicas[dep]
        running_now = status_counts[dep]["running"]
        paused_now = status_counts[dep]["paused"]
        unknown = max(total_expected - (running_now + paused_now), 0)
        status_counts[dep]["running"] += unknown

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["timestamp"]
            for dep in all_deployments:
                header += [f"{dep}_running", f"{dep}_paused"]
            header += ["optimize_flag", "pause_flag"]
            writer.writerow(header)

    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [now_iso]
        for dep in all_deployments:
            running = status_counts[dep]["running"]
            paused = status_counts[dep]["paused"]
            row += [running, paused]
        row += [0, 0]
        writer.writerow(row)
