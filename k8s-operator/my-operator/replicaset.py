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

KILL_PROBABILITY = 0.001
SERVER_AVAILABILITY = 1
algo_interval = 150
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
# Greedy_Redundancyアルゴリズムの実装
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

"""
def calculate_success_availability(services_dir, deployments, since_time=None):
    rates = []
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    for dep in deployments:
        service_csv = os.path.join(services_dir, f"{dep}.csv")
        if not os.path.exists(service_csv):
            rates.append(1.0)
            continue
        try:
            df = pd.read_csv(service_csv)
            if df.empty or 'timestamp' not in df.columns:
                rates.append(1.0)
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if since_time is not None:
                df = df[df['timestamp'] >= since_time]
            if len(df) < 2:
                rates.append(1.0)
                continue
            timestamps = df['timestamp'].unique()
            if len(timestamps) < 2:
                rates.append(1.0)
                continue
            latest_timestamp = timestamps[-1]
            previous_timestamp = timestamps[-2]
            latest_df = df[df['timestamp'] == latest_timestamp]
            previous_df = df[df['timestamp'] == previous_timestamp]
            # サービス全体の合計値で差分を取る
            def sum_counts(df, col):
                return df[col].sum() if col in df.columns else 0
            latest_success = sum_counts(latest_df, 'code_200s')
            previous_success = sum_counts(previous_df, 'code_200s')
            latest_fail = sum_counts(latest_df, 'code_400s') + sum_counts(latest_df, 'code_500s') + sum_counts(latest_df, 'code_timeout')
            previous_fail = sum_counts(previous_df, 'code_400s') + sum_counts(previous_df, 'code_500s') + sum_counts(previous_df, 'code_timeout')
            print(f"{dep}: Latest success: {latest_success}, Previous success: {previous_success}, Latest fail: {latest_fail}, Previous fail: {previous_fail}")
            success_count = latest_success - previous_success
            fail_count = latest_fail - previous_fail
            total_count = success_count + fail_count
            rate = success_count / total_count if total_count > 0 else 1.0
            print(f"{dep}: Success count: {success_count}, Fail count: {fail_count}, Rate: {rate}")
            rates.append(rate)
        except Exception as e:
            print(f"Error reading {service_csv}: {e}")
            rates.append(1.0)
    return rates
"""



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
        #service_avail = [random.uniform(0.95, 1.0) for i in range(len(all_deployments))]
        service_avail = [SERVICE_AVAILABILITY] * len(all_deployments)
    else:
        # === 可用性を前回optimize以降のlogから計算 =
        service_avail = calculate_service_availability(csv_filename, all_deployments)
        logger.info(f"Calculated service availabilities: {service_avail}")
    # 現在時刻を前回のOptimize時刻として記録
    last_optimize_time = datetime.now()


    # === r_addの値に応じてサービスグループを決定 ===
    logger.info(f"Current r_add value: {r_add}")
    
    if abs(r_add - 0.75) < 0.01:  # r_add = 0.75 (Mono architecture)
        logger.info("Using Monolithic architecture: All services in one group")
    
        best_solution_list = [[1] * len(all_deployments)] #[[1,1,1,1,1,1,1,1,1,1]]
        best_solution = best_solution_list  # リスト全体を渡す
        print("Monolithic architecture: best_solution_list:", best_solution_list)
        best_software_count = 1
        best_RUE = 1.0
        
    elif abs(r_add - 1.25) < 0.01:  # r_add = 1.25 (Micro architecture)
        logger.info("Using Microservices architecture: Each service in separate group")
        # 各サービスを個別のグループに配置
        best_solution_list = []
        for i in range(len(all_deployments)):
            group = [0] * len(all_deployments)
            group[i] = 1
            best_solution_list.append(group)
        best_solution = best_solution_list  # リスト全体を渡す
        print("Microservices architecture: best_solution_list:", best_solution_list)
        best_software_count = len(all_deployments)
        best_RUE = 0.0  # ダミー値
        
    else:  # r_add = 1.0 or other values (Hybrid architecture)
        logger.info("Using Hybrid architecture: Running optimization algorithm")
        # 既存の最適化アルゴリズムを実行
        best_matrices, best_counts, best_RUEs = multi_start_greedy(r_add, service_avail, server_avail, H, num_services, NUM_START)
        print("Hybrid architecture: best_matrices:", best_matrices[0],best_matrices[1],best_matrices[2])
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

    logger.info(f"Optimization result (grouping): best solution matrix: {best_solution_list}, software count: {best_software_count}, all redundancy list: {all_redundancy_list}")
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


