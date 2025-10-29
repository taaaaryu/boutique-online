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
all_deployments = ["frontend", "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice","productcatalogservice", "recommendationservice", "shippingservice","redis-cart"]
NAMESPACE = "default"

paused_pods = {}
service_groups = []  # グローバルなサービスグループ
pause_counts = {dep: 0 for dep in all_deployments}  # グローバルなpause回数辞書
# RM（Resilience Margin）サンプルを蓄積
rm_records = {dep: [] for dep in all_deployments}

SERVICE_AVAILABILITY=0.99

SERVER_AVAILABILITY = 1
algo_interval = 500
kill_interval = 40
pause_interval = 80
log_interval = 20
PROGRAM_START_TIME = datetime.now()
last_optimize_time = None  # 前回のOptimize時刻を記録
# pause_intervalごとにファイルを分けるjp:
# 実行ごとに一意なCSVファイル名を生成し、すべてのログをこのファイルに記録します
LOG_DIR = os.environ.get("LOG_DIR", ".")
RUN_NUM = os.environ.get("RUN_NUM", "0")

ARCH_TYPE = os.environ.get("ARCH_TYPE", "test")
CSV_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
csv_filename = f"{LOG_DIR}/pod_status-{ARCH_TYPE}-{CSV_TIMESTAMP}.csv"
pod_log_csv = f"{LOG_DIR}/pod_http_log_{ARCH_TYPE}_run_{RUN_NUM}.csv"

# デフォルトの冗長度（各サービスのレプリカ数）を初期化
all_redundancy_list = [1] * len(all_deployments)

def _parse_scale_steps(env_val: str):
    try:
        parts = [p.strip() for p in env_val.split(",")]
        steps = [int(p) for p in parts if p]
        return steps if steps else [1]
    except Exception:
        return [1]

# 観察モード（アルゴリズムを実行せず、全サービス個別グループ＋一括スケールシーケンス）
OBSERVE_ONLY = os.environ.get("OBSERVE_ONLY", "true").lower() in ("1", "true", "yes")
SCALE_STEPS = _parse_scale_steps(os.environ.get("SCALE_STEPS", "1"))
STEP_DURATION_SECONDS = int(os.environ.get("STEP_DURATION_SECONDS", "500"))
OBSERVE_DONE_FLAG = os.path.join(LOG_DIR, "observe_only_done.flag")

# スケール後に Ready を確認してからログ収集するかどうか（既定有効）
OBSERVE_WAIT_READY = os.environ.get("OBSERVE_WAIT_READY", "true").lower() in ("1", "true", "yes")
READINESS_TIMEOUT_SECONDS = int(os.environ.get("READINESS_TIMEOUT_SECONDS", "100"))
READINESS_POLL_INTERVAL_SECONDS = int(os.environ.get("READINESS_POLL_INTERVAL_SECONDS", "5"))

# メトリクス収集周期（秒）: 定期的に Prometheus のCPU/メモリ/ネットワークを収集
METRICS_SCRAPE_INTERVAL_SECONDS = int(os.environ.get("METRICS_SCRAPE_INTERVAL_SECONDS", "30"))

# ---------------------------


def collect_pod_logs_timer(spec, logger, **kwargs):
    """
    定期的にPodログを収集するタイマー
    """
    logger.info("Collecting pod logs...")
    
    # ログディレクトリを作成
    os.makedirs(LOG_DIR, exist_ok=True)

    # os.environ を用いた変数設定は禁止: collect_pod_logs のモジュール変数を直接上書きする
    try:
        collect_pod_logs.LOG_DIR = LOG_DIR
        collect_pod_logs.SERVICE_DIR = os.path.join(LOG_DIR, "services")
        collect_pod_logs.ARCH_TYPE = ARCH_TYPE
        collect_pod_logs.RUN_NUM = str(RUN_NUM)
        collect_pod_logs.NAMESPACE = NAMESPACE
    except Exception as e:
        logger.error(f"collect_pod_logs への設定反映に失敗しました: {e}")

    # collect_pod_logs.pyのmain()関数を呼び出し（環境変数ではなくモジュール変数で参照される）
    collect_pod_logs.main()
    logger.info(f"Pod logs collection completed. Files saved to {LOG_DIR}")


# 定期メトリクス収集タイマー（AppConfig 単位でintervalごとに実行）
@kopf.timer('myapp.example.com', 'v1alpha1', 'AppConfig', interval=METRICS_SCRAPE_INTERVAL_SECONDS)
def collect_metrics_timer(spec, logger, **kwargs):
    try:
        collect_pod_logs_timer(spec, logger, **kwargs)
    except Exception as e:
        logger.error(f"collect_metrics_timer failed: {e}")


def _micro_grouping():
    """全サービスを個別のグループに配置する行列（リスト）を返す。"""
    best_solution_list = []
    for i in range(len(all_deployments)):
        group = [0] * len(all_deployments)
        group[i] = 1
        best_solution_list.append(group)
    return best_solution_list

def _set_all_replicas(apps_api, replicas, logger):
    """全デプロイメントのreplicasを一括で設定する。"""
    for dep_name in all_deployments:
        body = {"spec": {"replicas": replicas}}
        try:
            apps_api.patch_namespaced_deployment(dep_name, NAMESPACE, body)
            logger.info(f"Set replicas for {dep_name} -> {replicas}")
        except kubernetes.client.exceptions.ApiException as e:
            logger.error(f"Failed to set replicas {replicas} for {dep_name}: {e}")

def _all_deployments_ready(apps_api, logger) -> bool:
    """全ターゲットデプロイメントが desired replicas に到達しているか確認する。"""
    for dep_name in all_deployments:
        try:
            dep = apps_api.read_namespaced_deployment(dep_name, NAMESPACE)
            desired = dep.spec.replicas or 0
            status = dep.status or None
            ready = (status.ready_replicas or 0) if status else 0
            updated = (status.updated_replicas or 0) if status else 0
            available = (status.available_replicas or 0) if status else 0
            if desired == 0:
                continue
            if updated < desired or ready < desired or available < desired:
                logger.debug(f"Not ready: {dep_name} desired={desired} ready={ready} updated={updated} available={available}")
                return False
        except kubernetes.client.exceptions.ApiException as e:
            logger.error(f"Failed to read deployment {dep_name}: {e}")
            return False
    return True

def _wait_all_ready(apps_api, timeout_s: int, poll_s: int, logger) -> bool:
    """全デプロイメントが Ready になるのを待機。タイムアウトしたら False を返す。"""
    start = time.time()
    while time.time() - start < timeout_s:
        if _all_deployments_ready(apps_api, logger):
            return True
        time.sleep(poll_s)
    return False


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
    
    # 観察モードではアルゴリズムは実行せず、全サービス個別グループ＋一括スケールシーケンスのみ実施
    if OBSERVE_ONLY:
        logger.info("OBSERVE_ONLY mode enabled: micro-grouping and uniform scaling sequence.")
        try:
            # 既に完了していたら何もしない
            if os.path.exists(OBSERVE_DONE_FLAG):
                logger.info("Observe-only sequence already completed. Skipping.")
                return

            # micro grouping を設定
            best_solution_list = _micro_grouping()
            service_groups = best_solution_list

            # APIクライアント
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            apps = client.AppsV1Api()

            # スケールシーケンス実行
            for replicas in SCALE_STEPS:
                logger.info(f"Uniformly scaling all services to replicas={replicas}")
                _set_all_replicas(apps, replicas, logger)

                # Ready 待ち（任意、有効が既定）
                if OBSERVE_WAIT_READY:
                    ok = _wait_all_ready(apps, READINESS_TIMEOUT_SECONDS, READINESS_POLL_INTERVAL_SECONDS, logger)
                    if ok:
                        logger.info("All deployments are Ready after scaling. Proceeding to log collection.")
                    else:
                        logger.warning("Readiness wait timed out. Proceeding to log collection anyway.")

                # 再計算された冗長度（replicas）を反映してログを出す
                all_redundancy_list = [replicas for _ in all_deployments]
                log_pod_status(spec, optimize_flag=1, service_groups=best_solution_list, service_availabilities=None)

                # Pod/Prometheusログ収集（スケール後 Ready 確認直後）
                collect_pod_logs_timer(spec, logger, **kwargs)

                # 次ステップまでの待機（Ready 待機とは独立して間隔を維持）
                logger.info(f"Sleeping {STEP_DURATION_SECONDS}s before next scaling step...")
                time.sleep(STEP_DURATION_SECONDS)

            # 完了フラグ
            with open(OBSERVE_DONE_FLAG, 'w') as f:
                f.write(datetime.now().isoformat())
            logger.info("Observe-only scaling sequence completed.")
        except Exception as e:
            logger.error(f"Error in observe-only sequence: {e}")
        finally:
            return



# ---- Helper ----
def get_deployment_name(pod):
    for owner in pod.metadata.owner_references or []:
        if owner.kind == "ReplicaSet":
            return owner.name.rsplit("-", 1)[0]
    return None


def scale_deployment(v1_apps, deployment_name, namespace, duration, logger):
    # 既存アルゴリズム関連のスケールダウンロジックは削除（観察モードのみ使用）
    logger.info("scale_deployment は未使用のためスキップします（観察モード）")


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
    desired_replicas = {dep.metadata.name: (dep.spec.replicas or 0) for dep in deployments if dep.metadata.name in all_deployments}

    pods = v1.list_namespaced_pod(namespace=NAMESPACE).items
    status_counts = {dep: {"running": 0, "paused": 0} for dep in all_deployments}

    # 旧アルゴリズムの一時停止管理は削除

    for pod in pods:
        deployment = get_deployment_name(pod)
        if deployment not in all_deployments:
            continue
        if not deployment:
            continue
        pod_name = pod.metadata.name
        if pod.status.phase == "Running":
            status_counts[deployment]["running"] += 1

    for dep_name in all_deployments:
        desired = desired_replicas.get(dep_name, 0)
        running_now = status_counts[dep_name]["running"]
        paused_now = max(0, desired - running_now)
        status_counts[dep_name]["paused"] = paused_now

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


