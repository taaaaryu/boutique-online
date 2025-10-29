#!/bin/bash
set -euo pipefail

# テストするユーザー数のリスト
USER_COUNTS=(50)

# 実行回数を指定
NUM_RUNS=3

# 各テストのランタイム
SPAWN_RATE=50
RUN_TIME="5m"

# テスト対象サービス一覧（明示的に限定してスケール制御）
SERVICES=(frontend adservice cartservice checkoutservice currencyservice emailservice paymentservice productcatalogservice recommendationservice shippingservice)

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile_test.py"

# テスト対象のホストURL（デフォルトは空。実行時に frontend Service から決定する）
HOST=""

# K8s Operator（観察モード実装版）のパス
OPERATOR_PATH_TEST="k8s-operator/my-operator/replicaset_test.py"

# 結果を保存するルートディレクトリ
RESULTS_DIR_TEST="locust_results/users/test"


# 観察モードの環境（必要に応じて外部から上書き可）
OBSERVE_ONLY="${OBSERVE_ONLY:-true}"
# 観察モードでは冗長化数は常に1に固定（必要なら外部から SCALE_STEPS を上書き可能）
SCALE_STEPS="${SCALE_STEPS:-1}"
STEP_DURATION_SECONDS="${STEP_DURATION_SECONDS:-300}"
OBSERVE_WAIT_READY="${OBSERVE_WAIT_READY:-true}"

# Prometheus 収集の周期と集計ウィンドウ（秒）
# 例: export METRICS_SCRAPE_INTERVAL_SECONDS=60 MONITOR_TIME_SECONDS=60
METRICS_SCRAPE_INTERVAL_SECONDS="${METRICS_SCRAPE_INTERVAL_SECONDS:-30}"
MONITOR_TIME_SECONDS="${MONITOR_TIME_SECONDS:-$METRICS_SCRAPE_INTERVAL_SECONDS}"

# 冗長化数ごとに個別の実験を行うための手順（スペース区切りで指定）
# 例: export REPLICA_STEPS="1 2 3 4"
REPLICA_STEPS=${REPLICA_STEPS:-"1 2 3 4"}

# frontend Service を NodePort に変更し、NodeIP:NodePort から HOST を決定する
derive_host_from_frontend() {
    echo "Determining HOST from 'frontend' Service (will set Service to NodePort if necessary)..."

    # Ensure the service exists
    if ! kubectl get svc frontend >/dev/null 2>&1; then
        echo "Error: Service 'frontend' not found in the current namespace"
        return 1
    fi

    # Ensure type is NodePort
    svc_type=$(kubectl get svc frontend -o jsonpath='{.spec.type}' 2>/dev/null || true)
    if [ "${svc_type}" != "NodePort" ]; then
        echo "Patching 'frontend' Service to NodePort..."
        kubectl patch svc frontend -p '{"spec":{"type":"NodePort"}}' || true
        # small wait for Kubernetes to apply the change
        sleep 3
    else
        echo "Service 'frontend' already NodePort"
    fi

    # Get nodePort (take first port's nodePort)
    node_port=$(kubectl get svc frontend -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true)
    if [ -z "$node_port" ]; then
        echo "Failed to obtain nodePort for 'frontend' service"
        return 1
    fi

    # Get a node IP: prefer ExternalIP, fall back to InternalIP
    node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || true)
    if [ -z "$node_ip" ]; then
        node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || true)
    fi
    if [ -z "$node_ip" ]; then
        echo "Failed to determine a node IP from 'kubectl get nodes'"
        return 1
    fi

    HOST="http://$node_ip:$node_port"
    echo "Derived HOST: $HOST"
    return 0
}



# 関数：デプロイメントの準備完了を確認
check_deployments_ready() {
    local max_wait=60
    local wait_time=0
    local check_interval=5
    
    echo "Checking if all deployments are ready..."
    
    while [ $wait_time -lt $max_wait ]; do
        # すべてのデプロイメントが準備完了しているかチェック
        if kubectl get deployments --no-headers | awk '{print $2}' | grep -v "0/0" | grep -q "/"; then
            ready_count=$(kubectl get deployments --no-headers | awk '{print $2}' | grep -E "^[0-9]+/[0-9]+$" | wc -l)
            total_count=$(kubectl get deployments --no-headers | wc -l)
            
            if [ $ready_count -eq $total_count ]; then
                echo "All deployments are ready"
                return 0
            fi
        fi
        
        echo "Deployments not yet ready, waiting... ($wait_time/$max_wait seconds)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    echo "Warning: Deployment readiness timeout after $max_wait seconds"
    return 1
}

# 関数：K8s Operatorを開始（観察モード）
start_operator() {
    local log_dir="$1"
    echo "==== Starting Kubernetes Operator (replicaset_test) ===="
    mkdir -p "$log_dir"
    # Derive namespace from current context if available
    local current_ns
    current_ns=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true)
    if [ -z "$current_ns" ]; then
        current_ns="default"
    fi
    # Pass through optional PROMETHEUS_URL if set in the environment
    LOG_DIR="$log_dir" ARCH_TYPE="users" NAMESPACE="$current_ns" \
    PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}" \
    METRICS_SCRAPE_INTERVAL_SECONDS="$METRICS_SCRAPE_INTERVAL_SECONDS" \
    MONITOR_TIME_SECONDS="$MONITOR_TIME_SECONDS" \
    OBSERVE_ONLY="$OBSERVE_ONLY" SCALE_STEPS="$SCALE_STEPS" \
    STEP_DURATION_SECONDS="$STEP_DURATION_SECONDS" OBSERVE_WAIT_READY="$OBSERVE_WAIT_READY" \
    kopf run "$OPERATOR_PATH_TEST" > "$log_dir/replicaset_operator.log" 2>&1 &
    OPERATOR_PID=$!
    echo "$OPERATOR_PID" > "replicaset_operator.pid"
    echo "Operator started with PID: $OPERATOR_PID (logs: $log_dir/replicaset_operator.log)"
    # Operatorの起動待ち
    sleep 15
}

# 関数：K8s Operatorを停止
stop_operator() {
    echo "==== Stopping Kubernetes Operator ===="
    if [ -f "replicaset_operator.pid" ]; then
        OPERATOR_PID=$(cat "replicaset_operator.pid")
        echo "Stopping operator with PID: $OPERATOR_PID"
        kill "$OPERATOR_PID" 2>/dev/null || echo "Process already stopped"
        rm -f "replicaset_operator.pid"
        # 待機して完全終了を確認、残っていれば強制終了
        for t in 1 2 3; do
            if pgrep -f "kopf run $OPERATOR_PATH_TEST" >/dev/null 2>&1; then
                sleep 3
            else
                break
            fi
        done
        if pgrep -f "kopf run $OPERATOR_PATH_TEST" >/dev/null 2>&1; then
            echo "Operator still running. Sending SIGKILL..."
            pkill -9 -f "kopf run $OPERATOR_PATH_TEST" || true
        fi
    else
        pkill -f "kopf run $OPERATOR_PATH_TEST" || true
    fi
    echo "Operator cleanup completed"
}

# 関数：Podを初期化（必要に応じて利用）
initialize_pods() {
    local note="${1:-}"
    echo "==== Initializing pods ${note} ===="
    echo "Deleting all pods..."
    kubectl delete pod --all || true
    echo "Waiting for pods to be recreated..."
    sleep 20
    if check_deployments_ready; then
        echo "Pod initialization completed successfully"
    else
        echo "Warning: Pod initialization may not have completed properly"
    fi
}

# 関数：特定のアーキテクチャラベルでテストを実行
run_tests_for_user_count() {
    local user_count="$1"
    # ディレクトリ構造: <RESULTS_ROOT>/<user_count>/replicas_<n>/run_<i>
    local user_dir="$RESULTS_DIR_TEST/${user_count}"

    echo "========================================"
    echo "Starting tests for ${user_count} users"
    echo "========================================"

    mkdir -p "$user_dir"

    # フェーズ1: Podを初期化（必要なら有効化）
    # initialize_pods "users=${user_count}"

    # フェーズ2: 準備完了チェック
    echo "==== Final preparation check ===="
    if check_deployments_ready; then
        echo "All systems ready for load testing"
    else
        echo "Warning: Some deployments may not be ready, but proceeding with test"
    fi

    # フェーズ3: 負荷テストを開始（Runごとに専用フォルダを作成し、OperatorもRun単位で起動/停止）
    echo "==== Starting load testing ===="
    for replica in $REPLICA_STEPS; do
        echo "---- Replica=$replica ----"
        for i in $(seq 1 "$NUM_RUNS"); do
            echo "==== ${user_count} users | replicas ${replica} | Run $i/$NUM_RUNS ===="
            replica_dir="$user_dir/replicas_${replica}"
            run_dir="$replica_dir/run_${i}"
            mkdir -p "$run_dir"

            # 事前に対象サービスのreplicasを明示的に合わせる（HPA等の影響を受けにくく）
            echo "Pre-scaling target services to replicas=${replica}"
            for svc in "${SERVICES[@]}"; do
                kubectl scale deployment "$svc" --replicas="$replica" || true
            done

            # RunごとにOperatorを起動（SCALE_STEPSを単一値にして、その値に固定）
            SCALE_STEPS="$replica" start_operator "$run_dir"

            LOG_DIR="$run_dir" RUN_NUM="$i" \
            locust -f "$LOCUSTFILE" --headless -u "${user_count}" -r "$SPAWN_RATE" \
                   --run-time "$RUN_TIME" --host "$HOST" \
                   --logfile "$run_dir/locust.log" \
                   --csv "$run_dir/locust" \
                   --csv-full-history

            # Run単位でオペレーターを停止
            stop_operator

            echo "Results saved to: $run_dir"
            echo "==== Finished ${user_count} | replicas ${replica} | Run $i ===="
            if [ "$i" -lt "$NUM_RUNS" ]; then
                echo "Waiting between test runs..."
                sleep 20
            fi
        done
    done
    echo "Completed tests for ${user_count} users"
}

# トラップ設定（スクリプト終了時に確実にクリーンアップ）
cleanup() {
    echo "==== Performing cleanup ===="
    stop_operator || true
    echo "Cleanup completed"
}
trap cleanup EXIT

# メイン処理開始
echo "==== Starting Locust User Scaling Test with Replicaset-Test Operator ===="

# ベース結果ディレクトリを作成
mkdir -p "$RESULTS_DIR_TEST"

# 動的に HOST を決定（frontend Service -> NodePort -> NodeIP:NodePort）
if [ -z "${HOST:-}" ]; then
    if ! derive_host_from_frontend; then
        echo "Warning: Could not derive HOST from frontend service; using fallback localhost:8080"
        HOST="http://127.0.0.1:8080"
    fi
fi

# ユーザー数ごとにループ
for user_count in "${USER_COUNTS[@]}"; do
    echo "****************************************"
    echo "Starting tests for $user_count users"
    echo "****************************************"
    # 各ラベル（アーキタイプ名）でテストを実行
    run_tests_for_user_count "$user_count"
done

echo "========================================"
echo "All user scaling tests completed!"
echo "Results saved under: $RESULTS_DIR_TEST"
echo "========================================"
