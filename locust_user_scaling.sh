#!/bin/bash

# テストするユーザー数のリスト
USER_COUNTS=(1000 1250 1500)

# 実行回数を指定
NUM_RUNS=3

# 各テストのランタイム
SPAWN_RATE=100
RUN_TIME="10m"

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile.py"

# テスト対象のホストURL
HOST="http://172.18.0.3:30957"

# K8s Operatorのパス
OPERATOR_PATH_RANDOM="k8s-operator/my-operator/replicaset_random.py"
OPERATOR_PATH_PROPOSE="k8s-operator/my-operator/replicaset.py"

# 結果を保存するルートディレクトリ
RESULTS_DIR_RANDOM="locust_results/users/random"
RESULTS_DIR_PROPOSE="locust_results/users/propose"

# アーキテクチャタイプとr_addの値の定義
declare -A ARCHITECTURES
ARCHITECTURES[Mono]=0.8
ARCHITECTURES[Hybrid]=1.0
ARCHITECTURES[Micro]=1.2

# テストする KILL_PROBABILITY の候補
KILL_PROB_LIST=(0.001 0.0005 0.0001)
KILL_TAG=""

# 関数：r_addの値を変更してファイルを編集
modify_r_add() {
    local r_add_value=$1
    echo "Modifying r_add to $r_add_value..."
    sed -i.tmp "s/^r_add=.*$/r_add=$r_add_value/" "$OPERATOR_PATH"
    rm -f "${OPERATOR_PATH}.tmp"
    echo "Modified replicaset.py (r_add=$r_add_value)"
}

# 関数：KILL_PROBABILITY を単一値に変更
modify_kill_probability() {
    local kill_prob_value=$1
    echo "Modifying KILL_PROBABILITY to $kill_prob_value..."
    sed -i.tmp "s/^KILL_PROBABILITY *=.*$/KILL_PROBABILITY = $kill_prob_value/" "$OPERATOR_PATH"
    rm -f "${OPERATOR_PATH}.tmp"
    echo "Modified replicaset.py (KILL_PROBABILITY=$kill_prob_value)"
}

# 関数：最適化完了を確認
check_optimization_complete() {
    local log_dir=$1
    local max_wait=120  # 最大待機時間（秒）
    local wait_time=0
    local check_interval=10  # チェック間隔（秒）
    
    echo "Waiting for optimization to complete..."
    
    while [ $wait_time -lt $max_wait ]; do
        # CSVファイルが存在し、最適化フラグが1の行があるかチェック
        if [ -f "$log_dir/pod_status-*.csv" ]; then
            csv_file=$(ls "$log_dir"/pod_status-*.csv | head -1)
            if [ -n "$csv_file" ] && [ -f "$csv_file" ]; then
                # 最適化フラグが1の行があるかチェック
                if grep -q ",1," "$csv_file"; then
                    echo "Optimization completed - found optimization flag in CSV"
                    return 0
                fi
            fi
        fi
        
        # ログファイルで最適化完了を確認
        if [ -f "$log_dir/replicaset_operator.log" ]; then
            if grep -q "Optimization result" "$log_dir/replicaset_operator.log"; then
                echo "Optimization completed - found optimization result in logs"
                return 0
            fi
        fi
        
        echo "Optimization not yet complete, waiting... ($wait_time/$max_wait seconds)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    echo "Warning: Optimization completion timeout after $max_wait seconds"
    return 1
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

# 関数：K8s Operatorを開始
start_operator() {
    local arch_type=$1
    local log_dir=$2
    echo "==== Starting Kubernetes Operator for $arch_type ===="
    echo "Starting replicaset operator for $arch_type..."
    LOG_DIR="$log_dir" ARCH_TYPE="$arch_type" kopf run "$OPERATOR_PATH" > "$log_dir/replicaset_operator.log" 2>&1 &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "replicaset_operator.pid"
    echo "Operator started with PID: $OPERATOR_PID for $arch_type"
    
    # Operatorの起動を待つ
    sleep 20
    
    # 最適化完了を待つ
    if check_optimization_complete "$log_dir"; then
        echo "Optimization phase completed successfully"
    else
        echo "Warning: Optimization phase may not have completed properly"
    fi
}

# 関数：K8s Operatorを停止
stop_operator() {
    echo "==== Stopping Kubernetes Operator ===="
    if [ -f "replicaset_operator.pid" ]; then
        OPERATOR_PID=$(cat "replicaset_operator.pid")
        echo "Stopping operator with PID: $OPERATOR_PID"
        kill $OPERATOR_PID 2>/dev/null || echo "Process already stopped"
        rm -f "replicaset_operator.pid"
        sleep 5
    else
        pkill -f "kopf run $OPERATOR_PATH"
    fi
    echo "Operator cleanup completed"
}

# 関数：Podを初期化
initialize_pods() {
    local arch_type=$1
    echo "==== Initializing pods for $arch_type ===="
    echo "Deleting all pods..."
    kubectl delete pod --all
    echo "Waiting for pods to be recreated..."
    sleep 20  # Podの再作成を待つ（時間を延長）
    
    # デプロイメントの準備完了を確認
    if check_deployments_ready; then
        echo "Pod initialization completed successfully"
    else
        echo "Warning: Pod initialization may not have completed properly"
    fi
}

# 関数：特定のアーキテクチャでテストを実行
run_tests_for_architecture() {
    local user_count=$1
    local arch_type=$2  
    local r_add_value=$3
    local arch_dir_random="$RESULTS_DIR_RANDOM/${KILL_TAG}/$user_count/$arch_type"
    local arch_dir_propose="$RESULTS_DIR_PROPOSE/${KILL_TAG}/$user_count/$arch_type"

    echo "========================================"
    echo "Starting tests for $user_count users, $arch_type Architecture (r_add=$r_add_value, ${KILL_TAG})"
    echo "========================================"

    # アーキテクチャごとのディレクトリを作成
    mkdir -p "$arch_dir_random"
    mkdir -p "$arch_dir_propose"
    echo "Results will be saved in: $arch_dir_random"

    # r_addの値を変更
    modify_r_add "$r_add_value"

    # フェーズ1: Podを初期化
    initialize_pods "$arch_type"

    # フェーズ2: K8s Operatorを開始して最適化アルゴリズムを実行
    start_operator "$arch_type" "$arch_dir_random"

    # フェーズ3: 最終的な準備完了確認
    echo "==== Final preparation check ===="
    if check_deployments_ready; then
        echo "All systems ready for load testing"
    else
        echo "Warning: Some deployments may not be ready, but proceeding with test"
    fi

    # フェーズ4: 負荷テストを開始
    echo "==== Starting load testing ===="
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type Run $i/$NUM_RUNS ===="
        LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i locust -f "$LOCUSTFILE" --headless -u $user_count -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "$arch_dir/locust_${arch_type}_run_${i}.log" --csv "$arch_dir/locust_${arch_type}_run_${i}"
        LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i python3 k8s-operator/my-operator/collect_pod_logs.py
        echo "==== Finished $arch_type Run $i ===="
        if [ $i -lt $NUM_RUNS ]; then
            echo "Waiting between test runs..."
            sleep 20
        fi
    done

    # オペレーターを停止
    stop_operator

    echo "Waiting between architecture tests..."
    sleep 30

    # フェーズ1: Podを初期化
    initialize_pods "$arch_type"

    # フェーズ2: K8s Operatorを開始して最適化アルゴリズムを実行
    start_operator "$arch_type" "$arch_dir_propose"

    # フェーズ3: 最終的な準備完了確認
    echo "==== Final preparation check ===="
    if check_deployments_ready; then
        echo "All systems ready for load testing"
    else
        echo "Warning: Some deployments may not be ready, but proceeding with test"
    fi

    # フェーズ4: 負荷テストを開始
    echo "Results will be saved in: $arch_dir_propose"
    echo "==== Starting load testing ===="
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type Run $i/$NUM_RUNS ===="
        LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i locust -f "$LOCUSTFILE" --headless -u $user_count -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "$arch_dir/locust_${arch_type}_run_${i}.log" --csv "$arch_dir/locust_${arch_type}_run_${i}"
        echo "==== Finished $arch_type Run $i ===="
    done
    stop_operator
    echo "Completed tests for $arch_type Architecture"
}

# トラップ設定（スクリプト終了時に確実にクリーンアップ）
cleanup() {
    echo "==== Performing cleanup ===="
    stop_operator
    echo "Cleanup completed"
}
trap cleanup EXIT

# メイン処理開始
echo "==== Starting Locust User Scaling Test with K8s Operator Integration ===="

# 結果ディレクトリを作成
mkdir -p "$RESULTS_DIR_RANDOM"
mkdir -p "$RESULTS_DIR_PROPOSE"

# KILL_PROBABILITY ごとにループ
for kill_prob in "${KILL_PROB_LIST[@]}"; do
    KILL_TAG="kill_${kill_prob}"
    echo "===== Testing with KILL_PROBABILITY=${kill_prob} (${KILL_TAG}) ====="
    modify_kill_probability "$kill_prob"

    # ユーザー数ごとにループ
    for user_count in "${USER_COUNTS[@]}"; do
        echo "****************************************"
        echo "Starting tests for $user_count users (${KILL_TAG})"
        echo "****************************************"
        
        # 各アーキテクチャタイプでテストを実行
        for arch_type in Mono Hybrid Micro; do
            r_add_value=${ARCHITECTURES[$arch_type]}
            run_tests_for_architecture "$user_count" "$arch_type" "$r_add_value"
        done
    done
done

echo "========================================"
echo "All user scaling tests completed!"
echo "Results saved in: $RESULTS_DIR_RANDOM"
echo "Results saved in: $RESULTS_DIR_PROPOSE"
echo "========================================"
