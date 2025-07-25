#!/bin/bash

# テストするユーザー数のリスト
USER_COUNTS=(1500 2000 2500 3000)

# 実行回数を指定
NUM_RUNS=1

# 各テストのランタイム
SPAWN_RATE=100
RUN_TIME="15m"

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile.py"

# テスト対象のホストURL
HOST="http://172.18.0.3:30716"

# K8s Operatorのパス
OPERATOR_PATH="k8s-operator/my-operator/replicaset.py"

# 結果を保存するルートディレクトリ
RESULTS_DIR="locust_results"

# アーキテクチャタイプとr_addの値の定義
declare -A ARCHITECTURES
ARCHITECTURES[Mono]=0.75
ARCHITECTURES[Hybrid]=1.0
ARCHITECTURES[Micro]=1.25

# 関数：r_addの値を変更してファイルを編集
modify_r_add() {
    local r_add_value=$1
    echo "Modifying r_add to $r_add_value..."
    sed -i.tmp "s/^r_add=.*$/r_add=$r_add_value/" "$OPERATOR_PATH"
    rm -f "${OPERATOR_PATH}.tmp"
    echo "Modified replicaset.py \(r_add=$r_add_value\)"
}

# 関数：K8s Operatorを開始
start_operator() {
    local arch_type=$1
    local log_dir=$2
    echo "==== Starting Kubernetes Operator for $arch_type ===="
    echo "Starting replicaset operator for $arch_type..."
    LOG_DIR="$log_dir" ARCH_TYPE="$arch_type" kopf run "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "replicaset_operator.pid"
    echo "Operator started with PID: $OPERATOR_PID for $arch_type"
    sleep 15  # Operatorの起動を待つ
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
    kubectl delete pod --all --namespace=default --ignore-not-found=true
    echo "Waiting for pods to be recreated..."
    sleep 10  # Podの再作成を待つ
    echo "Pod initialization completed for $arch_type"
}

# 関数：特定のアーキテクチャでテストを実行
run_tests_for_architecture() {
    local user_count=$1
    local arch_type=$2  
    local r_add_value=$3
    local arch_dir="$RESULTS_DIR/users/$user_count/$arch_type"

    echo "========================================"
    echo "Starting tests for $user_count users, $arch_type Architecture \(r_add=$r_add_value\)"
    echo "========================================"

    # アーキテクチャごとのディレクトリを作成
    mkdir -p "$arch_dir"
    echo "Results will be saved in: $arch_dir"

    # r_addの値を変更
    modify_r_add "$r_add_value"

    # Podを初期化
    initialize_pods "$arch_type"

    # K8s Operatorを開始
    start_operator "$arch_type" "$arch_dir"

    # 各テストラウンドを実行
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type Run $i/$NUM_RUNS ===="
        LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i locust -f "$LOCUSTFILE" --headless -u $user_count -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "$arch_dir/locust_${arch_type}_run_${i}.log" --csv "$arch_dir/locust_${arch_type}_run_${i}"
        #LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i python3 k8s-operator/my-operator/collect_pod_logs.py
        echo "==== Finished $arch_type Run $i ===="
        if [ $i -lt $NUM_RUNS ]; then
            echo "Waiting between test runs..."
            sleep 20
        fi
    done

    # オペレーターを停止
    stop_operator

    echo "Waiting between architecture tests..."
    sleep 60
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
mkdir -p "$RESULTS_DIR"

# ファイル存在チェック
if [ ! -f "$LOCUSTFILE" ]; then
    echo "Error: Locustfile not found at $LOCUSTFILE"
    exit 1
fi
if [ ! -f "$OPERATOR_PATH" ]; then
    echo "Error: Operator script not found at $OPERATOR_PATH"
    exit 1
fi

# ユーザー数ごとにループ
for user_count in "${USER_COUNTS[@]}"; do
    echo "****************************************"
    echo "Starting tests for $user_count users"
    echo "****************************************"
    
    # 各アーキテクチャタイプでテストを実行
    for arch_type in Mono Hybrid Micro; do
        r_add_value=${ARCHITECTURES[$arch_type]}
        run_tests_for_architecture "$user_count" "$arch_type" "$r_add_value"
    done
done

echo "========================================"
echo "All user scaling tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "========================================"
