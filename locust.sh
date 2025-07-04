"""# locust.sh

#!/bin/bash

# 実行回数を指定
NUM_RUNS=1

# 各テストのユーザー数やランタイムなどのパラメータ
USERS=100
SPAWN_RATE=10
RUN_TIME="1m"  # 15秒間テスト

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile.py"

# テスト対象のホストURL
HOST="http://172.18.0.2:32479"

# K8s Operatorのパス
OPERATOR_PATH="k8s-operator/my-operator/replicaset.py"

# PIDファイルの保存場所
OPERATOR_PID_FILE="replicaset_operator.pid"

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
    }

# 関数：K8s Operatorを開始
start_operator() {
    local arch_type=$1
    local log_dir=$2
    echo "==== Starting Kubernetes Operator for $arch_type ===="
    echo "Starting replicaset operator for $arch_type..."
    LOG_DIR="$log_dir" ARCH_TYPE="$arch_type" kopf run "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "$OPERATOR_PID_FILE"
    echo "Operator started with PID: $OPERATOR_PID for $arch_type"
    sleep 15
}

# 関数：K8s Operatorを停止
stop_operator() {
    echo "==== Stopping Kubernetes Operator ===="
    if [ -f "$OPERATOR_PID_FILE" ]; then
        OPERATOR_PID=$(cat "$OPERATOR_PID_FILE")
        echo "Stopping operator with PID: $OPERATOR_PID"
        kill $OPERATOR_PID 2>/dev/null || echo "Process already stopped"
        rm -f "$OPERATOR_PID_FILE"
        sleep 5
    fi
    echo "Operator cleanup completed"
}

# 関数：特定のアーキテクチャでテストを実行
run_tests_for_architecture() {
    local arch_type=$1
    local r_add_value=$2
    local arch_dir="$RESULTS_DIR/$arch_type"

    echo "========================================"
    echo "Starting tests for $arch_type Architecture (r_add=$r_add_value)"
    echo "========================================"

    # アーキテクチャごとのディレクトリを作成
    mkdir -p "$arch_dir"
    echo "Results will be saved in: $arch_dir"

    # r_addの値を変更
    modify_r_add "$r_add_value"

    # K8s Operatorを開始
    start_operator "$arch_type" "$arch_dir"

    # 各テストラウンドを実行
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type Run $i/$NUM_RUNS ===="
        LOG_DIR="$arch_dir" ARCH_TYPE="$arch_type" RUN_NUM=$i locust -f "$LOCUSTFILE" --headless -u $USERS -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "$arch_dir/locust_${arch_type}_run_${i}.log" --csv "$arch_dir/locust_${arch_type}_run_${i}"
        echo "==== Finished $arch_type Run $i ===="
        if [ $i -lt $NUM_RUNS ]; then
            echo "Waiting between test runs..."
            sleep 30
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
echo "==== Starting Locust Test with K8s Operator Integration ===="
echo "Testing 3 architectures: Mono (r_add=0.75), Hybrid (r_add=1.0), Micro (r_add=1.25)"

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

# 各アーキテクチャタイプでテストを実行
for arch_type in Mono Hybrid Micro; do
    r_add_value=${ARCHITECTURES[$arch_type]}
    run_tests_for_architecture "$arch_type" "$r_add_value"
done

echo "========================================"
echo "All architecture tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "========================================"
""
