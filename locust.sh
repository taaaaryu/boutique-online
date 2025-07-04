# run_locust_multiple_times.sh

#!/bin/bash

# 実行回数を指定（例: 5回）
NUM_RUNS=5

# 各テストのユーザー数やランタイムなどのパラメータ
USERS=1000
SPAWN_RATE=250
RUN_TIME="30m"  # 30分間テスト

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile.py"

# テスト対象のホストURL
HOST="http://172.18.0.2:32479"

# K8s Operatorのパス
OPERATOR_PATH="k8s-operator/my-operator/replicaset.py"

# PIDファイルの保存場所
OPERATOR_PID_FILE="replicaset_operator.pid"

# アーキテクチャタイプとr_addの値の定義
declare -A ARCHITECTURES
ARCHITECTURES[Mono]=0.75
ARCHITECTURES[Hybrid]=1.0
ARCHITECTURES[Micro]=1.25

# 関数：r_addの値を変更してファイルを編集
modify_r_add() {
    local r_add_value=$1
    local arch_type=$2
    
    echo "Modifying r_add to $r_add_value for $arch_type architecture..."
    
    # r_addの値を変更
    sed -i.tmp "s/r_add=r_adds\[0\]/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=0\.75/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=1\.0/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=1\.25/r_add=$r_add_value/" "$OPERATOR_PATH"
    
    # CSVファイル名にアーキテクチャタイプを含める
    sed -i.tmp "s/csv_filename = f\"pod_status-{pause_interval}-{CSV_TIMESTAMP}.csv\"/csv_filename = f\"pod_status-$arch_type-{pause_interval}-{CSV_TIMESTAMP}.csv\"/" "$OPERATOR_PATH"
    sed -i.tmp "s/csv_filename = f\"pod_status-[^-]*-{pause_interval}-{CSV_TIMESTAMP}.csv\"/csv_filename = f\"pod_status-$arch_type-{pause_interval}-{CSV_TIMESTAMP}.csv\"/" "$OPERATOR_PATH"
    
    # 一時ファイルを削除
    rm -f "${OPERATOR_PATH}.tmp"
    
    echo "Modified replicaset.py for $arch_type (r_add=$r_add_value)"
}

# 関数：K8s Operatorを開始
start_operator() {
    local arch_type=$1
    echo "==== Starting Kubernetes Operator for $arch_type ===="
    
    # kopf でオペレーターをバックグラウンド実行
    echo "Starting replicaset operator for $arch_type..."
    kopf run "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "$OPERATOR_PID_FILE"
    echo "Operator started with PID: $OPERATOR_PID for $arch_type"
    
    # オペレーターの初期化を待つ
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
        # プロセスが完全に停止するまで少し待つ
        sleep 5
    fi
    
    echo "Operator cleanup completed"
}

# 関数：特定のアーキテクチャでテストを実行
run_tests_for_architecture() {
    local arch_type=$1
    local r_add_value=$2
    
    echo "========================================"
    echo "Starting tests for $arch_type Architecture (r_add=$r_add_value)"
    echo "========================================"
    
    # r_addの値を変更
    modify_r_add "$r_add_value" "$arch_type"
    
    # K8s Operatorを開始
    start_operator "$arch_type"
    
    # 各テストラウンドを実行
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type Run $i/$NUM_RUNS ===="
        locust -f "$LOCUSTFILE" --headless -u $USERS -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "locust_${arch_type}_run_${i}.log"
        echo "==== Finished $arch_type Run $i ===="
        
        # 次のテストまでの待機時間（オペレーターが結果を処理する時間を確保）
        if [ $i -lt $NUM_RUNS ]; then
            echo "Waiting between test runs..."
            sleep 30
        fi
    done
    
    # オペレーターを停止
    stop_operator
    
    # テスト間の待機時間
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
echo "Results saved with prefixes: Mono, Hybrid, Micro"
echo "========================================"

# cleanup は EXIT トラップで自動実行される
