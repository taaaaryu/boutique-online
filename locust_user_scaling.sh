# locust_user_scaling.sh

#!/bin/bash

# 実行回数を指定（各ユーザー数・アーキテクチャ組み合わせごと）
NUM_RUNS=3

# 各テストのランタイムなどのパラメータ
SPAWN_RATE=250
RUN_TIME="15m"  # ユーザー数テストなので短縮

# locustfile.pyのパス
LOCUSTFILE="src/loadgenerator/locustfile.py"

# テスト対象のホストURL
HOST="http://172.18.0.2:32479"

# K8s Operatorのパス
OPERATOR_PATH="k8s-operator/my-operator/replicaset.py"

# PIDファイルの保存場所
OPERATOR_PID_FILE="replicaset_operator.pid"

# ユーザー数の配列（スケーリングテスト用）
USER_COUNTS=(100 500 1000 2000 5000)

# アーキテクチャタイプとr_addの値の定義
declare -A ARCHITECTURES
ARCHITECTURES[Mono]=0.75
ARCHITECTURES[Hybrid]=1.0
ARCHITECTURES[Micro]=1.25

# 結果ディレクトリの作成
RESULTS_DIR="scaling_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 関数：r_addの値を変更してファイルを編集
modify_r_add() {
    local r_add_value=$1
    local arch_type=$2
    local user_count=$3
    
    echo "Modifying r_add to $r_add_value for $arch_type architecture with $user_count users..."
    
    # r_addの値を変更
    sed -i.tmp "s/r_add=r_adds\[0\]/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=0\.75/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=1\.0/r_add=$r_add_value/" "$OPERATOR_PATH"
    sed -i.tmp "s/r_add=1\.25/r_add=$r_add_value/" "$OPERATOR_PATH"
    
    # CSVファイル名にアーキテクチャタイプとユーザー数を含める
    sed -i.tmp "s/csv_filename = f\"pod_status-{pause_interval}-{CSV_TIMESTAMP}.csv\"/csv_filename = f\"pod_status-$arch_type-$user_count-{pause_interval}-{CSV_TIMESTAMP}.csv\"/" "$OPERATOR_PATH"
    sed -i.tmp "s/csv_filename = f\"pod_status-[^-]*-{pause_interval}-{CSV_TIMESTAMP}.csv\"/csv_filename = f\"pod_status-$arch_type-$user_count-{pause_interval}-{CSV_TIMESTAMP}.csv\"/" "$OPERATOR_PATH"
    sed -i.tmp "s/csv_filename = f\"pod_status-[^-]*-[^-]*-{pause_interval}-{CSV_TIMESTAMP}.csv\"/csv_filename = f\"pod_status-$arch_type-$user_count-{pause_interval}-{CSV_TIMESTAMP}.csv\"/" "$OPERATOR_PATH"
    
    # 一時ファイルを削除
    rm -f "${OPERATOR_PATH}.tmp"
    
    echo "Modified replicaset.py for $arch_type with $user_count users (r_add=$r_add_value)"
}

# 関数：K8s Operatorを開始
start_operator() {
    local arch_type=$1
    local user_count=$2
    echo "==== Starting Kubernetes Operator for $arch_type with $user_count users ===="
    
    # kopf でオペレーターをバックグラウンド実行
    echo "Starting replicaset operator for $arch_type..."
    kopf run "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "$OPERATOR_PID_FILE"
    echo "Operator started with PID: $OPERATOR_PID for $arch_type with $user_count users"
    
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

# 関数：特定のアーキテクチャとユーザー数でテストを実行
run_tests_for_configuration() {
    local arch_type=$1
    local user_count=$2
    local r_add_value=$3
    
    echo "========================================"
    echo "Starting tests for $arch_type Architecture with $user_count users (r_add=$r_add_value)"
    echo "========================================"
    
    # r_addの値を変更
    modify_r_add "$r_add_value" "$arch_type" "$user_count"
    
    # K8s Operatorを開始
    start_operator "$arch_type" "$user_count"
    
    # 各テストラウンドを実行
    for i in $(seq 1 $NUM_RUNS)
    do
        echo "==== $arch_type with $user_count users - Run $i/$NUM_RUNS ===="
        
        # 結果ファイルのパス
        log_file="$RESULTS_DIR/locust_${arch_type}_${user_count}users_run_${i}.log"
        
        # Locustテストを実行
        locust -f "$LOCUSTFILE" --headless -u $user_count -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "$log_file"
        
        echo "==== Finished $arch_type with $user_count users - Run $i ===="
        
        # 次のテストまでの待機時間
        if [ $i -lt $NUM_RUNS ]; then
            echo "Waiting between test runs..."
            sleep 30
        fi
    done
    
    # オペレーターを停止
    stop_operator
    
    # 設定間の待機時間
    echo "Waiting between configurations..."
    sleep 60
    
    echo "Completed tests for $arch_type Architecture with $user_count users"
}

# 関数：結果の概要を生成
generate_summary() {
    echo "========================================"
    echo "Generating test summary..."
    echo "========================================"
    
    summary_file="$RESULTS_DIR/test_summary.txt"
    
    cat > "$summary_file" << EOF
Locust User Scaling Test Summary
Generated: $(date)

Test Configuration:
- Run time per test: $RUN_TIME
- Spawn rate: $SPAWN_RATE users/second
- Runs per configuration: $NUM_RUNS
- User counts tested: ${USER_COUNTS[@]}
- Architectures tested: Mono (r_add=0.75), Hybrid (r_add=1.0), Micro (r_add=1.25)

Results Directory Structure:
$RESULTS_DIR/
├── locust_{architecture}_{usercount}users_run_{run}.log
├── pod_status-{architecture}-{usercount}-{interval}-{timestamp}.csv
└── test_summary.txt

Total configurations tested: $((${#USER_COUNTS[@]} * 3))
Total test runs: $((${#USER_COUNTS[@]} * 3 * NUM_RUNS))

File Naming Convention:
- Locust logs: locust_{Mono|Hybrid|Micro}_{usercount}users_run_{1-$NUM_RUNS}.log
- Pod status: pod_status-{Mono|Hybrid|Micro}-{usercount}-{interval}-{timestamp}.csv

EOF

    echo "Summary generated at: $summary_file"
}

# トラップ設定（スクリプト終了時に確実にクリーンアップ）
cleanup() {
    echo "==== Performing cleanup ===="
    stop_operator
    generate_summary
    echo "Cleanup completed"
}
trap cleanup EXIT

# メイン処理開始
echo "========================================"
echo "Starting Locust User Scaling Test"
echo "========================================"
echo "Testing user counts: ${USER_COUNTS[@]}"
echo "Testing architectures: Mono (r_add=0.75), Hybrid (r_add=1.0), Micro (r_add=1.25)"
echo "Results will be saved to: $RESULTS_DIR"
echo "========================================"

# ファイル存在チェック
if [ ! -f "$LOCUSTFILE" ]; then
    echo "Error: Locustfile not found at $LOCUSTFILE"
    exit 1
fi

if [ ! -f "$OPERATOR_PATH" ]; then
    echo "Error: Operator script not found at $OPERATOR_PATH"
    exit 1
fi

# 各ユーザー数と各アーキテクチャタイプでテストを実行
total_configs=0
current_config=0

for user_count in "${USER_COUNTS[@]}"; do
    for arch_type in Mono Hybrid Micro; do
        ((total_configs++))
    done
done

for user_count in "${USER_COUNTS[@]}"; do
    for arch_type in Mono Hybrid Micro; do
        ((current_config++))
        r_add_value=${ARCHITECTURES[$arch_type]}
        
        echo "========================================"
        echo "Configuration $current_config/$total_configs"
        echo "========================================"
        
        run_tests_for_configuration "$arch_type" "$user_count" "$r_add_value"
    done
done

echo "========================================"
echo "All user scaling tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Check test_summary.txt for analysis guidance"
echo "========================================"

# cleanup は EXIT トラップで自動実行される 