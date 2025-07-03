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

# 関数：K8s Operatorを開始
start_operator() {
    echo "==== Setting up Kubernetes Operator ===="
 
    # replicaset.pyをバックグラウンドで開始
    echo "Starting replicaset operator..."
    python3 "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "$OPERATOR_PID_FILE"
    echo "Operator started with PID: $OPERATOR_PID"
    
    # オペレーターの初期化を待つ
    sleep 10
}
# K8s Operatorを開始
start_operator

# 各テストラウンドを実行
for i in $(seq 1 $NUM_RUNS)
do
    echo "==== Run $i/$NUM_RUNS ===="
    locust -f "$LOCUSTFILE" --headless -u $USERS -r $SPAWN_RATE --run-time $RUN_TIME --host "$HOST" --logfile "locust_run_${i}.log"
    echo "==== Finished Run $i ===="
    
    # 次のテストまでの待機時間（オペレーターが結果を処理する時間を確保）
    if [ $i -lt $NUM_RUNS ]; then
        echo "Waiting between test runs..."
        sleep 30
    fi
done

echo "All $NUM_RUNS runs completed."

# stop_operator は EXIT トラップで自動実行される
