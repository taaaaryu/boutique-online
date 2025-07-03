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
CRD_PATH="crd.json"
APPCONFIG_PATH="k8s-operator/my-appconfig.yaml"

# PIDファイルの保存場所
OPERATOR_PID_FILE="replicaset_operator.pid"

# 関数：依存関係をチェック
check_dependencies() {
    echo "==== Checking Dependencies ===="
    
    # kubectlの確認
    if ! command -v kubectl &> /dev/null; then
        echo "Error: kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # python3の確認
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed or not in PATH"
        exit 1
    fi
    
    # locustの確認
    if ! command -v locust &> /dev/null; then
        echo "Error: locust is not installed or not in PATH"
        exit 1
    fi
    
    # Pythonパッケージの確認
    echo "Checking Python packages..."
    python3 -c "
import sys
missing_packages = []

try:
    import kopf
except ImportError:
    missing_packages.append('kopf')

try:
    import kubernetes
except ImportError:
    missing_packages.append('kubernetes')
    
try:
    import numpy
except ImportError:
    missing_packages.append('numpy')
    
try:
    import pandas
except ImportError:
    missing_packages.append('pandas')

if missing_packages:
    print(f'Error: Missing Python packages: {missing_packages}')
    print('Please install them using: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All required Python packages are installed')
" || exit 1

    # Kubernetesクラスターへの接続確認
    echo "Checking Kubernetes cluster connection..."
    if ! kubectl cluster-info &> /dev/null; then
        echo "Warning: Cannot connect to Kubernetes cluster"
        echo "Please ensure your kubeconfig is properly configured"
        exit 1
    fi
    
    echo "All dependencies check passed!"
}

# 関数：K8s Operatorを開始
start_operator() {
    echo "==== Setting up Kubernetes Operator ===="
    
    # CRDを適用
    echo "Applying CRD..."
    kubectl apply -f "$CRD_PATH"
    
    # AppConfigリソースを適用
    echo "Applying AppConfig..."
    kubectl apply -f "$APPCONFIG_PATH"
    
    # 少し待つ（CRDが利用可能になるまで）
    sleep 5
    
    # replicaset.pyをバックグラウンドで開始
    echo "Starting replicaset operator..."
    python3 "$OPERATOR_PATH" &
    OPERATOR_PID=$!
    echo $OPERATOR_PID > "$OPERATOR_PID_FILE"
    echo "Operator started with PID: $OPERATOR_PID"
    
    # オペレーターの初期化を待つ
    sleep 10
}

# 関数：K8s Operatorを停止
stop_operator() {
    echo "==== Stopping Kubernetes Operator ===="
    
    if [ -f "$OPERATOR_PID_FILE" ]; then
        OPERATOR_PID=$(cat "$OPERATOR_PID_FILE")
        echo "Stopping operator with PID: $OPERATOR_PID"
        kill $OPERATOR_PID 2>/dev/null || echo "Process already stopped"
        rm -f "$OPERATOR_PID_FILE"
    fi
    
    # AppConfigリソースを削除
    echo "Cleaning up AppConfig..."
    kubectl delete -f "$APPCONFIG_PATH" --ignore-not-found=true
    
    echo "Operator cleanup completed"
}

# トラップ設定（スクリプト終了時に確実にオペレーターを停止）
trap stop_operator EXIT

# メイン処理開始
echo "==== Starting Locust Test with K8s Operator Integration ===="

# 依存関係のチェック
check_dependencies

# ファイル存在チェック
if [ ! -f "$LOCUSTFILE" ]; then
    echo "Error: Locustfile not found at $LOCUSTFILE"
    exit 1
fi

if [ ! -f "$OPERATOR_PATH" ]; then
    echo "Error: Operator script not found at $OPERATOR_PATH"
    exit 1
fi

if [ ! -f "$CRD_PATH" ]; then
    echo "Error: CRD file not found at $CRD_PATH"
    exit 1
fi

if [ ! -f "$APPCONFIG_PATH" ]; then
    echo "Error: AppConfig file not found at $APPCONFIG_PATH"
    exit 1
fi

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
