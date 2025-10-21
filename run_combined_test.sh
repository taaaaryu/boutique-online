#!/bin/bash

# 両方のスクリプトを実行するメインスクリプト
echo "=========================================="
echo "Combined Test Execution"
echo "=========================================="

# Default configuration (can be overridden by env)
SERVICES_DEFAULT="frontend"
USERS_DEFAULT="10 20 30"
REPLICAS_DEFAULT="1 2"
DURATION="${DURATION:-30}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"  # 同じパラメータで繰り返す回数
BASE_OUTPUT_DIR="${OUTPUT_DIR:-./each_test/individual_service_results}"

# 設定
# Kubernetes内部用（ポッド内からアクセス）
PROMETHEUS_URL_K8S="${PROMETHEUS_URL_K8S:-http://prometheus.istio-system.svc:9090}"
# 外部監視用（ホストマシンからポート転送経由でアクセス）
PROMETHEUS_URL_EXTERNAL="${PROMETHEUS_URL_EXTERNSSAL:-http://localhost:9090}"
TEST_DURATION="${TEST_DURATION:-$DURATION}"

# Build arrays from env overrides or defaults
# shellcheck disable=SC2206
SERVICES_ARRAY=(${SERVICES_DEFAULT})
if [[ ${#SERVICES_ARRAY[@]} -eq 0 ]]; then
  # shellcheck disable=SC2206
  SERVICES_ARRAY=(${SERVICES_DEFAULT})
fi
# shellcheck disable=SC2206
USERS_ARRAY=(${USERS_DEFAULT})
if [[ ${#USERS_ARRAY[@]} -eq 0 ]]; then
  # shellcheck disable=SC2206
  USERS_ARRAY=(${USERS_DEFAULT})
fi
# shellcheck disable=SC2206
REPLICAS_ARRAY=(${REPLICAS_DEFAULT})
if [[ ${#REPLICAS_ARRAY[@]} -eq 0 ]]; then
  # shellcheck disable=SC2206
  REPLICAS_ARRAY=(${REPLICAS_DEFAULT})
fi

# 出力ディレクトリを作成（権限エラー時は安全な相対パスにフォールバック）
OUTPUT_DIR="$BASE_OUTPUT_DIR"
if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
  echo "Warning: cannot write to $OUTPUT_DIR, falling back to ./each_test/individual_service_results"
  OUTPUT_DIR="./each_test/individual_service_results"
  BASE_OUTPUT_DIR="$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

echo "Configuration:"
echo "  Prometheus URL (K8s internal): $PROMETHEUS_URL_K8S"
echo "  Prometheus URL (External): $PROMETHEUS_URL_EXTERNAL"
echo "  Test Duration: ${TEST_DURATION}s"
echo "  Base Output Directory: $BASE_OUTPUT_DIR"
echo "  Services to test: ${SERVICES_ARRAY[*]}"
echo "  User counts to test: ${USERS_ARRAY[*]}"
echo "  Replica counts to test: ${REPLICAS_ARRAY[*]}"
echo "  Repeat count: ${REPEAT_COUNT}"
echo ""

# 必要ならPrometheusをポートフォワード（istio-system/prometheus → localhost:9090）
PORT_FORWARD_PID=""
if [[ "$PROMETHEUS_URL_EXTERNAL" == "http://localhost:9090" || "$PROMETHEUS_URL_EXTERNAL" == "http://127.0.0.1:9090" ]]; then
  echo "Starting kubectl port-forward for Prometheus (istio-system/prometheus) ..."
  # Kill any existing port-forward on 9090
  (pkill -f "kubectl port-forward .* 9090:9090" 2>/dev/null || true)
  sleep 1
  
  # Start new port-forward
  kubectl -n istio-system port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 &
  PORT_FORWARD_PID=$!
  
  # Wait and verify port-forward is working
  echo "Waiting for port-forward to be ready..."
  for i in {1..10}; do
    if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
      echo "Port-forward is ready!"
      break
    fi
    echo "  Attempt $i/10: waiting..."
    sleep 1
  done
  
  # Final check
  if ! curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
    echo "WARNING: Port-forward may not be ready, but continuing anyway..."
  fi
fi

# Function to ensure port-forward is running
ensure_port_forward() {
  if [[ "$PROMETHEUS_URL_EXTERNAL" == "http://localhost:9090" || "$PROMETHEUS_URL_EXTERNAL" == "http://127.0.0.1:9090" ]]; then
    # Check if port-forward is still alive
    if [[ -n "$PORT_FORWARD_PID" ]] && ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
      echo "Port-forward died, restarting..."
      PORT_FORWARD_PID=""
    fi
    
    # Check if port is responding
    if ! curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
      echo "Port-forward not responding, restarting..."
      (pkill -f "kubectl port-forward .* 9090:9090" 2>/dev/null || true)
      sleep 1
      kubectl -n istio-system port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 &
      PORT_FORWARD_PID=$!
      
      # Wait for it to be ready
      for i in {1..10}; do
        if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
          echo "Port-forward is ready!"
          break
        fi
        sleep 1
      done
    fi
  fi
}

# Run tests
FIRST_RUN=true
for REPLICA in "${REPLICAS_ARRAY[@]}"; do
     for SERVICE in "${SERVICES_ARRAY[@]}"; do
        for USER in "${USERS_ARRAY[@]}"; do
            for RUN in $(seq 1 $REPEAT_COUNT); do
                MONITOR_USERS="$USER"
                MONITOR_REPLICAS="$REPLICA"
                MONITOR_SERVICE="$SERVICE"
                MONITOR_NAMESPACE="${MONITOR_NAMESPACE:-default}"

                echo "=========================================="
                echo "Testing $SERVICE with $USER users, $REPLICA replicas (Run $RUN/$REPEAT_COUNT)"
                echo "=========================================="
                echo "Starting k8s test execution..."
                
                # 1テスト実行ごとのフォルダ（MMDD-<TargetService>）
                RUN_FOLDER="$(date +%m%d)-${SERVICE}"
                RUN_BASE_DIR="${BASE_OUTPUT_DIR}/${RUN_FOLDER}"
                mkdir -p "$RUN_BASE_DIR"

                # ログディレクトリの作成（新しい構造: {service}/{users}/replica{replicas}）
                LOG_DIR="${RUN_BASE_DIR}/${SERVICE}/${USER}/replica${REPLICA}"
                mkdir -p "$LOG_DIR"
                TEST_LOG_FILE="${LOG_DIR}/test_run${RUN}.log"
                echo "Test log will be saved to: $TEST_LOG_FILE"
                
                # k8s内でテストを実行（Frontendを経由してサービスにアクセス）
                export TARGET_HOST="http://frontend.default.svc.cluster.local"
                export PROMETHEUS_URL="$PROMETHEUS_URL_K8S"  # K8s内部用URL
                export SERVICES_STR="$SERVICE"
                export USERS_STR="$USER"
                export REPLICAS_STR="$REPLICA"
                export DURATION="$DURATION"
                export OUTPUT_DIR="$RUN_BASE_DIR"
                export RUN_NUMBER="$RUN"  # ファイル名に使用する実行番号
                
                # 最初の1回だけビルド、それ以降はスキップ
                if [[ "$FIRST_RUN" == "true" ]]; then
                    export SKIP_BUILD="false"
                    FIRST_RUN=false
                else
                    export SKIP_BUILD="true"
                fi
                
                # Ensure port-forward is still running
                echo "Verifying port-forward status..."
                ensure_port_forward

                # テスト開始情報をログファイルに記録
                {
                    echo "=========================================="
                    echo "Test Execution Log"
                    echo "=========================================="
                    echo "Service: $SERVICE"
                    echo "Users: $USER"
                    echo "Replicas: $REPLICA"
                    echo "Duration: $DURATION seconds"
                    echo "Run: $RUN/$REPEAT_COUNT"
                    echo "Started at: $(date)"
                    echo "=========================================="
                    echo ""
                } > "$TEST_LOG_FILE"
                
                # K8sテストをバックグラウンドで起動し、出力をキャプチャ（ログファイルとtempファイルの両方に記録）
                echo "Starting k8s test in background..."
                K8S_LOG=$(mktemp)
                kubectl scale deployment $SERVICE --replicas=$REPLICA -n default 2>&1 | tee -a "$TEST_LOG_FILE"
                ./run_k8s_test.sh > "$K8S_LOG" 2>&1 &
                K8S_PID=$!
                echo "K8s test started (PID: $K8S_PID)" | tee -a "$TEST_LOG_FILE"

                # JOB_STARTEDマーカーを待つ
                echo "Waiting for Job to start (watching for JOB_STARTED marker)..." | tee -a "$TEST_LOG_FILE"
                JOB_STARTED=false
                for i in {1..120}; do
                    if grep -q "JOB_STARTED" "$K8S_LOG" 2>/dev/null; then
                        echo "✓ Job started! Now starting external monitoring..." | tee -a "$TEST_LOG_FILE"
                        JOB_STARTED=true
                        break
                    fi
                    sleep 1
                    if ! kill -0 "$K8S_PID" 2>/dev/null; then
                        echo "WARNING: K8s test process ended before JOB_STARTED marker" | tee -a "$TEST_LOG_FILE"
                        break
                    fi
                done

                if [[ "$JOB_STARTED" == "false" ]]; then
                    echo "WARNING: JOB_STARTED marker not detected within 120 seconds, starting monitoring anyway..." | tee -a "$TEST_LOG_FILE"
                fi

                # 外部監視を開始（バックグラウンド実行）
                echo "Starting external Prometheus monitoring in background..." | tee -a "$TEST_LOG_FILE"
                echo "  Target service (load test): $MONITOR_SERVICE" | tee -a "$TEST_LOG_FILE"
                echo "  Monitoring: ALL services in namespace $MONITOR_NAMESPACE" | tee -a "$TEST_LOG_FILE"
                echo "  Users: $MONITOR_USERS  Replicas: $MONITOR_REPLICAS" | tee -a "$TEST_LOG_FILE"

                # 全サービス監視を1つのプロセスで実行
                RUN_NUMBER="$RUN" python3 "./each_test/external_prometheus_monitor.py" \
                    --prometheus-url "$PROMETHEUS_URL_EXTERNAL" \
                    --duration "$TEST_DURATION" \
                    --output-dir "$RUN_BASE_DIR" \
                    --service "$MONITOR_SERVICE" \
                    --namespace "${MONITOR_NAMESPACE:-default}" \
                    --users "$MONITOR_USERS" \
                    --replicas "$MONITOR_REPLICAS" \
                    --monitor-all-services &
                MONITOR_PID=$!
                echo "  Started monitoring (PID: $MONITOR_PID) for all services" | tee -a "$TEST_LOG_FILE"

                # K8sテストの完了を待つ（ログを表示して保存）
                echo "" | tee -a "$TEST_LOG_FILE"
                echo "Waiting for k8s test to complete..." | tee -a "$TEST_LOG_FILE"
                wait "$K8S_PID" 2>/dev/null || true
                echo "" | tee -a "$TEST_LOG_FILE"
                echo "==========================================" | tee -a "$TEST_LOG_FILE"
                echo "K8s test output:" | tee -a "$TEST_LOG_FILE"
                echo "==========================================" | tee -a "$TEST_LOG_FILE"
                cat "$K8S_LOG" | tee -a "$TEST_LOG_FILE"
                rm -f "$K8S_LOG"

                echo "" | tee -a "$TEST_LOG_FILE"
                echo "K8s test completed, waiting for external monitoring to finish..." | tee -a "$TEST_LOG_FILE"
                
                # 外部監視の完了を待つ
                if kill -0 "$MONITOR_PID" 2>/dev/null; then
                    echo "  Waiting for monitoring process (PID: $MONITOR_PID) to complete..." | tee -a "$TEST_LOG_FILE"
                    wait "$MONITOR_PID" 2>/dev/null || true
                    echo "  ✓ Monitoring process (PID: $MONITOR_PID) completed" | tee -a "$TEST_LOG_FILE"
                else
                    echo "  - Monitoring process (PID: $MONITOR_PID) already completed" | tee -a "$TEST_LOG_FILE"
                fi
            
            # 結果のサマリーを表示して保存
            {
                echo ""
                echo "=========================================="
                echo "Test Completed"
                echo "=========================================="
                echo "Finished at: $(date)"
                echo ""
                echo "Results summary (Run $RUN/$REPEAT_COUNT):"
                echo "  Results saved to: ${BASE_OUTPUT_DIR}/${SERVICE}/replica${REPLICA}/${USER}/*_run${RUN}.csv"
                echo "  Test log saved to: $TEST_LOG_FILE"
                echo "=========================================="
                echo ""
            } | tee -a "$TEST_LOG_FILE"
            
            # このRUNで作成されたジョブをクリーンアップ
            echo "" | tee -a "$TEST_LOG_FILE"
            echo "Cleaning up jobs for this run..." | tee -a "$TEST_LOG_FILE"
            # 成功したジョブを削除
            kubectl delete jobs -n default -l app=locust-runner --field-selector status.successful=1 2>&1 | tee -a "$TEST_LOG_FILE" && echo "  ✓ Successful jobs deleted" | tee -a "$TEST_LOG_FILE" || echo "  - No successful jobs to delete" | tee -a "$TEST_LOG_FILE"
            # 失敗したジョブも削除
            kubectl delete jobs -n default -l app=locust-runner --field-selector status.failed=1 2>&1 | tee -a "$TEST_LOG_FILE" && echo "  ✓ Failed jobs deleted" | tee -a "$TEST_LOG_FILE" || echo "  - No failed jobs to delete" | tee -a "$TEST_LOG_FILE"
            
            # 各実行の間にクールダウン期間を設ける
            if [[ $RUN -lt $REPEAT_COUNT ]]; then
                echo "Waiting 10 seconds before next run..." | tee -a "$TEST_LOG_FILE"
                sleep 10
            fi

        done  # RUN loop
        done  # REPLICA loop
    done  # USER loop
done  # SERVICE loop

# ポートフォワードのクリーンアップ
if [[ -n "$PORT_FORWARD_PID" ]]; then
  echo "Stopping Prometheus port-forward (pid=$PORT_FORWARD_PID) ..."
  kill "$PORT_FORWARD_PID" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Cleaning up completed jobs..."
echo "=========================================="

# 成功したジョブを削除
echo "Deleting successful jobs..."
kubectl delete jobs -n default -l app=locust-runner --field-selector status.successful=1 2>/dev/null || echo "No successful jobs to delete"

# 失敗したジョブも削除（オプション）
echo "Deleting failed jobs..."
kubectl delete jobs -n default -l app=locust-runner --field-selector status.failed=1 2>/dev/null || echo "No failed jobs to delete"

# 残っているジョブの数を確認
REMAINING_JOBS=$(kubectl get jobs -n default -l app=locust-runner --no-headers 2>/dev/null | wc -l)
if [[ $REMAINING_JOBS -gt 0 ]]; then
  echo "Warning: $REMAINING_JOBS job(s) still remaining"
  kubectl get jobs -n default -l app=locust-runner
else
  echo "All jobs cleaned up successfully"
fi

echo ""
echo "=========================================="
echo "Combined test execution completed!"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "Files are named with suffix: _run1.csv, _run2.csv, _run3.csv, _run4.csv, _run5.csv"
echo "=========================================="

