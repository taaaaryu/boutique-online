#!/bin/bash

# Default configuration
DEFAULT_USERS="450"
DEFAULT_REPLICAS="2"
DEFAULT_DURATION=30
DEFAULT_OUTPUT_DIR="/results"

# Test configuration
# 環境変数 SERVICES_STR/USERS_STR/REPLICAS_STR が設定されていればそれを優先（スペース区切り）
if [[ -n "$SERVICES_STR" ]]; then
    # shellcheck disable=SC2206
    SERVICES=($SERVICES_STR)
else
    SERVICES=("productcatalogservice")
fi

if [[ -n "$USERS_STR" ]]; then
    # shellcheck disable=SC2206
    USERS=($USERS_STR)
else
    USERS=($DEFAULT_USERS)
fi

if [[ -n "$REPLICAS_STR" ]]; then
    # shellcheck disable=SC2206
    REPLICAS=($REPLICAS_STR)
else
    REPLICAS=($DEFAULT_REPLICAS)
fi

DURATION=${DURATION:-$DEFAULT_DURATION}
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR

# Force venv python for consistent deps
export PATH="/opt/venv/bin:$PATH"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
# Raise file descriptor limit to avoid fsnotify watcher failures
ulimit -n 65536 2>/dev/null || true

echo "=========================================="
echo "Integrated Test System"
echo "=========================================="
echo "Services: ${SERVICES[*]}"
echo "User counts: ${USERS[*]}"
echo "Replica counts: ${REPLICAS[*]}"
echo "Duration: ${DURATION}s"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if Prometheus is accessible (env PROMETHEUS_URL or default in-cluster)
PROM_URL="${PROMETHEUS_URL:-http://prometheus.istio-system.svc:9090}"
echo "Checking Prometheus connection at $PROM_URL ..."
if ! curl -sf "$PROM_URL/api/v1/query?query=up" > /dev/null; then
    echo "Warning: Prometheus is not accessible at $PROM_URL"
    echo "Proceeding without interactive prompt (non-interactive mode)."
fi

echo ""

# Run tests
for SERVICE in "${SERVICES[@]}"; do
    for USER in "${USERS[@]}"; do
        for REPLICA in "${REPLICAS[@]}"; do
            echo "Testing $SERVICE with $USER users, $REPLICA replicas..."
            "$PYTHON_BIN" integrated_test_system.py \
                --service "$SERVICE" \
                --users $USER \
                --replicas $REPLICA \
                --duration "$DURATION" \
                --output-dir "$OUTPUT_DIR"
        done
    done
done

echo ""
echo "=========================================="
echo "Test completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

POD=$(kubectl get pod -n default -l job-name=locust-runner -o jsonpath='{.items[0].metadata.name}')
echo "Copying results from $POD to ./job_results"
kubectl cp -n default "$POD":/results ./job_results