#!/bin/bash

# k8s内でテストを実行するスクリプト
echo "=========================================="
echo "Starting k8s test execution"
echo "=========================================="

# Dockerイメージをビルド（SKIP_BUILD=trueでスキップ可能）
SKIP_BUILD="${SKIP_BUILD:-false}"

if [[ "$SKIP_BUILD" == "true" ]]; then
  echo "Skipping Docker build and image load (SKIP_BUILD=true)"
else
  echo "Building Docker image..."
  docker build -t locust-runner:v2 -f Dockerfile.locust-runner .
  
  echo "Loading image to kind cluster..."
  kind load docker-image --name microservice-demo locust-runner:v2
fi

# Generate unique job name with timestamp
JOB_NAME="locust-runner-$(date +%Y%m%d-%H%M%S)"
echo "Job name: ${JOB_NAME}"

# Clean up old completed jobs (optional, keeps last 5)
echo "Cleaning up old jobs..."
kubectl get jobs -n default -l app=locust-runner --sort-by=.metadata.creationTimestamp -o name | head -n -5 | xargs -r kubectl delete -n default 2>/dev/null || true

# 新しいジョブをデプロイ
echo "Deploying new job..."

# Ensure SA/RBAC exist (apply base manifest for non-Job resources)
kubectl apply -n default -f k8s/locust-runner.yaml --prune=false >/dev/null 2>&1 || true


# Parameters from environment (with defaults)
SERVICE_LIST="${SERVICES_STR:-productcatalogservice}"
USERS_STR_VAL="${USERS_STR:-450}"
REPLICAS_STR_VAL="${REPLICAS_STR:-2}"
DURATION_VAL="${DURATION:-30}"
TARGET_HOST_VAL="${TARGET_HOST:-http://frontend.default.svc.cluster.local}"
PROM_URL_VAL="${PROMETHEUS_URL:-http://prometheus.istio-system.svc:9090}"

cat <<EOF | kubectl apply -n default -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: default
  labels:
    app: locust-runner
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 30
  template:
    spec:
      serviceAccountName: locust-runner
      restartPolicy: Never
      volumes:
        - name: results
          emptyDir: {}
      containers:
        - name: runner
          image: locust-runner:v2
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: results
              mountPath: /results
          env:
            - name: TARGET_HOST
              value: "${TARGET_HOST_VAL}"
            - name: PROMETHEUS_URL
              value: "${PROM_URL_VAL}"
            - name: SERVICES_STR
              value: "${SERVICE_LIST}"
            - name: USERS_STR
              value: "${USERS_STR_VAL}"
            - name: DURATION
              value: "${DURATION_VAL}"
EOF

kubectl scale deployment/${SERVICE_LIST} --replicas=${REPLICAS_STR_VAL} -n default
echo "Waiting for pod to be created..."
# Wait for pod to be created
for i in {1..30}; do
  POD_NAME=$(kubectl get pods -n default -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [[ -n "$POD_NAME" ]]; then
    echo "Pod created: ${POD_NAME}"
    break
  fi
  sleep 1
done

if [[ -z "$POD_NAME" ]]; then
  echo "ERROR: Pod was not created within 30 seconds"
  exit 1
fi

echo "Waiting for pod to be running..."
# Wait for pod to be running (all containers started)
kubectl wait --for=condition=Ready --timeout=120s pod/${POD_NAME} -n default 2>/dev/null || true

# Check if pod is actually running
POD_STATUS=$(kubectl get pod ${POD_NAME} -n default -o jsonpath='{.status.phase}' 2>/dev/null)
echo "Pod status: ${POD_STATUS}"

if [[ "$POD_STATUS" == "Running" ]]; then
  echo "==========================================
JOB_STARTED
=========================================="
else
  echo "WARNING: Pod is not in Running state, but continuing..."
fi

# Wait for job to complete (not follow logs)
echo "Waiting for job to complete (timeout: DURATION + 60s)..."
# Calculate timeout based on DURATION + 60s buffer
WAIT_TIMEOUT=$((DURATION_VAL + 60))
kubectl wait --for=condition=complete --timeout=${WAIT_TIMEOUT}s job/${JOB_NAME} -n default 2>/dev/null || \
kubectl wait --for=condition=failed --timeout=5s job/${JOB_NAME} -n default 2>/dev/null || true

# Show logs after completion
echo "Job completed. Showing logs..."
kubectl logs -n default -l job-name=${JOB_NAME} --tail=50

echo "=========================================="
echo "k8s test execution completed"
echo "=========================================="
