set -euo pipefail

# ==========================================
# Configuration (edit as needed)
# ==========================================

# Base directory for results
BASE_DIR="./each_test/individual_service_results"

# Target service used in the test run folder name (MMDD-<TargetService>)
TARGET_SERVICE="frontend"

# Metrics to visualize across all services
METRICS=(
  "cpu_usage_percent"
  "memory_usage_percent"
  "network_receive_bytes_total"
  "network_transmit_bytes_total"
)

# Optional: manually set RUN_ROOT to a specific run folder
RUN_ROOT="${BASE_DIR}/1020-frontend"
# Auto-select the latest run folder matching pattern MMDD-<TargetService>
LATEST_RUN_DIR=$(ls -1t "${BASE_DIR}" 2>/dev/null | grep -E "^[0-9]{4}-${TARGET_SERVICE}$" | head -1 || true)
if [[ -z "${LATEST_RUN_DIR}" ]]; then
  echo "ERROR: No run folder found for pattern MMDD-${TARGET_SERVICE} under ${BASE_DIR}" >&2
  echo "Hint: Set RUN_ROOT manually above." >&2
  exit 1
fi
# If RUN_ROOT not manually set above, use the latest detected folder
if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="${BASE_DIR}/1015-frontend"
fi

# Ensure output directory exists
mkdir -p "${RUN_ROOT}/overview" || true

# Basic checks
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT does not exist: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PWD}/each_test/visualize_run_overview.py" ]]; then
  echo "ERROR: visualize_run_overview.py not found at ${PWD}/each_test/visualize_run_overview.py" >&2
  exit 1
fi


echo "Run root: ${RUN_ROOT}"
echo "Metrics: ${METRICS[*]}"

# ==========================================
# 1) All-services overview (per replicas, x=users, lines=services)
#    Output: ${RUN_ROOT}/overview/replicaN/all_services_<metric>_vs_users.png
# ==========================================
python3 "${PWD}/each_test/visualize_run_overview.py" \
  --run-root "${RUN_ROOT}" \
  --metrics "${METRICS[@]}" \
  --target-service "${TARGET_SERVICE}"

# ==========================================
# 2) Latency metrics overview (optional - only for Istio-enabled services)
#    Output: ${RUN_ROOT}/overview/replicaN/latency_<metric>_vs_users.png
#    Note: Some services (like redis-cart) may not have latency metrics
# ==========================================
LATENCY_METRICS=(
  "request_duration_avg"
  "request_rate_total"
  "response_size_bytes"
)

echo "Attempting to visualize latency metrics (may skip services without Istio metrics)..."
python3 "${PWD}/each_test/visualize_latency_overview.py" \
  --run-root "${RUN_ROOT}" \
  --metrics "${LATENCY_METRICS[@]}" \
  --target-service "${TARGET_SERVICE}" || echo "Warning: Latency visualization completed with warnings (some services may not have latency data)"

# ==========================================
# 2) (Optional) Per-service deep-dive using compare_system_metrics.py
#    NOTE: compare_system_metrics.py expects the legacy layout:
#          replica<replicas>/<users>/<service>/...
#    Our new layout is: <service>/<users>/replica<replicas>/...
#    If you still need per-service plots with this script, either
#    - run it on legacy runs, or
#    - migrate/augment the script to support the new layout.
#
# Uncomment and adapt ROOT to a legacy layout if needed.
#
# LEGACY_ROOT="./each_test/individual_service_results/front"  # legacy layout example
# SERVICES=("frontend")
# LEGACY_METRICS=(
#   "cpu_usage_seconds_total"
#   "cpu_usage_percent"
#   "memory_usage_percent"
#   "network_receive_bytes_total"
#   "network_transmit_bytes_total"
# )
# for service in "${SERVICES[@]}"; do
#   for metric in "${LEGACY_METRICS[@]}"; do
#     python3 each_test/compare_system_metrics.py \
#       --root "${LEGACY_ROOT}" \
#       --service "${service}" \
#       --allpods-overview \
#       --allpods-metric "${metric}" \
#       --plot || true
#   done
# done

echo "Visualization completed. Outputs are under: ${RUN_ROOT}/overview"