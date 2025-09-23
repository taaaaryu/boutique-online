set -e

SERVICES=(
    "productcatalogservice"
    "cartservice"
    "shippingservice"
)

METRICS=(
    "cpu_usage_seconds_total"
    "cpu_usage_percent"
    "memory_usage_percent"
    "network_receive_bytes_total"
    "network_transmit_bytes_total"
)

for service in "${SERVICES[@]}"; do
    for metric in "${METRICS[@]}"; do
    # Generate metric-specific overview plots (2 subplots: metric vs users, success rate vs users)
     python3 compare_system_metrics.py \
      --root ./individual_service_results \
      --service "${service}" \
      --allpods-overview \
      --allpods-metric "${metric}" \
      --outdir ./individual_service_results
    done
done
