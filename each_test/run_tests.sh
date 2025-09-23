#!/bin/bash

# Default configuration
DEFAULT_USERS="1000 2000 3000 4000 5000 6000"
DEFAULT_REPLICAS="1 2 3"
DEFAULT_DURATION=300
DEFAULT_OUTPUT_DIR="./individual_service_results"

# Test configuration
SERVICES=("productcatalogservice" "cartservice" "shippingservice")
USERS=($DEFAULT_USERS)
REPLICAS=($DEFAULT_REPLICAS)
DURATION=$DEFAULT_DURATION
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR

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

# Check if Prometheus is accessible
echo "Checking Prometheus connection..."
if ! curl -s http://localhost:8080/api/v1/query?query=up > /dev/null; then
    echo "Warning: Prometheus is not accessible at http://localhost:8080"
    echo "Please ensure Prometheus port-forwarding is running:"
    echo "  kubectl port-forward -n istio-system svc/prometheus 8080:9090"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Prometheus connection successful"
fi

echo ""

# Run tests
for SERVICE in "${SERVICES[@]}"; do
    for USER in "${USERS[@]}"; do
        for REPLICA in "${REPLICAS[@]}"; do
            echo "Testing $SERVICE with $USER users, $REPLICA replicas..."
            python3 integrated_test_system.py \
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
