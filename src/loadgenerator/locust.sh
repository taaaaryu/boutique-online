#!/bin/bash
# Run Locust load test with 500 users using locustfile.py
# Using port-forward to access service in default namespace

# Port forward the service (assuming frontend service on port 80)
# You can modify the service name and ports as needed
kubectl port-forward svc/frontend 8080:80 -n default &
PF_PID=$!

# Wait a moment for port-forward to establish
sleep 3

# Run locust
locust -f locustfile.py --headless -u 500 -r 50 -t 10m --host=http://localhost:8080

# Clean up port-forward
kill $PF_PID