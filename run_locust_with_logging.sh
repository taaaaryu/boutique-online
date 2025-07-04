#!/bin/bash
# This script runs the locust.sh script and captures all its output to a log file.

LOG_FILE="locust_full_run.log"

echo "Starting locust.sh and redirecting all output to ${LOG_FILE}..."

bash ./locust.sh > "${LOG_FILE}" 2>&1

echo "locust.sh finished. Full log saved to ${LOG_FILE}."
