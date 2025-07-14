import os
import csv
import re
from datetime import datetime
from kubernetes import client, config

LOG_DIR = os.environ.get("LOG_DIR", ".")
ARCH_TYPE = os.environ.get("ARCH_TYPE", "unknown")
RUN_NUM = os.environ.get("RUN_NUM", "0")
NAMESPACE = os.environ.get("NAMESPACE", "default")

OUTPUT_FILE = os.path.join(LOG_DIR, f"pod_http_log_{ARCH_TYPE}_run_{RUN_NUM}.csv")

status_re = re.compile(r"\b(\d{3})\b")


def parse_success(log_text):
    total = 0
    success = 0
    for line in log_text.splitlines():
        m = status_re.search(line)
        if not m:
            continue
        code = int(m.group(1))
        total += 1
        if code < 400:
            success += 1
    return success, total


def main():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=NAMESPACE).items

    headers = [
        "timestamp",
        "pod",
        "service",
        "total",
        "success",
        "failure",
        "success_rate",
    ]
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    rows = []
    now = datetime.utcnow().isoformat()
    for pod in pods:
        pod_name = pod.metadata.name
        service = pod.metadata.labels.get("app", "unknown")
        try:
            log_text = v1.read_namespaced_pod_log(name=pod_name, namespace=NAMESPACE)
        except Exception:
            log_text = ""
        success, total = parse_success(log_text)
        failure = total - success
        rate = success / total if total > 0 else 1.0
        rows.append([now, pod_name, service, total, success, failure, rate])

    if rows:
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


if __name__ == "__main__":
    main()