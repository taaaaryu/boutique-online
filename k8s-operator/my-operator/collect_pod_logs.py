import os
import csv
import re
import json
from datetime import datetime
import time
import requests
from kubernetes import client, config


LOG_DIR = os.environ.get("LOG_DIR", "pod_logs")
SERVICE_DIR = os.path.join(LOG_DIR, "services")

ARCH_TYPE = os.environ.get("ARCH_TYPE", "unknown")
RUN_NUM = os.environ.get("RUN_NUM", "0")
NAMESPACE = os.environ.get("NAMESPACE", "default")
DUMP_ENVOY_JSON = 1

def get_istio_proxy_container_name(pod):
    for c in pod.spec.containers:
        if c.name == "istio-proxy":
            return c.name
    return None

def main():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=NAMESPACE).items

    service_logs = {}
    all_codes = set()
    now = datetime.utcnow().isoformat()

    for pod in pods:
        pod_name = pod.metadata.name
        service = pod.metadata.labels.get("app", "unknown")
        container_name = get_istio_proxy_container_name(pod)
        if not container_name:
            continue
        try:
            log_text = v1.read_namespaced_pod_log(name=pod_name, namespace=NAMESPACE, container=container_name)
        except Exception:
            log_text = ""
        code_counts = {}
        timeout_count = 0  # タイムアウトカウント
        
        # Optionally write raw Envoy JSON access logs alongside other results
        # Append-only JSON Lines per pod to avoid mixing pod sources
        json_out_fh = None
        if DUMP_ENVOY_JSON and log_text:
            service_subdir = os.path.join(SERVICE_DIR, service)
            os.makedirs(service_subdir, exist_ok=True)
            json_out_path = os.path.join(service_subdir, f"envoy_access_{pod_name}.jsonl")
            try:
                json_out_fh = open(json_out_path, "a", encoding="utf-8")
            except Exception:
                json_out_fh = None

        for line in log_text.splitlines():
            # Try to persist raw json line if enabled
            if json_out_fh:
                ln = line.strip()
                # If the line is json (best-effort), write as-is to keep Envoy's schema
                if ln.startswith("{") and ln.endswith("}"):
                    try:
                        json.loads(ln)
                        json_out_fh.write(ln + "\n")
                    except Exception:
                        pass

            # レスポンス時間を先にチェック
            time_match = re.search(r'" \d{3} - [^-]+ - "[^"]*" \d+ \d+ (\d+) (\d+)', line)
            is_timeout = False
            
            if time_match:
                response_time = int(time_match.group(1))  # レスポンス時間（ミリ秒）
                #print(f"response_time: {response_time}")
                # 10秒（10000ミリ秒）を超えた場合をタイムアウトとしてカウント
                if response_time > 10000:
                    timeout_count += 1
                    is_timeout = True
                    # タイムアウト用のカスタムコードを追加
                    if 'timeout' not in code_counts:
                        code_counts['timeout'] = 0
                    code_counts['timeout'] += 1
            
            # タイムアウトでない場合のみHTTPステータスコードをカウント
            if not is_timeout:
                m = re.search(r'" \b(\d{3})\b', line)
                if m:
                    code = m.group(1)
                    code_counts[code] = code_counts.get(code, 0) + 1
        
        # タイムアウトカウントを別途出力
        if timeout_count > 0:
            print(f"{pod_name}: タイムアウト {timeout_count} 件")
        
        if json_out_fh:
            try:
                json_out_fh.close()
            except Exception:
                pass

        all_codes.update(code_counts.keys())
        if service not in service_logs:
            service_logs[service] = []
        service_logs[service].append((pod_name, code_counts))

    # サービス用サブディレクトリ作成
    os.makedirs(SERVICE_DIR, exist_ok=True)

    # サービスごとにファイル出力（Append方式）
    for service, pod_list in service_logs.items():
        # 統一されたヘッダーを定義
        headers = ["timestamp", "pod", "code_100s", "code_200s", "code_400s", "code_500s", "code_other", "code_timeout"]
        file_path = os.path.join(SERVICE_DIR, f"{service}.csv")
        
        # ファイルが存在しない場合はヘッダーを作成
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        # 新しい行をAppend
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            for pod_name, code_counts in pod_list:
                # 各範囲のカウントを集計
                code_100s = sum(code_counts.get(str(code), 0) for code in range(100, 200))
                code_200s = sum(code_counts.get(str(code), 0) for code in range(200, 300))
                code_400s = sum(code_counts.get(str(code), 0) for code in range(400, 500))
                code_500s = sum(code_counts.get(str(code), 0) for code in range(500, 600))
                code_other = sum(code_counts.get(str(code), 0) for code in code_counts.keys() 
                               if code.isdigit() and (int(code) < 100 or int(code) >= 600))
                code_timeout = code_counts.get('timeout', 0)
                
                row = [now, pod_name, code_100s, code_200s, code_400s, code_500s, code_other, code_timeout]
                writer.writerow(row)
        print(f"{service} のログを {file_path} に追加しました ({len(pod_list)} ポッド)")

    # --- Prometheus-based system metrics collection (CPU / Memory / Network) ---
    # Be robust to empty env values: treat empty string as unset and fall back.
    # If PROMETHEUS_URL is not provided, try PROMETHEUS, then default to localhost.
    PROM_URL = (
        os.environ.get('PROMETHEUS_URL')
        or os.environ.get('PROMETHEUS')
        or 'http://localhost:9090'
    )
    MONITOR_TIME = int(os.environ.get('MONITOR_TIME_SECONDS', '60'))

    def _prometheus_query(prom_url: str, query: str):
        try:
            resp = requests.get(f"{prom_url}/api/v1/query", params={'query': query}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Prometheus query failed: {e} (query={query})")
            return None

    def _prometheus_vector(prom_url: str, query: str) -> dict:
        data = _prometheus_query(prom_url, query)
        results = {}
        try:
            vec = data.get('data', {}).get('result', []) if data else []
            for item in vec:
                metric = item.get('metric', {})
                # prefer pod label if present
                key = (
                    metric.get('pod')
                    or metric.get('pod_name')
                    or metric.get('kubernetes_pod_name')
                    or metric.get('instance')
                    or metric.get('container')
                    or metric.get('container_name')
                    or 'unknown'
                )
                try:
                    value = float(item.get('value', [0, 0])[1])
                except Exception:
                    value = 0.0
                results[key] = value
        except Exception:
            pass
        return results

    def _prometheus_vector_by_service(prom_url: str, query: str) -> dict:
        """Return a mapping keyed by service-like labels (destination_workload/service_name)."""
        data = _prometheus_query(prom_url, query)
        results = {}
        try:
            vec = data.get('data', {}).get('result', []) if data else []
            for item in vec:
                metric = item.get('metric', {})
                key = (
                    metric.get('destination_workload')
                    or metric.get('destination_service_name')
                    or metric.get('service')
                    or metric.get('app')
                    or 'unknown'
                )
                try:
                    value = float(item.get('value', [0, 0])[1])
                except Exception:
                    value = 0.0
                results[key] = value
        except Exception:
            pass
        return results

    def _first_nonempty_vector(prom_url: str, queries: list[str]) -> dict:
        for q in queries:
            res = _prometheus_vector(prom_url, q)
            # consider non-empty if there is at least one key with non-zero or any key at all
            if any(abs(v) > 0 for v in res.values()) or len(res) > 0:
                return res
        return {}

    def _first_nonempty_service_vector(prom_url: str, queries: list[str]) -> dict:
        for q in queries:
            res = _prometheus_vector_by_service(prom_url, q)
            if any(abs(v) > 0 for v in res.values()) or len(res) > 0:
                return res
        return {}

    def collect_prometheus_system_metrics(service_name: str):
        ns = NAMESPACE
        pod_filter = ''
        if service_name and service_name != 'unknown':
            pod_filter = f',pod=~"{service_name}-.*"'

        # Queries with fallbacks (handle label variations and metric sources)
        # CPU rate (cores): preferred to compute avg cores directly
        cpu_rate_queries = [
            f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"{pod_filter}}}[{MONITOR_TIME}s]))',
            f'sum by (pod) (rate(container_cpu_usage_seconds_total{{kubernetes_namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"{pod_filter}}}[{MONITOR_TIME}s]))',
        ]
        # CPU increase (seconds) as alternative
        cpu_increase_queries = [
            f'sum by (pod) (increase(container_cpu_usage_seconds_total{{namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"{pod_filter}}}[{MONITOR_TIME}s]))',
            f'sum by (pod) (increase(container_cpu_usage_seconds_total{{kubernetes_namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"{pod_filter}}}[{MONITOR_TIME}s]))',
        ]
        # CPU cores (limits)
        cpu_cores_queries = [
            f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{ns}",resource="cpu",unit="core"{pod_filter}}})',
            f'sum by (pod) (container_spec_cpu_quota{{namespace="{ns}",container!="istio-proxy"{pod_filter}}} / container_spec_cpu_period{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})',
        ]
        # Memory working set and limit
        mem_working_queries = [
            f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})',
            f'sum by (pod) (container_memory_usage_bytes{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})',
        ]
        mem_limit_queries = [
            f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{ns}",resource="memory",unit="byte"{pod_filter}}})',
            f'sum by (pod) (container_spec_memory_limit_bytes{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})'
        ]
        # Network
        net_rx_queries = [
            f'sum by (pod) (increase(container_network_receive_bytes_total{{namespace="{ns}"{pod_filter}}}[{MONITOR_TIME}s]))',
            f'sum by (pod) (increase(container_network_receive_bytes_total{{kubernetes_namespace="{ns}"{pod_filter}}}[{MONITOR_TIME}s]))',
        ]
        net_tx_queries = [
            f'sum by (pod) (increase(container_network_transmit_bytes_total{{namespace="{ns}"{pod_filter}}}[{MONITOR_TIME}s]))',
            f'sum by (pod) (increase(container_network_transmit_bytes_total{{kubernetes_namespace="{ns}"{pod_filter}}}[{MONITOR_TIME}s]))',
        ]

        # Fetch vectors using fallbacks
        cpu_rate = _first_nonempty_vector(PROM_URL, cpu_rate_queries)
        cpu_increase = _first_nonempty_vector(PROM_URL, cpu_increase_queries)
        cpu_cores = _first_nonempty_vector(PROM_URL, cpu_cores_queries)
        mem_working = _first_nonempty_vector(PROM_URL, mem_working_queries)
        mem_limit = _first_nonempty_vector(PROM_URL, mem_limit_queries)
        net_rx = _first_nonempty_vector(PROM_URL, net_rx_queries)
        net_tx = _first_nonempty_vector(PROM_URL, net_tx_queries)

        pod_names = sorted(set().union(
            cpu_rate.keys(), cpu_increase.keys(), cpu_cores.keys(),
            mem_working.keys(), mem_limit.keys(), net_rx.keys(), net_tx.keys()
        ))

        # Aggregates
        total_cpu_rate = sum(float(cpu_rate.get(p, 0.0)) for p in pod_names)  # cores
        total_cpu_usage = sum(float(cpu_increase.get(p, 0.0)) for p in pod_names)
        # If increase() missing but rate present, derive seconds from rate * window
        if total_cpu_usage == 0.0 and total_cpu_rate > 0.0 and MONITOR_TIME > 0:
            total_cpu_usage = total_cpu_rate * float(MONITOR_TIME)
        total_cpu_cores = sum(float(cpu_cores.get(p, 0.0)) for p in pod_names)
        total_mem_working = sum(float(mem_working.get(p, 0.0)) for p in pod_names)
        total_mem_limit = sum(float(mem_limit.get(p, 0.0)) for p in pod_names)
        total_net_rx = sum(float(net_rx.get(p, 0.0)) for p in pod_names)
        total_net_tx = sum(float(net_tx.get(p, 0.0)) for p in pod_names)

        # CPU: avg cores during window and percent
        cpu_avg_cores = (
            float(total_cpu_rate) if total_cpu_rate > 0.0 else
            (float(total_cpu_usage) / float(MONITOR_TIME) if MONITOR_TIME > 0 else 0.0)
        )
        cpu_cores_missing = (total_cpu_cores == 0.0)
        cpu_pct = (cpu_avg_cores / total_cpu_cores) * 100.0 if total_cpu_cores > 0 else cpu_avg_cores * 100.0

        # Memory percent
        mem_pct = (total_mem_working / total_mem_limit * 100.0) if total_mem_limit > 0 else 0.0

        # Output path
        now_ts = datetime.utcnow().isoformat()
        service_subdir = os.path.join(SERVICE_DIR, service_name)
        os.makedirs(service_subdir, exist_ok=True)
        out_path = os.path.join(service_subdir, f"prom_system_{service_name}.csv")
        headers = [
            'timestamp', 'service_name',
            'cpu_usage_seconds', 'cpu_avg_cores', 'cpu_cores', 'cpu_usage_percent', 'cpu_cores_missing',
            'memory_working_set_bytes', 'memory_limit_bytes', 'memory_usage_percent',
            'network_receive_bytes_total', 'network_transmit_bytes_total'
        ]

        # Append
        try:
            if not os.path.exists(out_path):
                with open(out_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            with open(out_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_ts,
                    service_name,
                    f"{total_cpu_usage:.6f}",
                    f"{cpu_avg_cores:.6f}",
                    f"{total_cpu_cores:.6f}",
                    f"{cpu_pct:.2f}",
                    int(cpu_cores_missing),
                    int(total_mem_working),
                    int(total_mem_limit),
                    f"{mem_pct:.2f}",
                    f"{total_net_rx:.6f}",
                    f"{total_net_tx:.6f}",
                ])
            print(f"Appended Prometheus system metrics for {service_name} to {out_path}")
        except Exception as e:
            print(f"Failed to write Prometheus system metrics for {service_name}: {e}")

    def collect_prometheus_latency_metrics_for_services(services: list[str]):
        ns = NAMESPACE
        window = f"{MONITOR_TIME}s"

        # RPS and error ratio per destination_workload
        rps_queries = [
            f'sum by (destination_workload) (rate(istio_requests_total{{destination_workload_namespace="{ns}"}}[{window}]))',
            f'sum by (destination_workload) (rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{ns}"}}[{window}]))',
        ]
        err_ratio_queries = [
            (
                'sum by (destination_workload) '
                f'(rate(istio_requests_total{{destination_workload_namespace="{ns}", response_code=~"5.."}}[{window}])) '
                '/ '
                'sum by (destination_workload) '
                f'(rate(istio_requests_total{{destination_workload_namespace="{ns}"}}[{window}]))'
            ),
            (
                'sum by (destination_workload) '
                f'(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{ns}", response_code=~"5.."}}[{window}])) '
                '/ '
                'sum by (destination_workload) '
                f'(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{ns}"}}[{window}]))'
            ),
        ]

        # Latency quantiles using seconds buckets (preferred)
        def _hq(p: float, reporter: bool) -> str:
            rep = ', reporter="destination"' if reporter else ''
            return (
                f'histogram_quantile({p}, '
                'sum by (le, destination_workload) '
                f'(rate(istio_request_duration_seconds_bucket{{destination_workload_namespace="{ns}"{rep}}}[{window}])))'
            )

        p50_sec_queries = [_hq(0.5, False), _hq(0.5, True)]
        p90_sec_queries = [_hq(0.9, False), _hq(0.9, True)]
        p99_sec_queries = [_hq(0.99, False), _hq(0.99, True)]

        # Fallback to milliseconds buckets
        def _hq_ms(p: float, reporter: bool) -> str:
            rep = ', reporter="destination"' if reporter else ''
            return (
                f'histogram_quantile({p}, '
                'sum by (le, destination_workload) '
                f'(rate(istio_request_duration_milliseconds_bucket{{destination_workload_namespace="{ns}"{rep}}}[{window}])))'
            )

        p50_ms_queries = [_hq_ms(0.5, False), _hq_ms(0.5, True)]
        p90_ms_queries = [_hq_ms(0.9, False), _hq_ms(0.9, True)]
        p99_ms_queries = [_hq_ms(0.99, False), _hq_ms(0.99, True)]

        # Fetch vectors (first non-empty)
        rps = _first_nonempty_service_vector(PROM_URL, rps_queries)
        err_ratio = _first_nonempty_service_vector(PROM_URL, err_ratio_queries)

        p50_sec = _first_nonempty_service_vector(PROM_URL, p50_sec_queries)
        p90_sec = _first_nonempty_service_vector(PROM_URL, p90_sec_queries)
        p99_sec = _first_nonempty_service_vector(PROM_URL, p99_sec_queries)

        # If seconds buckets empty, fall back to ms buckets
        use_seconds = len(p50_sec) > 0 or len(p90_sec) > 0 or len(p99_sec) > 0
        if use_seconds:
            p50_ms_map = {k: v * 1000.0 for k, v in p50_sec.items()}
            p90_ms_map = {k: v * 1000.0 for k, v in p90_sec.items()}
            p99_ms_map = {k: v * 1000.0 for k, v in p99_sec.items()}
        else:
            p50_ms_map = _first_nonempty_service_vector(PROM_URL, p50_ms_queries)
            p90_ms_map = _first_nonempty_service_vector(PROM_URL, p90_ms_queries)
            p99_ms_map = _first_nonempty_service_vector(PROM_URL, p99_ms_queries)

        # Write per-service CSVs
        now_ts = datetime.utcnow().isoformat()
        for svc in services:
            service_subdir = os.path.join(SERVICE_DIR, svc)
            os.makedirs(service_subdir, exist_ok=True)
            out_path = os.path.join(service_subdir, f"prom_latency_{svc}.csv")
            headers = ['timestamp', 'service_name', 'rps', 'p50_ms', 'p90_ms', 'p99_ms', 'error_ratio']
            try:
                if not os.path.exists(out_path):
                    with open(out_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                with open(out_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now_ts,
                        svc,
                        f"{float(rps.get(svc, 0.0)):.6f}",
                        f"{float(p50_ms_map.get(svc, 0.0)):.2f}",
                        f"{float(p90_ms_map.get(svc, 0.0)):.2f}",
                        f"{float(p99_ms_map.get(svc, 0.0)):.2f}",
                        f"{float(err_ratio.get(svc, 0.0)):.6f}",
                    ])
                print(f"Appended Prometheus latency metrics for {svc} to {out_path}")
            except Exception as e:
                print(f"Failed to write Prometheus latency metrics for {svc}: {e}")

    def collect_prometheus_pod_metrics_for_services(services: list[str], service_pods: dict):
        ns = NAMESPACE
        window = f"{MONITOR_TIME}s"

        # Query per-pod maps once
        cpu_rate = _first_nonempty_vector(PROM_URL, [
            f'rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"}}[{window}])',
            f'rate(container_cpu_usage_seconds_total{{kubernetes_namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"}}[{window}])',
        ])
        cpu_cores = _first_nonempty_vector(PROM_URL, [
            f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{ns}",resource="cpu",unit="core"}})',
            f'(container_spec_cpu_quota{{namespace="{ns}",container!="istio-proxy"}} / container_spec_cpu_period{{namespace="{ns}",container!="istio-proxy"}})',
        ])
        mem_working = _first_nonempty_vector(PROM_URL, [
            f'container_memory_working_set_bytes{{namespace="{ns}",container!="istio-proxy"}}',
            f'container_memory_usage_bytes{{namespace="{ns}",container!="istio-proxy"}}',
        ])
        mem_limit = _first_nonempty_vector(PROM_URL, [
            f'kube_pod_container_resource_limits{{namespace="{ns}",resource="memory",unit="byte"}}',
            f'container_spec_memory_limit_bytes{{namespace="{ns}",container!="istio-proxy"}}',
        ])
        net_rx = _first_nonempty_vector(PROM_URL, [
            f'increase(container_network_receive_bytes_total{{namespace="{ns}"}}[{window}])',
            f'increase(container_network_receive_bytes_total{{kubernetes_namespace="{ns}"}}[{window}])',
        ])
        net_tx = _first_nonempty_vector(PROM_URL, [
            f'increase(container_network_transmit_bytes_total{{namespace="{ns}"}}[{window}])',
            f'increase(container_network_transmit_bytes_total{{kubernetes_namespace="{ns}"}}[{window}])',
        ])

        now_ts = datetime.utcnow().isoformat()
        for svc in services:
            pods = [p for p in service_pods.get(svc, [])]
            if not pods:
                continue
            service_subdir = os.path.join(SERVICE_DIR, svc)
            os.makedirs(service_subdir, exist_ok=True)
            out_path = os.path.join(service_subdir, f"prom_pods_{svc}.csv")
            headers = [
                'timestamp','service_name','pod',
                'cpu_rate_cores','cpu_cores_limit','cpu_util_percent',
                'memory_working_set_bytes','memory_limit_bytes','memory_usage_percent',
                'network_receive_bytes','network_transmit_bytes'
            ]
            try:
                if not os.path.exists(out_path):
                    with open(out_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                with open(out_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for pod in pods:
                        cr = float(cpu_rate.get(pod, 0.0))
                        cc = float(cpu_cores.get(pod, 0.0))
                        cpu_pct = (cr / cc * 100.0) if cc > 0 else (cr * 100.0)
                        mw = float(mem_working.get(pod, 0.0))
                        ml = float(mem_limit.get(pod, 0.0))
                        mem_pct = (mw / ml * 100.0) if ml > 0 else 0.0
                        rx = float(net_rx.get(pod, 0.0))
                        tx = float(net_tx.get(pod, 0.0))
                        writer.writerow([
                            now_ts, svc, pod,
                            f"{cr:.6f}", f"{cc:.6f}", f"{cpu_pct:.2f}",
                            int(mw), int(ml), f"{mem_pct:.2f}",
                            int(rx), int(tx)
                        ])
                print(f"Appended Prometheus pod metrics for {svc} ({len(pods)} pods) to {out_path}")
            except Exception as e:
                print(f"Failed to write Prometheus pod metrics for {svc}: {e}")

    # Collect Prometheus system metrics for each discovered service
    if PROM_URL:
        services = list(service_logs.keys())
        if services:
            print(f"Collecting Prometheus system metrics from {PROM_URL} for services: {services}")
            for svc in services:
                try:
                    collect_prometheus_system_metrics(svc)
                    # small delay to be gentle to Prometheus
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error collecting Prometheus system metrics for {svc}: {e}")

            # Build service->pods map for per-pod metrics
            service_pods = {svc: [pod for (pod, _codes) in pods] for svc, pods in service_logs.items()}

            # Collect latency quantiles and RPS per service
            try:
                collect_prometheus_latency_metrics_for_services(services)
            except Exception as e:
                print(f"Error collecting Prometheus latency metrics: {e}")

            # Collect per-pod metrics per service
            try:
                collect_prometheus_pod_metrics_for_services(services, service_pods)
            except Exception as e:
                print(f"Error collecting Prometheus pod metrics: {e}")


if __name__ == "__main__":
    main()