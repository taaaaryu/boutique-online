#!/usr/bin/env python3
"""
k8s外でPrometheusからメトリクスを取得するスクリプト
全サービスの監視をサポート
"""

import os
import sys
import time
import math
import requests
import json
import subprocess
import pandas as pd
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ExternalPrometheusMonitor:
    def __init__(self, prometheus_url: str, output_dir: str = "./external_monitoring_results", 
                 target_service: Optional[str] = None, namespace: str = "default", 
                 users: int = 0, replicas: int = 0, monitor_all_services: bool = False):
        self.prometheus_url = prometheus_url
        self.output_dir = output_dir
        self.namespace = namespace
        # テスト対象サービス（負荷をかけるサービス）
        self.target_service = target_service
        # 全サービス監視フラグ
        self.monitor_all_services = monitor_all_services
        self.users = users
        self.replicas = replicas
        # {(service_label, file_type): filepath}
        self.csv_files = {}
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_csv_files()
        
    def _setup_csv_files(self):
        """ルートディレクトリのみ作成（ファイルはサービス単位で遅延作成）"""
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_all_services(self) -> List[str]:
        """Kubernetesから全デプロイメントのサービス名を取得"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'deployments', '-n', self.namespace, '-o', 'jsonpath={.items[*].metadata.name}'],
                capture_output=True,
                text=True,
                check=True
            )
            services = result.stdout.strip().split()
            print(f"Found {len(services)} services in namespace {self.namespace}: {services}")
            return services
        except Exception as e:
            print(f"Error getting services from Kubernetes: {e}")
            return []

    def _ensure_files_for_service(self, service_label: str):
        """サービスごとのCSVファイルを準備（ヘッダー書き込み）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RUN_NUMBER環境変数を取得（設定されていない場合は空文字列）
        run_number = os.getenv("RUN_NUMBER", "")
        run_suffix = f"_run{run_number}" if run_number else ""
        
        # 新しい階層: <output_dir>/{target_service}/replica{replicas}/{users}/{monitored_service}
        subdirs = []
        if self.target_service:
            subdirs.append(self.target_service)
        if self.replicas:
            subdirs.append(f"replica{self.replicas}")
        if self.users:
            subdirs.append(str(self.users))
        subdirs.append(service_label)
        service_dir = os.path.join(self.output_dir, *subdirs)
        os.makedirs(service_dir, exist_ok=True)

        def ensure(file_type: str, headers: List[str]):
            key = (service_label, file_type)
            if key in self.csv_files:
                return
            filepath = os.path.join(service_dir, f"{file_type}_{timestamp}{run_suffix}.csv")
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.csv_files[key] = filepath
        # 統合フォーマット（integrated_test_system.py と同等）
        ensure(
            'system_metrics_full',
            [
                'timestamp',
                'pod_name',
                'cpu_usage_seconds_total',
                'cpu_usage_percent',
                'cpu_throttled_seconds_total',
                'memory_working_set_bytes',
                'memory_limit_bytes',
                'memory_usage_percent',
                'network_receive_bytes_total',
                'network_transmit_bytes_total',
            ],
        )
        
        # 遅延時間メトリクス
        ensure(
            'latency_metrics',
            [
                'timestamp',
                'service_name',
                'request_duration_avg',
                'request_rate_total',
                'response_size_bytes',
                # additional error/success metrics
                'request_error_rate',
                'request_success_rate_percent',
            ],
        )
        # downstream success metrics per source->destination
        ensure(
            'downstream_success',
            [
                'timestamp',
                'source_service',
                'destination_service',
                'request_rate',
                'error_rate',
                'success_rate_percent',
            ],
        )
    
    def query_prometheus(self, query: str) -> Optional[Dict]:
        """Prometheusにクエリを送信"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"Error querying Prometheus: {exc}")
            return None
    
    def write_metrics_to_csv(self, metrics: List[Dict], file_type: str, service_label: str):
        """メトリクスをCSVファイルに書き込み（サービス単位ファイル）"""
        key = (service_label, file_type)
        if key not in self.csv_files:
            self._ensure_files_for_service(service_label)
        filepath = self.csv_files[key]
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            for metric in metrics:
                writer.writerow(list(metric.values()))
    
    def monitor_once(self, for_service: str, duration: int):
        """指定されたサービスを監視"""
        start_time = time.time()
        service_label = for_service
        print(f"Monitoring service={service_label} for {duration}s ...")
        while time.time() - start_time < duration:
            try:
                # フルシステムメトリクス（integrated形式）を取得
                full_rows = self.get_system_metrics_full(for_service)
                if full_rows:
                    self.write_metrics_to_csv(full_rows, 'system_metrics_full', service_label)
                    print(f"[{service_label}] Collected full system metrics for {len(full_rows)} rows")

                # 遅延時間メトリクスを取得（Istio対応サービスのみ）
                latency_rows = self.get_latency_metrics(for_service)
                if latency_rows:
                    # compute error/success rates for the service and attach
                    err = self.get_error_rates(for_service)
                    if err is not None:
                        for r in latency_rows:
                            r['request_error_rate'] = err.get('error_rate', 0.0)
                            r['request_success_rate_percent'] = err.get('success_percent', 100.0)
                    self.write_metrics_to_csv(latency_rows, 'latency_metrics', service_label)
                    print(f"[{service_label}] Collected latency metrics for {len(latency_rows)} rows")
                else:
                    print(f"[{service_label}] No latency metrics available (may not be Istio-enabled or HTTP service)")

                # downstream success per destination from this service (if source label available)
                downstream_rows = self.get_downstream_success_rates(for_service)
                if downstream_rows:
                    self.write_metrics_to_csv(downstream_rows, 'downstream_success', service_label)
                    print(f"[{service_label}] Collected downstream success metrics for {len(downstream_rows)} rows")

                time.sleep(5)  # 5秒間隔で監視
                    
            except KeyboardInterrupt:
                print("Monitoring interrupted by user")
                break
            except Exception as e:
                print(f"Error during monitoring {service_label}: {e}")
                time.sleep(5)

    def monitor(self, duration: int = 60):
        """指定時間、全サービスまたは指定サービスを監視"""
        print(f"Starting external monitoring for {duration} seconds...")
        print(f"Prometheus URL: {self.prometheus_url}")
        print(f"Namespace: {self.namespace}")
        print(f"Target service (load test): {self.target_service}")
        print(f"Monitor all services: {self.monitor_all_services}")
        print(f"Output directory: {self.output_dir}")
        
        if self.monitor_all_services:
            # 全サービスを監視
            all_services = self._get_all_services()
            if not all_services:
                print("Warning: No services found, monitoring target service only")
                all_services = [self.target_service] if self.target_service else []
            
            print(f"Monitoring {len(all_services)} services concurrently...")
            # 各サービスを並行監視（簡易実装：順次だが同じ時間枠で）
            import threading
            threads = []
            for svc in all_services:
                thread = threading.Thread(target=self.monitor_once, args=(svc, duration))
                thread.start()
                threads.append(thread)
            
            # 全スレッドの完了を待つ
            for thread in threads:
                thread.join()
        else:
            # 指定サービスのみ監視
            if self.target_service:
                self.monitor_once(self.target_service, duration)
            else:
                print("No service specified for monitoring")
        
        print("External monitoring completed")
        print(f"Results saved to: {self.output_dir}")

    def _query_vector(self, query: str) -> Dict[str, float]:
        """Prometheus vector query, returns mapping key=pod or service value=float."""
        data = self.query_prometheus(query)
        results: Dict[str, float] = {}
        try:
            vec = data.get('data', {}).get('result', []) if data else []
            for item in vec:
                metric = item.get('metric', {})
                key = metric.get('pod') or metric.get('destination_service_name') or 'unknown'
                value = float(item['value'][1])
                results[key] = value
        except Exception:
            return {}
        return results

    def _get_label_values(self, label: str) -> List[str]:
        """Return the list of values for a Prometheus label key via the label API.
        Returns empty list on error or if no values.
        """
        try:
            resp = requests.get(f"{self.prometheus_url}/api/v1/label/{label}/values", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') == 'success':
                return data.get('data', []) or []
        except Exception:
            pass
        return []

    def _detect_label_variant(self, candidates: List[str]) -> Optional[str]:
        """Pick the first label that actually exists in Prometheus (has non-empty values).
        Returns None if none of the candidates are present.
        """
        for lbl in candidates:
            vals = self._get_label_values(lbl)
            if vals:
                print(f"Detected label '{lbl}' with {len(vals)} values (sample: {vals[:5]})")
                return lbl
        return None

    def get_error_rates(self, service: str) -> Optional[Dict[str, float]]:
        """Return error rate and success percent for a given destination service."""
        ns = self.namespace
        # total rate and 5xx rate
        total_q = f'sum(rate(istio_requests_total{{destination_service_name="{service}",reporter="destination"}}[1m]))'
        err_q = f'sum(rate(istio_requests_total{{destination_service_name="{service}",reporter="destination",response_code=~"5.."}}[1m]))'
        total = self._query_single_value(total_q) or 0.0
        errs = self._query_single_value(err_q) or 0.0
        try:
            success_percent = 100.0 * (1.0 - (errs / total)) if total > 0 else 100.0
        except Exception:
            success_percent = 100.0
        return {'request_rate': total, 'error_rate': errs, 'success_percent': success_percent}

    def get_downstream_success_rates(self, source_service: str) -> List[Dict]:
        """Return per-destination success rates for requests originating from source_service.
        Uses istio_requests_total with source_workload/destination_service_name labels when available.
        """
        rows: List[Dict] = []
        # Quick health check: if Prometheus is unreachable, return a sentinel row so
        # the CSV contains an explicit marker instead of remaining empty.
        prom_check = self.query_prometheus('up')
        prom_unreachable = False
        if not prom_check or prom_check.get('status') != 'success':
            prom_unreachable = True
            print("Prometheus appears unreachable (health check failed). Will emit sentinel downstream row.")

        rows: List[Dict] = []
        # Try to detect the correct src/dst label names from Prometheus before querying.
        ts = datetime.now().isoformat()
        src_candidates = ['source_workload', 'source_workload', 'source', 'source_workload_short']
        dst_candidates = ['destination_service_name', 'destination_service', 'destination', 'destination_workload']

        detected_src = self._detect_label_variant(src_candidates)
        detected_dst = self._detect_label_variant(dst_candidates)

        if detected_src and detected_dst:
            print(f"Using detected labels src={detected_src}, dst={detected_dst} for downstream queries")
            dest_rate_q = f'sum by ({detected_dst}) (rate(istio_requests_total{{{detected_src}="{source_service}",reporter="destination"}}[1m]))'
            dest_err_q = f'sum by ({detected_dst}) (rate(istio_requests_total{{{detected_src}="{source_service}",reporter="destination",response_code=~"5.."}}[1m]))'
            try:
                total_data = self.query_prometheus(dest_rate_q)
                err_data = self.query_prometheus(dest_err_q)
                if not total_data or total_data.get('status') != 'success':
                    print(f"No data for downstream totals with detected labels {detected_src}/{detected_dst}")
                else:
                    total_map = {}
                    err_map = {}
                    for item in total_data.get('data', {}).get('result', []):
                        dst = item.get('metric', {}).get(detected_dst) or item.get('metric', {}).get('destination_service_name') or 'unknown'
                        try:
                            total_map[dst] = float(item.get('value', [0, 0])[1])
                        except Exception:
                            total_map[dst] = 0.0

                    for item in err_data.get('data', {}).get('result', []):
                        dst = item.get('metric', {}).get(detected_dst) or item.get('metric', {}).get('destination_service_name') or 'unknown'
                        try:
                            err_map[dst] = float(item.get('value', [0, 0])[1])
                        except Exception:
                            err_map[dst] = 0.0

                    if total_map:
                        for dst, total in total_map.items():
                            errs = err_map.get(dst, 0.0)
                            success_pct = 100.0 * (1.0 - (errs / total)) if total > 0 else 100.0
                            rows.append({
                                'timestamp': ts,
                                'source_service': source_service,
                                'destination_service': dst,
                                'request_rate': total,
                                'error_rate': errs,
                                'success_rate_percent': success_pct,
                            })
                        return rows
                    else:
                        print(f"Detected labels {detected_src}/{detected_dst} returned no destinations")
            except Exception as e:
                print(f"Error querying downstream with detected labels {detected_src}/{detected_dst}: {e}")

        # If detection failed or returned no results, fall back to brute-force label variants
        print("Falling back to brute-force label variants")
        label_variants = [
            {'src_label': 'source_workload', 'dst_label': 'destination_service_name'},
            {'src_label': 'source_workload', 'dst_label': 'destination_service'},
            {'src_label': 'source_workload', 'dst_label': 'destination'},
            {'src_label': 'source_workload', 'dst_label': 'destination_service_name_unstable'},
        ]

        tried_any = False
        for labels in label_variants:
            src_label = labels['src_label']
            dst_label = labels['dst_label']
            # Build queries using the label names
            dest_rate_q = f'sum by ({dst_label}) (rate(istio_requests_total{{{src_label}="{source_service}",reporter="destination"}}[1m]))'
            dest_err_q = f'sum by ({dst_label}) (rate(istio_requests_total{{{src_label}="{source_service}",reporter="destination",response_code=~"5.."}}[1m]))'
            print(f"Trying downstream queries with src_label={src_label}, dst_label={dst_label}")
            try:
                total_data = self.query_prometheus(dest_rate_q)
                err_data = self.query_prometheus(dest_err_q)
                tried_any = True
                total_map = {}
                err_map = {}
                if not total_data or total_data.get('status') != 'success':
                    print(f"No data (or error) for total query using labels {src_label}/{dst_label}: {getattr(total_data, 'text', total_data)}")
                    continue
                if not err_data or err_data.get('status') != 'success':
                    print(f"No data (or error) for error query using labels {src_label}/{dst_label}: {getattr(err_data, 'text', err_data)}")

                for item in total_data.get('data', {}).get('result', []):
                    dst = item.get('metric', {}).get(dst_label) or item.get('metric', {}).get('destination_service_name') or 'unknown'
                    try:
                        total_map[dst] = float(item.get('value', [0, 0])[1])
                    except Exception:
                        total_map[dst] = 0.0

                for item in err_data.get('data', {}).get('result', []):
                    dst = item.get('metric', {}).get(dst_label) or item.get('metric', {}).get('destination_service_name') or 'unknown'
                    try:
                        err_map[dst] = float(item.get('value', [0, 0])[1])
                    except Exception:
                        err_map[dst] = 0.0

                # If we got any totals, build rows and return them (prefer first working label set)
                if total_map:
                    for dst, total in total_map.items():
                        errs = err_map.get(dst, 0.0)
                        success_pct = 100.0 * (1.0 - (errs / total)) if total > 0 else 100.0
                        rows.append({
                            'timestamp': ts,
                            'source_service': source_service,
                            'destination_service': dst,
                            'request_rate': total,
                            'error_rate': errs,
                            'success_rate_percent': success_pct,
                        })
                    # We prefer the first label variant that returns results
                    return rows
                else:
                    print(f"Label variant {src_label}/{dst_label} returned no destinations (empty result set)")

            except Exception as e:
                print(f"Error querying downstream success rates with labels {src_label}/{dst_label}: {e}")

        if prom_unreachable:
            # Emit a sentinel row so the CSV shows why there was no data
            rows = [{
                'timestamp': ts,
                'source_service': source_service,
                'destination_service': '__prometheus_unreachable__',
                'request_rate': 0.0,
                'error_rate': 0.0,
                'success_rate_percent': 0.0,
            }]
            print(f"Emitting sentinel downstream_success row for {source_service} due to Prometheus unreachability")
            return rows

        if not tried_any:
            print("No downstream queries were attempted (Prometheus may be unreachable)")
        else:
            print(f"Downstream success queries tried {len(label_variants)} label variants but found no results for source {source_service}")

        return rows

    def get_system_metrics_full(self, service: Optional[str] = None) -> List[Dict]:
        """integrated_test_system.py と同じ指標のセットを、Podごと（＋ALL_PODS）で返す"""
        pod_filter = ''
        if service:
            pod_filter = f',pod=~"{service}-.*"'
        ns = self.namespace
        # Vector queries grouped by pod (excluding istio-proxy to get only app metrics)
        cpu_usage_q = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!="",pod!="",container!="POD",container!="istio-proxy"{pod_filter}}}[1m]))'
        cpu_limit_q = f'sum by (pod) (container_spec_cpu_quota{{namespace="{ns}",container!="istio-proxy"{pod_filter}}} / container_spec_cpu_period{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})'
        cpu_throttle_q = f'sum by (pod) (rate(container_cpu_cfs_throttled_seconds_total{{namespace="{ns}",container!="istio-proxy"{pod_filter}}}[1m]))'
        mem_working_q = f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})'
        mem_limit_q = f'sum by (pod) (container_spec_memory_limit_bytes{{namespace="{ns}",container!="istio-proxy"{pod_filter}}})'
        net_rx_q = f'sum by (pod) (rate(container_network_receive_bytes_total{{namespace="{ns}"{pod_filter}}}[1m]))'
        net_tx_q = f'sum by (pod) (rate(container_network_transmit_bytes_total{{namespace="{ns}"{pod_filter}}}[1m]))'

        cpu_usage = self._query_vector(cpu_usage_q)
        cpu_limit = self._query_vector(cpu_limit_q)
        cpu_throttle = self._query_vector(cpu_throttle_q)
        mem_working = self._query_vector(mem_working_q)
        mem_limit = self._query_vector(mem_limit_q)
        net_rx = self._query_vector(net_rx_q)
        net_tx = self._query_vector(net_tx_q)

        pod_names = sorted(set().union(
            cpu_usage.keys(), cpu_limit.keys(), cpu_throttle.keys(),
            mem_working.keys(), mem_limit.keys(), net_rx.keys(), net_tx.keys()
        ))

        rows: List[Dict] = []
        now_ts = datetime.now().isoformat()

        total_cpu_usage = 0.0
        total_cpu_limit = 0.0
        total_cpu_throttle = 0.0
        total_mem_working = 0.0
        total_mem_limit = 0.0
        total_net_rx = 0.0
        total_net_tx = 0.0

        for pod in pod_names:
            c_usage = float(cpu_usage.get(pod, 0.0))
            c_limit = float(cpu_limit.get(pod, 0.0))
            c_throt = float(cpu_throttle.get(pod, 0.0))
            m_work = float(mem_working.get(pod, 0.0))
            m_lim = float(mem_limit.get(pod, 0.0))
            rx = float(net_rx.get(pod, 0.0))
            tx = float(net_tx.get(pod, 0.0))

            total_cpu_usage += c_usage
            total_cpu_limit += c_limit
            total_cpu_throttle += c_throt
            total_mem_working += m_work
            total_mem_limit += m_lim
            total_net_rx += rx
            total_net_tx += tx

            cpu_pct = (c_usage / c_limit * 100.0) if c_limit > 0 else 0.0
            mem_pct = (m_work / m_lim * 100.0) if m_lim > 0 else 0.0

            rows.append({
                'timestamp': now_ts,
                'pod_name': pod,
                'cpu_usage_seconds_total': c_usage,
                'cpu_usage_percent': cpu_pct,
                'cpu_throttled_seconds_total': c_throt,
                'memory_working_set_bytes': int(m_work),
                'memory_limit_bytes': int(m_lim),
                'memory_usage_percent': mem_pct,
                'network_receive_bytes_total': rx,
                'network_transmit_bytes_total': tx,
            })

        if service and pod_names:
            # ALL_PODS aggregate row
            total_cpu_pct = (total_cpu_usage / total_cpu_limit * 100.0) if total_cpu_limit > 0 else 0.0
            total_mem_pct = (total_mem_working / total_mem_limit * 100.0) if total_mem_limit > 0 else 0.0
            rows.append({
                'timestamp': now_ts,
                'pod_name': 'ALL_PODS',
                'cpu_usage_seconds_total': total_cpu_usage,
                'cpu_usage_percent': total_cpu_pct,
                'cpu_throttled_seconds_total': total_cpu_throttle,
                'memory_working_set_bytes': int(total_mem_working),
                'memory_limit_bytes': int(total_mem_limit),
                'memory_usage_percent': total_mem_pct,
                'network_receive_bytes_total': total_net_rx,
                'network_transmit_bytes_total': total_net_tx,
            })

        return rows

    def get_latency_metrics(self, service: Optional[str] = None) -> List[Dict]:
        """Istioメトリクスから遅延時間関連の指標を取得"""
        if not service:
            return []
        
        ns = self.namespace
        now_ts = datetime.now().isoformat()
        
        # Istioメトリクスクエリ（Istio sidecar経由のHTTPリクエスト）
        # frontendの場合はfrontend-externalも含める
        if service == "frontend":
            service_filter = f'destination_service_name=~"frontend.*",reporter="destination"'
        else:
            service_filter = f'destination_service_name="{service}",reporter="destination"'
        
        # 平均リクエスト持続時間
        duration_avg_q = f'sum(rate(istio_request_duration_milliseconds_sum{{{service_filter}}}[1m])) / sum(rate(istio_request_duration_milliseconds_count{{{service_filter}}}[1m]))'
        
        # リクエストレート
        request_rate_q = f'sum(rate(istio_requests_total{{{service_filter}}}[1m]))'
        
        # レスポンスサイズ
        response_size_q = f'sum(rate(istio_response_bytes_sum{{{service_filter}}}[1m])) / sum(rate(istio_response_bytes_count{{{service_filter}}}[1m]))'
        
        # クエリ実行
        duration_avg = self._query_single_value(duration_avg_q)
        request_rate = self._query_single_value(request_rate_q)
        response_size = self._query_single_value(response_size_q)
        
        # すべてのメトリクスが有効な値（0より大きく、nanでない）の場合のみ記録
        if (duration_avg and duration_avg > 0 and not math.isnan(duration_avg) and
            request_rate and request_rate > 0 and not math.isnan(request_rate) and
            response_size and response_size > 0 and not math.isnan(response_size)):
            return [{
                'timestamp': now_ts,
                'service_name': service,
                'request_duration_avg': duration_avg,
                'request_rate_total': request_rate,
                'response_size_bytes': response_size,
            }]
        
        return []

    def _query_single_value(self, query: str) -> Optional[float]:
        """単一の値を返すPrometheusクエリを実行"""
        data = self.query_prometheus(query)
        try:
            if data and data.get('status') == 'success':
                result = data.get('data', {}).get('result', [])
                if result and len(result) > 0:
                    value = result[0].get('value', [])
                    if len(value) > 1:
                        return float(value[1])
        except (ValueError, TypeError, KeyError):
            pass
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='External Prometheus Monitor')
    parser.add_argument('--prometheus-url', default='http://localhost:9090', 
                       help='Prometheus URL (default: http://localhost:9090)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in seconds (default: 60)')
    parser.add_argument('--output-dir', default='./external_monitoring_results',
                       help='Output directory for results (default: ./external_monitoring_results)')
    parser.add_argument('--service', default=None,
                       help='Target service name (service under load test, e.g. frontend)')
    parser.add_argument('--namespace', default='default',
                       help='Kubernetes namespace for metrics filters (default: default)')
    parser.add_argument('--users', type=int, default=0,
                       help='User count for directory nesting (default: 0 = skip)')
    parser.add_argument('--replicas', type=int, default=0,
                       help='Replica count for directory nesting (default: 0 = skip)')
    parser.add_argument('--monitor-all-services', action='store_true',
                       help='Monitor all services in the namespace (default: only target service)')
    
    args = parser.parse_args()
    
    monitor = ExternalPrometheusMonitor(
        prometheus_url=args.prometheus_url,
        output_dir=args.output_dir,
        target_service=args.service,
        namespace=args.namespace,
        users=args.users,
        replicas=args.replicas,
        monitor_all_services=args.monitor_all_services
    )
    monitor.monitor(args.duration)

if __name__ == "__main__":
    main()
