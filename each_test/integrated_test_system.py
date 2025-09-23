#!/usr/bin/env python3
"""
統合テストシステム
テスト実行→監視開始→テスト終了→結果表示（ログ作成、可視化）を一括管理
"""

import os
import sys
import time
import subprocess
import threading
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import signal
import atexit
import requests
import json
import csv
from typing import Dict, List, Optional

# gRPC imports disabled - using HTTP-based testing only
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# sys.path.append(os.path.join(project_root, 'src/paymentservice/proto'))
# sys.path.append(os.path.join(project_root, 'src/cartservice/src/protos'))

# try:
#     import demo_pb2
#     import demo_pb2_grpc
#     import Cart_pb2
#     import Cart_pb2_grpc
# except ImportError as e:
#     print(f"Error importing gRPC stubs: {e}")
#     sys.exit(1)

# Service configurations - HTTP-based only
SERVICES = {
    'productcatalogservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'cartservice': {
        'host': '172.18.0.3', 
        'port': 8080,
        'endpoint': '/cart'
    },
    'checkoutservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'paymentservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'shippingservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'currencyservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'recommendationservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    },
    'adservice': {
        'host': '172.18.0.3',
        'port': 8080,
        'endpoint': '/'
    }
}


class UnifiedMonitor:
    def __init__(self, service_name: str, output_dir: str, interval: int = 5):
        self.service_name = service_name
        self.output_dir = output_dir
        self.interval = interval
        self.monitoring = False
        self.csv_writers = {}
        self.csv_files = {}
        
        # Prometheus設定
        self.prometheus_url = "http://localhost:9090"
        
        os.makedirs(output_dir, exist_ok=True)
        self._setup_csv_files()

    def _setup_csv_files(self):
        """CSVファイルをセットアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CPU/メモリ/ネットワークメトリクス
        cpu_csv_path = f"{self.output_dir}/{self.service_name}_system_metrics_{timestamp}.csv"
        self.csv_files['system'] = open(cpu_csv_path, "w", newline="")
        self.csv_writers['system'] = csv.writer(self.csv_files['system'])
        self.csv_writers['system'].writerow([
            'timestamp', 'pod_name', 
            'cpu_usage_seconds_total', 'cpu_usage_percent', 'cpu_throttled_seconds_total',
            'memory_working_set_bytes', 'memory_limit_bytes', 'memory_usage_percent',
            'network_receive_bytes_total', 'network_transmit_bytes_total'
        ])
        
        print(f"Unified monitoring started for {self.service_name}")
        print(f"System metrics will be saved to: {cpu_csv_path}")
        print("Note: Istio metrics are disabled (Istio not properly installed)")

    def get_pod_names(self) -> List[str]:
        """対象サービスのPod名一覧を取得（レプリカ数に関係なくすべてのPodを取得）"""
        # 1) よくあるラベル: app=<service_name>
        selectors_to_try = [f'app={self.service_name}', f'app.kubernetes.io/name={self.service_name}']

        # 2) Serviceのselectorを流用（より確実）
        try:
            svc_result = subprocess.run(
                ['kubectl', 'get', 'svc', self.service_name, '-o', 'json'],
                capture_output=True, text=True, check=False
            )
            if svc_result.returncode == 0 and svc_result.stdout:
                import json as _json
                svc_obj = _json.loads(svc_result.stdout)
                selector = svc_obj.get('spec', {}).get('selector', {})
                if selector:
                    selector_str = ','.join([f"{k}={v}" for k, v in selector.items()])
                    selectors_to_try.insert(0, selector_str)
        except Exception as _:
            pass

        all_pods: List[str] = []
        for selector in selectors_to_try:
            try:
                cmd = [
                    'kubectl', 'get', 'pods',
                    '-l', selector,
                    '-o', 'jsonpath={.items[*].metadata.name}'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout:
                    pod_names = [p for p in result.stdout.strip().split() if p]
                    print(pod_names)
                    for p in pod_names:
                        pod_name = p.split('-')[0]
                        if pod_name == self.service_name:
                            all_pods.append(p)
            except Exception as _:
                continue

        if not all_pods:
            print(f"Warning: No pods found for selectors: {selectors_to_try}")
        else:
            print(f"Found {len(all_pods)} pods for {self.service_name}: {all_pods}")

        return all_pods

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

    def _query_scalar(self, query: str) -> float:
        """Run a Prometheus instant query and return its first scalar value, or 0.0 if missing."""
        data = self.query_prometheus(query)
        try:
            result = data.get('data', {}).get('result', []) if data else []
            if not result:
                return 0.0
            return float(result[0]['value'][1])
        except Exception:
            return 0.0

    def get_system_metrics(self, pod_name: str) -> Dict:
        """システムメトリクスを取得（Pod内全コンテナ合計で算出）"""
        metrics = {
            'cpu_usage_seconds_total': 0.0,
            'cpu_usage_percent': 0.0,
            'cpu_throttled_seconds_total': 0.0,
            'memory_working_set_bytes': 0,
            'memory_limit_bytes': 0,
            'memory_usage_percent': 0.0,
            'network_receive_bytes_total': 0,
            'network_transmit_bytes_total': 0
        }
        # CPU usage (cores/sec), 1m rate sum over containers
        cpu_query = (
            f'sum(rate(container_cpu_usage_seconds_total{{pod="{pod_name}",container!="",container!="POD"}}[1m]))'
        )
        metrics['cpu_usage_seconds_total'] = self._query_scalar(cpu_query)

        # CPU limit cores sum
        cpu_limit_query = (
            f'sum(container_spec_cpu_quota{{pod="{pod_name}"}} / container_spec_cpu_period{{pod="{pod_name}"}})'
        )
        cpu_limit_cores = self._query_scalar(cpu_limit_query)
        if cpu_limit_cores > 0:
            metrics['cpu_usage_percent'] = (metrics['cpu_usage_seconds_total'] / cpu_limit_cores) * 100.0

        # CPU throttled seconds (rate), sum
        throttle_query = (
            f'sum(rate(container_cpu_cfs_throttled_seconds_total{{pod="{pod_name}"}}[1m]))'
        )
        metrics['cpu_throttled_seconds_total'] = self._query_scalar(throttle_query)

        # Memory working set sum
        memory_query = f'sum(container_memory_working_set_bytes{{pod="{pod_name}"}})'
        metrics['memory_working_set_bytes'] = int(self._query_scalar(memory_query))

        # Memory limit sum
        memory_limit_query = f'sum(container_spec_memory_limit_bytes{{pod="{pod_name}"}})'
        metrics['memory_limit_bytes'] = int(self._query_scalar(memory_limit_query))
        if metrics['memory_limit_bytes'] > 0:
            metrics['memory_usage_percent'] = (metrics['memory_working_set_bytes'] / metrics['memory_limit_bytes']) * 100.0

        # Network RX/TX sum, 1m rate
        network_rx_query = f'sum(rate(container_network_receive_bytes_total{{pod="{pod_name}"}}[1m]))'
        network_tx_query = f'sum(rate(container_network_transmit_bytes_total{{pod="{pod_name}"}}[1m]))'
        metrics['network_receive_bytes_total'] = self._query_scalar(network_rx_query)
        metrics['network_transmit_bytes_total'] = self._query_scalar(network_tx_query)

        return metrics

    def get_system_metrics_total(self, pod_names: List[str]) -> Dict:
        """対象Pod群の合計メトリクスを取得（ALL_PODS用）"""
        metrics = {
            'cpu_usage_seconds_total': 0.0,
            'cpu_usage_percent': 0.0,
            'cpu_throttled_seconds_total': 0.0,
            'memory_working_set_bytes': 0,
            'memory_limit_bytes': 0,
            'memory_usage_percent': 0.0,
            'network_receive_bytes_total': 0,
            'network_transmit_bytes_total': 0
        }
        if not pod_names:
            return metrics
        pod_regex = "|".join(pod_names)
        cpu_query = (
            f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{pod_regex}",container!="",container!="POD"}}[1m]))'
        )
        cpu_limit_query = (
            f'sum(container_spec_cpu_quota{{pod=~"{pod_regex}"}} / container_spec_cpu_period{{pod=~"{pod_regex}"}})'
        )
        throttle_query = (
            f'sum(rate(container_cpu_cfs_throttled_seconds_total{{pod=~"{pod_regex}"}}[1m]))'
        )
        mem_query = f'sum(container_memory_working_set_bytes{{pod=~"{pod_regex}"}})'
        mem_limit_query = f'sum(container_spec_memory_limit_bytes{{pod=~"{pod_regex}"}})'
        rx_query = f'sum(rate(container_network_receive_bytes_total{{pod=~"{pod_regex}"}}[1m]))'
        tx_query = f'sum(rate(container_network_transmit_bytes_total{{pod=~"{pod_regex}"}}[1m]))'

        metrics['cpu_usage_seconds_total'] = self._query_scalar(cpu_query)
        cpu_limit_cores = self._query_scalar(cpu_limit_query)
        metrics['cpu_throttled_seconds_total'] = self._query_scalar(throttle_query)
        metrics['memory_working_set_bytes'] = int(self._query_scalar(mem_query))
        metrics['memory_limit_bytes'] = int(self._query_scalar(mem_limit_query))
        metrics['network_receive_bytes_total'] = self._query_scalar(rx_query)
        metrics['network_transmit_bytes_total'] = self._query_scalar(tx_query)
        if cpu_limit_cores > 0:
            metrics['cpu_usage_percent'] = (metrics['cpu_usage_seconds_total'] / cpu_limit_cores) * 100.0
        if metrics['memory_limit_bytes'] > 0:
            metrics['memory_usage_percent'] = (metrics['memory_working_set_bytes'] / metrics['memory_limit_bytes']) * 100.0
        return metrics

    def get_istio_metrics(self, pod_name: str) -> Dict:
        """Istioメトリクスを取得"""
        metrics = {
            'requests_total': 0,
            'requests_per_second': 0.0,
            'request_duration_milliseconds': 0.0,
            'request_duration_p50': 0.0,
            'request_duration_p95': 0.0,
            'request_duration_p99': 0.0,
            'request_messages_total': 0,
            'response_messages_total': 0,
            'error_rate_4xx': 0.0,
            'error_rate_5xx': 0.0
        }
        
        # リクエスト総数
        requests_query = f'sum(rate(istio_requests_total{{destination_workload="{self.service_name}"}}[5m]))'
        requests_data = self.query_prometheus(requests_query)
        if requests_data and requests_data.get('data', {}).get('result'):
            metrics['requests_per_second'] = float(requests_data['data']['result'][0]['value'][1])
        
        # リクエスト持続時間（P50）
        duration_query = f'histogram_quantile(0.5, rate(istio_request_duration_milliseconds_bucket{{destination_workload="{self.service_name}"}}[5m]))'
        duration_data = self.query_prometheus(duration_query)
        if duration_data and duration_data.get('data', {}).get('result'):
            metrics['request_duration_p50'] = float(duration_data['data']['result'][0]['value'][1])
        
        return metrics

    def monitor_metrics(self):
        """メトリクスを定期収集してCSVへ追記（すべてのPodを監視）"""
        while self.monitoring:
            try:
                pod_names = self.get_pod_names()
                timestamp = datetime.now().isoformat()
                
                if not pod_names:
                    print(f"[{timestamp}] {self.service_name}: No pods found, skipping monitoring cycle")
                    time.sleep(self.interval)
                    continue
                
                # すべてのPodのメトリクスを記録
                for pod_name in pod_names:
                    try:
                        system_metrics = self.get_system_metrics(pod_name)
                        self.csv_writers['system'].writerow([
                            timestamp, pod_name,
                            system_metrics['cpu_usage_seconds_total'],
                            system_metrics['cpu_usage_percent'],
                            system_metrics['cpu_throttled_seconds_total'],
                            system_metrics['memory_working_set_bytes'],
                            system_metrics['memory_limit_bytes'],
                            system_metrics['memory_usage_percent'],
                            system_metrics['network_receive_bytes_total'],
                            system_metrics['network_transmit_bytes_total']
                        ])
                    except Exception as pod_exc:
                        print(f"Error monitoring pod {pod_name}: {pod_exc}")
                        continue

                # 合計行（ALL_PODS）を追記
                try:
                    total_metrics = self.get_system_metrics_total(pod_names)
                    self.csv_writers['system'].writerow([
                        timestamp, 'ALL_PODS',
                        total_metrics['cpu_usage_seconds_total'],
                        total_metrics['cpu_usage_percent'],
                        total_metrics['cpu_throttled_seconds_total'],
                        total_metrics['memory_working_set_bytes'],
                        total_metrics['memory_limit_bytes'],
                        total_metrics['memory_usage_percent'],
                        total_metrics['network_receive_bytes_total'],
                        total_metrics['network_transmit_bytes_total']
                    ])
                except Exception as total_exc:
                    print(f"Error computing total metrics: {total_exc}")
                
                # ファイルに書き込み
                for csv_file in self.csv_files.values():
                    csv_file.flush()
                
                print(f"[{timestamp}] {self.service_name}: {len(pod_names)} pods monitored")
                    
            except Exception as exc:
                print(f"Error in metrics monitoring: {exc}")
            
            time.sleep(self.interval)

    def start_monitoring(self):
        """監視を開始"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_metrics)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """監視を停止"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.join()
        
        for csv_file in self.csv_files.values():
            csv_file.close()
        
        print(f"Unified monitoring stopped for {self.service_name}")


class ServiceLoadTester:
    def __init__(self, service_name, user_count, duration, output_dir):
        self.service_name = service_name
        self.user_count = user_count
        self.duration = duration
        self.output_dir = output_dir
        self.service_config = SERVICES[service_name]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{output_dir}/{service_name}_load_test_{timestamp}.csv"
        
    def create_locustfile(self):
        """Create a temporary locustfile for the specific service"""
        # Use HTTP-based Locust with e-commerce scenario from locustfile.py
        locust_content = f'''#!/usr/bin/env python3
import time
import random
from faker import Faker
from locust import HttpUser, task, between, events
import json
import datetime
import csv
import os

fake = Faker()

# Product list from original locustfile.py
products = [
    '0PUK6V6EV0',
    '1YMWWN1N4O', 
    '2ZYFJ3GM2N',
    '66VCHSJNUP',
    '6E92ZMYYFZ',
    '9SIQT8TOJO',
    'L9ECAV7KIM',
    'LS4PSXUNUM',
    'OLJCESPC7Z'
]

class {self.service_name.capitalize()}User(HttpUser):
    wait_time = between(1, 10)
    
    # Set host based on service - use NodePort
    host = "http://172.18.0.2:32075"
    
    @task
    def test_service(self):
        start_time = time.time()
        try:
            # Call different methods based on service
            if "{self.service_name}" == 'productcatalogservice':
                self._test_product_catalog()
            elif "{self.service_name}" == 'cartservice':
                self._test_cart_service()
            elif "{self.service_name}" == 'currencyservice':
                self._test_currency_service()
            elif "{self.service_name}" == 'recommendationservice':
                self._test_recommendation_service()
            elif "{self.service_name}" == 'adservice':
                self._test_ad_service()
            else:
                # Generic test for other services
                self._test_generic_service()
                
            response_time = (time.time() - start_time) * 1000

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

    
    def _test_product_catalog(self):
        # Test ListProducts via HTTP - use root endpoint
        response = self.client.get("/")
        return response
    
    def _test_cart_service(self):
        # Test cart via HTTP - use simple endpoint
        response = self.client.get("/cart")
        return response
    
    def _test_currency_service(self):
        # Test currency via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_recommendation_service(self):
        # Test recommendations via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_ad_service(self):
        # Test ads via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_generic_service(self):
        # Generic test - just make a simple HTTP request
        response = self.client.get("/")
        return response

# Create concrete user class
class TestUser({self.service_name.capitalize()}User):
    pass
'''
        locustfile_path = f"{self.output_dir}/{self.service_name}_locustfile.py"
        with open(locustfile_path, 'w') as f:
            f.write(locust_content)
        
        return locustfile_path
    
    def run_load_test(self):
        """Run the load test for the specific service"""
        print(f"Starting load test for {self.service_name} with {self.user_count} users for {self.duration} seconds")
        
        # Create locustfile
        locustfile_path = self.create_locustfile()
        
        # Run locust with CSV output for success rate metrics
        cmd = [
            'locust',
            '-f', locustfile_path,
            '--headless',
            '--users', str(self.user_count),
            '--spawn-rate', '10',
            '--run-time', f'{self.duration}s',
            '--csv', f'{self.output_dir}/{self.service_name}_results',
            '--logfile', f'{self.output_dir}/{self.service_name}_locust.log'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.duration + 60)
            if result.returncode == 0:
                print(f"Load test completed for {self.service_name}")
                return True
            else:
                print(f"Load test failed for {self.service_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"Load test timed out for {self.service_name}")
            return False


class IntegratedTestSystem:
    def __init__(self, service_name, user_count, duration, output_dir, replica_count=1):
        self.service_name = service_name
        self.user_count = user_count
        self.duration = duration
        self.output_dir = output_dir
        self.replica_count = replica_count
        
        # Create service-specific output directory
        self.service_output_dir = f"{output_dir}/replica{replica_count}/{user_count}/{service_name}"
        os.makedirs(self.service_output_dir, exist_ok=True)
        
        # Initialize components
        self.load_tester = ServiceLoadTester(service_name, user_count, duration, self.service_output_dir)
        self.monitor = UnifiedMonitor(service_name, self.service_output_dir, interval=1)
        
        # Control flags
        self.running = False
        self.load_test_completed = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\nReceived signal {signum}, stopping tests...")
        self.stop_tests()
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_tests()
    
    def start_monitoring(self):
        """Start monitoring in background"""
        print(f"Starting unified monitoring for {self.service_name}...")
        self.monitor.start_monitoring()
    
    def run_load_test(self):
        """Run the load test"""
        print(f"Starting load test for {self.service_name}...")
        success = self.load_tester.run_load_test()
        self.load_test_completed = True
        return success
    
    def run_tests(self):
        """Run both load test and monitoring"""
        self.running = True
        
        # Start monitoring first
        self.start_monitoring()
        
        # Wait a bit for monitoring to start
        time.sleep(5)
        
        # Run load test
        load_test_success = self.run_load_test()
        
        # Wait a bit more to capture post-load metrics
        time.sleep(10)
        
        # Stop monitoring
        self.stop_tests()
        
        return load_test_success
    
    def stop_tests(self):
        """Stop all tests"""
        if self.running:
            self.running = False
            self.monitor.stop_monitoring()
            print("All tests stopped")
    
    def create_visualization(self):
        """Create visualization of test results"""
        print(f"Creating visualization for {self.service_name}...")
        
        # Find CSV files
        system_files = [f for f in os.listdir(self.service_output_dir) if '_system_metrics_' in f and f.endswith('.csv')]
        locust_files = [f for f in os.listdir(self.service_output_dir) if f.endswith('_stats.csv')]
        
        if not system_files:
            print(f"No metric files found for visualization")
            return
        
        # Create figure with subplots (3x2 for success rate)
        fig, axes = plt.subplots(3, 1,figsize=(20, 18))
        fig.suptitle(f'Test Results - {self.service_name} (Users: {self.user_count}, Replica: {self.replica_count})', fontsize=16)
        
        # Plot CPU usage
        ax1 = axes[0, 0]
        if system_files:
            df_system = pd.read_csv(os.path.join(self.service_output_dir, system_files[0]))
            df_system['timestamp'] = pd.to_datetime(df_system['timestamp'])
            ax1.plot(df_system['timestamp'], df_system['cpu_usage_percent'], 
                    marker='o', markersize=3, label='CPU Usage (%)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Usage Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Memory usage
        ax2 = axes[0, 1]
        if system_files:
            ax2.plot(df_system['timestamp'], df_system['memory_usage_percent'], 
                    marker='s', markersize=3, label='Memory Usage (%)')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_title('Memory Usage Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Network Traffic
        ax3 = axes[1, 0]
        if system_files:
            ax3.plot(df_system['timestamp'], df_system['network_receive_bytes_total'], 
                    marker='^', markersize=3, label='Network RX (bytes/sec)')
            ax3.plot(df_system['timestamp'], df_system['network_transmit_bytes_total'], 
                    marker='v', markersize=3, label='Network TX (bytes/sec)')
        ax3.set_ylabel('Network Traffic (bytes/sec)')
        ax3.set_title('Network Traffic Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot CPU Throttling
        ax4 = axes[1, 1]
        if system_files:
            ax4.plot(df_system['timestamp'], df_system['cpu_throttled_seconds_total'], 
                    marker='d', markersize=3, label='CPU Throttling (sec/sec)')
        ax4.set_ylabel('CPU Throttling (sec/sec)')
        ax4.set_title('CPU Throttling Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot Success Rate (if Locust stats available)
        ax5 = axes[2, 0]
        if locust_files:
            df_locust = pd.read_csv(os.path.join(self.service_output_dir, locust_files[0]))
            # Calculate success rate
            df_locust['success_rate'] = ((df_locust['Request Count'] - df_locust['Failure Count']) / df_locust['Request Count'] * 100).fillna(0)
            ax5.bar(df_locust['Name'], df_locust['success_rate'], alpha=0.7, color='green')
            ax5.set_ylabel('Success Rate (%)')
            ax5.set_title('Request Success Rate by Endpoint')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Locust stats available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Request Success Rate by Endpoint')
        
        # Plot Response Time (if Locust stats available)
        ax6 = axes[2, 1]
        if locust_files:
            ax6.bar(df_locust['Name'], df_locust['Average Response Time'], alpha=0.7, color='blue')
            ax6.set_ylabel('Average Response Time (ms)')
            ax6.set_title('Average Response Time by Endpoint')
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Locust stats available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Average Response Time by Endpoint')
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.service_output_dir}/{self.service_name}_test_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
        plt.close()
    
    def generate_summary_report(self):
        """Generate summary report"""
        print(f"Generating summary report for {self.service_name}...")
        
        # Find CSV files
        system_files = [f for f in os.listdir(self.service_output_dir) if '_system_metrics_' in f and f.endswith('.csv')]
        locust_files = [f for f in os.listdir(self.service_output_dir) if f.endswith('_stats.csv')]
        
        report_path = f"{self.service_output_dir}/{self.service_name}_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"=== Test Summary Report ===\n")
            f.write(f"Service: {self.service_name}\n")
            f.write(f"User Count: {self.user_count}\n")
            f.write(f"Replica Count: {self.replica_count}\n")
            f.write(f"Test Duration: {self.duration} seconds\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            print(f"System files found: {system_files}")
            
            if system_files:
                df_system = pd.read_csv(os.path.join(self.service_output_dir, system_files[0]))
                f.write("=== System Metrics Summary ===\n")
                f.write(f"Max CPU Usage: {df_system['cpu_usage_percent'].max():.2f}%\n")
                f.write(f"Avg CPU Usage: {df_system['cpu_usage_percent'].mean():.2f}%\n")
                f.write(f"Max Memory Usage: {df_system['memory_usage_percent'].max():.2f}%\n")
                f.write(f"Avg Memory Usage: {df_system['memory_usage_percent'].mean():.2f}%\n")
                f.write(f"Max Network RX: {df_system['network_receive_bytes_total'].max():.2f} bytes/sec\n")
                f.write(f"Max Network TX: {df_system['network_transmit_bytes_total'].max():.2f} bytes/sec\n\n")
            else:
                f.write("=== System Metrics Summary ===\n")
                f.write("No system metrics data available\n\n")
            
            # Locust metrics (success rate and performance)
            if locust_files:
                df_locust = pd.read_csv(os.path.join(self.service_output_dir, locust_files[0]))
                f.write("=== Load Test Metrics Summary ===\n")
                
                # Calculate overall success rate
                total_requests = df_locust['Request Count'].sum()
                total_failures = df_locust['Failure Count'].sum()
                overall_success_rate = ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0
                
                f.write(f"Total Requests: {total_requests:,}\n")
                f.write(f"Total Failures: {total_failures:,}\n")
                f.write(f"Overall Success Rate: {overall_success_rate:.2f}%\n")
                f.write(f"Average Response Time: {df_locust['Average Response Time'].mean():.2f} ms\n")
                f.write(f"Max Response Time: {df_locust['Max Response Time'].max():.2f} ms\n")
                f.write(f"Average Requests/sec: {df_locust['Requests/s'].mean():.2f}\n")
                f.write(f"Max Requests/sec: {df_locust['Requests/s'].max():.2f}\n\n")
                
                # Per-endpoint breakdown
                f.write("=== Per-Endpoint Performance ===\n")
                for _, row in df_locust.iterrows():
                    if row['Name'] != 'Aggregated':
                        endpoint_success_rate = ((row['Request Count'] - row['Failure Count']) / row['Request Count'] * 100) if row['Request Count'] > 0 else 0
                        f.write(f"{row['Name']}: {endpoint_success_rate:.2f}% success, {row['Average Response Time']:.2f}ms avg\n")
            else:
                f.write("=== Load Test Metrics Summary ===\n")
                f.write("No Locust stats data available\n\n")
        
        print(f"Summary report saved to: {report_path}")


def run_service_test(service_name, user_counts, duration, output_dir, replica_counts):
    """Run tests for a service with different user counts and replica counts"""
    print(f"\n{'='*80}")
    print(f"Testing Service: {service_name}")
    print(f"{'='*80}")
    
    results = {}
    
    def _detect_namespace(name: str) -> str:
        # Try find namespace of the deployment via kubectl
        try:
            # Prefer deployment lookup
            cmd = ['kubectl', 'get', 'deploy', name, '-o', 'jsonpath={.metadata.namespace}']
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        # Fallback: default
        return 'default'

    def _scale_and_wait(name: str, replicas: int):
        ns = _detect_namespace(name)
        print(f"Scaling deployment/{name} to {replicas} in namespace {ns}...")
        # Scale
        subprocess.run(['kubectl', 'scale', f'deployment/{name}', f'--replicas={replicas}', '-n', ns],
                       check=False, capture_output=True, text=True)
        # Wait for rollout
        rollout = subprocess.run(['kubectl', 'rollout', 'status', f'deployment/{name}', '-n', ns, '--timeout=120s'],
                                 check=False, capture_output=True, text=True)
        if rollout.returncode != 0:
            print(f"Warning: Rollout wait failed: {rollout.stderr.strip()}")
        # Small settle delay
        time.sleep(5)

    for replica_count in replica_counts:
        print(f"\n--- Testing with {replica_count} replicas ---")
        # Ensure k8s deployment is scaled to desired replicas
        _scale_and_wait(service_name, replica_count)
        
        for user_count in user_counts:
            print(f"\n--- Testing with {user_count} users ---")
            
            # Create test instance
            test = IntegratedTestSystem(service_name, user_count, duration, output_dir, replica_count)
            
            # Run tests
            success = test.run_tests()
            
            # Create visualization and report
            if success:
                # Visualization disabled per user request
                test.generate_summary_report()
                print(f"✓ Test completed successfully for {user_count} users, {replica_count} replicas")
            else:
                print(f"✗ Test failed for {user_count} users, {replica_count} replicas")
            
            results[f"{replica_count}_{user_count}"] = success
            
            # Wait between tests
            if user_count != user_counts[-1] or replica_count != replica_counts[-1]:
                print("Waiting 10 seconds before next test...")
                time.sleep(10)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Integrated Test System with Monitoring and Visualization')
    parser.add_argument('--service', required=True, choices=list(SERVICES.keys()), 
                       help='Service to test')
    parser.add_argument('--users', type=int, nargs='+', default=[200, 400, 600, 800, 1000],
                       help='User counts to test')
    parser.add_argument('--replicas', type=int, nargs='+', default=[1, 2],
                       help='Replica counts to test')
    parser.add_argument('--duration', type=int, default=300,
                       help='Test duration in seconds')
    parser.add_argument('--output-dir', default='./individual_service_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Integrated Test System")
    print(f"Service: {args.service}")
    print(f"User counts: {args.users}")
    print(f"Replica counts: {args.replicas}")
    print(f"Duration: {args.duration} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # Run tests
    results = run_service_test(args.service, args.users, args.duration, args.output_dir, args.replicas)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Test Summary for {args.service}")
    print(f"{'='*80}")
    for test_key, success in results.items():
        replica, users = test_key.split('_')
        status = "PASS" if success else "FAIL"
        print(f"Replica {replica}, {users} users: {status}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
