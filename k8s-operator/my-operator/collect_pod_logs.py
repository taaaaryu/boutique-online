import os
import csv
import re
from datetime import datetime
from kubernetes import client, config


LOG_DIR = os.environ.get("LOG_DIR", "pod_logs")
SERVICE_DIR = os.path.join(LOG_DIR, "services")

ARCH_TYPE = os.environ.get("ARCH_TYPE", "unknown")
RUN_NUM = os.environ.get("RUN_NUM", "0")
NAMESPACE = os.environ.get("NAMESPACE", "default")

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
        
        for line in log_text.splitlines():
            # レスポンス時間を先にチェック
            time_match = re.search(r'" \d{3} - [^-]+ - "[^"]*" \d+ \d+ (\d+) (\d+)', line)
            is_timeout = False
            
            if time_match:
                response_time = int(time_match.group(1))  # レスポンス時間（ミリ秒）
                #print(f"response_time: {response_time}")
                # 3秒（3000ミリ秒）を超えた場合をタイムアウトとしてカウント
                if response_time > 3000:
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


if __name__ == "__main__":
    main()