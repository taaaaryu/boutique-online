import kopf
import kubernetes
import numpy as np
import random
import time
from itertools import combinations, chain

# ---------------------------
# ユーザー提供のアルゴリズムコード（グローバル定数はCRDから渡すので、ここではデフォルト値）
# ---------------------------
GENERATION = 10
NUM_NEXT = 10

def calc_software_av(services_group, service_avail, services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

def calc_software_av_matrix(services_in_sw, service_avail, server_avail):
    services_array = np.array(services_in_sw, dtype=int)
    #print("Services array:", services_array, "Service avail:", service_avail)   
    sw_avail_list = []
    count = 0
    for k in services_array:
        sw_avail = 1
        for i in range(k):
            sw_avail *= service_avail[count]
            count += 1
        sw_avail_list.append(sw_avail * server_avail)
    return sw_avail_list

def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    # 各グループの可用性の平均を評価指標として計算（簡易版）
    sum_matrix = np.sum(matrix, axis=1)
    software_availability = calc_software_av_matrix(sum_matrix, service_avail, server_avail)
    system_avail = np.prod(software_availability)
    matrix_resource = (r_add ** (sum_matrix - 1)) * sum_matrix * 1  # service_resource=1と仮定
    total_servers = np.sum(matrix_resource)
    return system_avail / total_servers if total_servers > 0 else 0

def make_matrix(service, software_count):
    # service: numpy array, software_count: int
    matrix = np.zeros((software_count, len(service) + 1), dtype=int)
    service_list = service.tolist()
    a = random.sample(service_list, software_count - 1)
    a.append(len(service) + 1)
    a.sort()
    idx = 0
    for i in range(software_count):
        for k in range(idx, a[i]):
            matrix[i][k] = 1
            idx += 1
    return matrix

def divide_sw(matrix, one_list):
    flag = 0
    cp_list = one_list.copy()
    while flag == 0:
        idx = random.randint(0, len(cp_list) - 2)
        start = cp_list[idx]
        end = cp_list[idx + 1]
        if end - start > 1:
            a = random.randint(start + 1, end - 1)
            div_matrix = np.insert(matrix, idx + 1, 0, axis=0)
            for i in range(a, cp_list[idx + 1]):
                div_matrix[idx][i] = 0
                div_matrix[idx + 1][i] = 1
            flag = 1
        else:
            continue
    return div_matrix

def integrate_sw(matrix, one_list):
    cp_list = one_list.copy()
    idx = random.randint(1, len(cp_list) - 2)
    start = cp_list[idx - 1]
    end = cp_list[idx + 1]
    for i in range(start, end):
        matrix[idx - 1][i] = 1
    new_matrix = np.delete(matrix, idx, 0)
    return new_matrix

def find_ones(matrix):
    arr = np.array(matrix)
    rows, cols = np.nonzero(arr)
    positions = [[col + 1 for col in cols[rows == row]] for row in np.unique(rows)]
    return positions

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H):
    best_RUEs = [-np.inf] * NUM_NEXT
    best_matrices = [None] * NUM_NEXT
    best_counts = [0] * NUM_NEXT

    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    for k in range(GENERATION):
        RUE_list = [best_RUE]
        matrix = best_matrix.copy()
        one_list = []
        col = 0
        for i in range(len(matrix[0])):
            if matrix[col][i] == 0:
                one_list.append(i)
                col += 1

        mini_RUE_list = [0]
        matrix_list = [[0]]
        for j in range(len(one_list)):
            a = one_list[j]
            one = matrix.copy()
            one[j][a - 1] = 0
            one[j][a] = 1

            one_new_RUE = calc_RUE(one, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(one_new_RUE)
            matrix_list.append(one)
            two = matrix.copy()
            two[j][a - 1] = 1
            two[j][a] = 0
            two_new_RUE = calc_RUE(two, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(two_new_RUE)
            matrix_list.append(two)

        new_RUE = max(mini_RUE_list)
        idx = mini_RUE_list.index(new_RUE)
        new_matrix = matrix_list[idx]
        RUE_list.append(new_RUE)

        one_list.append(len(matrix[0]))
        one_list.insert(0, 0)

        if software_count <= len(matrix[0]) - 1:
            new_sw_p_matrix = divide_sw(matrix, one_list)
            new_RUE_p = calc_RUE(new_sw_p_matrix, len(new_sw_p_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_p)
        else:
            new_RUE_p = 0

        if software_count >= 2:
            new_sw_n_matrix = integrate_sw(matrix, one_list)
            new_RUE_n = calc_RUE(new_sw_n_matrix, len(new_sw_n_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_n)
        else:
            new_RUE_n = 0

        max_RUE = max(RUE_list)

        if max_RUE > best_RUE:
            if max_RUE == new_RUE:
                best_RUE = new_RUE
                best_matrix = new_matrix
            elif max_RUE == new_RUE_p:
                best_RUE = max_RUE
                best_matrix = new_sw_p_matrix
                software_count += 1
            elif max_RUE == new_RUE_n:
                best_RUE = max_RUE
                best_matrix = new_sw_n_matrix
                software_count -= 1
        else:
            best_RUE = max_RUE

        if best_RUE > best_RUEs[0]:
            for i in range(NUM_NEXT - 1, 0, -1):
                best_RUEs[i] = best_RUEs[i - 1]
                best_matrices[i] = best_matrices[i - 1]
                best_counts[i] = best_counts[i - 1]
            best_RUEs[0] = best_RUE
            best_matrices[0] = best_matrix
            best_counts[0] = software_count
        else:
            for i in range(1, NUM_NEXT):
                if best_RUE > best_RUEs[i]:
                    for j in range(NUM_NEXT - 1, i, -1):
                        best_RUEs[j] = best_RUEs[j - 1]
                        best_matrices[j] = best_matrices[j - 1]
                        best_counts[j] = best_counts[j - 1]
                    best_RUEs[i] = best_RUE
                    best_matrices[i] = best_matrix
                    best_counts[i] = software_count
                    break
    return best_matrices, best_counts, best_RUEs, RUE_list

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, NUM_START):
    best_global_matrices = [None] * NUM_NEXT
    best_global_RUEs = [-np.inf] * NUM_NEXT
    best_global_counts = [0] * NUM_NEXT
    RUE_list = []
    x_gene = np.arange(1, GENERATION + 1)
    service = np.arange(1, num_service + 1)
    n = num_service  # n をサービス数とする

    software_count_float = np.random.normal(num_service / 2, 2, NUM_START)
    software_counts = np.clip(software_count_float.astype(int), 1, n)

    for software_count in software_counts:
        matrix = make_matrix(service, software_count)
        best_matrices, best_counts, best_RUEs_local, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H)
        RUE_list.append(RUE_each_list)
        for i in range(NUM_NEXT):
            if best_RUEs_local[i] > best_global_RUEs[i]:
                if best_matrices[i] is not None and (best_global_matrices[i] is None or not np.array_equal(best_matrices[i], best_global_matrices[i])):
                    best_global_matrices[i] = best_matrices[i]
                    best_global_counts[i] = best_counts[i]
                    best_global_RUEs[i] = best_RUEs_local[i]
    print("Best global RUEs:", best_global_RUEs)
    return best_global_matrices, best_global_counts, best_global_RUEs

# ---------------------------
# Greedy_Redundancyアルゴリズムの実装（シンプル版）
# ---------------------------
def greedy_redundancy(sw_avail, sw_resource, H, max_redundancy):
    num_sw = len(sw_avail)
    redundancy_list = [1] * num_sw
    sum_resource = np.sum(sw_resource)
    effective_avail = list(sw_avail)
    
    while sum_resource <= H:
        sorted_indices = np.argsort(effective_avail)
        updated = False
        for idx in sorted_indices:
            if redundancy_list[idx] >= max_redundancy:
                continue
            plus_resource = sw_resource[idx]
            if (sum_resource + plus_resource) <= H:
                redundancy_list[idx] += 1
                sum_resource += plus_resource
                effective_avail[idx] = 1 - (1 - sw_avail[idx]) ** redundancy_list[idx]
                updated = True
                break
        if not updated:
            break
    return redundancy_list

def parse_resource_limit(resource_limit_str, num_services):
    if resource_limit_str.endswith("n"):
        factor = int(resource_limit_str[:-1])
        return factor * num_services
    return int(resource_limit_str)

# ---------------------------
# Operator本体：CRD を監視して、最適化アルゴリズムを実行し、結果をCRD statusに反映
# ---------------------------
@kopf.on.create('myapp.example.com', 'v1alpha1', 'appconfigs')


# Kopf タイマー: 60秒ごとにCRDのspec.servicesのavailabilityをランダム更新
@kopf.timer('myapp.example.com', 'v1alpha1', 'appconfigs', interval=120.0)
def update_availability(spec, patch, logger, **kwargs):
    services_spec = spec.get('services', [])
    new_services = []
    for s in services_spec:
        s['availability'] = round(random.uniform(0.85, 0.9999), 4)
        new_services.append(s)
    patch.spec['services'] = new_services
    logger.info(f"Updated availabilities: {[s['availability'] for s in new_services]}")


@kopf.on.update('myapp.example.com', 'v1alpha1', 'appconfigs')
def optimize_appconfig(spec, meta, status, logger, **kwargs):
    namespace = meta.get('namespace', 'boutique')
    services_spec = spec.get('services', [])
    preferences = spec.get('preferences', {})

    # CRDから定数を読み取る
    r_adds = preferences.get('rAdds', [0.8, 1, 1.2])
    # ここでは任意のr_add値、例えばr_adds[1]を使用
    r_add = 1.05
    generation = int(preferences.get('generation', GENERATION))
    NUM_START = int(preferences.get('numStart', 50))
    NUM_NEXT_local = int(preferences.get('numNext', NUM_NEXT))
    average = int(preferences.get('average', 10))
    max_redundancy = int(preferences.get('maxReplicas', 3))  # CRDのmaxReplicasを冗長性上限として使用

    # 固定値（サーバ可用性、サービスリソース）
    server_avail = 0.95
    service_resource = 1

    # サービスは、CRD内のservicesのリスト全体を対象とする
    num_services = len(services_spec) - 1
    # ここでは H をサービス数*3 としているが、実際は parse_resource_limit を利用する場合もある
    H = num_services * 3
    
    # サービスの可用性配列
    service_avail = [s.get('availability', 0.99) for s in services_spec]
    # サービスIDリスト：1からnum_servicesまでの整数
    services = list(range(1, num_services + 1))

    # multi_start_greedy を実行して最適なグループ化結果を得る
    best_matrices, best_counts, best_RUEs = multi_start_greedy(r_add, service_avail, server_avail, H, num_services, NUM_START)
    best_solution = best_matrices[0]
    best_solution_list = best_solution.tolist() if isinstance(best_solution, np.ndarray) else best_solution
    best_software_count = int(best_counts[0])
    best_RUE = float(best_RUEs[0])
    
    # グループ化結果の各行を1グループとみなす
    groups = find_ones(best_solution)
    group_sizes = [sum(row) for row in best_solution]
    # 各グループの基礎可用性: 各グループのサービスの可用性の積 × server_avail
    group_avail = []
    size_start = 0
    size_end = 0
    for size in group_sizes:
        size_end += size
        prod = np.prod(service_avail[size_start:size_end])
        group_avail.append(prod * server_avail)
        size_start += size
    # 各グループのリソース要求は、サービス数×service_resource（簡易化）
    sw_resource = [size * service_resource for size in group_sizes]

    # Greedy_Redundancy を実行して、グループごとの最適な冗長化数（レプリカ数）を算出
    redundancy_list = greedy_redundancy(group_avail, sw_resource, H, max_redundancy)
    redundancy_list = [int(r) for r in redundancy_list]
    group_sizes = [int(size) for size in group_sizes]
    
    logger.info(f"Optimization result (grouping): best solution matrix: {best_solution_list}, software count: {best_software_count}, RUE: {best_RUE}")

    # Deployment更新用のグループ情報を作成
    final_group_details = []
    for idx, group in enumerate(groups):
        group_service_names = [services_spec[i]['name'] for i in [x - 1 for x in group]]
        redundancy_num = redundancy_list[idx] if idx < len(redundancy_list) else preferences.get('minReplicas', 1)
        final_group_details.append({
            "pod": f"group-{idx}",
            "services": group_service_names,
            "redundancy": redundancy_num
        })
    for i, detail in enumerate(final_group_details):
        logger.info(f"Group {i}: {detail}")
    # Deploymentの更新前に、既存の「group-」で始まるDeploymentを削除する


    # Deploymentの更新前に、既存の「group-」で始まるDeploymentを削除する
    api = kubernetes.client.AppsV1Api()
    try:
        existing_deployments = api.list_namespaced_deployment(namespace=namespace)
        for dep in existing_deployments.items:
            if dep.metadata.name.startswith("group-"):
                try:
                    api.delete_namespaced_deployment(name=dep.metadata.name, namespace=namespace)
                    logger.info(f"Deleted old deployment: {dep.metadata.name}")
                except Exception as e:
                    logger.error(f"Failed to delete deployment {dep.metadata.name}: {e}")
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")

    # Deploymentの更新：各グループごとにDeploymentを作成
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.config_exception.ConfigException:
        kubernetes.config.load_kube_config()

    api = kubernetes.client.AppsV1Api()
    core_api = kubernetes.client.CoreV1Api()

    for group in final_group_details:
        try:
            # 各サービスコンテナの定義
            containers = []
            for service_name in group['services']:
                # サービスごとにコンテナ定義を作成
                container = {
                    'name': service_name,
                    'image': f"gcr.io/google-samples/microservices-demo/{service_name}:v0.3.9",
                    'imagePullPolicy': 'IfNotPresent',  # 既存イメージを使用
                    'resources': {
                        'requests': {
                            'cpu': f"{next((s['requiredCPU'] for s in services_spec if s['name'] == service_name), '50m')}",
                            'memory': f"{next((s['requiredMemory'] for s in services_spec if s['name'] == service_name), '128Mi')}"
                        },
                        'limits': {
                            'cpu': f"{next((s['requiredCPU'] for s in services_spec if s['name'] == service_name), '50m')}",
                            'memory': f"{next((s['requiredMemory'] for s in services_spec if s['name'] == service_name), '128Mi')}"
                        }
                    },
                    'ports': [{'containerPort': 8080 + idx}],  # 各コンテナにユニークなポート番号
                    'env': [{
                        'name': 'PORT',
                        'value': str(8080 + idx)
                    }],
                    'readinessProbe': {
                        'httpGet': {
                            'path': '/health',
                            'port': 8080 + idx  # ポート番号を動的に設定
                        },
                        'initialDelaySeconds': 120,  # 初期遅延を延長
                        'periodSeconds': 30,  # チェック間隔を延長
                        'failureThreshold': 5,  # 失敗閾値を緩和
                        'timeoutSeconds': 10  # タイムアウトを設定
                    },
                    'livenessProbe': {
                        'httpGet': {
                            'path': '/health',
                            'port': 8080 + idx  # ポート番号を動的に設定
                        },
                        'initialDelaySeconds': 180,  # 初期遅延を延長
                        'periodSeconds': 30,  # チェック間隔を延長
                        'failureThreshold': 5,  # 失敗閾値を緩和
                        'timeoutSeconds': 10  # タイムアウトを設定
                    },
                    'startupProbe': {
                        'httpGet': {
                            'path': '/',
                            'port': 8080 + idx  # ポート番号を動的に設定
                        },
                        'failureThreshold': 30,  # 起動失敗閾値を大きく
                        'periodSeconds': 10  # チェック間隔
                    }
                }
                containers.append(container)

            # Initコンテナ定義 (依存関係解決用)
            init_containers = []
            if len(group['services']) > 1:
                init_containers.append({
                    'name': 'dependency-checker',
                    'image': 'busybox:1.35',
                    'command': ['sh', '-c', 
                        'for svc in $DEPENDENCIES; do port=$((${svc#*-} + 8080)); until nslookup $svc && curl -s http://$svc:$port/ | grep -q "OK"; do echo waiting for $svc; sleep 5; done; done'],
                    'env': [{
                        'name': 'DEPENDENCIES',
                        'value': ' '.join(group['services'])
                    }],
                    'resources': {
                        'requests': {'cpu': '10m', 'memory': '16Mi'},
                        'limits': {'cpu': '50m', 'memory': '32Mi'}
                    }
                })

            # サービスごとに個別のDeploymentを作成
            for service_name in group['services']:
                deployment = {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'metadata': {
                        'name': f"{group['pod']}-{service_name}",
                        'namespace': namespace,
                        'labels': {
                            'app': 'microservices-demo',
                            'service': service_name
                        }
                    },
                'spec': {
                    'replicas': group['redundancy'],
                    'selector': {
                        'matchLabels': {
                            'app': 'microservices-demo',
                            'group': group['pod']
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'microservices-demo',
                                'group': group['pod']
                            }
                        },
                        'spec': {
                            'initContainers': init_containers,
                            'containers': containers,
                            'restartPolicy': 'Always',
                            'terminationGracePeriodSeconds': 60,  # 延長
                            'dnsPolicy': 'ClusterFirst',
                            'securityContext': {
                                'runAsNonRoot': False,  # セキュリティ制限を緩和
                                'runAsUser': 0,  # rootユーザーで実行
                                'fsGroup': 0
                            },
                            'affinity': {
                                'podAntiAffinity': {
                                    'preferredDuringSchedulingIgnoredDuringExecution': [{
                                        'weight': 100,
                                        'podAffinityTerm': {
                                            'labelSelector': {
                                                'matchExpressions': [{
                                                    'key': 'app',
                                                    'operator': 'In',
                                                    'values': ['microservices-demo']
                                                }]
                                            },
                                            'topologyKey': 'kubernetes.io/hostname'
                                        }
                                    }]
                                }
                            }
                        }
                    },
                    'strategy': {
                        'type': 'RollingUpdate',
                        'rollingUpdate': {
                            'maxUnavailable': 0,  # 可用性向上
                            'maxSurge': 1
                        }
                    }
                }
            }

            # Deployment作成
            api.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            logger.info(f"Created deployment: {group['pod']}")

        except Exception as e:
            logger.error(f"Failed to create deployment {group['pod']}: {e}")
            raise kopf.TemporaryError(f"Deployment creation failed: {e}", delay=60)

    return {'message': 'Optimization completed', 'groups': final_group_details}



