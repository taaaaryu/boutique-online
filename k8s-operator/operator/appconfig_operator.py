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
@kopf.on.update('myapp.example.com', 'v1alpha1', 'appconfigs')
def optimize_appconfig(spec, meta, status, logger, **kwargs):
    namespace = meta.get('namespace', 'boutique')
    services_spec = spec.get('services', [])
    preferences = spec.get('preferences', {})

    # CRDから定数を読み取る
    r_adds = preferences.get('rAdds', [0.8, 1, 1.2])
    # ここでは任意のr_add値、例えばr_adds[1]を使用
    r_add = 1.1
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

    # DeploymentとServiceの更新：各グループごとにリソースを作成またはパッチ
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.config_exception.ConfigException:
        kubernetes.config.load_kube_config()

    # サービス名とポートのマッピング
    service_ports = {
        "adservice": 9555,
        "cartservice": 7070,
        "emailservice": 8080,
        "productcatalogservice": 3550,
        "shippingservice": 50051
    }

    for idx, group in enumerate(groups):
        deployment_name = f"group-{idx}"
        group_service_names = [services_spec[i]['name'] for i in [x - 1 for x in group]]
        containers = []
        for svc in group_service_names:
            containers.append({
                "name": svc,
                "image": f"us-central1-docker.pkg.dev/google-samples/microservices-demo/{svc}:v0.10.2",
                "resources": {
                    "requests": {
                        "cpu": next(s['requiredCPU'] for s in services_spec if s['name'] == svc),
                        "memory": next(s['requiredMemory'] for s in services_spec if s['name'] == svc),
                    }
                }
            })
        
        
        # Serviceの作成
        service_name = f"group-{idx}"
        service_ports_list = []
        for svc in group_service_names:
            if svc in service_ports:
                service_ports_list.append({
                    "name": svc,
                    "port": service_ports[svc],
                    "targetPort": service_ports[svc]
                })

        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "namespace": namespace,
                "labels": {
                    "group": service_name,
                    "app": f"group-{idx}",
                }
            },
            "spec": {
                "type": "ClusterIP",
                "selector": {"group": service_name},
                "ports": service_ports_list
            }
        }

        # Serviceの作成/更新
        core_api = kubernetes.client.CoreV1Api()
        try:
            core_api.patch_namespaced_service(service_name, namespace, service_manifest)
            logger.info(f"Patched service {service_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                core_api.create_namespaced_service(namespace, service_manifest)
                logger.info(f"Created service {service_name}")
            else:
                logger.error(f"Failed to update service {service_name}: {e}")

        # Deploymentの作成
        # Group Podにはistio-injectionを無効化、それ以外は有効化
        labels = {
            "group": deployment_name, 
            "app": f"group-{idx}",
            "sidecar-injection": "enabled"  # Group Podもsidecar有効化
        }
        
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": deployment_name, "namespace": namespace},
            "spec": {
                "replicas": redundancy_list[idx] if idx < len(redundancy_list) else preferences.get('minReplicas', 1),
                "selector": {"matchLabels": {"group": deployment_name, "app": f"group-{idx}"}},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "shareProcessNamespace": True,  # プロセス名前空間の共有を有効化
                        "containers": containers
                    },
                }
            }
        }
        try:
            api.patch_namespaced_deployment(deployment_name, namespace, deployment_manifest)
            logger.info(f"Patched deployment {deployment_name} to replicas {redundancy_list[idx]}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                api.create_namespaced_deployment(namespace, deployment_manifest)
                logger.info(f"Created deployment {deployment_name} with replicas {redundancy_list[idx]}")
            else:
                logger.error(f"Failed to update deployment {deployment_name}: {e}")

    # Istioリソースの作成/更新
    networking_api = kubernetes.client.CustomObjectsApi()
    
    # DestinationRuleの作成/更新
    for idx, group in enumerate(groups):
        service_name = f"group-{idx}"
        dr_body = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": service_name,
                "namespace": namespace
            },
            "spec": {
                "host": f"{service_name}.{namespace}.svc.cluster.local",
                "subsets": [
                    {
                        "name": "grouped",
                        "labels": {"app": service_name}
                    },
                    {
                        "name": "original",
                        "labels": {"app": service_name.replace("group-", "")}
                    }
                ]
            }
        }
        
        try:
            networking_api.patch_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                plural="destinationrules",
                name=service_name,
                namespace=namespace,
                body=dr_body
            )
        except kubernetes.client.rest.ApiException:
            networking_api.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                plural="destinationrules",
                namespace=namespace,
                body=dr_body
            )

    # VirtualServiceの作成/更新
    vs_body = {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "VirtualService",
        "metadata": {
            "name": "group-routing",
            "namespace": namespace
        },
        "spec": {
            "hosts": ["*"],
            "gateways": ["boutique/frontend-gateway"],
            "http": []
        }
    }

    # 各サービスごとのルーティングルールを追加（冗長化数に比例したweight設定）
    for idx, group in enumerate(groups):
        original_service = group[0]
        grouped_replicas = redundancy_list[idx] if idx < len(redundancy_list) else 1
        total_replicas = grouped_replicas+1
        weight_per_pod = int(100 / total_replicas) if total_replicas > 0 else 0
        
        routes = []
        for i in range(total_replicas):
            routes.append({
                "destination": {
                    "host": f"{original_service}.{namespace}.svc.cluster.local",
                    "subset": "grouped"
                },
                "weight": weight_per_pod
            })
        
        vs_body["spec"]["http"].append({
            "match": [{"uri": {"prefix": f"/{original_service}"}}],
            "route": routes
        })
        logger.info(routes)

    try:
        networking_api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            plural="virtualservices",
            name="group-routing",
            namespace=namespace,
            body=vs_body
        )
    except kubernetes.client.rest.ApiException:
        networking_api.create_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            plural="virtualservices",
            namespace=namespace,
            body=vs_body
        )

    result = {
        "optimizationResult": {
            "bestSolutionMatrix": best_solution_list,
            "softwareCount": int(best_software_count),
            "bestRUE": float(best_RUE),
            "redundancy": [int(x) for x in redundancy_list],
            "groups": [list(map(int, g)) for g in groups]
        }
    }
    return result

# Kopf タイマー: 60秒ごとにCRDのspec.servicesのavailabilityをランダム更新
@kopf.timer('myapp.example.com', 'v1alpha1', 'appconfigs', interval=300.0)
def update_availability(spec, patch, logger, **kwargs):
    services_spec = spec.get('services', [])
    new_services = []
    for s in services_spec:
        s['availability'] = round(random.uniform(0.85, 0.9999), 4)
        new_services.append(s)
    patch.spec['services'] = new_services
    logger.info(f"Updated availabilities: {[s['availability'] for s in new_services]}")
