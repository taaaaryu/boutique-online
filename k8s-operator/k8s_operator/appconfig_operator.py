import kopf
import kubernetes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Operator initialization started")
import numpy as np
from k8s_operator.algorithm import (
    calc_software_av,
    calc_software_av_matrix,
    calc_RUE,
    make_matrix,
    divide_sw,
    integrate_sw,
    greedy_search,
    multi_start_greedy,
    greedy_redundancy,
    find_ones
)

# アルゴリズムパラメータ
GENERATION = 10
NUM_NEXT = 10

def parse_resource_limit(resource_limit_str, num_services):
    if resource_limit_str.endswith("n"):
        factor = int(resource_limit_str[:-1])
        return factor * num_services
    return int(resource_limit_str)

@kopf.on.create('myapp.example.com', 'v1alpha1', 'appconfigs')
@kopf.on.update('myapp.example.com', 'v1alpha1', 'appconfigs')
def optimize_appconfig(spec, meta, status, logger, **kwargs):
    logger.info(f"AppConfig event triggered: {meta.get('name')}")
    logger.info(f"Operation type: {kwargs.get('operation')}")
    namespace = meta.get('namespace', 'boutique')
    services_spec = spec.get('services', [])
    preferences = spec.get('preferences', {})

    r_adds = preferences.get('rAdds', [0.8, 1, 1.2])
    r_add = 1.1
    generation = int(preferences.get('generation', GENERATION))
    NUM_START = int(preferences.get('numStart', 50))
    NUM_NEXT_local = int(preferences.get('numNext', NUM_NEXT))
    average = int(preferences.get('average', 10))
    max_redundancy = int(preferences.get('maxReplicas', 3))

    server_avail = 0.95
    service_resource = 1

    num_services = len(services_spec)
    H = int(preferences.get('H', num_services * 3))
    
    service_avail = [s.get('availability', 0.99) for s in services_spec]
    services = list(range(1, num_services + 1))

    best_matrices, best_counts, best_RUEs = multi_start_greedy(
        r_add, service_avail, server_avail, H, num_services, NUM_START, GENERATION, NUM_NEXT
    )
    best_solution = best_matrices[0]
    best_solution_list = best_solution.tolist() if isinstance(best_solution, np.ndarray) else best_solution
    best_software_count = int(best_counts[0])
    best_RUE = float(best_RUEs[0])
    
    groups = find_ones(best_solution)
    group_sizes = [sum(row) for row in best_solution]
    group_avail = []
    size_start = 0
    size_end = 0
    for size in group_sizes:
        size_end += size
        prod = np.prod(service_avail[size_start:size_end])
        group_avail.append(prod * server_avail)
        size_start += size
    sw_resource = [size * service_resource for size in group_sizes]

    redundancy_list = greedy_redundancy(group_avail, sw_resource, H, max_redundancy)
    redundancy_list = [int(r) for r in redundancy_list]
    group_sizes = [int(size) for size in group_sizes]
    
    logger.info(f"Optimization result (grouping): best solution matrix: {best_solution_list}, software count: {best_software_count}, RUE: {best_RUE}")

    api = kubernetes.client.AppsV1Api()
    try:
        existing_deployments = api.list_namespaced_deployment(namespace=namespace)
        print(existing_deployments)
        for dep in existing_deployments.items:
            if dep.metadata.name.startswith("group-"):
                try:
                    api.delete_namespaced_deployment(name=dep.metadata.name, namespace=namespace)
                    logger.info(f"Deleted old deployment: {dep.metadata.name}")
                except Exception as e:
                    logger.error(f"Failed to delete deployment {dep.metadata.name}: {e}")
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")

    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.config_exception.ConfigException:
        kubernetes.config.load_kube_config()

    # サービスとグループのマッピングを作成
    service_to_replicas = {}
    for idx, group in enumerate(groups):
        for service_idx in [x - 1 for x in group]:
            service_name = services_spec[service_idx]['name']
            service_to_replicas[service_name] = redundancy_list[idx] if idx < len(redundancy_list) else preferences.get('minReplicas', 1)

    # テスト用 - 固定で3レプリカに設定
    for service in services_spec:
        deployment_name = service['name']
        replicas = 3  # テスト用固定値
        
        logger.info(f"TEST MODE DEBUG: Creating deployment for {deployment_name} with {replicas} replicas")
        logger.info(f"Container details: {containers}")
        logger.info(f"Full deployment spec: {deployment}")
        
        # メインコンテナの定義
        containers = [{
            "name": deployment_name,
            "image": f"us-central1-docker.pkg.dev/google-samples/microservices-demo/{deployment_name}:v0.10.2",
            "resources": {
                "requests": {
                    "cpu": service['requiredCPU'],
                    "memory": service['requiredMemory']
                }
            },
            "readinessProbe": {
                "httpGet": {
                    "path": "/health",
                    "port": service.get('port', 8080)
                },
                "initialDelaySeconds": 5,
                "periodSeconds": 5
            },
            "livenessProbe": {
                "httpGet": {
                    "path": "/health",
                    "port": service.get('port', 8080)
                },
                "initialDelaySeconds": 15,
                "periodSeconds": 20
            }
        }]

        # デプロイメント定義
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "labels": {
                    "app": "microservices-demo",
                    "service": deployment_name
                }
            },
            "spec": {
                "replicas": service_to_replicas[deployment_name],
                "selector": {
                    "matchLabels": {
                        "app": "microservices-demo",
                        "service": deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "microservices-demo",
                            "service": deployment_name
                        }
                    },
                    "spec": {
                        "containers": containers,
                        "terminationGracePeriodSeconds": 5
                    }
                }
            }
        }

        try:
            api = kubernetes.client.AppsV1Api()
            # 既存Deploymentを取得しようと試みる
            try:
                api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
                # 存在する場合は更新
                api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                logger.info(f"Updated deployment: {deployment_name}")
            except kubernetes.client.exceptions.ApiException as e:
                if e.status == 404:
                    # 存在しない場合は作成
                    api.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment
                    )
                    logger.info(f"Created deployment: {deployment_name}")
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to apply deployment {deployment_name}: {e}")

    return {"message": "Optimization and deployment completed"}
