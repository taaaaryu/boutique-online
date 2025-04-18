import kopf
import kubernetes
import numpy as np
import random
import time

# Greedy_Redundancyアルゴリズムの実装例（シンプル版）
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

# CRD の作成・更新時に呼ばれるハンドラ
@kopf.on.create('myapp.example.com', 'v1alpha1', 'appconfigs')
@kopf.on.update('myapp.example.com', 'v1alpha1', 'appconfigs')
def reconcile_appconfig(spec, meta, status, logger, **kwargs):
    namespace = meta.get('namespace', 'default')
    services = spec.get('services', [])
    preferences = spec.get('preferences', {})

    min_replicas = int(preferences.get('minReplicas', 1))
    max_replicas = int(preferences.get('maxReplicas', 4))
    resource_limit_str = preferences.get('resourceLimit', "2n")
    num_services = len(services)
    H = parse_resource_limit(resource_limit_str, num_services)

    # sw_resource を各サービスについて1と仮定
    sw_resource = [1] * num_services
    # 各サービスのavailabilityを取得
    sw_avail = [s.get('availability') for s in services]

    redundancy_list = greedy_redundancy(sw_avail, sw_resource, H, max_replicas)
    logger.info(f"Computed redundancy: {redundancy_list}")

    # Kubernetes API を利用して各サービスに対応する Deployment のレプリカ数を更新
    api = kubernetes.client.AppsV1Api()
    for idx, service in enumerate(services):
        service_name = service.get('name')
        desired_replicas = redundancy_list[idx]
        patch_body = {"spec": {"replicas": desired_replicas}}
        try:
            api.patch_namespaced_deployment(
                name=service_name,
                namespace=namespace,
                body=patch_body
            )
            logger.info(f"Patched {service_name} to replicas {desired_replicas}")
        except Exception as e:
            logger.error(f"Failed to patch {service_name}: {e}")

    return {"redundancy": redundancy_list}

# Kopf タイマーを利用して30秒ごとにavailabilityをランダム更新
@kopf.timer('myapp.example.com', 'v1alpha1', 'appconfigs', interval=30.0)
def randomize_availability(spec, patch, logger, **kwargs):
    services = spec.get('services', [])
    new_services = []
    for s in services:
        s['availability'] = round(random.uniform(0.8, 0.999), 3)
        new_services.append(s)
    patch.spec['services'] = new_services
    logger.info(f"Randomized availabilities: {[s['availability'] for s in new_services]}")
