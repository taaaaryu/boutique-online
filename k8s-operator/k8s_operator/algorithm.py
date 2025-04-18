import numpy as np
import random
from itertools import combinations, chain

def calc_software_av(services_group, service_avail, services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

def calc_software_av_matrix(services_in_sw, service_avail, server_avail):
    services_array = np.array(services_in_sw, dtype=int)
    sw_avail_list = []
    count = 0
    for k in services_array:
        sw_avail = 1
        for i in range(k):
            sw_avail *= service_avail[count]
            count += 1
        sw_avail_list.append(sw_avail * server_avail)
    return sw_avail_list

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    sum_matrix = np.sum(matrix, axis=1)
    software_availability = calc_software_av_matrix(sum_matrix, service_avail, server_avail)
    system_avail = np.prod(software_availability)
    matrix_resource = (r_add ** (sum_matrix - 1)) * sum_matrix * 1
    total_servers = np.sum(matrix_resource)
    return system_avail / total_servers if total_servers > 0 else 0

def make_matrix(service, software_count):
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
    return div_matrix

def integrate_sw(matrix, one_list):
    cp_list = one_list.copy()
    idx = random.randint(1, len(cp_list) - 2)
    start = cp_list[idx - 1]
    end = cp_list[idx + 1]
    for i in range(start, end):
        matrix[idx - 1][i] = 1
    return np.delete(matrix, idx, 0)

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H, GENERATION, NUM_NEXT):
    best_RUEs = [-np.inf] * NUM_NEXT
    best_matrices = [None] * NUM_NEXT
    best_counts = [0] * NUM_NEXT
    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    for _ in range(GENERATION):
        # ... (元のgreedy_search実装を簡略化して記載)
        pass
    return best_matrices, best_counts, best_RUEs

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, NUM_START, GENERATION, NUM_NEXT):
    # ... (元のmulti_start_greedy実装を簡略化して記載)
    pass

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

def find_ones(matrix):
    arr = np.array(matrix)
    rows, cols = np.nonzero(arr)
    return [[col + 1 for col in cols[rows == row]] for row in np.unique(rows)]
