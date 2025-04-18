import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
import random
import csv

# パラメータ
r_adds = [0.8, 1, 1.2]  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数
num_service = [i for i in range(6,14)]  # サービス数
#num_service = [20, 40, 60, 80, 100]
server_avail = 0.99  # サーバの可用性 AWSEC2
service_resource = 1  # サービスリソース
GENERATION = 10
NUM_START = 50
NUM_NEXT = 10
average = 10
max_redundancy = 4

# ソフトウェアの可用性を計算する関数
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

def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H):
    best_RUEs = [-np.inf] * NUM_NEXT
    best_matrices = [None] * NUM_NEXT
    best_counts = [0] * NUM_NEXT

    list = []
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

            two = matrix.copy()
            two[j][a - 1] = 1
            two[j][a] = 0

            one_new_RUE = calc_RUE(one, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(one_new_RUE)
            matrix_list.append(one)
            two_new_RUE = calc_RUE(two, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(two_new_RUE)
            matrix_list.append(two)

        new_RUE = max(mini_RUE_list)
        idx = mini_RUE_list.index(new_RUE)
        new_matrix = matrix_list[idx]
        RUE_list.append(new_RUE)

        one_list.append(len(matrix[0]))
        one_list.insert(0, 0)

        if software_count <= n - 1:
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
        list.append(best_RUE)

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

    return best_matrices, best_counts, best_RUEs, list

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    avg_efficiency = []
    initial_redundancy = np.ones(software_count)
    sum_matrix = np.sum(matrix, axis=1)
    software_availability = calc_software_av_matrix(sum_matrix, service_avail, server_avail)
    sw_list = np.array(software_availability)
    system_avail = np.prod(sw_list)
    matrix_resource = (r_add ** (sum_matrix - 1)) * sum_matrix*service_resource
    total_servers = np.dot(initial_redundancy, matrix_resource)

    for i in range(software_count):
        initial_redundancy[i] += 1
        total_servers_red = np.dot(initial_redundancy, matrix_resource)
        total_servers_mask = total_servers_red <= H
        system_avail_red = np.prod([1 - ((1 - sa) ** int(r)) for sa, r in zip(software_availability, initial_redundancy)])
        redundancy_cost_efficiency = np.where(total_servers_mask, (system_avail_red - system_avail) / (total_servers_red - total_servers), 0)
        avg_efficiency.append(redundancy_cost_efficiency)
        initial_redundancy[i] = 1
    return np.mean(avg_efficiency)

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, NUM_START):
    best_global_matrices = [None] * NUM_NEXT
    best_global_RUEs = [-np.inf] * NUM_NEXT
    best_global_counts = [0] * NUM_NEXT
    RUE_list = []
    x_gene = np.arange(1, GENERATION + 1)
    service = np.arange(1, num_service)
    software_count_float = np.random.normal(num_service / 2, 2, NUM_START)
    software_counts = np.clip(software_count_float.astype(int), 1, n)

    for software_count in software_counts:
        matrix = make_matrix(service, software_count)
        best_matrices, best_counts, best_RUEs, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H)

        RUE_list.append(RUE_each_list)

        for i in range(NUM_NEXT):
            if best_RUEs[i] > best_global_RUEs[i]:
                if all(not np.array_equal(best_matrices[i], bm) for bm in best_global_matrices):
                    best_global_matrices[i] = best_matrices[i]
                    best_global_counts[i] = best_counts[i]
                    best_global_RUEs[i] = best_RUEs[i]

    return best_global_matrices, best_global_counts, best_global_RUEs

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

def Greedy_Redundancy(sw_avail, sw_resource):
    num_sw = len(sw_avail)
    redundancy_list = [1] * num_sw
    sum_Resource = np.sum(sw_resource)
    sw_avail_list = sw_avail

    calc = 0

    while sum_Resource <= H:
        sw_avail_sort, sw_resource, redundancy, sw_avail = zip(*sorted(zip(sw_avail_list, sw_resource, redundancy_list, sw_avail)))
        redundancy_list = list(redundancy)
        flag = 0
        i = 0
        for i in range(num_sw):
            if redundancy_list[i] >= max_redundancy:
                continue
            plus_resource = sw_resource[i]
            if (sum_Resource + plus_resource) <= H:
                redundancy_list[i] += 1
                sum_Resource += plus_resource
                sw_avail_list[i] = 1 - (1 - sw_avail[i]) ** int(redundancy_list[i])
                calc += 1
                flag += 1
                break
        if flag == 0:
            break

    system_av = np.prod([1 - ((1 - sa) ** int(r)) for sa, r in zip(sw_avail, redundancy_list)])
    return redundancy_list, sum_Resource, system_av, calc

# Prepare a list to store results for CSV output
csv_data = []

for r_add in r_adds:
    for n in num_service:
        softwares = [i for i in range(1, n+1)]
        services = [i for i in range(1, n + 1)]
        service_avail = [0.99]*n
        Resource = [service_resource*n*2]  # サーバリソース
        unav_list = []
        time_list = []

        
        for H in Resource:
            time_mean = []
            unav_mean = []
            calc_mean = [] #冗長化アルゴリズムの計算回数の平均

            for i in range(average):
                start = time.time()
                best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, len(services), NUM_START)

                min_unav = []
                best_combinations = []

                for p in best_matrix:
                    if p is not None:
                        best_combinations.append(find_ones(p))

                result_resource = []
                result_redundancy = []
                result_availabililty = []
                result_calc = []

                for comb in best_combinations:
                    software_availability = [calc_software_av(group, service_avail, services) * server_avail for group in comb]
                    sw_resource = np.array([service_resource*len(group) * (r_add ** (len(group) - 1)) for group in comb])
                    best_redundancy, best_resource, system_av, num_calc = Greedy_Redundancy(software_availability, sw_resource)

                    result_redundancy.append(best_redundancy)
                    result_resource.append(best_resource)
                    result_availabililty.append(system_av)
                    result_calc.append(num_calc)

                end = time.time()
                time_diff = end - start

                time_mean.append(time_diff)
                calc_mean.append(sum(result_calc))

                max_idx = result_availabililty.index(max(result_availabililty))
                unav_mean.append(1 - max(result_availabililty))
                print(r_add,time_diff,1 - max(result_availabililty))

            # Calculate max, min, and average for time and unav
            time_avg = sum(time_mean) / len(time_mean)
            time_max = max(time_mean)
            time_min = min(time_mean)
            unav_avg = sum(unav_mean) / len(unav_mean)
            unav_max = max(unav_mean)
            unav_min = min(unav_mean)

            # Append results to csv_data
            csv_data.append([r_add, n, time_avg, time_max, time_min, unav_avg, unav_max, unav_min])

#
# Write results to a CSV file
with open('results_提案手法.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['r_add', 'num_service', 'time_avg', 'time_max', 'time_min', 'unav_avg', 'unav_max', 'unav_min'])
    # Write the data
    csvwriter.writerows(csv_data)