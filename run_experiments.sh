#!/bin/bash

# 設定値
K=3  # 各pause_intervalでの測定回数
PAUSE_INTERVALS=(40 60 80)  # テストするpause_intervalの値
OUTPUT_DIR="experiment_results"
K6_OUTPUT_DIR="${OUTPUT_DIR}/k6_results"
CSV_OUTPUT_DIR="${OUTPUT_DIR}/csv_results"

# ディレクトリの作成
mkdir -p "${K6_OUTPUT_DIR}"
mkdir -p "${CSV_OUTPUT_DIR}"

# 結果を保存する配列
declare -a results

# 各pause_intervalに対して実験を実行
for pause_interval in "${PAUSE_INTERVALS[@]}"; do
    echo "Starting experiments for pause_interval=${pause_interval}"
    
    # pause_intervalの値をreplicaset.pyに反映
    sed -i.bak "s/pause_interval = [0-9.]*/pause_interval = ${pause_interval}/" k8s-operator/my-operator/replicaset.py
    
    # k回の測定を実行
    for ((i=1; i<=K; i++)); do
        echo "Running experiment ${i}/${K} for pause_interval=${pause_interval}"
        
        # kopfをバックグラウンドで起動
        kopf run k8s-operator/my-operator/replicaset.py --namespace boutique &
        KOPF_PID=$!
        
        # kopfが起動するまで少し待機
        sleep 10
        
        # k6テストを実行し、結果を保存
        k6 run k6/test.js > "${K6_OUTPUT_DIR}/k6_result_${pause_interval}_${i}.txt"
        
        # kopfを停止（Ctrl+Cを2回送信）
        kill -INT $KOPF_PID
        sleep 2
        kill -INT $KOPF_PID
        
        # kopfが完全に停止するまで待機
        wait $KOPF_PID
        
        # CSVファイルを保存
        latest_csv=$(ls -t pod_status-${pause_interval}-*.csv | head -n1)
        if [ -n "$latest_csv" ]; then
            cp "$latest_csv" "${CSV_OUTPUT_DIR}/csv_result_${pause_interval}_${i}.csv"
        fi
        
        # 次の実験の前に少し待機
        sleep 5
    done
    
    # このpause_intervalでの結果を集計
    echo "Results for pause_interval=${pause_interval}:"
    for ((i=1; i<=K; i++)); do
        result_file="${K6_OUTPUT_DIR}/k6_result_${pause_interval}_${i}.txt"
        if [ -f "$result_file" ]; then
            echo "Run ${i}:"
            grep "http_req_duration" "$result_file"
        fi
    done
    echo "----------------------------------------"
done

# 最終的な結果の表示
echo "Experiment completed. Results are saved in ${OUTPUT_DIR}"
echo "Please check the following directories for detailed results:"
echo "- K6 results: ${K6_OUTPUT_DIR}"
echo "- CSV results: ${CSV_OUTPUT_DIR}" 