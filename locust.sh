#!/usr/bin/env bash
set +e

# 実行回数
N=10

# 設定ファイル名
#CONF=locust.conf

for i in $(seq 1 "${N}"); do
  echo "==== Run #$i ===="
  locust \
    --locustfile src/loadgenerator/locustfile.py \
    --headless \
    --expect-workers 0 \
    --host http://localhost:31036 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 30m \
    --csv locust_results/result_$i \
    --reset-stats
done