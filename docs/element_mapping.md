# 要素間マッピング表

6 つの主要ブロックについて、機能・入出力・主要パラメータを一覧化しました。

| 要素 | 機能 (何をするか) | 主な入力 | 主な出力 | 主要パラメータ |
|------|------------------|----------|----------|----------------|
| **提案手法<br/>(Optimizer)** | サービス可用性・サーバ可用性を基に<br/>最適な実装形態 **c(I,k)** と冗長化度 **b(j,k)** を算出し Kubernetes に反映 | • サービス可用性 *a<sub>s</sub>*<br/>• サーバ可用性 *a<sub>sv</sub>*<br/>• 過去稼働 CSV | • *c(I,k)*, *b(j,k)*<br/>• Deployment replicas Patch<br/>• `optimize_flag=1` ログ行 | `REPLICA` : 初期レプリカ数<br/>`r_add` : 実装形態係数 (0.75/1.25…)<br/>`H` : 総リソース上限<br/>`NUM_START` : Greedy 開始点数<br/>`max_redundancy` : 冗長化の上限 |
| **設定ファイル** | すべてのモジュール共通の実行パラメータを集中管理し、<br/>Optimizer が生成したサービス実装形態・冗長度情報を保持して K8s へ反映 | 手動で記述した YAML / Python 定数<br/>• Optimizer 出力 *c(I,k)*, *b(j,k)* | • 更新済み ConfigMap / Helm values<br/>• Kubernetes へ渡す最終設定 | `kill_prob` : 障害発生確率<br/>`user_pattern` : ユーザー数シナリオ<br/>`algo_interval` : 最適化周期<br/>`log_interval` : ログ周期<br/>`service_list` : 対象サービス一覧<br/>`c(I,k)` & `b(j,k)` : 最新サービス実装/冗長度 |
| **障害注入モジュール<br/>(Injurer)** | 設定された確率・周期で Deployment を一時スケールダウンし擬似障害を発生 | • `kill_prob`, `kill_interval`<br/>• `service_groups` | • replicas −1 → 元に戻す Patch<br/>• `pause_counts` 更新 | `kill_interval` : 注入周期(sec)<br/>`duration` : 停止時間(sec)<br/>`max_kill_in_round` : 1 ラウンド最大停止数 |
| **アプリケーションシステム** | 9 サービスで構成されたオンラインショップ (microservices-demo) が HTTP リクエストを処理 | • 負荷テストからのリクエスト<br/>• replicas 更新 Patch | • HTTP レスポンス<br/>• Pod / Deployment ステータス | `CPU/Memory limits` : コンテナ資源上限<br/>`env vars` : DB 接続等環境変数 |
| **負荷テストソフト<br/>(Locust)** | ユーザー数を変化させながらリクエスト生成し、応答時間・スループットを計測 | • `user_pattern`, `spawn_rate`<br/>• Target URL | • `result_*_stats.csv`<br/>• `stats_history.csv` | `spawn_rate` : 1秒あたり生成数<br/>`run_time` : 総実行時間<br/>`think_time` : ユーザー待機時間 |
| **ログ収集モジュール** | 定期的に Pod 状態を取得し CSV に追記、可用性算出用履歴を提供 | • Kubernetes API からの状態情報<br/>• `optimize_flag` | • `pod_status-*.csv`（running / paused 数 等） | `log_interval` : 収集周期(sec)<br/>`csv_filename` : 出力 CSV 名<br/>`REPLICA` : 基準レプリカ数 | 