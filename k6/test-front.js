import http from 'k6/http';
import { check, sleep } from 'k6';

// 負荷のシナリオ設定例（VU数と時間を調整できます）
export let options = {
  stages: [
    { duration: '30s', target: 50 }, // 30秒間で50VUに増加
    { duration: '1m', target: 50 },  // 1分間50VUで負荷維持
    { duration: '30s', target: 0 }   // 30秒間で0VUに減少
  ],
};

export default function () {
  // クラスター内で k6 を実行する場合は、サービス名を利用してリクエスト可能
  // ※ 外部からアクセスする場合は、ノードIPとNodePort(31083)を利用するか、外部IPが割り当てられているか確認してください。
  let res = http.get('http://frontend-external.boutique.svc.cluster.local/');
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
  sleep(1); // 次のリクエストまで1秒待機（調整可）
}

