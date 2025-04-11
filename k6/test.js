import http from 'k6/http';
import { check, sleep } from 'k6';

// k6 のオプション設定：仮想ユーザ数 (VUs) とテスト期間
export let options = {
  vus: Number(__ENV.VUS) || 1000,
  duration: __ENV.DURATION || '5m',
};

// Ingress Gateway のホストとポート（ポートフォワーディングの場合は localhost と指定）
const ingressHost = __ENV.INGRESS_HOST || 'localhost';
// ポートは5080に変更
const ingressPort = __ENV.INGRESS_PORT || '80';

// VirtualService の hosts に合わせたターゲットホスト名
const targetHost = __ENV.TARGET_HOST || 'frontend.example.com';

// Ingress Gateway 経由の URL を作成（HTTP の場合）
const url = `http://${ingressHost}:${ingressPort}/`;

export default function () {
  // Istio のルーティングに合わせ、Host ヘッダーを指定

  const headers = { 'Host': targetHost };

  let res = http.get(url, { headers });
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
  sleep(1);
}


