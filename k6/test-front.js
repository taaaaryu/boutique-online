import http from 'k6/http';
import { check, sleep } from 'k6';

// 仮想ユーザ数 (VUs) とテスト期間
export let options = {
  vus: Number(__ENV.VUS) || 1000,
  duration: __ENV.DURATION || '5m',
};

// Node の IP と NodePort。必要に応じて ENV で上書き
const ingressHost = __ENV.INGRESS_HOST || '172.18.0.2';
const ingressPort = __ENV.INGRESS_PORT || '32410';

// VirtualService で定義したホスト名（Host ヘッダーに指定）
const targetHost = __ENV.TARGET_HOST || 'frontend.example.com';

// 実際に叩く URL
const url = `http://${ingressHost}:${ingressPort}/`;

export default function () {
  // Istio のルールに合わせて Host ヘッダーを付与
  const headers = { Host: targetHost };

  // GET リクエスト
  let res = http.get(url, { headers });

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(1);
}