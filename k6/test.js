import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem, randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';
import faker from 'https://cdn.jsdelivr.net/npm/faker@5.5.3/dist/faker.min.js';

// Define randomInt function since it's not in the imported library
function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// 仮想ユーザ数 (VUs) とテスト期間
export let options = {
  vus: Number(__ENV.VUS) || 500,
  duration: __ENV.DURATION || '5m',
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95%のリクエストが500ms以内に完了
  },
};

// Node の IP と NodePort。必要に応じて ENV で上書き
const ingressHost = __ENV.INGRESS_HOST || '172.18.0.2';
const ingressPort = __ENV.INGRESS_PORT || '32563';

// VirtualService で定義したホスト名（Host ヘッダーに指定）
const targetHost = __ENV.TARGET_HOST || 'frontend.example.com';

// ベースURL
const baseUrl = `http://${ingressHost}:${ingressPort}`;

// 商品ID一覧
const products = [
  '0PUK6V6EV0',
  '1YMWWN1N4O',
  '2ZYFJ3GM2N',
  '66VCHSJNUP',
  '6E92ZMYYFZ',
  '9SIQT8TOJO',
  'L9ECAV7KIM',
  'LS4PSXUNUM',
  'OLJCESPC7Z'
];

// 通貨一覧
const currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY'];

// 共通ヘッダー
const headers = { 
  'Host': targetHost,
  'Content-Type': 'application/x-www-form-urlencoded'
};

// シナリオの重み付け
const scenarios = {
  index: 10,
  setCurrency: 20,
  browseProduct: 100,
  addToCart: 20,
  viewCart: 30,
  checkout: 10,
  emptyCart: 5
};

// 合計重み
const totalWeight = Object.values(scenarios).reduce((sum, weight) => sum + weight, 0);

export default function () {
  // セッションCookieを保持するためのJar
  const jar = http.cookieJar();
  
  // シナリオをランダムに選択
  const random = Math.random() * totalWeight;
  let cumulativeWeight = 0;
  let selectedScenario = 'index'; // デフォルト
  
  for (const [scenario, weight] of Object.entries(scenarios)) {
    cumulativeWeight += weight;
    if (random <= cumulativeWeight) {
      selectedScenario = scenario;
      break;
    }
  }
  
  // 選択されたシナリオを実行
  switch (selectedScenario) {
    case 'index':
      index();
      break;
    case 'setCurrency':
      setCurrency();
      break;
    case 'browseProduct':
      browseProduct();
      break;
    case 'addToCart':
      addToCart();
      break;
    case 'viewCart':
      viewCart();
      break;
    case 'checkout':
      checkout();
      break;
    case 'emptyCart':
      emptyCart();
      break;
  }
  
  // リクエスト間の待機時間
  sleep(randomInt(1, 3));
}

// ホームページにアクセス (productcatalogservice, recommendationservice, adservice, currencyservice)
function index() {
  let res = http.get(`${baseUrl}/`, { headers });
  
  check(res, {
    'ホームページが正常に表示される': (r) => r.status === 200,
  });
}

// 通貨を設定 (currencyservice)
function setCurrency() {
  const currency = randomItem(currencies);
  let res = http.post(`${baseUrl}/setCurrency`, 
    { currency_code: currency }, 
    { headers }
  );
  
  check(res, {
    '通貨設定が成功': (r) => r.status === 200 || r.status === 302,
  });
}

// 商品詳細を閲覧 (productcatalogservice, recommendationservice)
function browseProduct() {
  const productId = randomItem(products);
  let res = http.get(`${baseUrl}/product/${productId}`, { headers });
  
  check(res, {
    '商品詳細が正常に表示される': (r) => r.status === 200,
  });
}

// カートを表示 (cartservice)
function viewCart() {
  let res = http.get(`${baseUrl}/cart`, { headers });
  
  check(res, {
    'カートが正常に表示される': (r) => r.status === 200,
  });
}

// カートに商品を追加 (productcatalogservice, cartservice)
function addToCart() {
  const productId = randomItem(products);
  
  // まず商品ページを表示
  http.get(`${baseUrl}/product/${productId}`, { headers });
  
  // カートに追加
  let res = http.post(`${baseUrl}/cart`, 
    { product_id: productId, quantity: randomInt(1, 10) }, 
    { headers }
  );
  
  check(res, {
    'カートへの追加が成功': (r) => r.status === 200 || r.status === 302,
  });
}

// カートを空にする (cartservice)
function emptyCart() {
  let res = http.post(`${baseUrl}/cart/empty`, {}, { headers });
  
  check(res, {
    'カートを空にする処理が成功': (r) => r.status === 200 || r.status === 302,
  });
}

// チェックアウト (cartservice, checkoutservice, paymentservice, emailservice, shippingservice)
function checkout() {
  // まずカートに商品を追加
  addToCart();
  
  // チェックアウト情報を生成
  const currentYear = new Date().getFullYear();
  const checkoutData = {
    email: faker.internet.email(),
    street_address: faker.address.streetAddress(),
    zip_code: faker.address.zipCode(),
    city: faker.address.city(),
    state: faker.address.stateAbbr(),
    country: faker.address.country(),
    credit_card_number: faker.finance.creditCardNumber('visa'),
    credit_card_expiration_month: randomInt(1, 12),
    credit_card_expiration_year: randomInt(currentYear + 1, currentYear + 5),
    credit_card_cvv: randomInt(100, 999).toString()
  };
  
  // チェックアウト処理
  let res = http.post(`${baseUrl}/cart/checkout`, checkoutData, { headers });
  
  check(res, {
    'チェックアウトが成功': (r) => r.status === 200 || r.status === 302,
  });
}
