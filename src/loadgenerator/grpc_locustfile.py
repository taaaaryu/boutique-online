#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
import grpc
import os
import csv
import datetime
from locust import User, task, between, events
from faker import Faker

# gRPC imports
import sys
sys.path.append('src/paymentservice/proto')
sys.path.append('src/cartservice/src/protos')

# Import generated gRPC stubs
try:
    import demo_pb2
    import demo_pb2_grpc
    import Cart_pb2
    import Cart_pb2_grpc
except ImportError as e:
    print(f"Warning: gRPC stubs not found: {e}")
    print("Run: ./generate_grpc.sh")

fake = Faker()

# Service configurations
SERVICES = {
    'productcatalogservice': {
        'host': 'productcatalogservice',
        'port': 3550,
        'stub_class': demo_pb2_grpc.ProductCatalogServiceStub
    },
    'cartservice': {
        'host': 'cartservice', 
        'port': 7070,
        'stub_class': Cart_pb2_grpc.CartServiceStub
    },
    'checkoutservice': {
        'host': 'checkoutservice',
        'port': 5050,
        'stub_class': demo_pb2_grpc.CheckoutServiceStub
    },
    'paymentservice': {
        'host': 'paymentservice',
        'port': 50051,
        'stub_class': demo_pb2_grpc.PaymentServiceStub
    },
    'shippingservice': {
        'host': 'shippingservice',
        'port': 50051,
        'stub_class': demo_pb2_grpc.ShippingServiceStub
    },
    'currencyservice': {
        'host': 'currencyservice',
        'port': 7000,
        'stub_class': demo_pb2_grpc.CurrencyServiceStub
    },
    'recommendationservice': {
        'host': 'recommendationservice',
        'port': 8080,
        'stub_class': demo_pb2_grpc.RecommendationServiceStub
    },
    'adservice': {
        'host': 'adservice',
        'port': 9555,
        'stub_class': demo_pb2_grpc.AdServiceStub
    },
    'emailservice': {
        'host': 'emailservice',
        'port': 5000,
        'stub_class': demo_pb2_grpc.EmailServiceStub
    }
}

# Product IDs for testing
products = [
    '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP', '6E92ZMYYFZ',
    '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z'
]

# Logging setup
LOG_DIR = os.environ.get("LOG_DIR", ".")
ARCH_TYPE = os.environ.get("ARCH_TYPE", "unknown")
RUN_NUM = os.environ.get("RUN_NUM", "0")

log_file_name = f"{LOG_DIR}/grpc_request_log_{ARCH_TYPE}_run_{RUN_NUM}.csv"
log_file = None
csv_writer = None

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global log_file, csv_writer
    log_file = open(log_file_name, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "service", "method", "response_time", "status"])

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    if csv_writer:
        status = "SUCCESS" if response and not hasattr(response, 'code') or getattr(response, 'code', 0) == 0 else "FAILURE"
        csv_writer.writerow([
            datetime.datetime.now().isoformat(),
            name.split('_')[0] if '_' in name else name,
            name,
            response_time,
            status
        ])

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if log_file:
        log_file.close()

class GrpcClient:
    """
    gRPC client wrapper for load testing
    """
    def __init__(self, service_name):
        self.service_name = service_name
        self.service_config = SERVICES[service_name]
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """Establish gRPC connection"""
        try:
            host = self.service_config['host']
            port = self.service_config['port']
            self.channel = grpc.insecure_channel(f'{host}:{port}')
            self.stub = self.service_config['stub_class'](self.channel)
        except Exception as e:
            print(f"Failed to connect to {self.service_name}: {e}")
    
    def _call_rpc(self, method_name, request, timeout=10):
        """Make gRPC call with timing"""
        start_time = time.time()
        try:
            method = getattr(self.stub, method_name)
            response = method(request, timeout=timeout)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            return response, response_time, "SUCCESS"
        except grpc.RpcError as e:
            response_time = (time.time() - start_time) * 1000
            return None, response_time, f"FAILURE: {e.code()}"
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return None, response_time, f"ERROR: {str(e)}"

class ProductCatalogUser(User):
    """User that tests ProductCatalogService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('productcatalogservice')
    
    @task(1)
    def get_product(self):
        """Get specific product"""
        product_id = random.choice(products)
        request = demo_pb2.GetProductRequest(id=product_id)
        response, response_time, status = self.client._call_rpc('GetProduct', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="productcatalogservice_GetProduct",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class CartUser(User):
    """User that tests CartService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('cartservice')
        self.user_id = fake.uuid4()
    
    @task(1)
    def add_item_to_cart(self):
        """Add item to cart"""
        product_id = random.choice(products)
        quantity = random.randint(1, 5)
        item = Cart_pb2.CartItem(product_id=product_id, quantity=quantity)
        request = Cart_pb2.AddItemRequest(user_id=self.user_id, item=item)
        response, response_time, status = self.client._call_rpc('AddItem', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="cartservice_AddItem",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class CurrencyUser(User):
    """User that tests CurrencyService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('currencyservice')
    
    @task(1)
    def get_supported_currencies(self):
        """Get supported currencies"""
        request = demo_pb2.Empty()
        response, response_time, status = self.client._call_rpc('GetSupportedCurrencies', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="currencyservice_GetSupportedCurrencies",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class RecommendationUser(User):
    """User that tests RecommendationService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('recommendationservice')
        self.user_id = fake.uuid4()
    
    @task(1)
    def get_recommendations(self):
        """Get product recommendations"""
        product_ids = random.sample(products, random.randint(1, 3))
        request = demo_pb2.ListRecommendationsRequest(
            user_id=self.user_id,
            product_ids=product_ids
        )
        response, response_time, status = self.client._call_rpc('ListRecommendations', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="recommendationservice_ListRecommendations",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class PaymentUser(User):
    """User that tests PaymentService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('paymentservice')
    
    @task(1)
    def charge_payment(self):
        """Charge payment"""
        amount = demo_pb2.Money(
            currency_code='USD',
            units=random.randint(1, 100),
            nanos=random.randint(0, 999999999)
        )
        credit_card = demo_pb2.CreditCardInfo(
            credit_card_number=fake.credit_card_number(),
            credit_card_cvv=random.randint(100, 999),
            credit_card_expiration_year=random.randint(2025, 2030),
            credit_card_expiration_month=random.randint(1, 12)
        )
        request = demo_pb2.ChargeRequest(amount=amount, credit_card=credit_card)
        response, response_time, status = self.client._call_rpc('Charge', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="paymentservice_Charge",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class ShippingUser(User):
    """User that tests ShippingService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('shippingservice')
    
    @task(1)
    def get_shipping_quote(self):
        """Get shipping quote"""
        address = demo_pb2.Address(
            street_address=fake.street_address(),
            city=fake.city(),
            state=fake.state_abbr(),
            country=fake.country(),
            zip_code=random.randint(10000, 99999)
        )
        items = [
            demo_pb2.CartItem(
                product_id=random.choice(products),
                quantity=random.randint(1, 5)
            ) for _ in range(random.randint(1, 3))
        ]
        request = demo_pb2.GetQuoteRequest(address=address, items=items)
        response, response_time, status = self.client._call_rpc('GetQuote', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="shippingservice_GetQuote",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

class AdUser(User):
    """User that tests AdService"""
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GrpcClient('adservice')
    
    @task(1)
    def get_ads(self):
        """Get ads"""
        context_keys = random.sample(['clothing', 'kitchen', 'electronics', 'books', 'sports'], 
                                   random.randint(1, 3))
        request = demo_pb2.AdRequest(context_keys=context_keys)
        response, response_time, status = self.client._call_rpc('GetAds', request)
        self.environment.events.request.fire(
            request_type="gRPC",
            name="adservice_GetAds",
            response_time=response_time,
            response_length=0,
            response=response,
            context={},
            exception=None
        )

# Concrete user classes
class ProductCatalogLoadUser(ProductCatalogUser):
    wait_time = between(1, 3)

class CartLoadUser(CartUser):
    wait_time = between(1, 3)

class CurrencyLoadUser(CurrencyUser):
    wait_time = between(1, 3)

class RecommendationLoadUser(RecommendationUser):
    wait_time = between(1, 3)

class PaymentLoadUser(PaymentUser):
    wait_time = between(1, 3)

class ShippingLoadUser(ShippingUser):
    wait_time = between(1, 3)

class AdLoadUser(AdUser):
    wait_time = between(1, 3)
