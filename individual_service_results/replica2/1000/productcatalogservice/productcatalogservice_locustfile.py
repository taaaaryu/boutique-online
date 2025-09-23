#!/usr/bin/env python3
import time
import random
from faker import Faker
from locust import HttpUser, task, between, events
import json
import datetime
import csv
import os

fake = Faker()

# Product list from original locustfile.py
products = [
    '0PUK6V6EV0',
    '1YMWWN1N4O', 
    '2ZYFJ3GM2N',
    '66VCHSJNUP',
    '6E92ZMYYFZ',
    '9SIQT8TOJO',
    'L9ECAV7KIM',
    'LS4PSXUNUM',
    'OLJCESPC7Z'
]

class ProductcatalogserviceUser(HttpUser):
    wait_time = between(1, 10)
    
    # Set host based on service - use localhost port-forwarding
    if "productcatalogservice" == 'productcatalogservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'cartservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'checkoutservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'paymentservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'shippingservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'currencyservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'recommendationservice':
        host = "http://localhost:8080"
    elif "productcatalogservice" == 'adservice':
        host = "http://localhost:8080"
    else:
        host = "http://localhost:8080"
    
    @task
    def test_service(self):
        start_time = time.time()
        try:
            # Call different methods based on service
            if "productcatalogservice" == 'productcatalogservice':
                self._test_product_catalog()
            elif "productcatalogservice" == 'cartservice':
                self._test_cart_service()
            elif "productcatalogservice" == 'currencyservice':
                self._test_currency_service()
            elif "productcatalogservice" == 'recommendationservice':
                self._test_recommendation_service()
            elif "productcatalogservice" == 'adservice':
                self._test_ad_service()
            else:
                # Generic test for other services
                self._test_generic_service()
                
            response_time = (time.time() - start_time) * 1000

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

    
    def _test_product_catalog(self):
        # Test ListProducts via HTTP - use root endpoint
        response = self.client.get("/")
        return response
    
    def _test_cart_service(self):
        # Test cart via HTTP - use simple endpoint
        response = self.client.get("/cart")
        return response
    
    def _test_currency_service(self):
        # Test currency via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_recommendation_service(self):
        # Test recommendations via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_ad_service(self):
        # Test ads via HTTP - use simple endpoint
        response = self.client.get("/")
        return response
    
    def _test_generic_service(self):
        # Generic test - just make a simple HTTP request
        response = self.client.get("/")
        return response

# Create concrete user class
class TestUser(ProductcatalogserviceUser):
    pass
