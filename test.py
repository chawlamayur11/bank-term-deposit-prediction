#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py - Prediction Service Verification Script
Course: DataTalksClub Machine Learning Zoomcamp (Midterm Project)
Author: Mayur Chawla

Tests the web service with two contrasting client profiles:
- Client A: High-propensity prospect (prior successful campaign, favorable economic conditions)
- Client B: Low-propensity prospect (no previous campaign history, higher interest rate environment)
"""

import sys
import json
import requests

# Test Client A: High likelihood of term deposit subscription
# Characteristics: Previously subscribed (poutcome=success), contacted in October, favorable economic environment
client_high_propensity = {
    "age": 45,
    "job": "management",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "oct",
    "day_of_week": "wed",
    "campaign": 1,
    "pdays": 3,
    "previous": 2,
    "poutcome": "success",
    "emp_var_rate": -3.4,
    "cons_price_idx": 92.431,
    "cons_conf_idx": -26.9,
    "euribor3m": 0.739,
    "nr_employed": 5017.5
}

# Test Client B: Low likelihood of term deposit subscription
# Characteristics: Repeatedly contacted in May, high interest rate environment, no previous contact
client_low_propensity = {
    "age": 32,
    "job": "blue-collar",
    "marital": "married",
    "education": "basic.9y",
    "default": "no",
    "housing": "yes",
    "loan": "yes",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 5,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
}


def test_via_api(url: str):
    """Send requests to live API endpoint."""
    print(f"Connecting to live API at {url}...")

    # Test Client A
    print("\n" + "=" * 60)
    print("Testing Client A (Expected: High Propensity / Subscription)")
    print("=" * 60)
    try:
        response_a = requests.post(f"{url}/predict", json=client_high_propensity, timeout=5)
        response_a.raise_for_status()
        print("Response A:")
        print(json.dumps(response_a.json(), indent=2))
    except Exception as e:
        print(f"Failed to query endpoint: {e}")
        return False

    # Test Client B
    print("\n" + "=" * 60)
    print("Testing Client B (Expected: Low Propensity / No Subscription)")
    print("=" * 60)
    try:
        response_b = requests.post(f"{url}/predict", json=client_low_propensity, timeout=5)
        response_b.raise_for_status()
        print("Response B:")
        print(json.dumps(response_b.json(), indent=2))
    except Exception as e:
        print(f"Failed to query endpoint: {e}")
        return False

    print("\nAll API tests completed successfully!")
    return True


def test_in_process():
    """Fallback local test using predict module directly."""
    print("\nLive server not detected on localhost:8000. Running in-process verification...")
    try:
        from predict import score_client
        print("\n" + "=" * 60)
        print("Testing Client A (In-Process)")
        print("=" * 60)
        res_a = score_client(client_high_propensity)
        print(json.dumps(res_a, indent=2))

        print("\n" + "=" * 60)
        print("Testing Client B (In-Process)")
        print("=" * 60)
        res_b = score_client(client_low_propensity)
        print(json.dumps(res_b, indent=2))

        print("\nIn-process test verification successful!")
    except Exception as e:
        print(f"In-process test failed: {e}")


def main():
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = False
    try:
        health = requests.get(f"{target_url}/health", timeout=2)
        if health.status_code == 200:
            success = test_via_api(target_url)
    except requests.exceptions.RequestException:
        pass

    if not success:
        test_in_process()
        print("\nTip: Start the server with 'python predict.py' or with Docker, then re-run 'python test.py' to test over HTTP.")


if __name__ == '__main__':
    main()
