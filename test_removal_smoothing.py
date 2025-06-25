#!/usr/bin/env python3
"""
Test script to verify removal smoothing implementation.
This script tests that the exponential probability distribution P(o) ∝ exp(-λo) 
produces the expected behavior where o=0 occurs ~98% of the time with λ=4.
"""

import random
import math
from collections import Counter

def sample_removal_offset(lam, max_offset=2):
    """
    Sample offset from exponential distribution P(o) ∝ exp(-λo)
    This is the same logic as implemented in run.py
    """
    # Calculate probabilities for offsets 0, 1, 2, ..., max_offset
    probs = []
    total_prob = 0
    for i in range(max_offset + 1):
        prob = math.exp(-lam * i)
        probs.append(prob)
        total_prob += prob
    
    # Normalize probabilities
    probs = [p / total_prob for p in probs]
    
    # Sample offset based on these probabilities
    rand_val = random.random()
    cumulative_prob = 0
    offset = 0
    for i, prob in enumerate(probs):
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            offset = i
            break
    
    return offset, probs

def test_removal_smoothing():
    """Test the removal smoothing implementation"""
    print("Testing Removal Smoothing Implementation")
    print("=" * 50)
    
    # Test with λ=4, max_offset=2 (as specified in the plan)
    lam = 4.0
    max_offset = 2
    num_samples = 100000
    
    # Sample many offsets to check the distribution
    counter = Counter()
    sample_probs = None
    
    for _ in range(num_samples):
        offset, probs = sample_removal_offset(lam, max_offset)
        counter[offset] += 1
        if sample_probs is None:
            sample_probs = probs
    
    print(f"Parameters: λ={lam}, max_offset={max_offset}")
    print(f"Number of samples: {num_samples}")
    print()
    
    print("Theoretical vs Empirical Probabilities:")
    print("-" * 40)
    
    for i in range(max_offset + 1):
        theoretical = sample_probs[i] * 100 if sample_probs else 0
        empirical = (counter[i] / num_samples) * 100
        print(f"Offset {i}: Theory={theoretical:.2f}%, Empirical={empirical:.2f}%")
    
    print()
    
    # Check if o=0 occurs ~98% of the time as expected
    prob_zero = (counter[0] / num_samples) * 100
    print(f"Probability of offset=0: {prob_zero:.2f}%")
    
    if prob_zero >= 95 and prob_zero <= 99:
        print("✓ SUCCESS: Offset=0 occurs ~98% of the time as expected!")
    else:
        print("✗ WARNING: Offset=0 probability is not as expected (~98%)")
    
    print()
    print("Expected behavior:")
    print("- With λ=4, offset=0 should occur ~98% of the time")
    print("- offset=1 should occur ~2% of the time") 
    print("- offset=2 should occur <1% of the time")
    print("- This creates smooth transitions between stages")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    test_removal_smoothing() 