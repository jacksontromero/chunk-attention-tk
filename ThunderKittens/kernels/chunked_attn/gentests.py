#!/usr/bin/env python3
"""
Test generator for chunked attention TK kernel.

Usage:
    python gentests.py                 # Generate all tests
    python gentests.py randn           # Generate single test
    python gentests.py --list          # List available tests
"""

import torch
import numpy as np
import sys
import math
import os
from pathlib import Path

# Test configuration - must match harness.impl compile-time constants
N_SEQS = 16
CHUNK_SIZE = 64
D_HEAD = 128
N_HEADS = 4
N_CHUNKS = 2

QUIET = True  # Suppress per-element progress bars


def compute_chunk_attention(q, k_chunks, v_chunks, softmax_scale):
    """Reference implementation matching attn_chunk_first_kernel_v2."""
    outputs, maxs, sums = [], [], []

    for k_chunk, v_chunk in zip(k_chunks, v_chunks):
        chunk_out, chunk_max, chunk_sum = [], [], []
        for h in range(q.shape[1]):
            scores = torch.matmul(q[:, h, :], k_chunk[h].T) * softmax_scale
            max_s = scores.max(dim=1, keepdim=True)[0]
            exp_s = torch.exp(scores - max_s)
            sum_s = exp_s.sum(dim=1, keepdim=True)
            out = torch.matmul(exp_s, v_chunk[h])
            chunk_out.append(out)
            chunk_max.append(max_s.squeeze())
            chunk_sum.append(sum_s.squeeze())
        outputs.append(torch.stack(chunk_out))
        maxs.append(torch.stack(chunk_max))
        sums.append(torch.stack(chunk_sum))

    return outputs, maxs, sums


def write_test_file(filename, q, k_chunks, v_chunks, outputs, maxs, sums):
    """Write test data to file."""
    with open(filename, 'w') as f:
        # Q, K chunks, V chunks, outputs, maxs, sums
        for arr in [q] + k_chunks + v_chunks + outputs + maxs + sums:
            data = arr.flatten().detach().cpu().numpy()
            f.write(' '.join(f'{x}' for x in data) + '\n')


# =============================================================================
# Test case definitions
# =============================================================================

TEST_CASES = {}

def register_test(name):
    """Decorator to register a test case."""
    def decorator(fn):
        TEST_CASES[name] = fn
        return fn
    return decorator


@register_test('randn')
def test_randn(seed=42):
    """Standard random normal."""
    torch.manual_seed(seed)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
    k = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    v = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('ones')
def test_ones(seed=42):
    """All ones - basic sanity check."""
    q = torch.ones((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
    k = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    v = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('small')
def test_small(seed=42):
    """Small values - less numerical issues."""
    torch.manual_seed(seed)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 0.1
    k = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
    v = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('large')
def test_large(seed=42):
    """Large values - stress numerical stability."""
    torch.manual_seed(seed)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 2.0
    k = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2.0 for _ in range(N_CHUNKS)]
    v = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2.0 for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('uniform')
def test_uniform(seed=42):
    """Uniform [-1, 1]."""
    torch.manual_seed(seed)
    q = torch.rand((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1
    k = [torch.rand((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1 for _ in range(N_CHUNKS)]
    v = [torch.rand((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1 for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('zeros_v')
def test_zeros_v(seed=42):
    """Zero V - output should be zero."""
    torch.manual_seed(seed)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
    k = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    v = [torch.zeros((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('identity_v')
def test_identity_v(seed=42):
    """Identity-like V matrix."""
    torch.manual_seed(seed)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 0.1
    k = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
    v = []
    for _ in range(N_CHUNKS):
        vt = torch.zeros((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')
        eye_size = min(CHUNK_SIZE, D_HEAD)
        for h in range(N_HEADS):
            vt[h, :eye_size, :eye_size] = torch.eye(eye_size, device='cuda')
        v.append(vt)
    return q, k, v


@register_test('positive')
def test_positive(seed=42):
    """All positive values."""
    torch.manual_seed(seed)
    q = torch.abs(torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda'))
    k = [torch.abs(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')) for _ in range(N_CHUNKS)]
    v = [torch.abs(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')) for _ in range(N_CHUNKS)]
    return q, k, v


@register_test('negative')
def test_negative(seed=42):
    """All negative values."""
    torch.manual_seed(seed)
    q = -torch.abs(torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda'))
    k = [-torch.abs(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')) for _ in range(N_CHUNKS)]
    v = [-torch.abs(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')) for _ in range(N_CHUNKS)]
    return q, k, v


def generate_seed_tests():
    """Generate tests with different random seeds."""
    for seed in [123, 456, 789, 1000, 2000]:
        name = f'seed{seed}'
        TEST_CASES[name] = lambda s=seed: test_randn(s)


generate_seed_tests()


# =============================================================================
# Main
# =============================================================================

def generate_test(name, verbose=False):
    """Generate a single test case."""
    if name not in TEST_CASES:
        print(f"Unknown test: {name}")
        return None

    q, k, v = TEST_CASES[name]()
    softmax_scale = 1.0 / math.sqrt(D_HEAD)
    outputs, maxs, sums = compute_chunk_attention(q, k, v, softmax_scale)

    filename = f'{name}_{N_SEQS}_{CHUNK_SIZE}_{D_HEAD}.txt'
    write_test_file(filename, q, k, v, outputs, maxs, sums)

    if verbose:
        print(f"Generated: {filename}")
        print(f"  Q range: [{q.min():.3f}, {q.max():.3f}]")
        print(f"  Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")

    return filename


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--list':
            print("Available tests:")
            for name in sorted(TEST_CASES.keys()):
                print(f"  {name}")
            return

        # Generate single test with verbose output
        filename = generate_test(arg, verbose=True)
        if filename:
            print(f"\nRun with: ./attn_chunk_first {filename}")
    else:
        # Generate all tests quietly
        print(f"Generating {len(TEST_CASES)} tests for dims ({N_SEQS}, {CHUNK_SIZE}, {D_HEAD})...")

        files = []
        for name in sorted(TEST_CASES.keys()):
            try:
                filename = generate_test(name, verbose=False)
                if filename:
                    files.append(filename)
            except Exception as e:
                print(f"  ERROR {name}: {e}")

        print(f"Generated {len(files)} test files")

        # Create test runner
        with open('run_tests.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# Run all tests and summarize results\n\n')
            f.write('PASSED=0\nFAILED=0\nFAILED_TESTS=""\n\n')
            for fn in files:
                f.write(f'./attn_chunk_first {fn} > /tmp/test_out.txt 2>&1\n')
                f.write('if [ $? -eq 0 ]; then\n')
                f.write(f'    echo "✓ {fn}"\n')
                f.write('    ((PASSED++))\n')
                f.write('else\n')
                f.write(f'    echo "✗ {fn}"\n')
                f.write('    ((FAILED++))\n')
                f.write(f'    FAILED_TESTS="$FAILED_TESTS {fn}"\n')
                f.write('fi\n\n')
            f.write('echo ""\necho "================================"\n')
            f.write('echo "PASSED: $PASSED / $((PASSED + FAILED))"\n')
            f.write('if [ $FAILED -gt 0 ]; then\n')
            f.write('    echo "FAILED:$FAILED_TESTS"\n')
            f.write('    exit 1\n')
            f.write('fi\n')

        os.chmod('run_tests.sh', 0o755)
        print("Run all tests with: ./run_tests.sh")


if __name__ == '__main__':
    main()
