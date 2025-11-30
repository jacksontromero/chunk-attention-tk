#!/usr/bin/env python3
"""
Test generator for chunked attention TK kernel.

Usage:
    python gentests.py                      # Generate full test grid
    python gentests.py --quick              # Quick test set (fewer configs)
    python gentests.py randn 8 4 2          # Single test: pattern n_seqs n_heads n_chunks
    python gentests.py --list               # List available data patterns
"""

import torch
import numpy as np
import sys
import math
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Fixed at compile time (must match harness.impl)
CHUNK_SIZE = 64
D_HEAD = 128
MAX_N_SEQS = 32

# Directories
TEST_DIR = Path("tests")
OUTPUT_DIR = Path("output")

# Number of parallel workers
N_WORKERS = min(cpu_count(), 16)


def compute_chunk_attention_cpu(q, k_chunks, v_chunks, softmax_scale):
    """Reference implementation - CPU version for parallel generation."""
    outputs, maxs, sums = [], [], []
    n_heads = q.shape[1]

    for k_chunk, v_chunk in zip(k_chunks, v_chunks):
        chunk_out, chunk_max, chunk_sum = [], [], []
        for h in range(n_heads):
            scores = np.matmul(q[:, h, :], k_chunk[h].T) * softmax_scale
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            sum_s = exp_s.sum(axis=1, keepdims=True)
            out = np.matmul(exp_s, v_chunk[h])
            chunk_out.append(out)
            chunk_max.append(max_s.squeeze(-1))
            chunk_sum.append(sum_s.squeeze(-1))
        outputs.append(np.stack(chunk_out))
        maxs.append(np.stack(chunk_max))
        sums.append(np.stack(chunk_sum))

    return outputs, maxs, sums


def write_test_file(filepath, n_seqs, n_heads, n_chunks, q, k_chunks, v_chunks, outputs, maxs, sums):
    """Write test data with header."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f'{n_seqs} {CHUNK_SIZE} {D_HEAD} {n_heads} {n_chunks}\n')
        for arr in [q] + k_chunks + v_chunks + outputs + maxs + sums:
            data = arr.flatten()
            f.write(' '.join(f'{x}' for x in data) + '\n')


# =============================================================================
# Data pattern generators (CPU/numpy versions for parallelism)
# =============================================================================

def pattern_randn(n_seqs, n_heads, n_chunks, seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(n_seqs, n_heads, D_HEAD).astype(np.float32)
    k = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) for _ in range(n_chunks)]
    v = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) for _ in range(n_chunks)]
    return q, k, v


def pattern_small(n_seqs, n_heads, n_chunks, seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(n_seqs, n_heads, D_HEAD).astype(np.float32) * 0.1
    k = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 0.1 for _ in range(n_chunks)]
    v = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 0.1 for _ in range(n_chunks)]
    return q, k, v


def pattern_ones(n_seqs, n_heads, n_chunks, seed=42):
    q = np.ones((n_seqs, n_heads, D_HEAD), dtype=np.float32)
    k = [np.ones((n_heads, CHUNK_SIZE, D_HEAD), dtype=np.float32) for _ in range(n_chunks)]
    v = [np.ones((n_heads, CHUNK_SIZE, D_HEAD), dtype=np.float32) for _ in range(n_chunks)]
    return q, k, v


def pattern_uniform(n_seqs, n_heads, n_chunks, seed=42):
    rng = np.random.RandomState(seed)
    q = (rng.rand(n_seqs, n_heads, D_HEAD).astype(np.float32) * 2 - 1)
    k = [(rng.rand(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 2 - 1) for _ in range(n_chunks)]
    v = [(rng.rand(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 2 - 1) for _ in range(n_chunks)]
    return q, k, v


def pattern_large(n_seqs, n_heads, n_chunks, seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(n_seqs, n_heads, D_HEAD).astype(np.float32) * 2.0
    k = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 2.0 for _ in range(n_chunks)]
    v = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) * 2.0 for _ in range(n_chunks)]
    return q, k, v


def pattern_positive(n_seqs, n_heads, n_chunks, seed=42):
    rng = np.random.RandomState(seed)
    q = np.abs(rng.randn(n_seqs, n_heads, D_HEAD).astype(np.float32))
    k = [np.abs(rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32)) for _ in range(n_chunks)]
    v = [np.abs(rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32)) for _ in range(n_chunks)]
    return q, k, v


DATA_PATTERNS = {
    'randn': pattern_randn,
    'small': pattern_small,
    'ones': pattern_ones,
    'uniform': pattern_uniform,
    'large': pattern_large,
    'positive': pattern_positive,
}


def generate_single_test(args):
    """Generate a single test - designed for multiprocessing."""
    pattern, n_seqs, n_heads, n_chunks = args

    if n_seqs > MAX_N_SEQS:
        return None

    try:
        q, k, v = DATA_PATTERNS[pattern](n_seqs, n_heads, n_chunks, seed=42)
        softmax_scale = 1.0 / math.sqrt(D_HEAD)
        outputs, maxs, sums = compute_chunk_attention_cpu(q, k, v, softmax_scale)

        filename = f'{pattern}_s{n_seqs}_h{n_heads}_c{n_chunks}.txt'
        filepath = TEST_DIR / filename
        write_test_file(filepath, n_seqs, n_heads, n_chunks, q, k, v, outputs, maxs, sums)
        return filename
    except Exception as e:
        print(f"ERROR {pattern} s{n_seqs} h{n_heads} c{n_chunks}: {e}")
        return None


def generate_grid(quick=False):
    """Generate test grid using multiprocessing."""
    if quick:
        n_seqs_values = [8, 16, 32]
        n_heads_values = [1, 4]
        n_chunks_values = [1, 2, 4]
        patterns = ['randn', 'small']
    else:
        n_seqs_values = [1, 2, 4, 8, 16, 24, 32]
        n_heads_values = [1, 2, 4, 8, 12, 16]
        n_chunks_values = [1, 2, 3, 4, 6, 8, 12, 16]
        patterns = ['randn', 'small', 'uniform', 'large', 'positive']

    # Create directories
    TEST_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build list of all test configurations
    configs = []
    for pattern in patterns:
        for n_seqs in n_seqs_values:
            for n_heads in n_heads_values:
                for n_chunks in n_chunks_values:
                    configs.append((pattern, n_seqs, n_heads, n_chunks))

    total = len(configs)
    print(f"Generating {total} tests using {N_WORKERS} workers...")

    start = time.time()

    # Use multiprocessing pool
    with Pool(N_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_single_test, configs, chunksize=32)):
            if result:
                results.append(result)
            # Progress update every 100 tests
            if (i + 1) % 100 == 0 or i + 1 == total:
                print(f"\r  [{i+1}/{total}] generated...", end='', flush=True)

    elapsed = time.time() - start
    print(f"\nGenerated {len(results)} tests in {elapsed:.1f}s ({len(results)/elapsed:.0f} tests/sec)")
    return results


def create_runner(files):
    """Create test runner script."""
    with open('run_tests.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Auto-generated test runner\n\n')
        f.write('mkdir -p output\n\n')
        f.write('PASSED=0\nFAILED=0\nFAILED_TESTS=""\n\n')
        f.write(f'TOTAL={len(files)}\n')
        f.write('COUNT=0\n\n')

        for fn in sorted(files):
            test_path = f'tests/{fn}'
            out_path = f'output/{fn.replace(".txt", ".log")}'
            f.write(f'((COUNT++))\n')
            f.write(f'./attn_chunk_first {test_path} > {out_path} 2>&1\n')
            f.write('if [ $? -eq 0 ]; then\n')
            f.write('    ((PASSED++))\n')
            f.write('else\n')
            f.write(f'    echo "✗ {fn}"\n')
            f.write('    ((FAILED++))\n')
            f.write(f'    FAILED_TESTS="$FAILED_TESTS\\n  {fn}"\n')
            f.write('fi\n')
            f.write('if [ $((COUNT % 50)) -eq 0 ]; then\n')
            f.write('    echo -ne "\\r[$COUNT/$TOTAL] $PASSED passed, $FAILED failed"\n')
            f.write('fi\n\n')

        f.write('echo -e "\\n\\n================================"\n')
        f.write('echo "PASSED: $PASSED / $((PASSED + FAILED))"\n')
        f.write('if [ $FAILED -gt 0 ]; then\n')
        f.write('    echo -e "FAILED:$FAILED_TESTS"\n')
        f.write('    exit 1\n')
        f.write('else\n')
        f.write('    echo "All tests passed!"\n')
        f.write('fi\n')

    os.chmod('run_tests.sh', 0o755)
    print("Created run_tests.sh")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--list':
            print("Data patterns:", ', '.join(sorted(DATA_PATTERNS.keys())))
            print(f"\nCompile-time: CHUNK_SIZE={CHUNK_SIZE}, D_HEAD={D_HEAD}, MAX_N_SEQS={MAX_N_SEQS}")
            print("\nUsage:")
            print("  python gentests.py                  # Full grid")
            print("  python gentests.py --quick          # Quick test set")
            print("  python gentests.py <pattern> <n_seqs> <n_heads> <n_chunks>")
            return

        if arg == '--quick':
            files = generate_grid(quick=True)
            create_runner(files)
            print(f"\nRun: ./run_tests.sh")
            return

        # Single test generation
        if arg in DATA_PATTERNS:
            pattern = arg
            n_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
            n_heads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
            n_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 2

            TEST_DIR.mkdir(exist_ok=True)
            result = generate_single_test((pattern, n_seqs, n_heads, n_chunks))
            if result:
                print(f"Generated: tests/{result}")
                print(f"Run: ./attn_chunk_first tests/{result} -v")
            return

        print(f"Unknown argument: {arg}")
        print("Use --list for help")
        return

    # Default: full grid
    files = generate_grid(quick=False)
    create_runner(files)
    print(f"\nRun: ./run_tests.sh")


if __name__ == '__main__':
    main()
