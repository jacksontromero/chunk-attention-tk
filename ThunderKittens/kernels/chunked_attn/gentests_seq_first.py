#!/usr/bin/env python3
"""
Test generator for attn_seq_first TK kernel.

Usage:
    python gentests_seq_first.py                      # Generate full test grid
    python gentests_seq_first.py --quick              # Quick test set (fewer configs)
    python gentests_seq_first.py randn 4 4 2          # Single test: pattern n_seqs n_heads n_chunks
    python gentests_seq_first.py randn 4 4 2 100      # With causal: 100 tokens (not 128)
    python gentests_seq_first.py --list               # List available data patterns

The seq_first kernel:
- Takes one query per sequence (n_seqs queries total)
- Processes all KV chunks for that sequence
- Applies causal masking on the last chunk if seq_tokens < n_chunks * chunk_size
- Outputs final normalized attention for each sequence
"""

import numpy as np
import sys
import math
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Fixed at compile time (must match harness_seq_first.impl)
CHUNK_SIZE = 64
D_HEAD = 128

# Directories
TEST_DIR = Path("tests_seq_first")
OUTPUT_DIR = Path("output_seq_first")

# Number of parallel workers
N_WORKERS = min(cpu_count(), 16)


def compute_seq_first_attention_cpu(q, k_chunks, v_chunks, softmax_scale, seq_tokens):
    """
    Reference implementation of seq_first attention.

    Args:
        q: [n_seqs, n_heads, d_head] - query for each sequence
        k_chunks: list of [n_heads, chunk_size, d_head] - K for each chunk
        v_chunks: list of [n_heads, chunk_size, d_head] - V for each chunk
        softmax_scale: 1/sqrt(d_head)
        seq_tokens: total number of tokens per sequence (for causal masking)

    Returns:
        output: [n_seqs, n_heads, d_head] - normalized attention output
    """
    n_seqs, n_heads, d_head = q.shape
    n_chunks = len(k_chunks)
    chunk_size = k_chunks[0].shape[1]

    output = np.zeros((n_seqs, n_heads, d_head), dtype=np.float32)

    for s in range(n_seqs):
        for h in range(n_heads):
            # Online softmax accumulation across chunks
            global_max = -np.inf
            global_sum = 0.0
            acc = np.zeros(d_head, dtype=np.float32)

            for c in range(n_chunks):
                # Compute how many tokens are valid in this chunk
                chunk_start = c * chunk_size
                chunk_end = min((c + 1) * chunk_size, seq_tokens)
                valid_tokens = chunk_end - chunk_start

                if valid_tokens <= 0:
                    continue

                k_chunk = k_chunks[c][h]  # [chunk_size, d_head]
                v_chunk = v_chunks[c][h]  # [chunk_size, d_head]

                # scores = Q @ K^T for this chunk
                scores = np.dot(q[s, h], k_chunk.T) * softmax_scale  # [chunk_size]

                # Apply causal mask: only first valid_tokens positions are valid
                if valid_tokens < chunk_size:
                    scores[valid_tokens:] = -np.inf

                # Online softmax update
                chunk_max = np.max(scores)
                new_max = max(global_max, chunk_max)

                # Rescale previous accumulation
                if global_sum > 0:
                    scale_old = np.exp(global_max - new_max)
                    acc = acc * scale_old
                    global_sum = global_sum * scale_old

                # Add current chunk contribution
                exp_scores = np.exp(scores - new_max)
                chunk_sum = np.sum(exp_scores)

                # scores @ V
                acc += np.dot(exp_scores, v_chunk)  # [d_head]
                global_sum += chunk_sum
                global_max = new_max

            # Normalize
            if global_sum > 0:
                output[s, h] = acc / global_sum
            else:
                output[s, h] = 0.0

    return output


def write_test_file(filepath, n_seqs, n_heads, n_chunks, seq_tokens, q, k_chunks, v_chunks, output):
    """Write test data with header."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        # Header: n_seqs chunk_size d_head n_heads n_chunks [seq_tokens]
        f.write(f'{n_seqs} {CHUNK_SIZE} {D_HEAD} {n_heads} {n_chunks} {seq_tokens}\n')
        # Q: [n_seqs, n_heads, d_head]
        f.write(' '.join(f'{x}' for x in q.flatten()) + '\n')
        # K chunks: each [n_heads, chunk_size, d_head]
        for k in k_chunks:
            f.write(' '.join(f'{x}' for x in k.flatten()) + '\n')
        # V chunks: each [n_heads, chunk_size, d_head]
        for v in v_chunks:
            f.write(' '.join(f'{x}' for x in v.flatten()) + '\n')
        # Expected output: [n_seqs, n_heads, d_head]
        f.write(' '.join(f'{x}' for x in output.flatten()) + '\n')


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
    pattern, n_seqs, n_heads, n_chunks, seq_tokens = args

    try:
        q, k, v = DATA_PATTERNS[pattern](n_seqs, n_heads, n_chunks, seed=42)
        softmax_scale = 1.0 / math.sqrt(D_HEAD)
        output = compute_seq_first_attention_cpu(q, k, v, softmax_scale, seq_tokens)

        # Filename includes causal info if applicable
        full_tokens = n_chunks * CHUNK_SIZE
        if seq_tokens < full_tokens:
            filename = f'{pattern}_s{n_seqs}_h{n_heads}_c{n_chunks}_t{seq_tokens}.txt'
        else:
            filename = f'{pattern}_s{n_seqs}_h{n_heads}_c{n_chunks}.txt'

        filepath = TEST_DIR / filename
        write_test_file(filepath, n_seqs, n_heads, n_chunks, seq_tokens, q, k, v, output)
        return filename
    except Exception as e:
        print(f"ERROR {pattern} s{n_seqs} h{n_heads} c{n_chunks} t{seq_tokens}: {e}")
        return None


def generate_grid(quick=False):
    """Generate test grid using multiprocessing."""
    if quick:
        n_seqs_values = [1, 4, 8]
        n_heads_values = [1, 4]
        n_chunks_values = [1, 2, 4]
        patterns = ['randn', 'small']
        # Causal token counts (as fraction of last chunk)
        causal_fractions = [1.0, 0.5]  # full, half of last chunk valid
    else:
        n_seqs_values = [1, 2, 4, 8, 16]
        n_heads_values = [1, 2, 4, 8]
        n_chunks_values = [1, 2, 4, 8]
        patterns = ['randn', 'small', 'uniform', 'large', 'positive']
        # Causal: full, 3/4, 1/2, 1/4 of last chunk valid
        causal_fractions = [1.0, 0.75, 0.5, 0.25]

    # Create directories
    TEST_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build list of all test configurations
    configs = []
    for pattern in patterns:
        for n_seqs in n_seqs_values:
            for n_heads in n_heads_values:
                for n_chunks in n_chunks_values:
                    for causal_frac in causal_fractions:
                        # Calculate seq_tokens
                        if causal_frac >= 1.0:
                            seq_tokens = n_chunks * CHUNK_SIZE
                        else:
                            # (n_chunks-1) full chunks + partial last chunk
                            last_chunk_tokens = max(1, int(CHUNK_SIZE * causal_frac))
                            seq_tokens = (n_chunks - 1) * CHUNK_SIZE + last_chunk_tokens

                        configs.append((pattern, n_seqs, n_heads, n_chunks, seq_tokens))

    # Deduplicate (some configs may be identical)
    configs = list(set(configs))
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
    print(f"\nGenerated {len(results)} tests in {elapsed:.1f}s ({len(results)/max(1,elapsed):.0f} tests/sec)")
    return results


def create_runner(files):
    """Create test runner script with parallel execution support."""
    with open('run_tests_seq_first.sh', 'w') as f:
        f.write(f'''#!/bin/bash
# Auto-generated test runner for seq_first kernel
# Usage: ./run_tests_seq_first.sh [JOBS]   (default: 4 parallel jobs)

JOBS=${{1:-4}}
TOTAL={len(files)}
RESULTS_FILE=$(mktemp)

echo "Running $TOTAL seq_first tests with $JOBS parallel jobs..."
echo ""

# Function to run a single test and output result (streams to terminal)
run_test() {{
    TEST="$1"
    RESULTS_FILE="$2"
    NAME=$(basename "$TEST")
    OUTPUT=$(./attn_seq_first "$TEST" 2>&1)
    if [ $? -eq 0 ]; then
        echo "✓ $NAME"
        echo "PASS" >> "$RESULTS_FILE"
    else
        SUMMARY=$(echo "$OUTPUT" | grep "SUMMARY:" | head -1)
        echo "✗ $NAME - $SUMMARY"
        echo "FAIL:$NAME" >> "$RESULTS_FILE"
    fi
}}
export -f run_test

# Run tests in parallel, streaming output
cat << 'TESTLIST' | xargs -P "$JOBS" -I {{}} bash -c 'run_test "{{}}" "'$RESULTS_FILE'"'
''')
        for fn in sorted(files):
            f.write(f'tests_seq_first/{fn}\n')
        f.write('''TESTLIST

# Count results
PASSED=$(grep -c "^PASS$" "$RESULTS_FILE" || true)
PASSED=${PASSED:-0}
FAILED=0
FAILED_TESTS=""

while IFS= read -r line; do
    if [[ "$line" == FAIL:* ]]; then
        ((FAILED++))
        NAME="${line#FAIL:}"
        FAILED_TESTS="$FAILED_TESTS\\n  $NAME"
    fi
done < <(grep "^FAIL:" "$RESULTS_FILE" 2>/dev/null || true)

rm -f "$RESULTS_FILE"

TOTAL_RUN=$((PASSED + FAILED))
echo ""
echo "================================"
echo "PASSED: $PASSED / $TOTAL_RUN"
if [ "$FAILED" -gt 0 ]; then
    echo -e "FAILED:$FAILED_TESTS"
    exit 1
else
    echo "All tests passed!"
fi
''')

    os.chmod('run_tests_seq_first.sh', 0o755)
    print("Created run_tests_seq_first.sh")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--list':
            print("Data patterns:", ', '.join(sorted(DATA_PATTERNS.keys())))
            print(f"\nCompile-time: CHUNK_SIZE={CHUNK_SIZE}, D_HEAD={D_HEAD}")
            print("\nUsage:")
            print("  python gentests_seq_first.py                  # Full grid")
            print("  python gentests_seq_first.py --quick          # Quick test set")
            print("  python gentests_seq_first.py <pattern> <n_seqs> <n_heads> <n_chunks> [seq_tokens]")
            print("\nCausal masking:")
            print("  - If seq_tokens < n_chunks * chunk_size, last chunk is partially masked")
            print("  - Example: 2 chunks, 100 tokens = first chunk full (64), second has 36 valid tokens")
            return

        if arg == '--quick':
            files = generate_grid(quick=True)
            create_runner(files)
            print(f"\nRun: ./run_tests_seq_first.sh")
            return

        # Single test generation
        if arg in DATA_PATTERNS:
            pattern = arg
            n_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            n_heads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
            n_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            # seq_tokens: defaults to full (n_chunks * CHUNK_SIZE)
            full_tokens = n_chunks * CHUNK_SIZE
            seq_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else full_tokens

            TEST_DIR.mkdir(exist_ok=True)
            result = generate_single_test((pattern, n_seqs, n_heads, n_chunks, seq_tokens))
            if result:
                print(f"Generated: tests_seq_first/{result}")
                print(f"Run: ./attn_seq_first tests_seq_first/{result} -v")
            return

        print(f"Unknown argument: {arg}")
        print("Use --list for help")
        return

    # Default: full grid
    files = generate_grid(quick=False)
    create_runner(files)
    print(f"\nRun: ./run_tests_seq_first.sh")


if __name__ == '__main__':
    main()
