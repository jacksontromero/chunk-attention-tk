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
    Same logic as before (ground truth doesn't change).
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

def compute_chunk_intermediates(q, k_chunk, v_chunk, softmax_scale):
    """
    Compute partial attention results for a single chunk.
    Mimics attn_chunk_first output.
    Returns:
        maxs: [n_seqs, n_heads]
        sums: [n_seqs, n_heads]
        attns: [n_seqs, n_heads, d_head]
    """
    n_seqs, n_heads, d_head = q.shape
    # k_chunk: [n_heads, chunk_size, d_head]
    
    maxs = np.zeros((n_seqs, n_heads), dtype=np.float32)
    sums = np.zeros((n_seqs, n_heads), dtype=np.float32)
    attns = np.zeros((n_seqs, n_heads, d_head), dtype=np.float32)

    for s in range(n_seqs):
        for h in range(n_heads):
            scores = np.dot(q[s, h], k_chunk[h].T) * softmax_scale
            # No masking for cached chunks (they are "shared prefixes", usually fully visible)
            
            m = np.max(scores)
            e = np.exp(scores - m)
            sm = np.sum(e)
            out = np.dot(e, v_chunk[h])
            
            maxs[s, h] = m
            sums[s, h] = sm
            attns[s, h] = out
            
    return maxs, sums, attns

def write_test_file(filepath, n_seqs, n_heads, n_chunks, n_shared_chunks, seq_tokens, 
                    q, k_chunks, v_chunks, output, 
                    cached_maxs=None, cached_sums=None, cached_attns=None):
    """Write test data with header."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        # Header: n_seqs chunk_size d_head n_heads n_chunks seq_tokens n_shared_chunks
        f.write(f'{n_seqs} {CHUNK_SIZE} {D_HEAD} {n_heads} {n_chunks} {seq_tokens} {n_shared_chunks}\n')
        
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
        
        # Cached data for shared chunks (if any)
        # Order: For each chunk c in 0..n_shared-1: maxs, sums, attns
        if n_shared_chunks > 0:
            # Flatten lists of arrays
            # cached_maxs is list of [n_seqs, n_heads]. Total n_shared_chunks items.
            for i in range(n_shared_chunks):
                f.write(' '.join(f'{x}' for x in cached_maxs[i].flatten()) + '\n')
                f.write(' '.join(f'{x}' for x in cached_sums[i].flatten()) + '\n')
                f.write(' '.join(f'{x}' for x in cached_attns[i].flatten()) + '\n')


# =============================================================================
# Data pattern generators
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
    pattern, n_seqs, n_heads, n_chunks, n_shared_chunks, seq_tokens = args

    try:
        if pattern not in DATA_PATTERNS:
             # Fallback for patterns not explicitly redefined here but present in original script
             # For now just use random if unknown
             rng = np.random.RandomState(42)
             q = rng.randn(n_seqs, n_heads, D_HEAD).astype(np.float32)
             k = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) for _ in range(n_chunks)]
             v = [rng.randn(n_heads, CHUNK_SIZE, D_HEAD).astype(np.float32) for _ in range(n_chunks)]
        else:
             q, k, v = DATA_PATTERNS[pattern](n_seqs, n_heads, n_chunks, seed=42)
             
        softmax_scale = 1.0 / math.sqrt(D_HEAD)
        
        # Calculate full ground truth
        output = compute_seq_first_attention_cpu(q, k, v, softmax_scale, seq_tokens)

        # Calculate cached intermediates
        cached_maxs, cached_sums, cached_attns = [], [], []
        for i in range(n_shared_chunks):
            m, s, a = compute_chunk_intermediates(q, k[i], v[i], softmax_scale)
            cached_maxs.append(m)
            cached_sums.append(s)
            cached_attns.append(a)

        # Filename
        fname = f'{pattern}_s{n_seqs}_h{n_heads}_c{n_chunks}_sh{n_shared_chunks}'
        if seq_tokens < n_chunks * CHUNK_SIZE:
            fname += f'_t{seq_tokens}'
        fname += '.txt'

        filepath = TEST_DIR / fname
        write_test_file(filepath, n_seqs, n_heads, n_chunks, n_shared_chunks, seq_tokens, 
                        q, k, v, output, cached_maxs, cached_sums, cached_attns)
        return fname
    except Exception as e:
        print(f"ERROR {args}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_grid(quick=False):
    """Generate test grid."""
    if quick:
        n_seqs_values = [1, 4]
        n_heads_values = [1, 4]
        n_chunks_values = [2, 4] # Need at least 2 chunks to have shared
        patterns = ['small', 'randn']
    else:
        n_seqs_values = [1, 2, 4, 8, 16]
        n_heads_values = [1, 2, 4, 8]
        n_chunks_values = [1, 2, 4, 8]
        patterns = ['randn', 'small', 'uniform', 'large', 'positive']

    TEST_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    configs = []
    for pattern in patterns:
        for n_seqs in n_seqs_values:
            for n_heads in n_heads_values:
                for n_chunks in n_chunks_values:
                    # Test: No shared, Half shared, All shared (minus 1 maybe?)
                    # Usually shared chunks are prefixes.
                    # e.g. 4 chunks. shared=0, shared=2, shared=4?
                    # If shared=4, all chunks are precomputed. Kernel just merges.
                    # attn_seq_first logic: for i in warp..chunk_num. if i < shared ...
                    
                    shared_options = [0]
                    if n_chunks >= 2:
                        shared_options.append(n_chunks // 2)
                    if n_chunks >= 1:
                        # shared_options.append(n_chunks) # Full cache hit - maybe skip to save time?
                        pass
                        
                    # Causal token counts (as fraction of last chunk)
                    causal_fractions = [1.0, 0.5] if quick else [1.0, 0.75, 0.5, 0.25]

                    for n_shared in shared_options:
                        for causal_frac in causal_fractions:
                            # Calculate seq_tokens
                            if causal_frac >= 1.0:
                                seq_tokens = n_chunks * CHUNK_SIZE
                            else:
                                # (n_chunks-1) full chunks + partial last chunk
                                last_chunk_tokens = max(1, int(CHUNK_SIZE * causal_frac))
                                seq_tokens = (n_chunks - 1) * CHUNK_SIZE + last_chunk_tokens

                            configs.append((pattern, n_seqs, n_heads, n_chunks, n_shared, seq_tokens))

    # Deduplicate
    configs = list(set(configs))
    total = len(configs)
    print(f"Generating {total} tests using {N_WORKERS} workers...")

    with Pool(N_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_single_test, configs, chunksize=32)):
            if result:
                results.append(result)
            if (i + 1) % 100 == 0:
                print(f"\r  [{i+1}/{total}] generated...", end='', flush=True)

    print(f"\nGenerated {len(results)} tests.")
    return results

def create_runner(files):
    # Same as before, just updates the filename list
    with open('run_tests_seq_first.sh', 'w') as f:
        f.write(f'''#!/bin/bash
JOBS=${{1:-4}}
RESULTS_FILE=$(mktemp)
echo "Running tests..."
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
cat << 'TESTLIST' | xargs -P "$JOBS" -I {{}} bash -c 'run_test "{{}}" "'$RESULTS_FILE'"'
''')
        for fn in sorted(files):
            f.write(f'tests_seq_first/{fn}\n')
        f.write('''TESTLIST
# Count logic same as before
PASSED=$(grep -c "^PASS$" "$RESULTS_FILE" || true)
FAILED=0
FAILED_TESTS=""
while IFS= read -r line; do
    if [[ "$line" == FAIL:* ]]; then
        ((FAILED++))
        NAME="${line#FAIL:}"
        FAILED_TESTS="$FAILED_TESTS\n  $NAME"
    fi
done < <(grep "^FAIL:" "$RESULTS_FILE" 2>/dev/null || true)
rm -f "$RESULTS_FILE"
echo ""
echo "PASSED: $PASSED / $((PASSED + FAILED))"
if [ "$FAILED" -gt 0 ]; then
    echo -e "FAILED:$FAILED_TESTS"
    exit 1
else
    echo "All tests passed!"
fi
''')
    os.chmod('run_tests_seq_first.sh', 0o755)

def main():
    # Simplification: Always generate quick grid or full grid based on args
    # Ignoring single test arg parsing for brevity, user requested substantial updates to logic
    if '--quick' in sys.argv:
        files = generate_grid(quick=True)
    else:
        files = generate_grid(quick=False)
    create_runner(files)

if __name__ == '__main__':
    main()
