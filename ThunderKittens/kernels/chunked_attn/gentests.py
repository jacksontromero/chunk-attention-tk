import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import sys
import math
import os

# Test configuration - these must match harness.impl compile-time constants
N_SEQS = 16      # number of sequences attending to chunk
CHUNK_SIZE = 64  # size of KV chunk
D_HEAD = 128     # head dimension
N_HEADS = 4      # number of attention heads
N_CHUNKS = 2     # number of chunks to test

def compute_chunk_attention(q, k_chunks, v_chunks, softmax_scale):
    """
    Reference implementation matching attn_chunk_first_kernel_v2.

    Args:
        q: [N_SEQS, N_HEADS, D_HEAD]
        k_chunks: list of [N_HEADS, CHUNK_SIZE, D_HEAD]
        v_chunks: list of [N_HEADS, CHUNK_SIZE, D_HEAD]
        softmax_scale: 1/sqrt(d_head)

    Returns:
        outputs: list of [N_HEADS, N_SEQS, D_HEAD] per chunk
        maxs: list of [N_HEADS, N_SEQS] per chunk
        sums: list of [N_HEADS, N_SEQS] per chunk
    """
    outputs_ref = []
    maxs_ref = []
    sums_ref = []

    n_chunks = len(k_chunks)
    n_heads = q.shape[1]

    for chunk_idx in range(n_chunks):
        k_chunk = k_chunks[chunk_idx]
        v_chunk = v_chunks[chunk_idx]

        chunk_outputs = []
        chunk_maxs = []
        chunk_sums = []

        for head_idx in range(n_heads):
            q_head = q[:, head_idx, :]  # [N_SEQS, D_HEAD]
            k_head = k_chunk[head_idx]   # [CHUNK_SIZE, D_HEAD]
            v_head = v_chunk[head_idx]   # [CHUNK_SIZE, D_HEAD]

            # Compute attention scores: [N_SEQS, CHUNK_SIZE]
            scores = torch.matmul(q_head, k_head.T) * softmax_scale

            # Compute max for numerical stability
            max_scores = scores.max(dim=1, keepdim=True)[0]  # [N_SEQS, 1]

            # Compute exp and sum (NOT normalized - matches kernel behavior)
            exp_scores = torch.exp(scores - max_scores)  # [N_SEQS, CHUNK_SIZE]
            sum_scores = exp_scores.sum(dim=1, keepdim=True)  # [N_SEQS, 1]

            # Compute attention output (NOT normalized - matches kernel behavior)
            attn_output = torch.matmul(exp_scores, v_head)

            chunk_outputs.append(attn_output)
            chunk_maxs.append(max_scores.squeeze())
            chunk_sums.append(sum_scores.squeeze())

        outputs_ref.append(torch.stack(chunk_outputs, dim=0))  # [N_HEADS, N_SEQS, D_HEAD]
        maxs_ref.append(torch.stack(chunk_maxs, dim=0))         # [N_HEADS, N_SEQS]
        sums_ref.append(torch.stack(chunk_sums, dim=0))         # [N_HEADS, N_SEQS]

    return outputs_ref, maxs_ref, sums_ref


def write_test_file(filename, q, k_chunks, v_chunks, outputs_ref, maxs_ref, sums_ref):
    """Write test data to file."""
    print(f"Writing to {filename}")

    with open(filename, 'w') as f:
        # Write Q: [N_SEQS, N_HEADS, D_HEAD]
        qf = q.flatten().detach().cpu().numpy()
        for i in trange(len(qf), desc="Writing Q", leave=False):
            f.write(f"{qf[i]} ")
        f.write('\n')

        # Write K chunks: N_CHUNKS * [N_HEADS, CHUNK_SIZE, D_HEAD]
        for chunk_idx, k_chunk in enumerate(k_chunks):
            kf = k_chunk.flatten().detach().cpu().numpy()
            for i in trange(len(kf), desc=f"Writing K chunk {chunk_idx}", leave=False):
                f.write(f"{kf[i]} ")
            f.write('\n')

        # Write V chunks: N_CHUNKS * [N_HEADS, CHUNK_SIZE, D_HEAD]
        for chunk_idx, v_chunk in enumerate(v_chunks):
            vf = v_chunk.flatten().detach().cpu().numpy()
            for i in trange(len(vf), desc=f"Writing V chunk {chunk_idx}", leave=False):
                f.write(f"{vf[i]} ")
            f.write('\n')

        # Write expected outputs: N_CHUNKS * [N_HEADS, N_SEQS, D_HEAD]
        for chunk_idx, out in enumerate(outputs_ref):
            of = out.flatten().detach().cpu().numpy()
            for i in trange(len(of), desc=f"Writing O chunk {chunk_idx}", leave=False):
                f.write(f"{of[i]} ")
            f.write('\n')

        # Write expected maxs: N_CHUNKS * [N_HEADS, N_SEQS]
        for chunk_idx, maxs in enumerate(maxs_ref):
            mf = maxs.flatten().detach().cpu().numpy()
            for i in trange(len(mf), desc=f"Writing maxs chunk {chunk_idx}", leave=False):
                f.write(f"{mf[i]} ")
            f.write('\n')

        # Write expected sums: N_CHUNKS * [N_HEADS, N_SEQS]
        for chunk_idx, sums in enumerate(sums_ref):
            sf = sums.flatten().detach().cpu().numpy()
            for i in trange(len(sf), desc=f"Writing sums chunk {chunk_idx}", leave=False):
                f.write(f"{sf[i]} ")
            f.write('\n')


def generate_test(test_name, seed=42):
    """Generate a single test case."""
    print(f"\n{'='*60}")
    print(f"Generating test: {test_name} (seed={seed})")
    print(f"N_SEQS={N_SEQS}, CHUNK_SIZE={CHUNK_SIZE}, D_HEAD={D_HEAD}, N_HEADS={N_HEADS}, N_CHUNKS={N_CHUNKS}")
    print('='*60)

    softmax_scale = 1.0 / math.sqrt(D_HEAD)
    torch.manual_seed(seed)

    if test_name == 'ones':
        # All ones - tests basic computation
        q = torch.ones((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
        k_chunks = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
        v_chunks = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]

    elif test_name == 'randn':
        # Standard random normal
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
        k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
        v_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]

    elif test_name == 'small_randn':
        # Small values - less numerical issues
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 0.1
        k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
        v_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]

    elif test_name == 'identity_v':
        # Identity-like V - helps debug output computation
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 0.1
        k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
        # V is eye-like: first D_HEAD rows are identity, rest are zeros
        v_chunks = []
        for _ in range(N_CHUNKS):
            v = torch.zeros((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda')
            for h in range(N_HEADS):
                eye_size = min(CHUNK_SIZE, D_HEAD)
                v[h, :eye_size, :eye_size] = torch.eye(eye_size, device='cuda')
            v_chunks.append(v)

    elif test_name == 'uniform':
        # Uniform distribution
        q = torch.rand((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1
        k_chunks = [torch.rand((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1 for _ in range(N_CHUNKS)]
        v_chunks = [torch.rand((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2 - 1 for _ in range(N_CHUNKS)]

    elif test_name == 'large_randn':
        # Larger values - stress test numerical stability
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 2.0
        k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2.0 for _ in range(N_CHUNKS)]
        v_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 2.0 for _ in range(N_CHUNKS)]

    elif test_name == 'mixed_chunks':
        # Different patterns per chunk
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
        k_chunks = []
        v_chunks = []
        for i in range(N_CHUNKS):
            if i % 2 == 0:
                k_chunks.append(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda'))
                v_chunks.append(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda'))
            else:
                k_chunks.append(torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.5)
                v_chunks.append(torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda'))

    elif test_name.startswith('seed_'):
        # Test with specific seed
        seed_num = int(test_name.split('_')[1])
        torch.manual_seed(seed_num)
        q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
        k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
        v_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]

    else:
        print(f'Unknown test name: {test_name}')
        return None

    # Compute reference outputs
    outputs_ref, maxs_ref, sums_ref = compute_chunk_attention(q, k_chunks, v_chunks, softmax_scale)

    # Verify reference computation looks reasonable
    print(f"  Q range: [{q.min():.4f}, {q.max():.4f}]")
    print(f"  Output range: [{outputs_ref[0].min():.4f}, {outputs_ref[0].max():.4f}]")
    print(f"  Max range: [{maxs_ref[0].min():.4f}, {maxs_ref[0].max():.4f}]")
    print(f"  Sum range: [{sums_ref[0].min():.4f}, {sums_ref[0].max():.4f}]")

    # Write to file
    filename = f'{test_name}_{N_SEQS}_{CHUNK_SIZE}_{D_HEAD}.txt'
    write_test_file(filename, q, k_chunks, v_chunks, outputs_ref, maxs_ref, sums_ref)

    return filename


def main():
    if len(sys.argv) > 1:
        # Generate specific test
        test_name = sys.argv[1]
        generate_test(test_name)
    else:
        # Generate all standard tests
        test_cases = [
            'randn',
            'ones',
            'small_randn',
            'identity_v',
            'uniform',
            'large_randn',
            'mixed_chunks',
            'seed_123',
            'seed_456',
            'seed_789',
        ]

        print("Generating all test cases...")
        generated_files = []
        for test_name in test_cases:
            try:
                filename = generate_test(test_name)
                if filename:
                    generated_files.append(filename)
            except Exception as e:
                print(f"ERROR generating {test_name}: {e}")

        print(f"\n{'='*60}")
        print(f"Generated {len(generated_files)} test files:")
        for f in generated_files:
            print(f"  {f}")
        print('='*60)

        # Write a test runner script
        with open('run_all_tests.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# Auto-generated test runner\n\n')
            f.write('PASSED=0\n')
            f.write('FAILED=0\n\n')
            for filename in generated_files:
                f.write(f'echo "\\n=== Running {filename} ==="\n')
                f.write(f'./attn_chunk_first {filename}\n')
                f.write('if [ $? -eq 0 ]; then\n')
                f.write('    ((PASSED++))\n')
                f.write('else\n')
                f.write('    ((FAILED++))\n')
                f.write('fi\n\n')
            f.write('echo "\\n========================================"\n')
            f.write('echo "RESULTS: $PASSED passed, $FAILED failed"\n')
            f.write('echo "========================================"\n')
            f.write('exit $FAILED\n')

        os.chmod('run_all_tests.sh', 0o755)
        print("\nCreated run_all_tests.sh - run with: ./run_all_tests.sh")


if __name__ == '__main__':
    main()
