import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import sys
import math

# Simple test configuration for chunked attention
# We'll test with a single chunk scenario first
N_SEQS = 16      # number of sequences attending to chunk
CHUNK_SIZE = 64  # size of KV chunk
D_HEAD = 128     # head dimension
N_HEADS = 4      # number of attention heads
N_CHUNKS = 2     # number of chunks to test

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn'

print(f"Generating test: {TESTNAME}")
print(f"N_SEQS={N_SEQS}, CHUNK_SIZE={CHUNK_SIZE}, D_HEAD={D_HEAD}, N_HEADS={N_HEADS}, N_CHUNKS={N_CHUNKS}")

softmax_scale = 1.0 / math.sqrt(D_HEAD)

if TESTNAME == 'ones':
    torch.manual_seed(42)
    q = torch.ones((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
    k_chunks = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    v_chunks = [torch.ones((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
elif TESTNAME == 'randn':
    torch.manual_seed(42)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda')
    k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
    v_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') for _ in range(N_CHUNKS)]
elif TESTNAME == 'simple':
    torch.manual_seed(42)
    q = torch.randn((N_SEQS, N_HEADS, D_HEAD), dtype=torch.float32, device='cuda') * 0.1
    k_chunks = [torch.randn((N_HEADS, CHUNK_SIZE, D_HEAD), dtype=torch.float32, device='cuda') * 0.1 for _ in range(N_CHUNKS)]
    v_chunks = [torch.eye(D_HEAD)[:CHUNK_SIZE].unsqueeze(0).repeat(N_HEADS, 1, 1).cuda() for _ in range(N_CHUNKS)]
else:
    print('Invalid test name')
    sys.exit(1)

# Compute reference output for each chunk
# For each chunk, compute: softmax(Q @ K.T / sqrt(d)) @ V
outputs_ref = []
maxs_ref = []
sums_ref = []

for chunk_idx in range(N_CHUNKS):
    k_chunk = k_chunks[chunk_idx]  # [N_HEADS, CHUNK_SIZE, D_HEAD]
    v_chunk = v_chunks[chunk_idx]  # [N_HEADS, CHUNK_SIZE, D_HEAD]

    chunk_outputs = []
    chunk_maxs = []
    chunk_sums = []

    for head_idx in range(N_HEADS):
        q_head = q[:, head_idx, :]  # [N_SEQS, D_HEAD]
        k_head = k_chunk[head_idx]  # [CHUNK_SIZE, D_HEAD]
        v_head = v_chunk[head_idx]  # [CHUNK_SIZE, D_HEAD]

        # Compute attention scores: [N_SEQS, CHUNK_SIZE]
        scores = torch.matmul(q_head, k_head.T) * softmax_scale

        # Compute max for numerical stability
        max_scores = scores.max(dim=1, keepdim=True)[0]  # [N_SEQS, 1]

        # Compute exp and sum
        exp_scores = torch.exp(scores - max_scores)  # [N_SEQS, CHUNK_SIZE]
        sum_scores = exp_scores.sum(dim=1, keepdim=True)  # [N_SEQS, 1]

        # Compute attention output: [N_SEQS, D_HEAD]
        attn_output = torch.matmul(exp_scores, v_head)

        chunk_outputs.append(attn_output)
        chunk_maxs.append(max_scores.squeeze())
        chunk_sums.append(sum_scores.squeeze())

    outputs_ref.append(torch.stack(chunk_outputs, dim=0))  # [N_HEADS, N_SEQS, D_HEAD]
    maxs_ref.append(torch.stack(chunk_maxs, dim=0))  # [N_HEADS, N_SEQS]
    sums_ref.append(torch.stack(chunk_sums, dim=0))  # [N_HEADS, N_SEQS]

# Write test data to file
fn = f'{TESTNAME}_{N_SEQS}_{CHUNK_SIZE}_{D_HEAD}.txt'
print(f"Writing to {fn}")

with open(fn, 'w') as f:
    # Write Q: [N_SEQS, N_HEADS, D_HEAD]
    qf = q.flatten().detach().cpu().numpy()
    for i in trange(len(qf), desc="Writing Q"):
        f.write(f"{qf[i]} ")
    f.write('\n')

    # Write K chunks: N_CHUNKS * [N_HEADS, CHUNK_SIZE, D_HEAD]
    for chunk_idx in range(N_CHUNKS):
        kf = k_chunks[chunk_idx].flatten().detach().cpu().numpy()
        for i in trange(len(kf), desc=f"Writing K chunk {chunk_idx}"):
            f.write(f"{kf[i]} ")
        f.write('\n')

    # Write V chunks: N_CHUNKS * [N_HEADS, CHUNK_SIZE, D_HEAD]
    for chunk_idx in range(N_CHUNKS):
        vf = v_chunks[chunk_idx].flatten().detach().cpu().numpy()
        for i in trange(len(vf), desc=f"Writing V chunk {chunk_idx}"):
            f.write(f"{vf[i]} ")
        f.write('\n')

    # Write expected outputs: N_CHUNKS * [N_HEADS, N_SEQS, D_HEAD]
    for chunk_idx in range(N_CHUNKS):
        of = outputs_ref[chunk_idx].flatten().detach().cpu().numpy()
        for i in trange(len(of), desc=f"Writing O chunk {chunk_idx}"):
            f.write(f"{of[i]} ")
        f.write('\n')

    # Write expected maxs: N_CHUNKS * [N_HEADS, N_SEQS]
    for chunk_idx in range(N_CHUNKS):
        mf = maxs_ref[chunk_idx].flatten().detach().cpu().numpy()
        for i in trange(len(mf), desc=f"Writing maxs chunk {chunk_idx}"):
            f.write(f"{mf[i]} ")
        f.write('\n')

    # Write expected sums: N_CHUNKS * [N_HEADS, N_SEQS]
    for chunk_idx in range(N_CHUNKS):
        sf = sums_ref[chunk_idx].flatten().detach().cpu().numpy()
        for i in trange(len(sf), desc=f"Writing sums chunk {chunk_idx}"):
            f.write(f"{sf[i]} ")
        f.write('\n')

print(f"Test data written to {fn}")
print(f"Total elements written: Q={len(qf)}, K={len(kf)*N_CHUNKS}, V={len(vf)*N_CHUNKS}")

