// attn_seq_first_kernel

/**
 * ThunderKittens implementation of attn_seq_first_kernel
 *
 * Original: chunk_attn/cpp/chunk_attn/kernel_cuda.cu (lines 886-1087)
 *
 * Processes attention for one query sequence across all its KV chunks:
 *   - For shared chunks: merges cached partial results from chunk_first
 *   - For unique chunks: computes Q @ K^T, softmax, scores @ V
 *   - Uses online softmax to accumulate across chunks
 *   - Finally normalizes and outputs the attention result
 *
 * Grid: (n_heads, n_seqs)
 * Block: 128 threads (4 warps), each warp processes chunks in round-robin
 *
 * Differences from original:
 *   - Uses bf16 instead of half
 *   - Uses TK primitives for vector-matrix operations
 *   - Uses TK primitives for reductions (max, sum, exp)
 */

#include "kittens.cuh"
#include "../../include/kittens.cuh"
using namespace kittens;

// Configuration: 4 warps for chunk parallelism (matching original)
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

/**
 * Kernel globals - mirrors original kernel parameters
 */
template<int chunk_size, int d_head>
struct attn_seq_first_globals {
    // Final output
    gl<bf16, -1, -1, -1, -1> output;    // [n_seqs, n_heads, d_head]

    // Cached partial results from chunk_first kernel
    gl<bf16, -1, -1, -1, -1> attns;     // partial attention outputs
    gl<float, -1, -1, -1, -1> maxs;     // row maxima
    gl<float, -1, -1, -1, -1> sums;     // row sums

    // Chunk metadata
    gl<int, -1, -1, -1, -1> offsets;    // [n_chunks] output offset per chunk
    gl<int, -1, -1, -1, -1> begins;     // [n_chunks] seq range start
    gl<int, -1, -1, -1, -1> ends;       // [n_chunks] seq range end

    // Inputs
    const bf16* Q;                       // [n_seqs, n_heads, d_head]
    void** keys;                         // [n_chunks] -> [n_heads, chunk_size, d_head]
    void** values;                       // [n_chunks] -> [n_heads, chunk_size, d_head]

    // Sequence-to-chunk mapping
    const int* seq_chunk_map;            // [n_seqs, max_chunks_per_seq]
    const int* seq_n_tokens;             // [n_seqs] number of tokens per sequence
    int seq_chunk_map_stride;            // stride for seq_chunk_map

    float scale;                         // 1/sqrt(d_head)
    int n_heads;
    int n_shared_chunks;                 // chunks with cached results
    int delta_tokens;                    // token offset adjustment
};

template<int chunk_size, int d_head>
__global__ void __launch_bounds__(BLOCK_SIZE)
attn_seq_first_tk(const __grid_constant__ attn_seq_first_globals<chunk_size, d_head> g) {

    // Thread indexing
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int warp = kittens::warpid();
    const int lane = kittens::laneid();

    // Compute sequence length and chunk count
    const int seq_length = g.seq_n_tokens[seq_idx] + g.delta_tokens;
    const int last_chunk_unmask_tokens = seq_length % chunk_size;
    const int chunk_num = (seq_length + chunk_size - 1) / chunk_size;

    // Offsets for Q and KV access
    const int q_row_offset = seq_idx * g.n_heads * d_head + head_idx * d_head;
    const int kv_row_offset = head_idx * chunk_size * d_head;

    // Pointers
    const bf16* q_ptr = g.Q + q_row_offset;
    bf16* output_ptr = g.output.raw_ptr + q_row_offset;
    const int* seq_mapping = g.seq_chunk_map + seq_idx * g.seq_chunk_map_stride;

    // =========================================================================
    // Shared memory allocation
    // =========================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Q vector shared by all warps
    sv<bf16, d_head> &Q_sv = al.allocate<sv<bf16, d_head>>();

    // Per-warp output accumulators (in bf16 for merging)
    auto &out_sv = al.allocate<sv<bf16, d_head>, NUM_WARPS>()[warp];

    // Per-warp attention scores scratch
    auto &scores_sv = al.allocate<sv<bf16, chunk_size>, NUM_WARPS>()[warp];

    // Per-warp KV tile scratch (can be reused for K then V)
    auto &KV_s = al.allocate<st<bf16, chunk_size, d_head>, NUM_WARPS>()[warp];

    // Per-warp max and sum (only need one value per warp, but use array for sync)
    __shared__ float warp_max[NUM_WARPS];
    __shared__ float warp_sum[NUM_WARPS];

    // =========================================================================
    // Load Q into shared memory (all threads cooperate)
    // =========================================================================
    for (int i = threadIdx.x; i < d_head; i += BLOCK_SIZE) {
        Q_sv[i] = q_ptr[i];
    }

    // Initialize per-warp output to zero
    for (int i = lane; i < d_head; i += 32) {
        out_sv[i] = __float2bfloat16(0.0f);
    }

    // Initialize per-warp max and sum
    if (lane == 0) {
        warp_max[warp] = -INFINITY;
        warp_sum[warp] = 0.0f;
    }
    __syncthreads();

    // Per-warp running state (in registers)
    float score_max = -INFINITY;
    float score_sum = 0.0f;

    // =========================================================================
    // Chunk loop: each warp processes chunks in round-robin
    // Warp 0: chunks 0, 4, 8, ...
    // Warp 1: chunks 1, 5, 9, ...
    // etc.
    // =========================================================================
    for (int i = warp; i < chunk_num; i += NUM_WARPS) {
        const int chunk_idx = seq_mapping[i];

        // ---------------------------------------------------------------------
        // PATH A: Merge cached results from chunk_first kernel
        // ---------------------------------------------------------------------
        if (chunk_idx < g.n_shared_chunks) {
            // Compute offset into cached results
            const int result_offset = g.offsets[chunk_idx];
            const int seq_begin = g.begins[chunk_idx];
            const int seq_end = g.ends[chunk_idx];
            const int max_sum_offset = result_offset * g.n_heads +
                                       head_idx * (seq_end - seq_begin) +
                                       seq_idx - seq_begin;
            const int attn_offset = max_sum_offset * d_head;

            // Load cached max, sum (lane 0 loads, then broadcasts)
            float cached_max = 0.0f, cached_sum = 0.0f;
            if (lane == 0) {
                cached_max = g.maxs.raw_ptr[max_sum_offset];
                cached_sum = g.sums.raw_ptr[max_sum_offset];
            }
            cached_max = __shfl_sync(0xffffffff, cached_max, 0);
            cached_sum = __shfl_sync(0xffffffff, cached_sum, 0);

            // Compute new max and scales for online softmax merge
            float new_score_max = fmaxf(score_max, cached_max);
            float cached_scale = __expf(cached_max - new_score_max);
            float scale_old = __expf(score_max - new_score_max);

            // Update running state
            score_max = new_score_max;
            score_sum = cached_sum * cached_scale + score_sum * scale_old;

            // Merge cached attention output into running output
            // out_sv = out_sv * scale_old + cached_attn * cached_scale
            const bf16* cached_attn = g.attns.raw_ptr + attn_offset;
            for (int j = lane; j < d_head; j += 32) {
                float old_val = __bfloat162float(out_sv[j]) * scale_old;
                float cached_val = __bfloat162float(cached_attn[j]) * cached_scale;
                out_sv[j] = __float2bfloat16(old_val + cached_val);
            }
            __syncwarp();
            continue;
        }

        // ---------------------------------------------------------------------
        // PATH B: Compute fresh attention for non-shared chunk
        // ---------------------------------------------------------------------
        const bf16* K_ptr = reinterpret_cast<const bf16*>(g.keys[chunk_idx]) + kv_row_offset;
        const bf16* V_ptr = reinterpret_cast<const bf16*>(g.values[chunk_idx]) + kv_row_offset;

        // Load K into shared memory for this warp
        for (int j = lane; j < chunk_size * d_head; j += 32) {
            KV_s[{j / d_head, j % d_head}] = K_ptr[j];
        }
        __syncwarp();

        // ---------------------------------------------------------------------
        // Compute Q @ K^T using broadcast pattern
        // Q is [d_head], K is [chunk_size, d_head], result is [chunk_size]
        // For each k in chunk_size: score[k] = sum_d(Q[d] * K[k,d])
        // ---------------------------------------------------------------------
        // We'll compute this using register operations
        // Load Q into register vector
        rv<bf16, d_head, ducks::rv_layout::naive> Q_rv;
        for (int j = 0; j < d_head; j += 32) {
            if (j + lane < d_head) {
                Q_rv[j / 32][0] = Q_sv[j + lane];
            }
        }

        // Compute dot products: score[k] = Q . K[k, :]
        float chunk_scores[chunk_size / 32];  // Each thread handles chunk_size/32 scores
        #pragma unroll
        for (int k = 0; k < chunk_size / 32; k++) {
            chunk_scores[k] = 0.0f;
        }

        // Each thread computes partial dot products, then reduce
        for (int k_base = 0; k_base < chunk_size; k_base += 32) {
            int k = k_base + lane;  // This thread handles score[k]
            if (k < chunk_size) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < d_head; d++) {
                    dot += __bfloat162float(Q_sv[d]) * __bfloat162float(KV_s[{k, d}]);
                }
                chunk_scores[k_base / 32] = dot * g.scale;
            }
        }

        // Apply causal mask for last chunk
        if (i == chunk_num - 1 && last_chunk_unmask_tokens != 0) {
            #pragma unroll
            for (int k = 0; k < chunk_size / 32; k++) {
                int global_k = k * 32 + lane;
                if (global_k >= last_chunk_unmask_tokens) {
                    chunk_scores[k] = -INFINITY;
                }
            }
        }

        // ---------------------------------------------------------------------
        // Softmax: find max, compute exp, find sum
        // ---------------------------------------------------------------------
        // Find chunk max (include running max for stability)
        float chunk_max = score_max;
        #pragma unroll
        for (int k = 0; k < chunk_size / 32; k++) {
            chunk_max = fmaxf(chunk_max, chunk_scores[k]);
        }

        // Warp reduce max
        #pragma unroll
        for (int mask = 16; mask >= 1; mask /= 2) {
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffff, chunk_max, mask));
        }

        // Compute exp(score - max) and sum
        float chunk_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < chunk_size / 32; k++) {
            chunk_scores[k] = __expf(chunk_scores[k] - chunk_max);
            chunk_sum += chunk_scores[k];
        }

        // Warp reduce sum
        #pragma unroll
        for (int mask = 16; mask >= 1; mask /= 2) {
            chunk_sum += __shfl_xor_sync(0xffffffff, chunk_sum, mask);
        }

        // Store scores to shared memory for scores @ V computation
        #pragma unroll
        for (int k = 0; k < chunk_size / 32; k++) {
            int global_k = k * 32 + lane;
            if (global_k < chunk_size) {
                scores_sv[global_k] = __float2bfloat16(chunk_scores[k]);
            }
        }
        __syncwarp();

        // Compute scale for merging old results
        float scale_old = __expf(score_max - chunk_max);

        // Update running state
        score_max = chunk_max;
        score_sum = score_sum * scale_old + chunk_sum;

        // ---------------------------------------------------------------------
        // Compute scores @ V and merge into output
        // scores is [chunk_size], V is [chunk_size, d_head], result is [d_head]
        // output[d] = scale_old * output[d] + sum_k(scores[k] * V[k,d])
        // ---------------------------------------------------------------------
        // Load V into shared memory (reuse KV_s)
        for (int j = lane; j < chunk_size * d_head; j += 32) {
            KV_s[{j / d_head, j % d_head}] = V_ptr[j];
        }
        __syncwarp();

        // Compute scores @ V for each output dimension
        for (int d = lane; d < d_head; d += 32) {
            float result = 0.0f;
            #pragma unroll
            for (int k = 0; k < chunk_size; k++) {
                result += __bfloat162float(scores_sv[k]) * __bfloat162float(KV_s[{k, d}]);
            }
            // Merge into output with scaling
            float old_val = __bfloat162float(out_sv[d]) * scale_old;
            out_sv[d] = __float2bfloat16(old_val + result);
        }
        __syncwarp();
    }

    // Store final per-warp max and sum for cross-warp reduction
    if (lane == 0) {
        warp_max[warp] = score_max;
        warp_sum[warp] = score_sum;
    }
    __syncthreads();

    // =========================================================================
    // Cross-warp reduction: combine results from all 4 warps
    // =========================================================================
    float scale[NUM_WARPS];
    float total_sum = 0.0f;

    // Compute global max and scales (one thread does this, then broadcasts)
    if (lane == 0) {
        float global_max = warp_max[0];
        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++) {
            global_max = fmaxf(global_max, warp_max[w]);
        }

        total_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            scale[w] = __expf(warp_max[w] - global_max);
            total_sum += warp_sum[w] * scale[w];
        }
    }
    __syncwarp();

    // Broadcast scales and total_sum to all threads in warp
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) {
        scale[w] = __shfl_sync(0xffffffff, scale[w], 0);
    }
    total_sum = __shfl_sync(0xffffffff, total_sum, 0);

    float inv_sum = __fdividef(1.0f, total_sum + 1e-6f);

    // =========================================================================
    // Normalize and store final output
    // output[d] = sum_w(out_sv[w][d] * scale[w]) / total_sum
    // =========================================================================
    // Get pointers to all warp outputs for reduction
    sv<bf16, d_head>* all_out_sv = reinterpret_cast<sv<bf16, d_head>*>(
        &al.allocate<sv<bf16, d_head>, NUM_WARPS>()[0]) - NUM_WARPS;  // Rewind to get base

    // Actually we need to access shared memory differently - use explicit addressing
    // The out_sv arrays are contiguous in shared memory
    extern __shared__ char smem_raw[];
    // Skip Q_sv (d_head * sizeof(bf16)) to get to out_sv base
    bf16* out_sv_base = reinterpret_cast<bf16*>(smem_raw + sizeof(sv<bf16, d_head>));

    for (int d = threadIdx.x; d < d_head; d += BLOCK_SIZE) {
        float output_val = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            bf16* warp_out = out_sv_base + w * d_head;
            output_val += __bfloat162float(warp_out[d]) * scale[w];
        }
        output_ptr[d] = __float2bfloat16(output_val * inv_sum);
    }
}

#include "harness.impl"
