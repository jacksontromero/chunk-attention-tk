# Chunked Attention ThunderKittens Kernel

This directory contains a ThunderKittens implementation of the chunked attention kernel for testing and benchmarking.

## Files

- `attn_chunk_first.cu` - Main kernel implementation
- `harness.impl` - Test harness for running the kernel
- `gentests.py` - Python script to generate test data
- `Makefile` - Build configuration

## Building

```bash
# Set THUNDERKITTENS_ROOT environment variable if not already set
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens

# Build the test binary
make
```

## Running Tests

1. Generate test data:
```bash
python gentests.py randn
# This creates: randn_16_64_128.txt
```

2. Run the test:
```bash
./attn_chunk_first randn_16_64_128.txt
```

## Test Configuration

The test is configured for:
- `N_SEQS = 16`: Number of sequences attending to each chunk
- `CHUNK_SIZE = 64`: Size of K/V chunks
- `D_HEAD = 128`: Head dimension
- `N_HEADS = 4`: Number of attention heads
- `N_CHUNKS = 2`: Number of chunks to test

These can be modified in both `gentests.py` and `harness.impl` (they must match).

## Test Types

- `randn`: Random normal data (default)
- `ones`: All ones
- `simple`: Simple test with smaller values and identity V matrices

## Implementation Notes

This ThunderKittens version makes several simplifying assumptions compared to the full CUDA implementation:

1. **Data type**: Currently uses `float` instead of `half` for all computations
2. **Precision**: Single precision throughout, not using double-buffer approach
3. **Memory**: Allocates separate buffers for K and V (not reusing memory)
4. **Simplified calling pattern**: Tests individual chunk attention without full pipeline integration

These simplifications are documented in code comments and are acceptable for testing the core kernel logic.

## Expected Output

The test checks three outputs against reference values:
- Attention outputs: `softmax(Q @ K.T) @ V`
- Max values: Maximum attention scores per row (for numerical stability)
- Sum values: Sum of exp(attention scores) per row (for normalization)

Success criteria:
- Maximum difference < 0.1 (loose tolerance due to float precision)
- No NaN values

## Troubleshooting

If the build fails:
- Ensure `THUNDERKITTENS_ROOT` is set correctly
- Check that you have CUDA toolkit installed
- Verify GPU architecture in Makefile matches your GPU

If tests fail:
- Check that test data was generated successfully
- Verify dimensions match between gentests.py and harness.impl
- Check shared memory limits for your GPU

