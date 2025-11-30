#!/bin/bash
# Auto-generated test runner

PASSED=0
FAILED=0

echo "\n=== Running randn_16_64_128.txt ==="
./attn_chunk_first randn_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running ones_16_64_128.txt ==="
./attn_chunk_first ones_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running small_randn_16_64_128.txt ==="
./attn_chunk_first small_randn_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running identity_v_16_64_128.txt ==="
./attn_chunk_first identity_v_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running uniform_16_64_128.txt ==="
./attn_chunk_first uniform_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running large_randn_16_64_128.txt ==="
./attn_chunk_first large_randn_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running mixed_chunks_16_64_128.txt ==="
./attn_chunk_first mixed_chunks_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running seed_123_16_64_128.txt ==="
./attn_chunk_first seed_123_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running seed_456_16_64_128.txt ==="
./attn_chunk_first seed_456_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n=== Running seed_789_16_64_128.txt ==="
./attn_chunk_first seed_789_16_64_128.txt
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi

echo "\n========================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "========================================"
exit $FAILED
