#!/bin/bash
# Run all tests and summarize results

PASSED=0
FAILED=0
FAILED_TESTS=""

./attn_chunk_first identity_v_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ identity_v_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ identity_v_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS identity_v_16_64_128.txt"
fi

./attn_chunk_first large_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ large_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ large_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS large_16_64_128.txt"
fi

./attn_chunk_first negative_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ negative_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ negative_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS negative_16_64_128.txt"
fi

./attn_chunk_first ones_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ ones_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ ones_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS ones_16_64_128.txt"
fi

./attn_chunk_first positive_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ positive_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ positive_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS positive_16_64_128.txt"
fi

./attn_chunk_first randn_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ randn_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ randn_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS randn_16_64_128.txt"
fi

./attn_chunk_first seed1000_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ seed1000_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ seed1000_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS seed1000_16_64_128.txt"
fi

./attn_chunk_first seed123_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ seed123_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ seed123_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS seed123_16_64_128.txt"
fi

./attn_chunk_first seed2000_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ seed2000_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ seed2000_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS seed2000_16_64_128.txt"
fi

./attn_chunk_first seed456_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ seed456_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ seed456_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS seed456_16_64_128.txt"
fi

./attn_chunk_first seed789_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ seed789_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ seed789_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS seed789_16_64_128.txt"
fi

./attn_chunk_first small_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ small_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ small_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS small_16_64_128.txt"
fi

./attn_chunk_first uniform_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ uniform_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ uniform_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS uniform_16_64_128.txt"
fi

./attn_chunk_first zeros_v_16_64_128.txt > /tmp/test_out.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ zeros_v_16_64_128.txt"
    ((PASSED++))
else
    echo "✗ zeros_v_16_64_128.txt"
    ((FAILED++))
    FAILED_TESTS="$FAILED_TESTS zeros_v_16_64_128.txt"
fi

echo ""
echo "================================"
echo "PASSED: $PASSED / $((PASSED + FAILED))"
if [ $FAILED -gt 0 ]; then
    echo "FAILED:$FAILED_TESTS"
    exit 1
fi
