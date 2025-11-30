#!/bin/bash
# Auto-generated test runner
# Usage: ./run_tests.sh [JOBS]   (default: 4 parallel jobs)

JOBS=${1:-4}
TOTAL=36
RESULTS_FILE=$(mktemp)

echo "Running $TOTAL tests with $JOBS parallel jobs..."
echo ""

# Function to run a single test and output result (streams to terminal)
run_test() {
    TEST="$1"
    RESULTS_FILE="$2"
    NAME=$(basename "$TEST")
    OUTPUT=$(./attn_chunk_first "$TEST" 2>&1)
    if [ $? -eq 0 ]; then
        echo "✓ $NAME"
        echo "PASS" >> "$RESULTS_FILE"
    else
        SUMMARY=$(echo "$OUTPUT" | grep "SUMMARY:" | head -1)
        echo "✗ $NAME - $SUMMARY"
        echo "FAIL:$NAME" >> "$RESULTS_FILE"
    fi
}
export -f run_test

# Run tests in parallel, streaming output
cat << 'TESTLIST' | xargs -P "$JOBS" -I {} bash -c 'run_test "{}" "'$RESULTS_FILE'"'
tests/randn_s16_h1_c1.txt
tests/randn_s16_h1_c2.txt
tests/randn_s16_h1_c4.txt
tests/randn_s16_h4_c1.txt
tests/randn_s16_h4_c2.txt
tests/randn_s16_h4_c4.txt
tests/randn_s32_h1_c1.txt
tests/randn_s32_h1_c2.txt
tests/randn_s32_h1_c4.txt
tests/randn_s32_h4_c1.txt
tests/randn_s32_h4_c2.txt
tests/randn_s32_h4_c4.txt
tests/randn_s8_h1_c1.txt
tests/randn_s8_h1_c2.txt
tests/randn_s8_h1_c4.txt
tests/randn_s8_h4_c1.txt
tests/randn_s8_h4_c2.txt
tests/randn_s8_h4_c4.txt
tests/small_s16_h1_c1.txt
tests/small_s16_h1_c2.txt
tests/small_s16_h1_c4.txt
tests/small_s16_h4_c1.txt
tests/small_s16_h4_c2.txt
tests/small_s16_h4_c4.txt
tests/small_s32_h1_c1.txt
tests/small_s32_h1_c2.txt
tests/small_s32_h1_c4.txt
tests/small_s32_h4_c1.txt
tests/small_s32_h4_c2.txt
tests/small_s32_h4_c4.txt
tests/small_s8_h1_c1.txt
tests/small_s8_h1_c2.txt
tests/small_s8_h1_c4.txt
tests/small_s8_h4_c1.txt
tests/small_s8_h4_c2.txt
tests/small_s8_h4_c4.txt
TESTLIST

# Count results
PASSED=$(grep -c "^PASS$" "$RESULTS_FILE" || true)
PASSED=${PASSED:-0}
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
