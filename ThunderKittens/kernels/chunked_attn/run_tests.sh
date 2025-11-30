#!/bin/bash
# Fast parallel test runner with streaming output

# Number of parallel jobs (adjust based on GPU memory)
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

# Get list of test files
TESTS=(tests/*.txt)
TOTAL=${#TESTS[@]}

echo "Running $TOTAL tests with $PARALLEL_JOBS parallel jobs..."
echo ""

# Temp file for results
RESULTS_FILE=$(mktemp)
trap "rm -f $RESULTS_FILE" EXIT

# Function to run a single test
run_test() {
    local TEST_FILE="$1"
    local TEST_NAME=$(basename "$TEST_FILE")

    # Run test, capture output in memory
    OUTPUT=$(./attn_chunk_first "$TEST_FILE" 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $TEST_NAME"
        echo "PASS" >> "$RESULTS_FILE"
    else
        # Extract summary
        SUMMARY=$(echo "$OUTPUT" | grep "^SUMMARY:" | head -1)
        echo "✗ $TEST_NAME - ${SUMMARY#SUMMARY: }"
        echo "FAIL:$TEST_NAME" >> "$RESULTS_FILE"
    fi
}
export -f run_test
export RESULTS_FILE

# Run tests in parallel, output streams as jobs complete
printf '%s\n' "${TESTS[@]}" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_test "$@"' _ {}

# Count results
PASSED=$(grep -c "^PASS" "$RESULTS_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -c "^FAIL" "$RESULTS_FILE" 2>/dev/null || echo 0)

echo ""
echo "================================"
echo "PASSED: $PASSED / $((PASSED + FAILED))"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    grep "^FAIL:" "$RESULTS_FILE" | cut -d: -f2 | sed 's/^/  /'
    exit 1
else
    echo "All tests passed!"
fi
