#!/bin/bash
#
# run-all-tests.sh
#
# Runs NumericSwift tests across all compilation configurations:
# - Standalone (no optional dependencies)
# - With ArraySwift only
# - With PlotSwift only
# - With both ArraySwift and PlotSwift
#
# Usage: ./scripts/run-all-tests.sh [--verbose] [--skip-build] [--config CONFIG]
#
# Options:
#   --verbose     Show full test output
#   --skip-build  Skip the build step (run tests only)
#   --config      Run only specified config: standalone, arrayswift, plotswift, combined
#

set -e

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Default options
VERBOSE=false
SKIP_BUILD=false
SINGLE_CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --config|-c)
            SINGLE_CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--skip-build] [--config CONFIG]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Show full test output"
            echo "  --skip-build     Skip the build step (run tests only)"
            echo "  --config, -c     Run only specified config:"
            echo "                   standalone, arrayswift, plotswift, combined"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Track results
declare -A RESULTS
FAILED=0

# Function to run tests with specific configuration
run_tests() {
    local config_name=$1
    local arrayswift=$2
    local plotswift=$3

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running tests: ${config_name}${NC}"
    echo -e "${BLUE}  NUMERICSWIFT_INCLUDE_ARRAYSWIFT=${arrayswift}${NC}"
    echo -e "${BLUE}  NUMERICSWIFT_INCLUDE_PLOTSWIFT=${plotswift}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Set environment variables
    export NUMERICSWIFT_INCLUDE_ARRAYSWIFT=$arrayswift
    export NUMERICSWIFT_INCLUDE_PLOTSWIFT=$plotswift

    # Clean build directory for fresh compilation with new flags
    if [ "$SKIP_BUILD" = false ]; then
        echo "Cleaning build directory..."
        swift package clean 2>/dev/null || true
    fi

    # Build
    if [ "$SKIP_BUILD" = false ]; then
        echo "Building..."
        if [ "$VERBOSE" = true ]; then
            if ! swift build 2>&1; then
                echo -e "${RED}Build failed for ${config_name}${NC}"
                RESULTS[$config_name]="BUILD_FAILED"
                FAILED=$((FAILED + 1))
                return 1
            fi
        else
            if ! swift build 2>&1 | tail -5; then
                echo -e "${RED}Build failed for ${config_name}${NC}"
                RESULTS[$config_name]="BUILD_FAILED"
                FAILED=$((FAILED + 1))
                return 1
            fi
        fi
    fi

    # Run tests
    echo "Running tests..."
    local test_output
    local test_exit_code

    if [ "$VERBOSE" = true ]; then
        swift test 2>&1
        test_exit_code=$?
    else
        test_output=$(swift test 2>&1)
        test_exit_code=$?
        # Show summary
        echo "$test_output" | grep -E "(Test Suite|Executed|passed|failed|error:)" || true
    fi

    if [ $test_exit_code -eq 0 ]; then
        echo -e "${GREEN}PASSED: ${config_name}${NC}"
        RESULTS[$config_name]="PASSED"
    else
        echo -e "${RED}FAILED: ${config_name}${NC}"
        if [ "$VERBOSE" = false ]; then
            echo "Run with --verbose for full output"
        fi
        RESULTS[$config_name]="FAILED"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    return $test_exit_code
}

# Verify we're in the right directory
if [ ! -f "Package.swift" ]; then
    echo -e "${RED}Error: Must be run from NumericSwift root directory${NC}"
    exit 1
fi

# Verify optional dependencies exist
ARRAYSWIFT_PATH="../ArraySwift"
PLOTSWIFT_PATH="../PlotSwift"

ARRAYSWIFT_AVAILABLE=true
PLOTSWIFT_AVAILABLE=true

if [ ! -d "$ARRAYSWIFT_PATH" ]; then
    echo -e "${YELLOW}Warning: ArraySwift not found at ${ARRAYSWIFT_PATH}${NC}"
    echo -e "${YELLOW}Skipping ArraySwift-related tests${NC}"
    ARRAYSWIFT_AVAILABLE=false
fi

if [ ! -d "$PLOTSWIFT_PATH" ]; then
    echo -e "${YELLOW}Warning: PlotSwift not found at ${PLOTSWIFT_PATH}${NC}"
    echo -e "${YELLOW}Skipping PlotSwift-related tests${NC}"
    PLOTSWIFT_AVAILABLE=false
fi

echo ""
echo -e "${BLUE}NumericSwift Test Runner${NC}"
echo "========================="
echo ""

# Run configurations based on single config option or all
if [ -n "$SINGLE_CONFIG" ]; then
    case $SINGLE_CONFIG in
        standalone)
            run_tests "Standalone" "0" "0" || true
            ;;
        arrayswift)
            if [ "$ARRAYSWIFT_AVAILABLE" = true ]; then
                run_tests "ArraySwift" "1" "0" || true
            else
                echo -e "${YELLOW}Skipping ArraySwift config (not available)${NC}"
            fi
            ;;
        plotswift)
            if [ "$PLOTSWIFT_AVAILABLE" = true ]; then
                run_tests "PlotSwift" "0" "1" || true
            else
                echo -e "${YELLOW}Skipping PlotSwift config (not available)${NC}"
            fi
            ;;
        combined)
            if [ "$ARRAYSWIFT_AVAILABLE" = true ] && [ "$PLOTSWIFT_AVAILABLE" = true ]; then
                run_tests "Combined" "1" "1" || true
            else
                echo -e "${YELLOW}Skipping Combined config (dependencies not available)${NC}"
            fi
            ;;
        *)
            echo -e "${RED}Unknown config: $SINGLE_CONFIG${NC}"
            echo "Valid configs: standalone, arrayswift, plotswift, combined"
            exit 1
            ;;
    esac
else
    # Run all available configurations
    run_tests "Standalone" "0" "0" || true

    if [ "$ARRAYSWIFT_AVAILABLE" = true ]; then
        run_tests "ArraySwift" "1" "0" || true
    fi

    if [ "$PLOTSWIFT_AVAILABLE" = true ]; then
        run_tests "PlotSwift" "0" "1" || true
    fi

    if [ "$ARRAYSWIFT_AVAILABLE" = true ] && [ "$PLOTSWIFT_AVAILABLE" = true ]; then
        run_tests "Combined" "1" "1" || true
    fi
fi

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"

for config in "${!RESULTS[@]}"; do
    result=${RESULTS[$config]}
    if [ "$result" = "PASSED" ]; then
        echo -e "  ${GREEN}$config: $result${NC}"
    else
        echo -e "  ${RED}$config: $result${NC}"
    fi
done

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED configuration(s) failed${NC}"
    exit 1
fi
