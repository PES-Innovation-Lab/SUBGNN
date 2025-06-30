#!/bin/bash

# Run Glasgow Subgraph Solver tests
# This script runs the subgraph solver for all combinations of subgraph and target graph files

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if the solver executable exists
if [ ! -f "./build/glasgow_subgraph_solver" ]; then
    echo "Error: glasgow_subgraph_solver not found in ./build/"
    echo "Please build the project first using 'make' or 'cmake --build build'"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p results

# Function to run a single test
run_test() {
    local subgraph_file="$1"
    local target_file="$2"
    local result_file="$3"
    
    echo "Running: subgraph=$subgraph_file, target=$target_file"
    
    # Run the solver and capture output
    timeout 300s ./build/glasgow_subgraph_solver "$subgraph_file" "$target_file" > "$result_file" 2>&1 --format lad
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT" >> "$result_file"
        echo "  -> TIMEOUT"
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR (exit code: $exit_code)" >> "$result_file"
        echo "  -> ERROR"
    else
        echo "  -> COMPLETED"
    fi
}

# Main execution
echo "Starting Glasgow Subgraph Solver tests..."
echo "Subgraph directory: finalsubgraph/"
echo "Target graph directory: finaltargetgraph/"
echo "Results will be saved in: results/"
echo ""

# Count total number of tests
total_subgraphs=$(ls finalsubgraph/subgraph*.lad 2>/dev/null | wc -l)
total_targets=$(ls finaltargetgraph/targetgraphs*.lad 2>/dev/null | wc -l)
total_tests=$((total_subgraphs * total_targets))

echo "Found $total_subgraphs subgraph files and $total_targets target graph files"
echo "Total tests to run: $total_tests"
echo ""

# Initialize counters
test_count=0
start_time=$(date +%s)

# Run tests for all combinations
for subgraph_file in finalsubgraph/subgraph*.lad; do
    if [ ! -f "$subgraph_file" ]; then
        echo "No subgraph files found in finalsubgraph/ directory"
        exit 1
    fi
    
    # Extract subgraph number from filename
    subgraph_name=$(basename "$subgraph_file" .lad)
    
    for target_file in finaltargetgraph/targetgraphs*.lad; do
        if [ ! -f "$target_file" ]; then
            echo "No target graph files found in finaltargetgraph/ directory"
            exit 1
        fi
        
        # Extract target graph number from filename
        target_name=$(basename "$target_file" .lad)
        
        # Create result filename
        result_file="results/${subgraph_name}_vs_${target_name}.out"
        
        # Run the test
        run_test "$subgraph_file" "$target_file" "$result_file"
        
        # Update progress
        test_count=$((test_count + 1))
        if [ $((test_count % 100)) -eq 0 ]; then
            current_time=$(date +%s)
            elapsed=$((current_time - start_time))
            echo "Progress: $test_count/$total_tests tests completed (${elapsed}s elapsed)"
        fi
    done
done

# Final summary
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo ""
echo "All tests completed!"
echo "Total tests run: $test_count"
echo "Total time: ${total_time}s"
echo "Results saved in: results/"

# Generate summary report
echo ""
echo "Generating summary report..."
summary_file="results/summary.txt"
{
    echo "Glasgow Subgraph Solver Test Summary"
    echo "Generated on: $(date)"
    echo "Total tests: $test_count"
    echo "Total time: ${total_time}s"
    echo ""
    echo "Result breakdown:"
    echo "Completed: $(grep -l "COMPLETED" results/*.out 2>/dev/null | wc -l)"
    echo "Timeouts: $(grep -l "TIMEOUT" results/*.out 2>/dev/null | wc -l)"
    echo "Errors: $(grep -l "ERROR" results/*.out 2>/dev/null | wc -l)"
} > "$summary_file"

echo "Summary report saved to: $summary_file"

