#!/bin/bash
# Run all no-learning tests for all schedule permutations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running All No-Learning Test Permutations"
echo "=========================================="
echo ""

# Run all permutations: (a1,a2), (a1,a3), (a2,a1), (a2,a3), (a3,a1), (a3,a2)
# Note: (a1,a1) etc. would have KL=0, so we skip those

echo "Running (a1, a2)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a1 --schedule_q a2 --skip_ode

echo ""
echo "Running (a1, a3)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a1 --schedule_q a3 --skip_ode

echo ""
echo "Running (a2, a1)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a2 --schedule_q a1 --skip_ode

echo ""
echo "Running (a2, a3)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a2 --schedule_q a3 --skip_ode

echo ""
echo "Running (a3, a1)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a3 --schedule_q a1 --skip_ode

echo ""
echo "Running (a3, a2)..."
python "$SCRIPT_DIR/nolearning_test.py" --schedule_p a3 --schedule_q a2 --skip_ode

echo ""
echo "=========================================="
echo "All no-learning tests complete!"
echo "=========================================="

