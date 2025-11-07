#!/bin/bash
# Run all cross-schedule evaluation experiments
# Uses trained models to evaluate KL identity across different schedules

echo "=========================================="
echo "Cross-Schedule KL Identity Evaluation"
echo "=========================================="
echo ""

MODEL_A1="data/models/vtheta_schedule_a1_mse_0-05_20251027_231222.pth"
MODEL_A2="data/models/vtheta_schedule_a2_mse_0-05_20251027_231335.pth"
MODEL_A3="data/models/vtheta_schedule_a3_mse_0-05_20251027_231356.pth"

# Run experiments: schedule_i vs trained schedule_j where i ≠ j

echo "Running (a1 vs a2 model) ..."
python experiment.py --schedule a1 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A2"

echo ""
echo "Running (a1 vs a3 model) ..."
python experiment.py --schedule a1 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A3"

echo ""
echo "Running (a2 vs a1 model) ..."
python experiment.py --schedule a2 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A1"

echo ""
echo "Running (a2 vs a3 model) ..."
python experiment.py --schedule a2 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A3"

echo ""
echo "Running (a3 vs a1 model) ..."
python experiment.py --schedule a3 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A1"

echo ""
echo "Running (a3 vs a2 model) ..."
python experiment.py --schedule a3 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A2"

echo ""
echo "=========================================="
echo "All cross-evaluation experiments complete!"
echo "=========================================="

