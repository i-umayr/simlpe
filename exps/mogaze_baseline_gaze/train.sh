#!/bin/bash

echo "🚀 FIX 2: DUAL OUTPUT GAZE SUPERVISION TRAINING"
echo "=============================================="

# Set environment for deterministic training
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Training parameters
SEED=888
GAZE_WEIGHT=0.1
NUM_LAYERS=48

echo "🔧 Training Configuration:"
echo "   Seed: $SEED"
echo "   Gaze Loss Weight: $GAZE_WEIGHT"  
echo "   MLP Layers: $NUM_LAYERS"
echo "   Expected Training Time: ~2 hours"
echo "   Expected Improvement: 3-8mm MPJPE"

echo ""
echo "📊 Baseline to Beat:"
echo "   3D Pose Only:        73.3 mm MPJPE"
echo "   Previous Gaze Try:   73.8 mm MPJPE (no improvement)"
echo "   Target:              65-70 mm MPJPE"

echo ""
echo "🎯 Fix 2 Strategy:"
echo "   • Dual output heads: pose (63D) + gaze (3D)"
echo "   • Combined loss: pose loss + weighted gaze loss"
echo "   • Forces model to learn gaze patterns through supervision"
echo "   • Shared backbone learns gaze-pose relationships"

echo ""
echo "🚀 Starting training..."
python train_mogaze_gaze.py \
    --seed $SEED \
    --with-normalization \
    --num $NUM_LAYERS \
    --gaze-weight $GAZE_WEIGHT \
    --exp-name mogaze_gaze_fix2_experiment.txt

echo ""
echo "✅ Training completed!"
echo ""
echo "🧪 Next steps:"
echo "   1. Run evaluation: python test_mogaze_gaze_fix2.py"
echo "   2. Check results vs 73.3mm baseline"
echo "   3. If successful: Combine with other fixes"
echo "   4. If not: Try Fix 1 (Separate DCT) or Fix 3 (Dynamic Gaze)"

echo ""
echo "📁 Files generated:"
echo "   • Model: log_gaze_fix2/snapshot/model-iter-40000.pth"
echo "   • Logs: log_gaze_fix2/log_*.log"
echo "   • Results: mogaze_gaze_fix2_experiment.txt"