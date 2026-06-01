#!/bin/bash
#SBATCH --job-name=nba_stacking_fast
#SBATCH --output=slurm-nba-stacking-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --partition=RM
#SBATCH --account=cis240116p
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bfb5559@psu.edu

echo "========================================"
echo "NBA STACKING ENSEMBLE - OPTIMIZED"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Set threading environment variables for maximum parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Performance Configuration:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  Available CPUs: $(nproc)"
echo ""

# Load Anaconda module
echo "Loading Anaconda module..."
module load anaconda3

# Activate environment
echo "Activating nba_env environment..."
source activate nba_env

# Verify and install/upgrade packages
echo ""
echo "Verifying Python packages..."
pip install --quiet --upgrade pandas numpy scipy scikit-learn joblib xgboost matplotlib seaborn

echo ""
echo "Package versions:"
python << 'PYEOF'
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  xgboost: {xgb.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
PYEOF

# Navigate to project directory
echo ""
echo "Navigating to project directory..."
cd $PROJECT/nba_modeling || { 
    echo "ERROR: Directory $PROJECT/nba_modeling not found"
    echo "Current directory: $(pwd)"
    echo "PROJECT variable: $PROJECT"
    exit 1
}

echo "Working directory: $(pwd)"

# Verify input file exists
echo ""
echo "Checking for input data..."
if [ ! -f "nba_pbp.csv" ]; then
    echo "ERROR: nba_pbp.csv not found!"
    echo "Files in current directory:"
    ls -lh *.csv 2>/dev/null || echo "  No CSV files found"
    exit 1
fi

FILE_SIZE=$(du -h nba_pbp.csv | cut -f1)
FILE_LINES=$(wc -l < nba_pbp.csv)
echo "✓ Input file found: nba_pbp.csv"
echo "  Size: $FILE_SIZE"
echo "  Lines: $FILE_LINES"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p model_plots
mkdir -p backup_models

# Verify Python script exists
if [ ! -f "nba_model.py" ]; then
    echo "ERROR: nba_model.py not found!"
    echo "Python files in directory:"
    ls -lh *.py 2>/dev/null || echo "  No Python files found"
    exit 1
fi

echo "✓ Training script found: nba_model.py"

# Run training with detailed timing
echo ""
echo "========================================"
echo "STARTING MODEL TRAINING"
echo "========================================"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Use 'time' command for detailed resource usage
/usr/bin/time -v python nba_model.py 2>&1 | tee training_output.log

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    echo "========================================"
    
    echo ""
    echo "Generated Model Files:"
    ls -lh *.pkl *.json 2>/dev/null || echo "  No model files found"
    
    echo ""
    echo "Generated Visualizations:"
    ls -lh model_plots/*.png 2>/dev/null || echo "  No plots found"
    
    echo ""
    echo "Backing up results..."
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    cp nba_stacking_model.pkl backup_models/nba_model_${TIMESTAMP}.pkl 2>/dev/null
    cp stacking_model_results.json backup_models/results_${TIMESTAMP}.json 2>/dev/null
    echo "✓ Backup created with timestamp: $TIMESTAMP"
    
    echo ""
    echo "Model Summary:"
    python << 'PYEOF'
import json
try:
    with open('stacking_model_results.json', 'r') as f:
        results = json.load(f)
    print(f"  Training samples: {results['n_samples_train']:,}")
    print(f"  Test samples: {results['n_samples_test']:,}")
    print(f"  Features: {results['n_features']}")
    print(f"  Training time: {results['training_time_seconds']:.1f}s ({results['training_time_seconds']/60:.1f} min)")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"  Brier Score: {results['test_metrics']['brier_score']:.4f}")
    print(f"  ECE: {results['test_metrics']['expected_calibration_error']:.4f}")
except Exception as e:
    print(f"  Could not load results: {e}")
PYEOF
    
else
    echo "✗ TRAINING FAILED"
    echo "========================================"
    echo ""
    echo "Error details:"
    echo "  Exit code: $EXIT_CODE"
    echo ""
    echo "Last 50 lines of output:"
    tail -50 training_output.log
    exit 1
fi

# Calculate total runtime
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - $(date -d "$(head -n 6 $0 | tail -n 1 | cut -d':' -f2-)" +%s 2>/dev/null || echo $END_TIME)))
TOTAL_MINUTES=$((SECONDS / 60))
REMAINING_SECONDS=$((SECONDS % 60))

echo ""
echo "========================================"
echo "JOB COMPLETE"
echo "========================================"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${TOTAL_MINUTES}m ${REMAINING_SECONDS}s"
echo ""
echo "Output files saved in: $(pwd)"
echo "SLURM output: slurm-nba-stacking-${SLURM_JOB_ID}.out"
echo "Training log: training_output.log"
echo ""
echo "To view results:"
echo "  cat stacking_model_results.json"
echo "  ls -lh model_plots/"
echo ""
echo "To download results:"
echo "  scp bfb5559@bridges2.psc.edu:$PROJECT/nba_modeling/*.pkl ."
echo "  scp bfb5559@bridges2.psc.edu:$PROJECT/nba_modeling/*.json ."
echo "  scp bfb5559@bridges2.psc.edu:$PROJECT/nba_modeling/model_plots/*.png ."
echo ""