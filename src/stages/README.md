# Pipeline Stages

This directory contains the three-stage feature selection pipeline:

## Stage 1 - Fast Filtering (stage1/)
- GPU-accelerated univariate feature filtering
- Mutual information calculation
- Correlation filtering
- Variance thresholding
- Reduces 5000 features to 500 in under 30 seconds

## Stage 2 - GPU-MCTS Selection (stage2/)
- Monte Carlo Tree Search with GPU acceleration
- Ensemble of 100+ trees
- Neural network metamodel for fast evaluation
- Reduces 500 features to 50

## Stage 3 - Precise Evaluation (stage3/)
- Full model training with XGBoost/RandomForest
- Comprehensive cross-validation
- SHAP values and permutation importance
- Final selection of 10-20 optimal features