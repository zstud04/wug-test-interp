# Results
results for train, evaluation, and interp (causal interventions) on embeddings.

## Structure for all embeddings folders
Embeddings are written in a folder which indicates the training conditions:
`{modality}_{interleaving_method}_lr{lr}_seed{seed}_ep{epochs}`
e.g:
`syntax_alternating_lr0p001_seed14010_ep50`

Each embeddings folder output contains:
- epoch_stats.csv: epoch-by-epoch loss, gradients, and eval scores
- eval_item_scores.csv: scores on dev set by item. Long format, with scores for each item for each epoch
- init_pair_distances.csv: L2 distance between singular/plural natural embeddings.
- learned_embedings.pt: The tensorfile with the final output embeddings
- logit_gap_stats.csv: Difference in output logits for singular/plural on singular/plural dev trials, at each step
- run_config.csv: Shows the hyperparameters that were used for training
- step_stats.csv: More granular version of epoch_stats.csv. Shows CE loss, grad norms for each individual weight update.
- trajectory_stats.csv: Shows how [wug]/[wugs] project onto singular/plural embeddings directions at each epoch. Useful for debugging/checking trajectories.

## train/
Results for training/evaluating embeddings to identify best LR.

- CI_seed_runs/: Runs for many (50) seeds to generate CIs for embeddings at specific LRs.
- LR_sweeps/: Runs for 5 seeds per LR to identify best LR for a given model.
- final_embedding_results: Embedding results for the chosen embeddings (specific seed/LR) used for downstream eval and interp.

## eval/
Eval results for selected embeddings.
- generalization_constructions/: results for "OOD" constructions
- movement_analysis/: embeddings trajectories during training.
- attractors/: attractors results. **these are used as inputs directly into causal interp. scripts**

## interp/
Results for all interpretability (causal intervention) methods.
- main_runs/: Runs for all conditions/attractors for *all* intervention methods.
- seed_replicate/: Replicate main experiments across k=5 seeds (for subset of methods)
- hook_comparisons/: Compare circuit patching and ablation methods for different hook points. 

## deprecated/
Results from previous/defunct iterations of interpretability methods.


