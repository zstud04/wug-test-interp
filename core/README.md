# Core scripts/functionality

All scripts for training and evaluating embeddings are available here.

## eval/
- embed_eval.py: evaluate trained wug/wugs embeddings on minimal difference pairs of stimuli.

## interp/
Code for all intervention methods.

Abstract classes:
- intervention.py: abstract class for ALL intervention methods
- circuit_intervention.py: abstract class for CIRCUIT BASED intervention methods (inherits from Intervention class)

Interventions:
- ablation.py
- circuit.py
- das.py
- diffmean.py
- linear_probe.py

## train/
- embed_train.py: train image/text based [wug] and [wugs] embeddings.