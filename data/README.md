# Input data/stimuli for train/eval

## embeddings/
Inputs for TRAINING embeddings, which consists of initializing, training, and dev set evaluation at each epoch.

### init/
- noun_init.txt: unordered txt of single-token singular and plural nouns for initializing [wug] and [wugs]. Evenly balanced between singular and plural so not biased in either direction.

### train/
Image and text inputs for training embeddings. 
- im/creature_1...creature_5: image stimuli for image condition. Different "creatures" generated via OpenAI API
- text/image_train_1.csv: text stimuli for image condition. 
- text/syntax_train_1.csv: text stimuli for syntax condition.
- text/syntax_train_2.csv: [DEPRECATED] alternative text stimuli for syntax condition.

### dev/
- dev_eval.csv: a two column (good/bad) CSV with wug/wugs examples. Evaluated at each epoch and used for stopping protocol

## eval/
Inputs for downstream evaluation, stimuli which are not used for dev set or interchange evals (e.g., attractor stimuli).
- generalization_constructions/generalization_constructions.csv: OOD constructions different from those seen in training for syntax condition.
- attractors/attractor_stimuli.csv: [DEPRECATED] old attractor stimuli.

## interp/
Inputs for interpretability analyses, including attractors.
- agreement_target_natural.csv: 'natural' sentences used as source for interventions. Includes both a train and test split with disjoint verbs, and n. attractors 0...3. All sentences end with a singular/plural verb
- agreement_target_wug.csv: 'wug' sentences. Test split only, identical to the natural test split but with [wug]/[wugs] substituted.


