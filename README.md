# A wug test for mechanistic interpretability

## Setup
Run ./setup.sh to automatically install required packages via conda. This will prompt you for a huggingface token and create a new env wug_test_env.

## Commands


### Training embeddings
Use /core/embed_train.py to train newly initialized embeddings [wug]/[wugs] for a VLM.

Args:
- lr (float): learning rate
- seed (int): random seed for training/embed initialization
- epochs (int): number of training epochs
- training_condition (str): "image" (train with real images from image_dir) or "syntax" (text-only, or with filler images if filler_image_dir is set)
- train_csv (str): training csv with cols "type" (singular/plural) and "sentence" (raw sentence str)
- eval_csv (str): eval csv with minimal pairs in each row (good_sentence) and (bad_sentence)
- batch_mode (str): "alternating" (separate singular/plural optimizer steps) or "joint" (one step with summed per-example CE). Default "alternating"
- batch_size (int): number of singular/plural pairs per optimizer step. Default 1
- model (str): HF model name/path. Must have tied embed/lm head weights. Default "Qwen/Qwen3-VL-4B-Instruct"
- cache_dir (str): HF cache directory. Default "/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
- image_dir (str): directory of images, used in the "image" condition. Default "../images/snarples"
- out_dir (str): root output dir; a per-run subfolder is created inside it. Default "wug_lr_sweep_results"
- write_embeddings (flag): if set, saves learned [wug]/[wugs] embeddings to learned_embeddings.pt. Default off
- filler_image_dir (str): directory of filler images to pair with sentences in the "syntax" condition. Default None
- embed_init (str): txt file of noun words (one per line) for embed init. If supplied, [wug]/[wugs] are initialized near the mean of these embeddings with noise scaled to the mean 
singular-plural distance. If omitted, uses default random init. Default None
- write_embeddings: set this flag to write out final embeddings into the save dir.
- write_all: set this flag to write intermediate embeddings from all epochs in training.
- terminate_cond_epochs (int): value of n epochs after which to terminate training if no improvement. Default 5



*Example syntax condition training:*

`python3 -m core.train.embed_train     --training_condition syntax     --train_csv data/embeddings/train/text/syntax_train_1.csv     --eval_csv data/embeddings/eval/singular_plural_eval.csv     --embed_init data/embeddings/init/noun_init.txt     --lr 0.001 --seed 40 --epochs 50 --write_embeddings`



*Example image condition training:*

`python3 -m core.train.embed_train     --training_condition image     --image_dir data/embeddings/train/im/creature_1/     --train_csv data/embeddings/train/text/image_train_1.csv     --eval_csv data/embeddings/eval/singular_plural_eval.csv     --embed_init data/embeddings/init/noun_init.txt     --lr 0.001 --seed 40 --epochs 10 --write_embeddings`
