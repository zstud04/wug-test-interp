import argparse
import csv
import os
import random
from abc import ABC, abstractmethod

import torch
from minicons import scorer


class Intervention(ABC):


    OUTPUT_FIELDS = [
        "split",
        "tok",
        "layer",
        "source_input",
        "base_input",
        "source_logp_A",
        "source_logp_B",
        "base_logp_A",
        "base_logp_B",
        "base_intervention_logp_A",
        "base_intervention_logp_B",
    ]

    # Novel tokens are always added.
    ADDED_TOKENS = [" [wug]", " [wugs]"]

    def __init__(self, args=None):
        self.args = self.parse_args(args)
        self.train_rows = None
        self.test_rows = None
        self.lm = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        self.load_model()

    @staticmethod
    def parse_args(args=None):
        p = argparse.ArgumentParser()

        p.add_argument("--model_path", required=True,
                       help="Model name/path passed to VLMScorer.")
        p.add_argument("--cache_dir",
                       default="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford",
                       help="HF cache directory.")
        p.add_argument("--embeddings_path", default=None,
                       help="Optional path to a learned_embeddings.pt with "
                            "'wug_embedding'/'wugs_embedding'. If set, inject "
                            "these into the [wug]/[wugs] rows.")

        # Either provide train/test directly, or provide src_csv + a "split" col.
        p.add_argument("--train_csv", default=None,
                       help="CSV with train rows.")
        p.add_argument("--test_csv", default=None,
                       help="CSV with test rows.")
        p.add_argument("--src_csv", default=None,
                       help="CSV with a 'split' column (train/test) used when "
                            "--train_csv/--test_csv are not provided.")

        p.add_argument("--source_input_col", required=True,
                       help="Column with sentences to inject FROM (source).")
        p.add_argument("--base_input_col", required=True,
                       help="Column with sentences to inject INTO (base).")
        p.add_argument("--source_completion_A", required=True,
                       help="Column with the (good) completion as a FULL string: "
                            "the BASE input with the correct token appended, e.g. "
                            "base 'The dog' -> 'The dog runs'. The comparison "
                            "token is the first token beyond the base input.")
        p.add_argument("--source_completion_B", required=True,
                       help="Column with the (bad) completion as a FULL string: "
                            "the BASE input with the incorrect token appended. "
                            "The comparison token is the first token beyond the "
                            "base input.")

        # Optional position restrictions; empty => all.
        p.add_argument("--toks", nargs="*", type=int, default=None,
                       help="Token positions to intervene on. Empty => all.")
        p.add_argument("--layers", nargs="*", type=int, default=None,
                       help="Layer positions to intervene on. Empty => all.")

        p.add_argument("--out_csv", default="intervention_results.csv",
                       help="Path for the single output CSV.")

        
        p.add_argument("--filter", nargs="*", default=None, metavar="COL=VALUE",
                       help="Keep only rows matching every COL=VALUE (case-"
                            "insensitive). Applied to train and test alike.")

   
        p.add_argument("--n_sample", type=int, default=None,
                       help="Randomly sample this many rows from EACH split "
                            "(after --filter). Overridden per-split by "
                            "--n_sample_train / --n_sample_test.")
        p.add_argument("--n_sample_train", type=int, default=None,
                       help="Sample size for the train split (overrides "
                            "--n_sample).")
        p.add_argument("--n_sample_test", type=int, default=None,
                       help="Sample size for the test split (overrides "
                            "--n_sample).")
        p.add_argument("--seed", type=int, default=0,
                       help="Random seed for subsampling.")

        return p.parse_args(args)


    def load_model(self):
        """
        Load the VLMScorer, register [wug]/[wugs], and (optionally) inject
        learned embeddings from --embeddings_path.
        """
        self.lm = scorer.VLMScorer(
            self.args.model_path,
            device=self.device,
            torch_dtype=torch.bfloat16,
            cache_dir=self.args.cache_dir,
        )
        self.model = self.lm.model
        self.tokenizer = self.lm.tokenizer.tokenizer

        self._add_wug_tokens()
        self.wug_id, self.wugs_id = [
            self.tokenizer(t, add_special_tokens=False).input_ids[0]
            for t in self.ADDED_TOKENS
        ]

        if self.args.embeddings_path:
            self._inject_embeddings(self.args.embeddings_path)

        return self.lm

    def _add_wug_tokens(self):
        vocab = self.tokenizer.get_vocab()
        to_add = [t for t in self.ADDED_TOKENS if t not in vocab]
        if to_add:
            self.tokenizer.add_tokens(to_add)
            old_len = self.model.resize_token_embeddings().weight.shape[0]
            self.model.resize_token_embeddings(old_len + len(to_add))

    def _embed(self):

        return self.model.model.language_model.embed_tokens

    def _inject_embeddings(self, path):
        rec = torch.load(path, map_location="cpu")
        emb = self._embed()
        with torch.no_grad():
            emb.weight.data[self.wug_id] = rec["wug_embedding"].to(
                emb.weight.device, dtype=emb.weight.dtype)
            emb.weight.data[self.wugs_id] = rec["wugs_embedding"].to(
                emb.weight.device, dtype=emb.weight.dtype)

    
    def chat_template(self, text, noimage=True, assistant=False):

        if noimage:
            context = [{"role": "user",
                        "content": [{"type": "text", "text": text}]}]
        else:
            context = [{"role": "user",
                        "content": [{"type": "image"},
                                    {"type": "text", "text": text}]}]

        if assistant:
            enc = self.tokenizer.apply_chat_template(
                context, add_generation_prompt=True)
        else:
            enc = self.tokenizer.apply_chat_template(
                context, continue_final_message=True)
        return self._coerce_ids(enc)

    @staticmethod
    def _coerce_ids(enc):
        """Normalize apply_chat_template output to a flat list[int]."""
        if hasattr(enc, "ids"):            # tokenizers.Encoding
            return list(enc.ids)
        if hasattr(enc, "input_ids"):      # BatchEncoding
            ids = enc.input_ids
            if len(ids) and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return list(ids)
        if isinstance(enc, dict):          # plain dict
            ids = enc["input_ids"]
            if len(ids) and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return list(ids)
        return list(enc)                   # already a list of ints

    def tokenize(self, text):
        """Chat-templated input ids as a [1, T] tensor on device."""
        ids = self.chat_template(text, noimage=True, assistant=False)
        return torch.tensor([ids], device=self.device)

    def token_id(self, completion):
        """First token id of a completion string (single-token assumed)."""
        return self.tokenizer(completion, add_special_tokens=False).input_ids[0]

    def completion_token_id(self, source_input_text, completion_text):
      
        source_ids = self.chat_template(source_input_text, noimage=True, assistant=False)
        comp_ids = self.chat_template(completion_text, noimage=True, assistant=False)
        i = 0
        while (i < len(source_ids) and i < len(comp_ids)
               and source_ids[i] == comp_ids[i]):
            i += 1
        return comp_ids[i]

   
    @torch.no_grad()
    def logprobs_at_last(self, input_ids):

        out = self.model(input_ids)
        logits = out.logits[0, -1, :].float()
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def token_logprobs(self, text, completion_ids):
        """
        Logprobs of each id in `completion_ids` as the next token after `text`
        (chat-templated). Returns a list of floats aligned with completion_ids.
        """
        logp = self.logprobs_at_last(self.tokenize(text))
        return [logp[cid].item() for cid in completion_ids]

    def AB_logprobs(self, text, id_A, id_B):
        """Convenience: (logp_A, logp_B) for one input string."""
        lp_A, lp_B = self.token_logprobs(text, [id_A, id_B])
        return lp_A, lp_B

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def _parse_filters(self):
        """[(col, normalized_value), ...] from --filter (empty if none)."""
        filters = []
        for item in (self.args.filter or []):
            sep = "==" if "==" in item else "="
            col, val = item.split(sep, 1)
            filters.append((col.strip(), val.strip().casefold()))
        return filters

    def _apply_filters(self, rows):
        """Keep rows matching ALL filters (stripped, case-insensitive)."""
        filters = self._parse_filters()
        if not filters:
            return rows
        kept = []
        for r in rows:
            if all(str(r.get(col, "")).strip().casefold() == val
                   for col, val in filters):
                kept.append(r)
        return kept

    def _sample_n(self, split):
        """Resolve the sample size for a split: per-split flag else --n_sample."""
        per_split = getattr(self.args, f"n_sample_{split}")
        return per_split if per_split is not None else self.args.n_sample

    @staticmethod
    def _sample_rows(rows, n, rng):
        """Random subsample of n rows (all rows if n is None or n >= len)."""
        if n is None or n >= len(rows):
            return rows
        return rng.sample(rows, n)

    @staticmethod
    def _keep_split(rows, split):
        """Within a CSV, keep only rows of the given split (if a split col exists)."""
        if rows and "split" in rows[0]:
            return [r for r in rows if r["split"] == split]
        return rows

    def load_rows(self):
        if self.args.train_csv is not None and self.args.test_csv is not None:
            self.train_rows = self._keep_split(
                self._read_csv(self.args.train_csv), "train")
            self.test_rows = self._keep_split(
                self._read_csv(self.args.test_csv), "test")
        else:
            rows = self._read_csv(self.args.src_csv)
            self.train_rows = [r for r in rows if r["split"] == "train"]
            self.test_rows = [r for r in rows if r["split"] == "test"]

        # Filter first, then subsample the filtered subset.
        self.train_rows = self._apply_filters(self.train_rows)
        self.test_rows = self._apply_filters(self.test_rows)

        rng = random.Random(self.args.seed)
        self.train_rows = self._sample_rows(
            self.train_rows, self._sample_n("train"), rng)
        self.test_rows = self._sample_rows(
            self.test_rows, self._sample_n("test"), rng)

        return self.train_rows, self.test_rows

    # ------------------------------------------------------------------
    # Position grids
    # ------------------------------------------------------------------
    def get_layers(self):
        if self.args.layers:
            return list(self.args.layers)
        return self.all_layers()

    def get_toks(self):
        if self.args.toks:
            return list(self.args.toks)
        return self.all_toks()

    def all_layers(self):
        """Every layer in the language model."""
        n = self.model.model.language_model.config.num_hidden_layers
        return list(range(n))

    def all_toks(self):
        """
        Every token position of a base input. Sentences are assumed equal
        length (per the I/O contract), so the first train base row defines
        the grid.
        """
        row = self.train_rows[0]
        ids = self.chat_template(row[self.args.base_input_col],
                                 noimage=True, assistant=False)
        return list(range(len(ids)))

    # ------------------------------------------------------------------
    # The method-specific work
    # ------------------------------------------------------------------
    @abstractmethod
    def intervention(self, train_rows, eval_rows, layer, tok, split_name):
        """
        Perform the intervention for one (layer, tok) cell on `eval_rows`
        (which belong to `split_name`, "train" or "test"). Fitting always uses
        `train_rows` regardless of split.

        Return a list of dicts (one per eval row) with the value columns of
        OUTPUT_FIELDS:
            source_input, base_input,
            source_logp_A, source_logp_B,
            base_logp_A, base_logp_B,
            base_intervention_logp_A, base_intervention_logp_B

        `split`, `tok`, `layer` are filled by `main`.
        """
        ...

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def main(self):
        self.load_rows()

        layers = self.get_layers()
        toks = self.get_toks()
        splits = [("train", self.train_rows), ("test", self.test_rows)]

        out_rows = []
        for layer in layers:
            for tok in toks:
                for split_name, eval_rows in splits:
                    results = self.intervention(
                        self.train_rows, eval_rows, layer, tok, split_name
                    )
                    for res in results:
                        row = dict(res)
                        row["split"] = split_name
                        row["layer"] = layer
                        row["tok"] = tok
                        out_rows.append(row)

        self.write_output(out_rows)
        return out_rows

    def write_output(self, out_rows):
        out_dir = os.path.dirname(self.args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.OUTPUT_FIELDS)
            w.writeheader()
            for row in out_rows:
                w.writerow({k: row.get(k) for k in self.OUTPUT_FIELDS})


if __name__ == "__main__":
    raise SystemExit("Intervention is abstract; run a concrete subclass.")