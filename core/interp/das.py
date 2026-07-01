import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvene as pv
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from intervention import Intervention


class DAS(Intervention):

    # TODO: make training hyperparams passable as cmd args
    N_STEPS = 200
    LR = 5e-3

    # Warn (pre-intervention) if A is not among the base's top-K next tokens,
    # or B is not among the source's top-K next tokens.
    #TODO: remove
    TOPK_WARN = 20

    def __init__(self, args=None):
        super().__init__(args)
        self._rotation_cache = {}          
        for p in self.model.parameters():  
            p.requires_grad = False

   
    def _lm_config(self):
        return self.model.model.language_model.config

    def _hidden_size(self):
        return self._lm_config().hidden_size

    def _component(self, layer):

        return f"model.language_model.layers[{layer}].output"

    def _pv_inputs(self, text):
        """Chat-templated inputs as a pyvene-ready dict."""
        ids = self.chat_template(text, noimage=True, assistant=False)
        ids = torch.tensor([ids], device=self.device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    
    @staticmethod
    def _make_das_fn(rotate_layer):
        def das_fn(b, s):
            w = rotate_layer.weight
            a = (w / w.norm()).squeeze(0)            # unit direction [hidden]
            bp = (b * a).sum(-1, keepdim=True)       # base proj
            sp = (s * a).sum(-1, keepdim=True)       # source proj
            return b + (sp - bp) * a                 # swap base's value for source's
        return das_fn

    def _build_pv(self, rotate_layer, layer):
        return pv.IntervenableModel({
            "component": self._component(layer),
            "intervention": self._make_das_fn(rotate_layer),
        }, model=self.model)

    def _warn_oob(self, base_logp, src_logp, id_A, id_B, base_text, src_text):
        """
        Warn if p(A | source) < p(B | source) (pre-intervention).
        """
        if src_logp[id_A].item() < src_logp[id_B].item():
            print(f"  WARNING: logp(A|src)={src_logp[id_A].item():.3f} < "
                  f"logp(B|src)={src_logp[id_B].item():.3f} for source: {src_text!r}")


    def _fit(self, train_rows, layer, tok):
        key = (layer, tok)
        if key in self._rotation_cache:
            return self._rotation_cache[key]

        rotate_layer = nn.Linear(self._hidden_size(), 1, bias=False)
        nn.init.orthogonal_(rotate_layer.weight)
        rotate_layer = rotate_layer.to(device=self.device, dtype=torch.bfloat16)

        pv_das = self._build_pv(rotate_layer, layer)
        optimizer = torch.optim.Adam(rotate_layer.parameters(), lr=self.LR)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(0.1 * self.N_STEPS), self.N_STEPS
        )

        # Precompute pyvene inputs + per-row flip target. The target is the
        # comparison token of source_completion_A, diffed against the BASE input.
        items = []
        for r in train_rows:
            base_text = r[self.args.base_input_col]
            base = self._pv_inputs(base_text)
            src = self._pv_inputs(r[self.args.source_input_col])
            src_text = r[self.args.source_input_col]
            target_id = self.completion_token_id(
                src_text, r[self.args.source_completion_A])
            print(f"  [fit] target A token: "
                  f"{self.tokenizer.decode([target_id])!r} (id={target_id})")
            items.append((base, src, target_id))

        pbar = tqdm(range(self.N_STEPS),
                    desc=f"DAS fit L{layer} t{tok}", leave=False)
        for step in pbar:
            base, src, target_id = items[step % len(items)]
            _, out = pv_das(
                base, [src],
                {"sources->base": ([[[tok]]], [[[tok]]])},
            )
            logits = out.logits[:, -1, :]
            target = torch.tensor([target_id], device=self.device)
            loss = F.cross_entropy(logits.float(), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        del pv_das
        self._rotation_cache[key] = rotate_layer
        return rotate_layer

    def _score(self, rotate_layer, eval_rows, layer, tok, split_name):
        pv_das = self._build_pv(rotate_layer, layer)
        results = []
        with torch.no_grad():
            for r in eval_rows:
                src_text = r[self.args.source_input_col]

                base_text = r[self.args.base_input_col]
                id_A = self.completion_token_id(
                    src_text, r[self.args.source_completion_A])
                id_B = self.completion_token_id(
                    src_text, r[self.args.source_completion_B])
               
                # Clean (pre-intervention) full next-token logprobs.
                base_logp = self.logprobs_at_last(self.tokenize(base_text))
                src_logp = self.logprobs_at_last(self.tokenize(src_text))
                base_lp_A, base_lp_B = base_logp[id_A].item(), base_logp[id_B].item()
                src_lp_A, src_lp_B = src_logp[id_A].item(), src_logp[id_B].item()

                self._warn_oob(base_logp, src_logp, id_A, id_B,
                               base_text, src_text)

                # Patched base forward.
                base = self._pv_inputs(base_text)
                src = self._pv_inputs(src_text)
                _, out = pv_das(
                    base, [src],
                    {"sources->base": ([[[tok]]], [[[tok]]])},
                )
                logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
                base_int_lp_A = logp[id_A].item()
                base_int_lp_B = logp[id_B].item()

                results.append({
                    "source_input": src_text,
                    "base_input": base_text,
                    "source_logp_A": src_lp_A,
                    "source_logp_B": src_lp_B,
                    "base_logp_A": base_lp_A,
                    "base_logp_B": base_lp_B,
                    "base_intervention_logp_A": base_int_lp_A,
                    "base_intervention_logp_B": base_int_lp_B,
                })
        del pv_das
        return results

    
    #intervention implemented from abstract
    def intervention(self, train_rows, eval_rows, layer, tok, split_name):
        rotate_layer = self._fit(train_rows, layer, tok)
        results = self._score(rotate_layer, eval_rows, layer, tok, split_name)
        torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    DAS().main()