"""
LinearProbe — concrete Intervention.

At a given (layer, tok), fits a logistic-regression probe that classifies the
residual-stream-post activation of source inputs as class 1 and base inputs as
class 0. The learned weight vector defines a direction in activation space.

At apply time the unit-normalized probe direction is ADDED (not swapped, unlike
DAS) to the base run's residual-stream-post activation at (layer, tok), scaled
by a coefficient c, and the same A/B continuation logprobs are read.

Hook site is identical to DAS: model.language_model.layers[L].output, i.e. the
post-residual hidden state at layer L. A/B comparison tokens are derived exactly
as in DAS via completion_token_id(source_input, completion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvene as pv
from tqdm import tqdm

from intervention import Intervention


class LinearProbe(Intervention):


    N_STEPS = 200
    LR = 5e-3

    # coeff for linear combination of probe direction
    COEFF = 50.0

    # Warn (pre-intervention) if p(A | source) < p(B | source).
    def __init__(self, args=None):
        super().__init__(args)
        self._probe_cache = {}             # (layer, tok) -> unit direction [hidden]
        for p in self.model.parameters():  # only the probe is trained
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Model-structure helpers (same depth/site as DAS)
    # ------------------------------------------------------------------
    def _lm_config(self):
        return self.model.model.language_model.config

    def _hidden_size(self):
        return self._lm_config().hidden_size

    def _component(self, layer):
        # Residual-stream-post = decoder layer output. Path is relative to
        # self.model (= lm.model, the top Qwen3VLForConditionalGeneration), so
        # one fewer `model` hop than the Python attribute access.
        return f"model.language_model.layers[{layer}].output"

    def _pv_inputs(self, text):
        """Chat-templated inputs as a pyvene-ready dict."""
        ids = self.chat_template(text, noimage=True, assistant=False)
        ids = torch.tensor([ids], device=self.device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


    def _collect_act(self, text, layer, tok):
        """
        Residual-stream-post activation [hidden] at position `tok` for `text`.
        Uses a no-op collecting intervention so the hook site matches apply time.
        """
        captured = {}

        def collect_fn(b, s):

            captured["act"] = b.detach().to(torch.float32).clone()
            return b

        pv_model = pv.IntervenableModel({
            "component": self._component(layer),
            "intervention": collect_fn,
        }, model=self.model)

        inputs = self._pv_inputs(text)
        with torch.no_grad():
            pv_model(inputs, unit_locations={"base": [[[tok]]]})

        del pv_model
        return captured["act"].reshape(-1)   # [hidden]

    #additive probe intervention
    def _make_add_fn(self, direction):
        """Add COEFF * unit(direction) to the base activation at the hooked pos."""
        d = direction.to(self.device)
        d = d / d.norm()
        delta = (self.COEFF * d).to(torch.bfloat16)

        def add_fn(b, s):
            return b + delta

        return add_fn

    def _build_pv(self, direction, layer):
        return pv.IntervenableModel({
            "component": self._component(layer),
            "intervention": self._make_add_fn(direction),
        }, model=self.model)

   
    def _fit(self, train_rows, layer, tok):
        key = (layer, tok)
        if key in self._probe_cache:
            return self._probe_cache[key]

        # Collect activations: source -> label 1, base -> label 0.
        feats, labels = [], []
        for r in train_rows:
            src_text = r[self.args.source_input_col]
            base_text = r[self.args.base_input_col]
            feats.append(self._collect_act(src_text, layer, tok))
            labels.append(1.0)
            feats.append(self._collect_act(base_text, layer, tok))
            labels.append(0.0)

        X = torch.stack(feats).to(self.device, torch.float32)   # [2N, hidden]
        y = torch.tensor(labels, device=self.device)            # [2N]

        probe = nn.Linear(self._hidden_size(), 1, bias=True).to(
            self.device, torch.float32)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.LR)

        pbar = tqdm(range(self.N_STEPS),
                    desc=f"Probe fit L{layer} t{tok}", leave=False)
        for _ in pbar:
            logits = probe(X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        direction = probe.weight.detach().squeeze(0).clone()    # [hidden]
        self._probe_cache[key] = direction
        return direction

    # ------------------------------------------------------------------
    # Warning helper (same semantics as DAS)
    # ------------------------------------------------------------------
    def _warn_oob(self, src_logp, id_A, id_B, src_text):
        """Warn if p(A | source) < p(B | source) (pre-intervention)."""
        if src_logp[id_A].item() < src_logp[id_B].item():
            print(f"  WARNING: logp(A|src)={src_logp[id_A].item():.3f} < "
                  f"logp(B|src)={src_logp[id_B].item():.3f} for source: {src_text!r}")

    # ------------------------------------------------------------------
    # Score eval rows for one (layer, tok)
    # ------------------------------------------------------------------
    def _score(self, direction, eval_rows, layer, tok, split_name):
        pv_add = self._build_pv(direction, layer)
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

                self._warn_oob(src_logp, id_A, id_B, src_text)

                # Patched base forward: base_act += COEFF * unit(direction).
                base = self._pv_inputs(base_text)
                _, out = pv_add(base, unit_locations={"base": [[[tok]]]})
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
        del pv_add
        return results

    def intervention(self, train_rows, eval_rows, layer, tok, split_name):
        direction = self._fit(train_rows, layer, tok)
        results = self._score(direction, eval_rows, layer, tok, split_name)
        torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    LinearProbe().main()