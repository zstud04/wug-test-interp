"""
DiffMean 
"""

import torch
import pyvene as pv
from tqdm import tqdm

from intervention import Intervention


class DiffMean(Intervention):

    # Steering magnitude, in units of the unit-normalized direction. Kept
    # identical to LinearProbe.COEFF so the two methods differ only in `w`.
    # Note diff-in-means has a natural scale of its own (||mu_src - mu_base||);
    # NORMALIZE=False uses that instead and ignores COEFF.
    COEFF = 50.0
    NORMALIZE = True

    def __init__(self, args=None):
        super().__init__(args)
        self._dir_cache = {}               # (layer, tok) -> direction [hidden]
        for p in self.model.parameters():  # nothing is trained; freeze anyway
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Model-structure helpers (same depth/site as DAS and LinearProbe)
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

    # ------------------------------------------------------------------
    # Additive intervention (signed)
    # ------------------------------------------------------------------
    def _make_add_fn(self, direction, sign):
        """Add sign * COEFF * unit(direction) to the base act at the hooked pos."""
        d = direction.to(self.device)
        if self.NORMALIZE:
            d = self.COEFF * (d / d.norm())
        delta = (sign * d).to(torch.bfloat16)

        def add_fn(b, s):
            return b + delta

        return add_fn

    def _build_pv(self, direction, layer, sign=1.0):
        return pv.IntervenableModel({
            "component": self._component(layer),
            "intervention": self._make_add_fn(direction, sign),
        }, model=self.model)

    # ------------------------------------------------------------------
    # Fit: closed-form, forward rows only (see module docstring)
    # ------------------------------------------------------------------
    def _fit(self, train_rows, layer, tok):
        key = (layer, tok)
        if key in self._dir_cache:
            return self._dir_cache[key]

        rows = self.forward_rows(train_rows)
        if not rows:
            raise ValueError(
                "DiffMean._fit found no forward (inverse=0) train rows."
            )
        if len(rows) < len(train_rows):
            print(f"  [fit] dropping {len(train_rows) - len(rows)} inverse train "
                  f"rows; the class means are role-based and would cancel.")

        src_sum = torch.zeros(self._hidden_size(), device=self.device,
                              dtype=torch.float32)
        base_sum = torch.zeros_like(src_sum)

        pbar = tqdm(rows, desc=f"DiffMean fit L{layer} t{tok}", leave=False)
        for r in pbar:
            src_sum += self._collect_act(
                r[self.args.source_input_col], layer, tok).to(self.device)
            base_sum += self._collect_act(
                r[self.args.base_input_col], layer, tok).to(self.device)

        n = len(rows)
        direction = (src_sum / n) - (base_sum / n)    # [hidden]

        if direction.norm() == 0:
            raise ValueError(
                f"DiffMean direction is exactly zero at L{layer} t{tok}; "
                f"source and base activations have identical means."
            )
        print(f"  [fit] L{layer} t{tok}: ||w|| = {direction.norm().item():.4f} "
              f"over n={n} pairs")

        self._dir_cache[key] = direction
        return direction

    # ------------------------------------------------------------------
    # Warning helper (same semantics as DAS and LinearProbe)
    # ------------------------------------------------------------------
    def _warn_oob(self, src_logp, id_A, id_B, src_text):
        """Warn if p(A | source) < p(B | source) (pre-intervention)."""
        if src_logp[id_A].item() < src_logp[id_B].item():
            print(f"  WARNING: logp(A|src)={src_logp[id_A].item():.3f} < "
                  f"logp(B|src)={src_logp[id_B].item():.3f} for source: {src_text!r}")

    def _score_one(self, pv_add, r, tok):
        """Clean + patched logprobs for a single eval row."""
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

        # Patched base forward: base_act += sign * COEFF * unit(direction).
        base = self._pv_inputs(base_text)
        _, out = pv_add(base, unit_locations={"base": [[[tok]]]})
        logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)

        return {
            "source_input": src_text,
            "base_input": base_text,
            "source_logp_A": src_lp_A,
            "source_logp_B": src_lp_B,
            "base_logp_A": base_lp_A,
            "base_logp_B": base_lp_B,
            "base_intervention_logp_A": logp[id_A].item(),
            "base_intervention_logp_B": logp[id_B].item(),
        }

    # ------------------------------------------------------------------
    # Score eval rows for one (layer, tok)
    #
    # Forward rows get +w, inverse rows get -w. Rows are grouped by sign so
    # only one IntervenableModel is registered on the model at a time; results
    # are scattered back into eval-row order.
    # ------------------------------------------------------------------
    def _score(self, direction, eval_rows, layer, tok, split_name):
        results = [None] * len(eval_rows)

        for sign in (1.0, -1.0):
            want_inverse = sign < 0
            idxs = [i for i, r in enumerate(eval_rows)
                    if self.is_inverse(r) == want_inverse]
            if not idxs:
                continue

            pv_add = self._build_pv(direction, layer, sign)
            with torch.no_grad():
                for i in idxs:
                    results[i] = self._score_one(pv_add, eval_rows[i], tok)
            del pv_add
            torch.cuda.empty_cache()

        assert all(r is not None for r in results)
        return results

    # intervention implemented from abstract
    def intervention(self, train_rows, eval_rows, layer, tok, split_name):
        direction = self._fit(train_rows, layer, tok)
        results = self._score(direction, eval_rows, layer, tok, split_name)
        torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    DiffMean().main()