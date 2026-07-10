"""
Ablation — concrete AblationIntervention.

"""

import torch
from tqdm import tqdm

from ablation_intervention import AblationIntervention
from circuit import Circuit


class Ablation(AblationIntervention, Circuit):

    # Number of (hook, layer, tok, unit) nodes in the circuit. --top_k overrides.
    TOP_K = 64

    def __init__(self, args=None):
        super().__init__(args)
        self._values = None
        self._masks = None
        self._circuit = None
        self._tok_idx = None

    # ------------------------------------------------------------------
    # The search space, resolved once
    # ------------------------------------------------------------------
    def _keys(self):
        layers = sorted({l for l, _ in self.candidate_cells()})
        return [(h, l) for h in self.hook_points for l in layers]

    def _toks(self):
        return sorted({t for _, t in self.candidate_cells()})

    def _tok_tensor(self):
        if self._tok_idx is None:
            self._tok_idx = torch.tensor(self._toks(), device=self.device)
        return self._tok_idx

    # ------------------------------------------------------------------
    # Ablation values: mean activation per (hook, layer, tok), or zeros
    # ------------------------------------------------------------------
    def ablation_values(self, train_rows):
        keys, toks = self._keys(), self._toks()

        if self.args.ablation == "zero":
            print("  ablation values: zeros")
            return {k: torch.zeros(len(toks), self.width(k[0]),
                                   device=self.device, dtype=torch.float32)
                    for k in keys}

        tok_idx = self._tok_tensor()
        acc = {k: torch.zeros(len(toks), self.width(k[0]),
                              device=self.device, dtype=torch.float32)
               for k in keys}

        for r in tqdm(train_rows, desc="ablation means", leave=False):
            acts = self._run_capture(self.tokenize(self.eval_text(r)), keys)
            for k in keys:
                acc[k] += acts[k][0, tok_idx, :]
            del acts

        for k in keys:
            acc[k] /= len(train_rows)

        torch.cuda.empty_cache()
        n_fwd = len(self.forward_rows(train_rows))
        print(f"  ablation values: means over {len(train_rows)} train rows "
              f"({n_fwd} forward, {len(train_rows) - n_fwd} inverse)")
        return acc

    # ------------------------------------------------------------------
    # Masks: which (tok, unit) coordinates each mode ablates
    # ------------------------------------------------------------------
    def _build_masks(self, circuit):
        keys, toks = self._keys(), self._toks()
        pos_of = {t: i for i, t in enumerate(toks)}

        circ = {k: torch.zeros(len(toks), self.width(k[0]),
                               device=self.device, dtype=torch.bool)
                for k in keys}
        for key, (node_toks, node_units) in circuit["nodes"].items():
            rows = torch.tensor([pos_of[int(t)] for t in node_toks.tolist()],
                                device=self.device)
            circ[key][rows, node_units] = True

        n_circ = sum(int(v.sum()) for v in circ.values())
        n_total = sum(v.numel() for v in circ.values())
        print(f"  masks: circuit {n_circ} / complement {n_total - n_circ} "
              f"/ all {n_total} coordinates")

        return {
            "circuit": circ,
            "complement": {k: ~v for k, v in circ.items()},
            "all": {k: torch.ones_like(v) for k, v in circ.items()},
        }

    # ------------------------------------------------------------------
    # Hook: overwrite masked coordinates with the ablation value
    # ------------------------------------------------------------------
    def _register_ablate(self, key, mask, vals, tok_idx):
        hook, layer = key
        mod = self._module(hook, layer)

        def _apply(h):
            h = h.clone()
            cur = h[0, tok_idx, :]
            h[0, tok_idx, :] = torch.where(mask, vals.to(h.dtype), cur)
            return h

        if self._is_input_site(hook):
            def pre(module, args):
                return (_apply(args[0]),)
            return mod.register_forward_pre_hook(pre)

        def post(module, args, output):
            return self._put(output, _apply(self._get(output)))
        return mod.register_forward_hook(post)

    # ------------------------------------------------------------------
    # One run
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_ablated(self, row, circuit, values, mode):
        if self._masks is None or self._circuit is not circuit:
            self._masks = self._build_masks(circuit)
            self._circuit = circuit

        input_ids = self.tokenize(self.eval_text(row))
        id_A, id_B = self.AB_ids(row)

        if mode == "none":
            handles = []
        else:
            masks = self._masks[mode]
            tok_idx = self._tok_tensor()
            handles = [self._register_ablate(k, masks[k], values[k], tok_idx)
                       for k in self._keys() if masks[k].any()]

        try:
            out = self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        return logp[id_A].item(), logp[id_B].item()

    # ------------------------------------------------------------------
    # Progress bar around the inherited four-run score()
    # ------------------------------------------------------------------
    def score(self, circuit, eval_rows, split_name):
        results = super().score(
            circuit,
            list(tqdm(eval_rows, desc=f"ablate [{split_name}]", leave=False)),
            split_name,
        )
        torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    Ablation().main()