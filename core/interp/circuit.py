"""
Circuit — concrete CircuitIntervention.

Finds a circuit of activations whose swap from source into base is most
responsible for flipping the model's verb-agreement prediction. A node is

    (hook_point, layer, tok, unit)

and the circuit is the top-k of them, selected in ONE ranking over the whole
hook x layer x token grid.

HOOK POINTS. Four sites, selectable with --hook_points:

    mlp_act    input to down_proj: act_fn(gate_proj(x)) * up_proj(x)   [d_ffn]
    mlp_out    output of the MLP block                                 [d_model]
    attn_out   output of the attention block                           [d_model]
    resid      output of the decoder layer (residual stream post)      [d_model]

`mlp_act` is the default and the interesting one: the elementwise nonlinearity
makes it a privileged basis, and Arora et al. (2025) find circuits there to be
roughly 100x sparser than at `mlp_out`. The others have no preferred axes, so a
single coordinate of them is not expected to mean much on its own -- they are
here as baselines, and as the sites DAS and LinearProbe operate on (`resid`).

ATTRIBUTION. For a node v, the first-order estimate of what patching it from
source into base does to the metric m = logit_A - logit_B (A agrees with the
source, B with the base) is

    dm  ~=  ( v(source) - v(base) ) * dm/dv |_base

i.e. activation-delta times gradient, with the gradient taken at the base run.
This is attribution patching: a single backward pass estimates every candidate
interchange intervention at once. Positive score means swapping this node in
pushes the base toward the source's verb.

BIDIRECTIONALITY IS FREE. Swap the roles of source and base. The metric negates
(A and B trade places) and the activation delta negates too, so the product is
unchanged -- only the point at which the gradient is evaluated moves, from the
plural sentence to the singular one. Under --add_inverse, forward and inverse
rows therefore contribute the SAME functional sampled at both sentences, and the
mean over all train rows is a single direction-agnostic attribution. No sign
handling, no forward_rows() filter. Contrast LinearProbe / DiffMean, whose
additive steer is signed and must consult `inverse` at every step.

MIXING HOOK POINTS. Nodes from different sites compete in one top-k, but
attribution is in raw activation units and those units are not comparable across
sites: `mlp_act` is post-SwiGLU and unbounded, `resid` grows by an order of
magnitude across depth. A naive mixed ranking is therefore dominated by whichever
site happens to have the largest numbers, not by causal importance. Set
NORM_PER_HOOK to z-score attributions within each site before ranking, which
makes the competition meaningful at the cost of a `top_k` that no longer has an
absolute-effect-size interpretation. For a clean comparison ACROSS sites, prefer
running each alone and overlaying the sparsity curves.

APPLICATION. Pure interchange: for each selected node, overwrite the base run's
activation with the source run's at that exact coordinate. Nothing is scaled,
nothing is added, no coefficient. Like DAS's projection swap and unlike an
additive steer, this is symmetric under exchanging source and base, so the same
circuit applies unmodified to both directions of the test set.

GRADIENTS. Model parameters keep requires_grad=True so autograd builds a graph
through the network; `torch.autograd.grad` is used rather than `.backward()`, so
no parameter `.grad` buffers are ever populated. Detaching the captured
activations would sever gradient paths that pass through later blocks, which is
the paper's edge-attribution variant, not what we want for nodes.
"""

import torch
from tqdm import tqdm

from circuit_intervention import CircuitIntervention


class Circuit(CircuitIntervention):

    HOOK_POINTS = ("mlp_act", "mlp_out", "attn_out", "resid")
    DEFAULT_HOOK_POINTS = ("mlp_act",)

    # Number of (hook, layer, tok, unit) nodes in the circuit. --top_k overrides.
    TOP_K = 64

    # Rank by signed attribution (nodes that drive the flip) or by magnitude
    # (nodes that matter, in either sign).
    RANK_ABS = False

    # Z-score attributions within each hook point before ranking. Only relevant
    # when more than one hook point is selected. See module docstring.
    NORM_PER_HOOK = False

    EXTRA_FIELDS = ["n_nodes"]

    # Sites whose activation is the module's INPUT rather than its output.
    _INPUT_SITES = ("mlp_act",)

    # ------------------------------------------------------------------
    # Model structure
    # ------------------------------------------------------------------
    def _lm(self):
        return self.model.model.language_model

    def _config(self):
        return self._lm().config

    def _module(self, hook, layer):
        block = self._lm().layers[layer]
        if hook == "mlp_act":
            return block.mlp.down_proj      # its INPUT is the intermediate act
        if hook == "mlp_out":
            return block.mlp
        if hook == "attn_out":
            return block.self_attn
        if hook == "resid":
            return block
        raise ValueError(f"unknown hook point {hook!r}")

    def width(self, hook):
        cfg = self._config()
        return cfg.intermediate_size if hook == "mlp_act" else cfg.hidden_size

    @classmethod
    def _is_input_site(cls, hook):
        return hook in cls._INPUT_SITES

    # Modules differ in whether they return a bare tensor or a tuple whose
    # first element is the hidden state; normalize both directions.
    @staticmethod
    def _get(out):
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _put(out, new):
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _register_capture(self, store, key):
        """Record the activation at `key = (hook, layer)` without modifying it."""
        hook, layer = key
        mod = self._module(hook, layer)

        if self._is_input_site(hook):
            def pre(module, args, _k=key):
                store[_k] = args[0]
                return None
            return mod.register_forward_pre_hook(pre)

        def post(module, args, output, _k=key):
            store[_k] = self._get(output)
            return None
        return mod.register_forward_hook(post)

    def _register_patch(self, src_acts, nodes, key):
        """Overwrite the base activation at `key` with the source's."""
        hook, layer = key
        mod = self._module(hook, layer)
        toks, units = nodes[key]
        src = src_acts[key]                       # [1, T, width]

        if self._is_input_site(hook):
            def pre(module, args):
                h = args[0].clone()
                h[0, toks, units] = src[0, toks, units].to(h.dtype)
                return (h,)
            return mod.register_forward_pre_hook(pre)

        def post(module, args, output):
            h = self._get(output).clone()
            h[0, toks, units] = src[0, toks, units].to(h.dtype)
            return self._put(output, h)
        return mod.register_forward_hook(post)

    def _run_capture(self, input_ids, keys):
        """Forward pass; returns {(hook, layer): [1, T, width]} detached fp32."""
        store = {}
        handles = [self._register_capture(store, k) for k in keys]
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()
        return {k: v.detach().float() for k, v in store.items()}

    # ------------------------------------------------------------------
    # Fit: attribution patching, averaged over both directions
    # ------------------------------------------------------------------
    def fit_circuit(self, train_rows):
        cells = self.candidate_cells()
        layers = sorted({l for l, _ in cells})
        toks = sorted({t for _, t in cells})
        tok_idx = torch.tensor(toks, device=self.device)

        # Block order defines the flat index; fixed once, reused for decoding.
        keys = [(h, l) for h in self.hook_points for l in layers]
        scores = {k: torch.zeros(len(toks), self.width(k[0]),
                                 device=self.device, dtype=torch.float32)
                  for k in keys}

        self.model.eval()

        for r in tqdm(train_rows, desc=f"attribution (k={self.top_k})",
                      leave=False):
            src_ids = self.tokenize(r[self.args.source_input_col])
            base_ids = self.tokenize(r[self.args.base_input_col])
            id_A, id_B = self.AB_ids(r)

            if max(toks) >= base_ids.shape[1]:
                raise ValueError(
                    f"--toks max {max(toks)} exceeds sequence length "
                    f"{base_ids.shape[1]} for {r[self.args.base_input_col]!r}"
                )

            # v(source): clean activations that would be swapped in.
            src_acts = self._run_capture(src_ids, keys)

            # v(base) and dm/dv|_base: one forward, one backward.
            store = {}
            handles = [self._register_capture(store, k) for k in keys]
            try:
                with torch.enable_grad():
                    out = self.model(base_ids)
                    logits = out.logits[0, -1, :].float()
                    m = logits[id_A] - logits[id_B]
                    grads = torch.autograd.grad(m, [store[k] for k in keys])
            finally:
                for h in handles:
                    h.remove()

            for k, g in zip(keys, grads):
                base_act = store[k].detach().float()[0, tok_idx, :]
                delta = src_acts[k][0, tok_idx, :] - base_act
                scores[k] += delta * g.detach().float()[0, tok_idx, :]

            del store, grads, src_acts
            torch.cuda.empty_cache()

        for k in keys:
            scores[k] /= len(train_rows)

        return self._select_topk(scores, keys, toks)

    # ------------------------------------------------------------------
    # Top-k across every (hook, layer, tok, unit)
    # ------------------------------------------------------------------
    def _select_topk(self, scores, keys, toks):
        rank_blocks = []
        for k in keys:
            b = scores[k]
            if self.NORM_PER_HOOK:
                b = (b - b.mean()) / (b.std() + 1e-8)
            rank_blocks.append(b.reshape(-1))

        flat = torch.cat([scores[k].reshape(-1) for k in keys])   # raw scores
        rank = torch.cat(rank_blocks)                             # ranking basis
        rank = rank.abs() if self.RANK_ABS else rank

        k_sel = min(self.top_k, flat.numel())
        top = torch.topk(rank, k_sel).indices

        # Decode flat index -> (block, tok, unit). Blocks vary in width, so
        # bucketize against cumulative sizes rather than dividing by a constant.
        sizes = torch.tensor([len(toks) * self.width(h) for h, _ in keys],
                             device=flat.device)
        starts = torch.cat([torch.zeros(1, dtype=sizes.dtype,
                                        device=flat.device), sizes.cumsum(0)])
        block_of = torch.bucketize(top, starts[1:], right=True)
        rem = top - starts[block_of]

        widths = torch.tensor([self.width(h) for h, _ in keys],
                              device=flat.device)[block_of]
        tok_of = rem // widths
        unit_of = rem % widths

        toks_t = torch.tensor(toks, device=flat.device)
        node_tok = toks_t[tok_of]
        node_score = flat[top]

        # nodes[(hook, layer)] -> (tok LongTensor, unit LongTensor)
        nodes, detail = {}, []
        for b, key in enumerate(keys):
            sel = block_of == b
            if not sel.any():
                continue
            t_sel = node_tok[sel].to(self.device)
            u_sel = unit_of[sel].to(self.device)
            nodes[key] = (t_sel, u_sel)
            for t, u, s in zip(t_sel.tolist(), u_sel.tolist(),
                               node_score[sel].tolist()):
                detail.append((key[0], key[1], t, u, s))

        detail.sort(key=lambda d: -abs(d[4]))
        circuit = {"nodes": nodes, "n_nodes": int(k_sel), "detail": detail}

        by_hook = {}
        for h, l, _, _, _ in detail:
            by_hook.setdefault(h, set()).add(l)
        print(f"  circuit: {k_sel} nodes")
        for h, ls in by_hook.items():
            print(f"    {h}: {sum(1 for d in detail if d[0] == h)} nodes "
                  f"across layers {sorted(ls)}")
        print(f"  score range: [{node_score.min():.4f}, {node_score.max():.4f}]")
        return circuit

    # ------------------------------------------------------------------
    # Score: interchange the selected coordinates, source -> base
    # ------------------------------------------------------------------
    def score(self, circuit, eval_rows, split_name):
        nodes = circuit["nodes"]
        keys = list(nodes.keys())
        results = []

        for r in tqdm(eval_rows, desc=f"patch [{split_name}]", leave=False):
            src_text = r[self.args.source_input_col]
            base_text = r[self.args.base_input_col]
            id_A, id_B = self.AB_ids(r)

            # Clean (pre-intervention) logprobs.
            base_logp = self.logprobs_at_last(self.tokenize(base_text))
            src_logp = self.logprobs_at_last(self.tokenize(src_text))

            if src_logp[id_A].item() < src_logp[id_B].item():
                print(f"  WARNING: logp(A|src)={src_logp[id_A].item():.3f} < "
                      f"logp(B|src)={src_logp[id_B].item():.3f} for source: "
                      f"{src_text!r}")

            # Patched base forward: swap the circuit's coordinates in.
            src_acts = self._run_capture(self.tokenize(src_text), keys)
            handles = [self._register_patch(src_acts, nodes, k) for k in keys]
            try:
                with torch.no_grad():
                    out = self.model(self.tokenize(base_text))
            finally:
                for h in handles:
                    h.remove()

            logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)

            results.append({
                "source_input": src_text,
                "base_input": base_text,
                "source_logp_A": src_logp[id_A].item(),
                "source_logp_B": src_logp[id_B].item(),
                "base_logp_A": base_logp[id_A].item(),
                "base_logp_B": base_logp[id_B].item(),
                "base_intervention_logp_A": logp[id_A].item(),
                "base_intervention_logp_B": logp[id_B].item(),
                "n_nodes": circuit["n_nodes"],
            })

        torch.cuda.empty_cache()
        return results

    # ------------------------------------------------------------------
    # Circuit dump for --circuit_out
    # ------------------------------------------------------------------
    def circuit_rows(self, circuit):
        fieldnames = ["hook", "layer", "tok", "unit", "attribution"]
        rows = [{"hook": h, "layer": l, "tok": t, "unit": u, "attribution": s}
                for h, l, t, u, s in circuit["detail"]]
        return fieldnames, rows


if __name__ == "__main__":
    Circuit().main()