"""
CircuitIntervention
"""

import csv
import os
from abc import abstractmethod

from intervention import Intervention


class CircuitIntervention(Intervention):

    BASE_FIELDS = [
        "split",
        "inverse",
        "source_input",
        "base_input",
        "source_logp_A",
        "source_logp_B",
        "base_logp_A",
        "base_logp_B",
        "base_intervention_logp_A",
        "base_intervention_logp_B",
    ]

    # Subclasses append their own per-row columns here (e.g. ["n_nodes"]).
    EXTRA_FIELDS = []

    # Subclass default for circuit size; overridden by --top_k. Subclasses that
    # leave this None require --top_k to be passed explicitly.
    TOP_K = None

    # Sites at which circuit nodes may live, as a subclass-declared vocabulary.
    # A node is then (hook_point, layer, tok, unit). Populating this enables
    # --hook_points; leaving it empty disables the flag entirely.
    #
    # Names are subclass-defined, but a useful convention (cf. Arora et al.
    # 2025, who compare exactly these) is:
    #
    #   "mlp_act"   input to down_proj: act_fn(gate_proj(x)) * up_proj(x)
    #               -- the privileged, elementwise-nonlinear basis
    #   "mlp_out"   output of down_proj
    #   "attn_out"  output of the attention block
    #   "resid"     the decoder layer's output (residual stream post)
    #
    # These are NOT interchangeable: sparsity and interpretability differ by
    # roughly two orders of magnitude between mlp_act and mlp_out, so a circuit
    # is only comparable to another circuit found at the same site.
    HOOK_POINTS = ()
    DEFAULT_HOOK_POINTS = ()

    def __init__(self, args=None):
        super().__init__(args)
        if not self.args.add_inverse:
            print("  WARNING: --add_inverse is not set. Circuit attribution is "
                  "naturally bidirectional; without it the circuit is fit and "
                  "evaluated on one direction only.")

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @classmethod
    def build_parser(cls):
        p = super().build_parser()
        p.add_argument("--top_k", type=int, default=None,
                       help="Number of nodes in the circuit. Overrides the "
                            "subclass's TOP_K default.")
        if cls.HOOK_POINTS:
            p.add_argument("--hook_points", nargs="*", default=None,
                           choices=list(cls.HOOK_POINTS),
                           help="Sites at which circuit nodes may live. "
                                f"Supported: {', '.join(cls.HOOK_POINTS)}. "
                                f"Default: "
                                f"{', '.join(cls.DEFAULT_HOOK_POINTS) or 'all'}. "
                                "Nodes from several sites compete in one ranking, "
                                "so pass more than one only if their activation "
                                "scales are comparable.")
        p.add_argument("--circuit_out", default=None,
                       help="Optional path for a CSV describing the fitted "
                            "circuit itself (one row per node). Written only if "
                            "the subclass implements circuit_rows().")
        return p

    @property
    def top_k(self):
        """Resolved circuit size: --top_k if given, else the subclass TOP_K."""
        k = self.args.top_k if self.args.top_k is not None else self.TOP_K
        if k is None:
            raise ValueError(
                f"{type(self).__name__} defines no TOP_K default; pass --top_k.")
        if k < 1:
            raise ValueError(f"--top_k must be >= 1, got {k}")
        return k

    @property
    def hook_points(self):
        """
        Resolved hook points, deduplicated and ordered as in HOOK_POINTS.
        Falls back to DEFAULT_HOOK_POINTS, then to every supported site.
        """
        if not self.HOOK_POINTS:
            return ()
        chosen = getattr(self.args, "hook_points", None)
        if not chosen:
            chosen = self.DEFAULT_HOOK_POINTS or self.HOOK_POINTS
        unknown = set(chosen) - set(self.HOOK_POINTS)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} does not support hook point(s) "
                f"{sorted(unknown)}; supported: {list(self.HOOK_POINTS)}")
        return tuple(h for h in self.HOOK_POINTS if h in set(chosen))

    @property
    def OUTPUT_FIELDS(self):
        return self.BASE_FIELDS + list(self.EXTRA_FIELDS)

    # ------------------------------------------------------------------
    # Search space: the grid is where nodes may live, not a loop
    # ------------------------------------------------------------------
    def candidate_cells(self):
        """
        [(layer, tok), ...] over which circuit nodes are searched. Restricted by
        --layers / --toks; defaults to every layer x every token position.
        """
        return [(l, t) for l in self.get_layers() for t in self.get_toks()]

    def candidate_sites(self):
        """
        [(hook_point, layer, tok), ...]: the full search space. Equals
        candidate_cells() crossed with hook_points, or the bare cells when the
        subclass declares no HOOK_POINTS.
        """
        if not self.hook_points:
            return [(None, l, t) for l, t in self.candidate_cells()]
        return [(h, l, t) for h in self.hook_points
                for l, t in self.candidate_cells()]

    # ------------------------------------------------------------------
    # The method-specific work
    # ------------------------------------------------------------------
    @abstractmethod
    def fit_circuit(self, train_rows):
        """
        Identify the circuit from `train_rows`, searching over
        `candidate_sites()` and keeping `self.top_k` nodes. Called exactly once.

        Returns an opaque object handed back to `score` and `circuit_rows`.
        Rows carry an `inverse` flag; attribution should be averaged over both
        directions rather than filtered (see module docstring).
        """
        ...

    @abstractmethod
    def score(self, circuit, eval_rows, split_name):
        """
        Evaluate `circuit` on `eval_rows` (belonging to `split_name`).

        Return a list of dicts, ONE PER EVAL ROW AND IN EVAL-ROW ORDER, with the
        value columns of BASE_FIELDS (minus `split` and `inverse`, which `main`
        fills) plus any EXTRA_FIELDS the subclass declares.
        """
        ...

    def circuit_rows(self, circuit):
        """
        Optional. Return a (fieldnames, rows) pair describing the circuit, one
        row per node. Written to --circuit_out if both are provided.
        """
        return None, None

    # ------------------------------------------------------------------
    # Unused: this class replaces the per-cell driver
    # ------------------------------------------------------------------
    def intervention(self, train_rows, eval_rows, layer, tok, split_name):
        raise RuntimeError(
            f"{type(self).__name__} is a CircuitIntervention; its unit of "
            f"analysis spans layers and positions. Implement fit_circuit() and "
            f"score() instead of intervention()."
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def main(self):
        self.load_rows()

        cells = self.candidate_cells()
        print(f"  searching {len(cells)} candidate cells "
              f"({len(self.get_layers())} layers x {len(self.get_toks())} toks)")
        if self.hook_points:
            print(f"  hook points: {', '.join(self.hook_points)}")
        print(f"  circuit size k = {self.top_k}")
        print(f"  train: {len(self.train_rows)} rows "
              f"({len(self.forward_rows(self.train_rows))} forward)")
        print(f"  test:  {len(self.test_rows)} rows "
              f"({len(self.forward_rows(self.test_rows))} forward)")

        circuit = self.fit_circuit(self.train_rows)

        out_rows = []
        for split_name, eval_rows in [("train", self.train_rows),
                                      ("test", self.test_rows)]:
            if not eval_rows:
                continue
            results = self.score(circuit, eval_rows, split_name)
            if len(results) != len(eval_rows):
                raise RuntimeError(
                    f"{type(self).__name__}.score returned {len(results)} "
                    f"results for {len(eval_rows)} eval rows; results must be "
                    f"one-per-row and in order."
                )
            for res, src_row in zip(results, eval_rows):
                row = dict(res)
                row["split"] = split_name
                row["inverse"] = "1" if self.is_inverse(src_row) else "0"
                out_rows.append(row)

        self.write_output(out_rows)
        self.write_circuit(circuit)
        return out_rows

    def write_circuit(self, circuit):
        if not self.args.circuit_out:
            return
        fieldnames, rows = self.circuit_rows(circuit)
        if not fieldnames or rows is None:
            print("  note: --circuit_out given but circuit_rows() is not "
                  "implemented; skipping.")
            return
        out_dir = os.path.dirname(self.args.circuit_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.args.circuit_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        print(f"  wrote {len(rows)} circuit nodes to {self.args.circuit_out}")


if __name__ == "__main__":
    raise SystemExit(
        "CircuitIntervention is abstract; run a concrete subclass.")