"""
AblationIntervention
"""

from abc import abstractmethod

from circuit_intervention import CircuitIntervention


class AblationIntervention(CircuitIntervention):

    # Modes handed to `run_ablated`. Order fixes the column order below.
    MODES = ("none", "complement", "circuit", "all")

    BASE_FIELDS = [
        "split",
        "inverse",
        "eval_input",
        "counterfactual_input",
        "clean_logp_A",
        "clean_logp_B",
        "complement_ablation_logp_A",
        "complement_ablation_logp_B",
        "circuit_ablation_logp_A",
        "circuit_ablation_logp_B",
        "all_ablation_logp_A",
        "all_ablation_logp_B",
    ]

    EXTRA_FIELDS = ["n_nodes"]

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @classmethod
    def build_parser(cls):
        p = super().build_parser()
        p.add_argument("--ablation", default="mean", choices=["mean", "zero"],
                       help="Value substituted for an ablated node: its mean "
                            "activation over the train rows, or zero. Zero is "
                            "simpler but off-distribution.")
        return p

    # ------------------------------------------------------------------
    # Which sentence the metric is read from
    #
    # `source_completion_A` agrees with the source, so evaluating on the source
    # makes A the correct token and m = logit_A - logit_B positive when clean.
    # The base sentence is carried through as `counterfactual_input` -- it is
    # used to compute the attribution baseline v(x'), never run for the metric.
    # ------------------------------------------------------------------
    def eval_text(self, row):
        return row[self.args.source_input_col]

    def counterfactual_text(self, row):
        return row[self.args.base_input_col]

    # ------------------------------------------------------------------
    # The method-specific work
    # ------------------------------------------------------------------
    @abstractmethod
    def ablation_values(self, train_rows):
        """
        Compute the value each node takes when ablated: the mean activation at
        every (hook, layer, tok) over `train_rows`, or zeros if --ablation zero.

        Called once, before scoring. Returns an opaque object handed to
        `run_ablated`.
        """
        ...

    @abstractmethod
    def run_ablated(self, row, circuit, values, mode):
        """
        Run the model on `eval_text(row)` with nodes ablated according to `mode`:

            "none"        no ablation (the clean run)
            "complement"  ablate every candidate node NOT in the circuit
            "circuit"     ablate exactly the circuit's nodes
            "all"         ablate every candidate node

        "complement" and "all" range over `candidate_sites()`, so what counts as
        the complement depends on --hook_points, --layers and --toks. A circuit
        can only be faithful relative to the search space it was drawn from; a
        narrow --layers makes faithfulness easy and means little.

        Return (logp_A, logp_B) at the last position.
        """
        ...

    # ------------------------------------------------------------------
    # Score: four runs per row, no interchange anywhere
    # ------------------------------------------------------------------
    def score(self, circuit, eval_rows, split_name):
        values = self._values
        results = []

        for r in eval_rows:
            out = {}
            for mode in self.MODES:
                lp_A, lp_B = self.run_ablated(r, circuit, values, mode)
                prefix = "clean" if mode == "none" else f"{mode}_ablation"
                out[f"{prefix}_logp_A"] = lp_A
                out[f"{prefix}_logp_B"] = lp_B

            if out["clean_logp_A"] < out["clean_logp_B"]:
                print(f"  WARNING: clean logp(A)={out['clean_logp_A']:.3f} < "
                      f"logp(B)={out['clean_logp_B']:.3f} on eval input: "
                      f"{self.eval_text(r)!r}")

            out["eval_input"] = self.eval_text(r)
            out["counterfactual_input"] = self.counterfactual_text(r)
            out["n_nodes"] = circuit["n_nodes"]
            results.append(out)

        return results

    # ------------------------------------------------------------------
    # Driver: compute ablation values once, between fit and score
    # ------------------------------------------------------------------
    def main(self):
        self.load_rows()

        cells = self.candidate_cells()
        print(f"  searching {len(cells)} candidate cells "
              f"({len(self.get_layers())} layers x {len(self.get_toks())} toks)")
        if self.hook_points:
            print(f"  hook points: {', '.join(self.hook_points)}")
        print(f"  circuit size k = {self.top_k}")
        print(f"  ablation: {self.args.ablation}")
        print(f"  train: {len(self.train_rows)} rows "
              f"({len(self.forward_rows(self.train_rows))} forward)")
        print(f"  test:  {len(self.test_rows)} rows "
              f"({len(self.forward_rows(self.test_rows))} forward)")

        circuit = self.fit_circuit(self.train_rows)
        self._values = self.ablation_values(self.train_rows)

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


if __name__ == "__main__":
    raise SystemExit(
        "AblationIntervention is abstract; run a concrete subclass.")