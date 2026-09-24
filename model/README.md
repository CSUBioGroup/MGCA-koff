# Final MGCA model contract

`model.py` is the frozen final architecture used by the experiment workflow.
The development tag `v10_unbounded` is provenance only; the manuscript name is
MGCA.

Protein expert windows use `even_span_v2`. For an encoder with depth `N` and
window width `k`, the four starts are
`1 + round(i * (N - k) / 3)` for `i = 0, 1, 2, 3`, implemented with
deterministic integer half-up rounding. For ESM2-t36 this gives windows
`1-2, 12-13, 24-25, 35-36` when `k=2`, and
`1-8, 10-17, 20-27, 29-36` when `k=8`. The historical
`legacy_anchors_v1` option is retained only for explicit compatibility with
archived runs.

Cache files produced by `even_span_v2` always include the explicit
`__wleven_span_v2` suffix. Historical `legacy_anchors_v1` caches retain their
original `__wsN` names, preventing silent reuse across layouts.

For a batch of protein and ligand expert tensors, the network constructs:

1. a protein anchor `a_p` from gated ESM2 experts;
2. a ligand correction `Delta_d` from gated Morgan experts;
3. a joint correction `Delta_j` from low-rank bidirectional cross-attention;
4. `h_p = N(a_p)`;
5. `h_d = N(a_p + alpha_d * Delta_d)`;
6. `h_j = N(a_p + alpha_d * Delta_d + m_j * alpha_j * Delta_j)`.

`alpha_d` and `alpha_j` are positive global softplus coefficients without an
upper cap. `m_j` implements training-only joint-branch dropout and equals one
at inference.

One shared head `g_theta` is called on all three states:

```text
y_p = g_theta(h_p)
y_d = g_theta(h_d)
y_j = g_theta(h_j)
```

These are three calls to one parameter-sharing MLP within one forward pass,
not three models or three training runs. Training performs one backward pass
and one optimizer step per batch. `y_j` is the final inference output.

The head is `Linear(512,512) -> LayerNorm -> SiLU -> Linear(512,256) ->
SiLU -> Linear(256,1)` with no active head dropout. The main objective is final
MSE. Protein auxiliary MSE and two masked correction-utility terms provide weak
training regularization with fixed weights 0.10, 0.02 and 0.02.

The complete constructor, ablation switches and returned diagnostic tensors
are documented directly in `model.py`.
