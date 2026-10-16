# Improving latent-space organization — findings & plan

Context: goal is to improve *organization* of the VAE latent space for the VCTK-words
model. We tried an NF **posterior** (`vctk_words_nf`, `nf_posterior_layers=10`) hoping it
would help, but FAD/KID/Vendi + unconditional-generation metrics are ~equal to no-NF, and
latent probes are identical on gender/accent and slightly *worse* on word decodability.
The diagnosis below explains why, and what to try instead.

## Why the NF posterior didn't help (from `flow_posterior_analysis.ipynb`)

The posterior flow trains **correctly** — it is not a bug:
- **Not near-identity.** `mean|log det J| ≈ 7.5`, relative displacement `||z_flow−z||/||z|| ≈ 0.78`.
- **Lowers the latent loss** ≈ −233 nats/sample, almost all via the prior-match term.
- **Gaussianizes + centres** the aggregate posterior (mean|μ| 0.67 → 0.14, std ≈ 1).
- **Spreads** the latent: participation ratio 80 → 95 (matches higher fidelity in the NF dump).
- **But it barely mixes dimensions:** per-dim `corr(z, z_flow) ≈ 0.997`. So it's a
  *monotonic per-dimension reparametrization* (shift/rescale/mild nonlinearity), not a rotation.

**Conclusion:** the NF posterior is added capacity without benefit on this data. Its only
footprint is a higher-dimensional, less axis-aligned latent — which is why fine-grained
**linear** decodability (word) drops slightly while gender (big margin) is unaffected, and
why perceptual metrics don't move (a bijection preserves what the decoder can use).

## The key insight — why a posterior flow is the wrong tool for organization

A posterior flow is a **bijection after the encoder**, trained only to match `q(z|x)` to
`N(0,I)` (its loss is `KL(q ‖ N(0,I))`). It has **no incentive to organize anything
perceptually**, and being invertible it **cannot disentangle** — it can't create/destroy
statistical dependence between factors in a way a downstream decoder/probe can't undo. A
*more* expressive flow (autoregressive, more coupling capacity) would Gaussianize *better*
and mix dims more, but that is orthogonal to — and can even hurt — linear organization.

### The paradox and its resolution (β vs. flow both "spread variance / Gaussianize")

Both raising β and adding a flow push the aggregate posterior toward `N(0,I)` and spread
variance — same end-state histograms. They differ in **what must change to get there**:

- **Raising β (no-NF):** increases the weight on `KL(q(z|x) ‖ p(z))`. `q` is produced by the
  encoder, so the only way to lower KL is to **change the encoder** — it must give up
  information, collapse unjustified dims, and keep factors that earn their KL cost. That
  reshaping of the data→latent map **is** disentanglement; Gaussianization is a side effect.
- **Adding a posterior flow:** lowers KL by inserting a bijection *after* the encoder and
  measuring KL on `z_flow`. The flow pays the KL bill, so the **encoder is never pressured to
  economize** — organization of the representation is untouched; only the coordinates are
  warped. (Exactly what the diagnostic showed.)

So: "spread variance via reparametrization (flow)" ≠ "spread variance via re-encoding (β)".
Only β acts on the encoder — the thing we want to change. **A bijection cannot; β can.**

## Recommended experiments (in order of expected payoff)

First decide what "organization" means for the thesis — it forks the plan:

1. **Disentanglement / smoothness / less collapse → β / KL-schedule sweep (highest leverage).**
   Sweep β (KL weight) on the **no-NF** model. Fidelity analysis suggests the latent is
   under-compressed (high-dim, diffuse); more KL pressure pushes toward fewer, cleaner,
   more axis-aligned factors (β-VAE). **Caveat:** β-VAE disentanglement is finicky and
   dataset-dependent — too high → posterior collapse + worse reconstruction, not clean
   factors. Treat as a bet to measure, not a guarantee.
   - Measure at each β with the existing probes: gender/accent/duration decodability,
     word classifier (robust N-seed version), silhouette, fidelity/participation ratio.

2. **Linear decodability of attributes specifically → supervision in the objective.**
   Add a light auxiliary loss on the latent: a classification/regression head (predict
   word/gender, backprop into encoder) or a contrastive term (same-word-different-speaker
   latents pulled together). This directly optimizes what the probes measure; an
   unsupervised bijection cannot.

3. **Only if the thesis needs the flow to *be* the mechanism → add permutations.**
   The `0.997` per-dim correlation means the RealNVP couplings don't circulate information
   across dims. Add **permutations / 1×1 invertible convs (Glow-style) between couplings** —
   cheapest change that directly attacks the "doesn't rotate the code" finding.
   - **Autoregressive (IAF/MAF):** strictly more expressive but heavy, and improves the
     Gaussian match with **no** guarantee of better perceptual organization — high cost,
     uncertain payoff. Skip unless maximal posterior flexibility is the explicit claim.
   - Richer coupling `hidden_features`: marginal; doesn't address mixing.

## Bottom line

Don't reach for a fancier flow first. The flow is downstream of the encoder and can't change
organization by construction (confirmed empirically). β is the only knob here that acts on
the encoder, so it's the right first experiment — but verify with the probes rather than
assuming it works. Use supervision if the goal is specifically linear attribute structure.
