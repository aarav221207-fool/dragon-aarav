# 🐉 BDH Inspector 

A small web microscope for Pathway's **Dragon Hatchling (BDH)** architecture. It visualizes real per-layer activation statistics like shapes, sparsity, mean / std / min / max, and samples, captured by instrumenting the official public model with forward hooks.

**🔗 Live demo: https://brainy-joseph.github.io/-dragonforge-bdh-inspector/**
*(no install, no local server, just open the link)*


---

## What is this?

[BDH (Dragon Hatchling)](https://github.com/pathwaycom/bdh) is a biologically inspired, post-Transformer architecture ([Kosowski et al., 2025](https://arxiv.org/abs/2509.26507)). This project runs the official public implementation, hooks every module (`embed`, `ln`, `attn`, `drop`), records summary statistics of each tensor as the model processes text, and renders them in a simple scrollable inspector.

## What you can see

- **Pipeline** — the layer sequence across all 6 repeated levels, with tensor shapes.
- **Sparsity** — fraction of zero activations per layer (BDH's sparse, ReLU-gated latent space).
- **Activations** — side-by-side comparison of the same statistics across 3 different prompts.
- **Layer stats** — click any layer in the tree to see numel, mean, std, min/max, sparsity and a sample sparkline.

## What the data shows (measured on the public model)

- **Strictly-causal attention fingerprint** — attention-output sparsity is exactly `1/T` (1/11, 1/19, 1/43 across the three prompts): the first token sees nothing, matching the `tril(diagonal=-1)` mask in `bdh.py`.
- **Gated sparsity ≈ 0.76–0.77** at every level consistent with two ~50%-sparse ReLU latents being multiplied (1 − 0.5² = 0.75).
- **Signal amplification across repeated levels** — attention-output std grows level by level (≈309 → 442 → 581 → 685 on "The cat sat"), even without training.

## An honest note

The official `pathwaycom/bdh` repository ships **no pretrained weights**, so every number here describes the *freshly initialized* architecture. Trained-model phenomena from the paper (e.g. monosemantic synapses) are **established in the paper, not measured here**. The same instrumentation script can be re-run on any trained checkpoint to extend these views.

## How it works

1. `instrument_bdh_v7.py` (run in a codespace on the official repo) loads `bdh.py`, registers forward hooks on every module, runs 3 prompts, and writes `bdh_microscope_rich.json`.
2. The JSON is converted to `data.js` (`const BDH_DATA = {...}`).
3. A dependency-free static site (`index.html` + `styles.css` + `js/`) renders the views, hosted on GitHub Pages.

## Roadmap

- [ ] Attention score heatmaps (raw strictly-causal Q·Kᵀ, per head and level)
- [ ] RoPE phase visualization from the `freqs` buffer
- [ ] Before/after training comparison (diff two instrumentation runs)
- [ ] More prompts and longer sequences

## Credits

- Architecture & code: [pathwaycom/bdh](https://github.com/pathwaycom/bdh) (MIT)
- Paper: *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* — [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)

## License

MIT
