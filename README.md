# anvil-decision

**Local-first model workbench in the browser.** Probe your GPU, pick a small
model, run typed-decision inference, and fine-tune on your own data — nothing
leaves the tab.

Built around Fastino's [GLiNER2.5](https://huggingface.co/fastino) decision
models and Jev-style typed outputs (a `state` + structured questions → one
forward pass → calibrated answers).

## What's live

| | |
|---|---|
| Hardware probe | WebGPU adapter → tier → which models fit |
| Typed-decision inference | GLiNER2.5-Decide (custom `gliner2` ONNX driver) + open-jev family (`kev-0.6b`, `open-jev`, `kev-4b`), WebGPU or WASM |
| Decoder-LM inference | transformers.js text-generation for decoder roster entries (e.g. SmolLM2) — labeled generation, device/dtype shown |
| Dataset builder | `.jsonl` / `.csv` / `.json` / `.txt` import, sector templates, column mapping — plus bundled starter sets and Hugging Face import (public datasets → rows in-tab), health stats (dupes, class balance, label detection), dedupe |
| **Training engine v1** | Linear head over frozen MiniLM-L6 embeddings — WGSL compute or CPU, real loss/accuracy, exports `.safetensors` |
| **Training engine v2** | Our own BERT encoder forward **and backward** (attention, LayerNorm, GELU, LoRA, AdamW) implemented in WGSL — LoRA and full fine-tune of MiniLM-L6 in a Worker. WGSL output is parity-checked against the CPU reference on-device before training/inference proceeds; failure → automatic CPU fallback |
| **DeBERTa-v3-small substrate** | Second fine-tune substrate: real DeBERTa-v3 disentangled attention (qk + c2p + p2c, shared pos projections) fwd+bwd, CPU — verified to bit-level logits parity vs the reference ONNX model. ~570 MB weight download |
| **Multi-task training** | Pick several label columns → one shared encoder + one head per task; shared gradients flow through the encoder |
| Held-out eval | Seeded stratified 80/20 train/val split; `val` accuracy is on rows the model never trained on. Optional eval report compares the trained artifact against the selected roster baseline on held-out rows — accuracy, macro-F1, per-class P/R/F1, confusion matrix, latency, agreement |
| Artifacts | One portable `.zip` per run: `weights/*.safetensors` (adapter or full params + heads), `model.card.json`, `weights/predict.mjs` runnable inference snippet |

## What's not

- LoRA/full fine-tune on roster models themselves — the roster pick is the
  inference/eval baseline and question pack; the fine-tune substrate is
  MiniLM-L6 or DeBERTa-v3-small.
- DeBERTa substrate is CPU-only (no WGSL disentangled-attention backend yet).
- OPFS checkpoint persistence (in-session only).

## Quickstart

```sh
npm i
npm run dev   # → http://localhost:5173
```

Studio flow: `#/studio` → **probe → model → dataset → run → configure/train**.

Verification scripts (Node ≥22, strips TS types):

```sh
npm test                       # parsers, artifacts, zip, bench + encoder grad checks
npm run test:weights           # real-weight parity vs HF (MiniLM + DeBERTa, big downloads)
npm run test:e2e               # Playwright studio-flow smoke (needs browsers installed)
node scripts/parity.mts        # encoder forward == transformers.js MiniLM (cos 1.0)
node scripts/parity-deberta.mts  # deberta encoder → NLI logits == reference ONNX
node scripts/ft-smoke.mts      # CPU LoRA fine-tune: loss 0.93 → 0.009, acc → 1.0
node scripts/xformer-smoke.mts # finite-difference grad check (minilm)
node scripts/deberta-smoke.mts # finite-difference grad check (deberta)
node scripts/fetch-starter.mts # regenerate public/datasets/* from HF
```

## Stack

React 19 · TypeScript · Vite · Tailwind v4 · transformers.js (ONNX Runtime
WebGPU/WASM) · open-jev · custom WGSL compute engine · three.js.

## Licensing

Code: MIT (see `LICENSE`). Model weights keep their own licenses —
GLiNER2.5-* are Apache-2.0 (fine-tuning allowed; Fastino trademarks not
included), MiniLM Apache-2.0, kev models per their repos. Bundled starter
datasets in `public/datasets/` are subsets of public Hugging Face datasets —
each entry in `index.json` names its source and license. Anvil is an
independent project, not affiliated with Fastino.
