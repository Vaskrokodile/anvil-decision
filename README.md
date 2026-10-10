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
| Dataset builder | `.jsonl` / `.csv` / `.json` / `.txt` import, sector templates, column mapping — plus bundled starter sets and Hugging Face import (public datasets → rows in-tab), health stats (dupes, class balance, label detection), dedupe |
| **Training engine v1** | Linear head over frozen MiniLM-L6 embeddings — WGSL compute or CPU, real loss/accuracy, exports `.safetensors` |
| **Training engine v2** | Our own BERT encoder forward **and backward** (attention, LayerNorm, GELU, LoRA, AdamW) implemented in WGSL — LoRA and full fine-tune of MiniLM-L6 in a Worker |
| Held-out eval | Seeded stratified 80/20 train/val split in both engines; reported `val` accuracy is on rows the model never trained on |
| Artifacts | `.safetensors` export + reload; trained heads/adapters run in the Run step |

## What's not

- LoRA/full fine-tune on models *other* than MiniLM-L6 (roster models are
  probe/fit metadata for now).
- OPFS checkpoint persistence (in-session only).
- The WGSL backend is verified structurally; numerical parity vs the CPU
  reference is proven on CPU — GPU-path bugs would surface as a training
  failure, and the worker auto-falls back to CPU.

## Quickstart

```sh
npm i
npm run dev   # → http://localhost:5173
```

Studio flow: `#/studio` → **probe → model → dataset → run → configure/train**.

Verification scripts (Node ≥22, strips TS types):

```sh
node scripts/parity.mts     # encoder forward == transformers.js MiniLM (cos 1.0)
node scripts/ft-smoke.mts   # CPU LoRA fine-tune: loss 0.93 → 0.009, acc → 1.0
node scripts/xformer-smoke.mts  # finite-difference grad check
node scripts/fetch-starter.mts  # regenerate public/datasets/* from HF
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
