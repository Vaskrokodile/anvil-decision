# Anvil

In-browser fine-tuning studio for sub-1B specialist models. WebGPU probe → model
selection → dataset builder → job spec. Landing + Studio SPA.

## Environment

- Project lives on `E:\projects\anvil` — **C: has ~0 free space; do not install or write there.**
- npm cache is on `E:\npm-cache`.
- Shell is Windows PowerShell — no `&&`, use `;`.

## Commands

- `npm run dev` — Vite dev server (http://localhost:5173)
- `npm run build` — `tsc -b` + production build
- `npm run lint` — oxlint

## Structure

- `src/data/models.ts` — model roster (real HF ids, params, tiers). Edit candidates here.
- `src/data/sectors.ts` — sector definitions + dataset schemas.
- `src/lib/gpu.ts` — WebGPU probe + tier classification (T0–T4).
- `src/lib/dataset.ts` — .jsonl/.csv/.json/.txt parsers + `analyzeDataset()` health stats (dupes, per-column uniques, label guess, warnings) + `dedupeDataset()`.
- `src/lib/hfDatasets.ts` — datasets-server client: `parseDatasetRef` (id or URL), `fetchSplits`, `fetchRows` (ClassLabel→names, retry on 429). Shared with `scripts/fetch-starter.mts`, which regenerates `public/datasets/*.jsonl` + `index.json` (bundled starter sets).
- `src/lib/split.ts` — seeded stratified train/val split + `mulberry` rng. Both workers hold out 20% (`valFrac`) and report `valAcc`/`valN`/`trainN` in their `done` messages.
- `src/components/AuroraCanvas.tsx` — three.js star field over CSS aurora (lazy chunk).
- `src/studio/` — 5-step console: probe → model → dataset → run → configure/train.
- `src/lib/jev.ts` — open-jev wrapper (`kev-0.6b` / `open-jev` / `kev-4b`).
  NOTE: `gliner2`-config ONNX exports (e.g. GLiNER2.5-Decide-ONNX) are NOT
  supported by open-jev 0.1.2 — they need a dedicated encoder driver.
- `src/lib/embed.ts` — frozen MiniLM-L6 embedder (transformers.js).
- `src/lib/head.ts` + `src/lib/gpuTrain.ts` + `src/lib/train.worker.ts` —
  training engine v1: linear head over frozen embeddings, CPU + WGSL backends.
- `src/lib/safetensors.ts` / `trainedHead.ts` — artifact export & session store.
- `src/lib/xformer/` — own encoder implementation for real fine-tune:
  `spec.ts` (EncoderBackend contract + MINILM_L6 + paramShape),
  `weights.ts` (fetch MiniLM safetensors ~91MB), `cpu.ts` (reference backend),
  `wgsl.ts` (WebGPU compute backend, fwd+bwd), `ft.worker.ts` (train loop),
  `ft.ts` (artifact store + tuned-encoder inference). Modes: head (frozen
  embeddings), lora (adapters on q/k/v/o/f1/f2), full (all encoder params).

## Conventions

- "Aurora lab" design: full-bleed dark aurora hero (CSS `.aurora-sky` /
  `.aurora-dome` / `.planet` / `.dot-matrix` layers + lazy `AuroraCanvas` star
  field), then a paper-white body. One accent only — `blue` (#3b82f6). No green,
  no orange, no purple.
- Tokens in `src/index.css` `@theme`: light body = paper/card/line/fg/dim/faint;
  dark hero = abyss/navy + ice/ice-dim/ice-faint/ice-line.
- `Tag` tones: solid / blue / line / dim / faint — fill level carries meaning.
  "Over budget" uses hatched `.bar-over`. Cards can aurora-fill on hover via
  `.card-aurora`.
- Type: Instrument Serif (`font-display`) for headings, Space Grotesk body,
  JetBrains Mono for data/labels. Labels are bracketed: `[ 01 — roster ]`.
- Honesty rule: UI labels mark what is live vs. planned — keep it that way.
