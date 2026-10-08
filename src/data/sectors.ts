import type { Sector } from './models';

export interface SectorDef {
  id: Sector | 'custom';
  code: string;
  name: string;
  desc: string;
  /** Example record shapes the dataset builder scaffolds */
  schema: { field: string; role: 'input' | 'label' }[];
  examples: string[];
  suggested: string[]; // hf ids
}

export const SECTORS: SectorDef[] = [
  {
    id: 'decision',
    code: 'SEC-01',
    name: 'General Decision',
    desc: 'Typed decisions with confidence — routing, approvals, accept/reject, escalation. The JEPA-shaped problem: predict the outcome, not the text.',
    schema: [
      { field: 'context', role: 'input' },
      { field: 'decision', role: 'label' },
      { field: 'confidence', role: 'label' },
    ],
    examples: ['approve / hold / escalate', 'route to team A–F', 'ship or block'],
    suggested: ['fastino/GLiNER2.5-Decide', 'fastino/GLiNER2.5-multi-Decide'],
  },
  {
    id: 'scientific',
    code: 'SEC-02',
    name: 'Scientific',
    desc: 'Claims, methods, evidence strength, citation roles. Structure papers and reports into typed, checkable records.',
    schema: [
      { field: 'passage', role: 'input' },
      { field: 'claim_type', role: 'label' },
      { field: 'evidence_span', role: 'label' },
    ],
    examples: ['method / result / limitation', 'supporting vs. contradicting citation'],
    suggested: ['fastino/gliner2.5-multi-v1', 'Qwen/Qwen3-0.6B'],
  },
  {
    id: 'labelling',
    code: 'SEC-03',
    name: 'Data Labelling',
    desc: 'Turn raw text into training data: entities, spans, taxonomy tags, attributes. Fine-tune the labeller on your annotation guide.',
    schema: [
      { field: 'text', role: 'input' },
      { field: 'entities', role: 'label' },
      { field: 'taxonomy', role: 'label' },
    ],
    examples: ['product NER for catalog', 'topic taxonomy tagging', 'span-level annotation'],
    suggested: ['fastino/gliner2.5-small-v1', 'fastino/gliner2.5-base-v1'],
  },
  {
    id: 'trust',
    code: 'SEC-04',
    name: 'PII & Moderation',
    desc: 'Redaction and guardrails before anything leaves the machine. Train the filter on what your org counts as sensitive.',
    schema: [
      { field: 'text', role: 'input' },
      { field: 'pii_spans', role: 'label' },
      { field: 'policy_verdict', role: 'label' },
    ],
    examples: ['names, IDs, accounts', 'jailbreak / injection screening'],
    suggested: ['fastino/GLiNER2-Guardrails-PII-Multi'],
  },
  {
    id: 'custom',
    code: 'SEC-05',
    name: 'Your Sector',
    desc: 'Bring your own task and your own schema. Import JSONL / CSV, define the labels, pick any model that fits your GPU.',
    schema: [
      { field: 'input', role: 'input' },
      { field: 'label', role: 'label' },
    ],
    examples: ['your domain, your labels'],
    suggested: [],
  },
];
