import { useEffect, useState } from 'react';
import { probeGpu, type GpuReport } from './gpu';

/** Module-level memo so every component shares one probe. */
let probePromise: Promise<GpuReport> | null = null;

export function useGpuReport(): GpuReport | null {
  const [report, setReport] = useState<GpuReport | null>(null);
  useEffect(() => {
    probePromise ??= probeGpu();
    let live = true;
    probePromise.then((r) => live && setReport(r));
    return () => {
      live = false;
    };
  }, []);
  return report;
}
