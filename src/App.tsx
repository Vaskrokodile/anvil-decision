import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { ArrowLeft, ArrowUpRight } from '@phosphor-icons/react';
import Nav from './components/Nav';
import Hero from './components/Hero';
import Landing from './components/Landing';
import Footer from './components/Footer';
import { useGpuReport } from './lib/useGpu';
import { TIER_LABELS } from './data/models';

const Studio = lazy(() => import('./studio/Studio'));

type Route = '/' | '/studio';

function routeFromHash(): Route {
  return location.hash.replace('#', '') === '/studio' ? '/studio' : '/';
}

/** Minimal light-theme top bar for the studio route (Nav is hero-styled/dark). */
function StudioBar() {
  const gpu = useGpuReport();
  return (
    <div className="mx-auto flex h-16 max-w-[1340px] items-center justify-between px-6">
      <a href="#/" className="flex items-center gap-3">
        <span className="inline-block h-3.5 w-3.5 rotate-45 rounded-[3px] bg-fg" />
        <span className="font-sans text-[15px] font-semibold tracking-[0.22em] text-fg">
          ANVIL
        </span>
        <span className="font-mono text-[10px] tracking-widest text-faint">studio</span>
      </a>
      <div className="flex items-center gap-4">
        <span className="hidden items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-faint sm:flex">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${
              gpu === null ? 'animate-pulse bg-faint' : gpu.ok ? 'bg-blue' : 'bg-line'
            }`}
          />
          {gpu === null ? 'probing…' : gpu.ok ? TIER_LABELS[gpu.tier] : 'no webgpu'}
        </span>
        <a
          href="#/"
          className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.15em] text-dim transition-colors hover:text-fg"
        >
          <ArrowLeft size={12} />
          site
        </a>
      </div>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);

  useEffect(() => {
    const onHash = () => {
      setRoute(routeFromHash());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const goStudio = useCallback(() => {
    location.hash = '/studio';
  }, []);

  if (route === '/studio') {
    return (
      <div className="min-h-svh bg-paper text-fg">
        <StudioBar />
        <Suspense fallback={<div className="min-h-svh" />}>
          <Studio />
        </Suspense>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-svh bg-paper text-fg">
      <Nav onStudio={goStudio} />
      <Hero onStudio={goStudio} />
      <Landing />
      <Footer />
      {/* floating studio shortcut mirrors the ref's persistent CTA */}
      <button
        onClick={goStudio}
        className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full bg-fg px-5 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-card shadow-lg transition-opacity hover:opacity-85"
      >
        Studio
        <ArrowUpRight size={13} />
      </button>
    </div>
  );
}
