export default function Footer() {
  return (
    <footer className="border-t border-line bg-paper">
      <div className="mx-auto flex max-w-[1340px] flex-wrap items-end justify-between gap-4 px-6 py-10">
        <div>
          <p className="font-sans text-[15px] font-semibold tracking-[0.22em] text-fg">
            ANVIL
          </p>
          <p className="mt-1 font-mono text-[11px] text-faint">
            small models, hot metal, your machine
          </p>
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-faint">
          <span>local-first</span>
          <span className="mx-3 text-line">/</span>
          <span>webgpu</span>
          <span className="mx-3 text-line">/</span>
          <span>≤ 1.24B params</span>
        </div>
      </div>
    </footer>
  );
}
