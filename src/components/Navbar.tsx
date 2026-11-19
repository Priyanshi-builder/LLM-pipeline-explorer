interface NavbarProps {
  activeSection: string;
  onResetLab: () => void;
}

const NAV_LINKS = [
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'vocabulary', label: 'Vocabulary' },
  { id: 'embeddings', label: 'Embeddings' },
];

export function Navbar({ activeSection, onResetLab }: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#070B14]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between gap-8 px-6 py-3.5">
        <a
          href="#pipeline"
          className="text-base font-semibold tracking-tight text-slate-100 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-sm"
        >
          LLM Pipeline Explorer
        </a>

        <nav
          aria-label="Primary sections"
          className="hidden md:flex items-center gap-7 text-sm font-medium text-slate-400"
        >
          {NAV_LINKS.map((link) => {
            const isActive = activeSection === link.id;
            return (
              <a
                key={link.id}
                href={`#${link.id}`}
                className={`relative py-1 transition-colors duration-150 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-sm ${
                  isActive ? 'text-cyan-300' : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {link.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 -bottom-[15px] h-0.5 bg-cyan-400"
                  />
                )}
              </a>
            );
          })}
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onResetLab}
            className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-xs font-medium text-cyan-200 transition-colors duration-150 hover:bg-cyan-500/20 hover:border-cyan-400/70 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
          >
            Reset Lab
          </button>
        </div>
      </div>
    </header>
  );
}
