import { useMemo, useState } from 'react';
import {
  ENCODING_METADATA,
  type TokenizerEncodingId,
  describeTokenCategory,
  loadEncoding,
  tokenizeTextSync,
} from '../lib/tokenizer';

interface TokenizerSectionProps {
  text: string;
  onTextChange: (next: string) => void;
  encodingId: TokenizerEncodingId;
  onEncodingChange: (next: TokenizerEncodingId) => void;
  onLookupVocabToken: (tokenId: number) => void;
}

const TOKENIZER_PRESETS = [
  {
    label: 'Case & Spacing',
    text: 'Apple vs apple vs  apple — leading spaces change token IDs!',
  },
  {
    label: 'Subword Morphology',
    text: 'Uncharacteristically antidisestablishmentarianism tokenization.',
  },
  {
    label: 'Code & Indentation',
    text: 'function softmax(logits: number[]): number[] {\n  return logits.map(Math.exp);\n}',
  },
  {
    label: 'Numbers & Unicode',
    text: '2026-11-19 · $1,048,576 · Café ☕ · 東京',
  },
];

const TOKEN_PALETTE = [
  {
    box: 'border-cyan-500/45 bg-cyan-500/12 text-cyan-100',
    active: 'border-cyan-300 bg-cyan-500/28 ring-2 ring-cyan-400/50 text-white',
    idText: 'text-cyan-300',
  },
  {
    box: 'border-blue-500/45 bg-blue-500/12 text-blue-100',
    active: 'border-blue-300 bg-blue-500/28 ring-2 ring-blue-400/50 text-white',
    idText: 'text-blue-300',
  },
  {
    box: 'border-violet-500/45 bg-violet-500/12 text-violet-100',
    active: 'border-violet-300 bg-violet-500/28 ring-2 ring-violet-400/50 text-white',
    idText: 'text-violet-300',
  },
  {
    box: 'border-indigo-500/45 bg-indigo-500/12 text-indigo-100',
    active: 'border-indigo-300 bg-indigo-500/28 ring-2 ring-indigo-400/50 text-white',
    idText: 'text-indigo-300',
  },
  {
    box: 'border-teal-500/45 bg-teal-500/12 text-teal-100',
    active: 'border-teal-300 bg-teal-500/28 ring-2 ring-teal-400/50 text-white',
    idText: 'text-teal-300',
  },
];

export function TokenizerSection({
  text,
  onTextChange,
  encodingId,
  onEncodingChange,
  onLookupVocabToken,
}: TokenizerSectionProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);
  const [loadingEncoding, setLoadingEncoding] = useState<TokenizerEncodingId | null>(null);

  const result = useMemo(
    () => tokenizeTextSync(text, encodingId),
    [text, encodingId],
  );

  const activeIdx =
    hoveredIdx !== null
      ? hoveredIdx
      : pinnedIdx !== null && pinnedIdx < result.tokens.length
        ? pinnedIdx
        : result.tokens.length > 0
          ? 0
          : null;

  const activeToken =
    activeIdx !== null ? (result.tokens[activeIdx] ?? null) : null;

  const handleSelectEncoding = async (nextId: TokenizerEncodingId) => {
    if (nextId === encodingId) return;
    setLoadingEncoding(nextId);
    try {
      await loadEncoding(nextId);
      onEncodingChange(nextId);
    } finally {
      setLoadingEncoding(null);
    }
  };

  const encodingMeta = ENCODING_METADATA[encodingId];

  return (
    <section id="tokenizer" className="border-b border-slate-800/80 bg-[#070B14]">
      <div className="mx-auto max-w-[1360px] px-6 py-16">
        {/* Section Header + Encoding Switcher */}
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-8">
          <div>
            <p className="text-xs font-mono text-cyan-400 mb-2">
              Real Byte-Pair Encoding (BPE) Workbench
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-50">
              02. Tokenizer Explorer
            </h2>
            <p className="mt-2 max-w-2xl text-[15px] text-slate-300 leading-relaxed">
              LLMs read subword chunks rather than whole words or characters. Type any
              sentence to observe real BPE merges, token ID assignments, and UTF-8
              byte splits.
            </p>
          </div>

          <div className="flex flex-col sm:items-end gap-2">
            <div
              role="group"
              aria-label="BPE Tokenizer Model"
              className="inline-flex rounded-lg border border-slate-800 bg-[#0D1322] p-1"
            >
              {(['cl100k_base', 'o200k_base', 'r50k_base'] as TokenizerEncodingId[]).map(
                (id) => {
                  const isActive = encodingId === id;
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => handleSelectEncoding(id)}
                      disabled={loadingEncoding !== null}
                      className={`rounded-md px-3 py-1.5 font-mono text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        isActive
                          ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/40'
                          : 'text-slate-400 hover:text-slate-100 border border-transparent'
                      }`}
                    >
                      {loadingEncoding === id ? `Loading ${id}…` : id}
                    </button>
                  );
                },
              )}
            </div>
            <div className="text-xs text-slate-400 font-mono">
              {encodingMeta.models} · {encodingMeta.vocabSize.toLocaleString()} vocab size
            </div>
          </div>
        </div>

        {/* Main 2-Column Split */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left: Input + Colored Token Boxes + Integer ID Array */}
          <div className="lg:col-span-8 space-y-5">
            <div className="rounded-xl border border-slate-800 bg-[#0D1322] p-5">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <label
                  htmlFor="tokenizer-textarea"
                  className="text-xs font-medium text-slate-300"
                >
                  Input Text
                </label>
                <div className="flex flex-wrap items-center gap-1.5">
                  {TOKENIZER_PRESETS.map((preset) => (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => {
                        setPinnedIdx(null);
                        onTextChange(preset.text);
                      }}
                      className="rounded-md border border-slate-700/80 bg-slate-900/70 px-2.5 py-1 text-xs text-slate-300 transition-colors hover:border-cyan-500/60 hover:text-cyan-200 whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              <textarea
                id="tokenizer-textarea"
                rows={3}
                value={text}
                onChange={(e) => onTextChange(e.target.value)}
                placeholder="Type sentences, code, numbers, or multilingual text..."
                className="w-full rounded-lg border border-slate-700/80 bg-[#070B14] px-4 py-3 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
              />

              {/* Unboxed Tabular Metrics */}
              <div className="mt-4 pt-4 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <div className="text-xs text-slate-400">Total Tokens</div>
                  <div className="mt-0.5 font-mono text-2xl font-semibold tabular-nums text-cyan-300">
                    {result.totalTokens}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Characters</div>
                  <div className="mt-0.5 font-mono text-2xl font-semibold tabular-nums text-slate-100">
                    {result.totalChars}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">UTF-8 Bytes</div>
                  <div className="mt-0.5 font-mono text-2xl font-semibold tabular-nums text-slate-100">
                    {result.totalBytes}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Chars / Token</div>
                  <div className="mt-0.5 font-mono text-2xl font-semibold tabular-nums text-violet-300">
                    {result.charsPerToken.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>

            {/* Colored Token Boxes */}
            <div className="rounded-xl border border-slate-800 bg-[#0D1322] p-5">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <h3 className="text-sm font-semibold text-slate-100">
                  Tokenized Sequence
                </h3>
                <span className="text-xs text-slate-400">
                  Hover or click any token box · <span className="font-mono">·</span> = space,{' '}
                  <span className="font-mono">↵</span> = newline
                </span>
              </div>

              {result.tokens.length === 0 ? (
                <div className="rounded-lg border border-dashed border-slate-800 py-10 text-center text-sm text-slate-400">
                  Enter text above to inspect live BPE tokens and IDs.
                </div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {result.tokens.map((tok, idx) => {
                    const palette = TOKEN_PALETTE[idx % TOKEN_PALETTE.length];
                    const isHighlighted = activeIdx === idx;
                    return (
                      <button
                        key={`${tok.id}-${idx}`}
                        type="button"
                        onMouseEnter={() => setHoveredIdx(idx)}
                        onMouseLeave={() => setHoveredIdx(null)}
                        onClick={() => setPinnedIdx(idx)}
                        className={`group flex flex-col items-start rounded-md border px-2.5 py-1.5 font-mono transition-all duration-100 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
                          isHighlighted ? palette.active : palette.box
                        }`}
                      >
                        <span className="text-sm leading-snug whitespace-pre">
                          {tok.displayText}
                        </span>
                        <span
                          className={`mt-0.5 text-[11px] tabular-nums ${
                            isHighlighted ? 'text-white' : palette.idText
                          }`}
                        >
                          {tok.id}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Synced Integer ID Array */}
              {result.tokenIds.length > 0 && (
                <div className="mt-5 pt-4 border-t border-slate-800/80">
                  <div className="text-xs font-medium text-slate-400 mb-2">
                    Encoded Token ID Vector
                  </div>
                  <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-[#070B14] p-3 font-mono text-xs">
                    <span className="text-slate-500">[</span>
                    {result.tokens.map((tok, idx) => {
                      const isHighlighted = activeIdx === idx;
                      return (
                        <span key={idx} className="inline-flex items-center">
                          <button
                            type="button"
                            onMouseEnter={() => setHoveredIdx(idx)}
                            onMouseLeave={() => setHoveredIdx(null)}
                            onClick={() => setPinnedIdx(idx)}
                            className={`rounded px-1 py-0.5 tabular-nums transition-colors cursor-pointer ${
                              isHighlighted
                                ? 'bg-cyan-500/30 text-cyan-100 font-semibold'
                                : 'text-slate-300 hover:text-cyan-300'
                            }`}
                          >
                            {tok.id}
                          </button>
                          {idx < result.tokens.length - 1 && (
                            <span className="text-slate-600">,</span>
                          )}
                        </span>
                      );
                    })}
                    <span className="text-slate-500">]</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right: Live Token Inspector */}
          <aside className="lg:col-span-4 rounded-xl border border-slate-800 bg-[#0D1322] p-5 space-y-4">
            <div className="flex items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <h3 className="text-sm font-semibold text-slate-100">
                Token Inspector
              </h3>
              {activeToken && (
                <span className="font-mono text-xs tabular-nums text-cyan-300">
                  Position #{activeToken.index}
                </span>
              )}
            </div>

            {activeToken ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-slate-800 bg-[#070B14] p-3">
                    <div className="text-[11px] text-slate-400">Decoded Text</div>
                    <div className="mt-1 font-mono text-base font-semibold text-slate-100 break-all">
                      "{activeToken.displayText}"
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-[#070B14] p-3">
                    <div className="text-[11px] text-slate-400">Token ID</div>
                    <div className="mt-1 font-mono text-base font-semibold tabular-nums text-cyan-300">
                      {activeToken.id}
                    </div>
                  </div>
                </div>

                <div className="space-y-2.5 text-xs">
                  <div className="flex items-center justify-between py-1.5 border-b border-slate-800/70">
                    <span className="text-slate-400">Category</span>
                    <span className="font-mono text-slate-100">
                      {activeToken.category}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1.5 border-b border-slate-800/70">
                    <span className="text-slate-400">Leading Space Merged</span>
                    <span className="font-mono text-slate-100">
                      {activeToken.startsWithSpace ? 'Yes (0x20)' : 'No'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1.5 border-b border-slate-800/70">
                    <span className="text-slate-400">UTF-8 Byte Length</span>
                    <span className="font-mono tabular-nums text-slate-100">
                      {activeToken.byteLength} byte
                      {activeToken.byteLength === 1 ? '' : 's'}
                    </span>
                  </div>
                  <div className="py-1.5">
                    <div className="text-slate-400 mb-1.5">UTF-8 Hex Octets</div>
                    <div className="flex flex-wrap gap-1.5">
                      {activeToken.utf8Hex.map((hex, i) => (
                        <span
                          key={i}
                          className="rounded border border-slate-700 bg-[#070B14] px-2 py-0.5 font-mono text-xs text-violet-300"
                        >
                          0x{hex}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed pt-2 border-t border-slate-800/80">
                  {describeTokenCategory(
                    activeToken.category,
                    activeToken.text,
                    activeToken.id,
                  )}
                </p>

                <button
                  type="button"
                  onClick={() => onLookupVocabToken(activeToken.id)}
                  className="w-full rounded-lg border border-blue-500/40 bg-blue-500/12 px-4 py-2 text-xs font-medium text-blue-200 transition-colors hover:bg-blue-500/22 cursor-pointer"
                >
                  Inspect ID {activeToken.id} in Vocabulary Explorer ↓
                </button>
              </>
            ) : (
              <p className="text-xs text-slate-400">
                Enter text to inspect individual tokens.
              </p>
            )}
          </aside>
        </div>
      </div>
    </section>
  );
}
