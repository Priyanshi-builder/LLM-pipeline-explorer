import { useEffect, useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import {
  ENCODING_METADATA,
  type TokenCategory,
  type TokenizerEncodingId,
  type VocabItem,
  getNeighboringVocabItems,
  searchVocabulary,
  tokenizeTextSync,
} from '../lib/tokenizer';

interface VocabularySectionProps {
  encodingId: TokenizerEncodingId;
  focusedTokenId: number | null;
}

const CATEGORY_FILTERS: Array<{ id: TokenCategory | 'all'; label: string }> = [
  { id: 'all', label: 'All Tokens' },
  { id: 'leading-space-word', label: 'Space + Word' },
  { id: 'word', label: 'Standalone Word' },
  { id: 'subword', label: 'Subword' },
  { id: 'number', label: 'Numbers' },
  { id: 'punctuation', label: 'Symbols' },
  { id: 'special', label: 'Special' },
];

export function VocabularySection({
  encodingId,
  focusedTokenId,
}: VocabularySectionProps) {
  const [searchQuery, setSearchQuery] = useState<string>('transformer');
  const [phraseInput, setPhraseInput] = useState<string>(
    'Neural transformers embed tokens cleanly.',
  );
  const [categoryFilter, setCategoryFilter] = useState<TokenCategory | 'all'>('all');
  const [selectedItem, setSelectedItem] = useState<VocabItem | null>(null);

  const searchResults = useMemo(() => {
    return searchVocabulary(searchQuery, encodingId, categoryFilter, 48);
  }, [searchQuery, encodingId, categoryFilter]);

  const phraseTokens = useMemo(() => {
    if (!phraseInput.trim()) return [];
    return tokenizeTextSync(phraseInput, encodingId).tokens;
  }, [phraseInput, encodingId]);

  useEffect(() => {
    if (focusedTokenId !== null) {
      setCategoryFilter('all');
      setSearchQuery(String(focusedTokenId));
      const res = searchVocabulary(String(focusedTokenId), encodingId, 'all', 10);
      const exact = res.items.find((it) => it.id === focusedTokenId);
      if (exact) {
        setSelectedItem(exact);
      }
    }
  }, [focusedTokenId, encodingId]);

  useEffect(() => {
    if (!selectedItem && searchResults.items.length > 0) {
      setSelectedItem(searchResults.items[0]);
    }
  }, [searchResults.items, selectedItem]);

  const neighbors = useMemo(() => {
    if (!selectedItem) return [];
    return getNeighboringVocabItems(selectedItem.id, encodingId, 3);
  }, [selectedItem, encodingId]);

  const encodingMeta = ENCODING_METADATA[encodingId];

  return (
    <section id="vocabulary" className="border-b border-slate-800/80 bg-[#090E1A]">
      <div className="mx-auto max-w-[1360px] px-6 py-16">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 mb-8">
          <div>
            <p className="text-xs font-mono text-blue-400 mb-2">
              Searchable BPE Dictionary · {encodingMeta.vocabSize.toLocaleString()} Entries
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-50">
              03. Vocabulary Explorer
            </h2>
            <p className="mt-2 max-w-2xl text-[15px] text-slate-300 leading-relaxed">
              Every LLM has a finite vocabulary mapping subwords to integer IDs.
              Search the real {encodingMeta.label} dictionary by word, suffix, or
              numeric ID, or enter a phrase to inspect its constituent vocabulary entries.
            </p>
          </div>
          <div className="font-mono text-xs tabular-nums text-slate-400">
            Showing {searchResults.items.length} of{' '}
            {searchResults.totalMatches.toLocaleString()} matches
          </div>
        </div>

        {/* Phrase Breakdown Strip */}
        <div className="rounded-xl border border-slate-800 bg-[#0D1322] p-5 mb-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
            <div className="lg:col-span-5">
              <label
                htmlFor="vocab-phrase-input"
                className="block text-xs font-medium text-slate-300 mb-1.5"
              >
                Explore Vocabulary From Any Phrase
              </label>
              <input
                id="vocab-phrase-input"
                type="text"
                value={phraseInput}
                onChange={(e) => setPhraseInput(e.target.value)}
                placeholder="Enter text to inspect its vocabulary entries..."
                className="w-full rounded-lg border border-slate-700/80 bg-[#070B14] px-3.5 py-2 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
              />
            </div>
            <div className="lg:col-span-7">
              <div className="text-xs text-slate-400 mb-1.5">
                Matched Vocabulary Entries (click any token to inspect)
              </div>
              <div className="flex flex-wrap gap-1.5">
                {phraseTokens.map((tok, i) => (
                  <button
                    key={`${tok.id}-${i}`}
                    type="button"
                    onClick={() => {
                      const found = searchVocabulary(String(tok.id), encodingId, 'all', 5).items.find(
                        (v) => v.id === tok.id,
                      );
                      if (found) setSelectedItem(found);
                    }}
                    className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-xs transition-colors cursor-pointer ${
                      selectedItem?.id === tok.id
                        ? 'border-blue-400 bg-blue-500/20 text-blue-100'
                        : 'border-slate-700/80 bg-[#070B14] text-slate-200 hover:border-blue-500/50'
                    }`}
                  >
                    <span>{tok.displayText}</span>
                    <span className="text-blue-300 tabular-nums">#{tok.id}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Main Search + Table & Detail Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left: Search + Results Table */}
          <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-[#0D1322] p-5">
            <div className="flex flex-col sm:flex-row gap-3 mb-4">
              <div className="relative flex-1">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500"
                  aria-hidden="true"
                />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder='Search token text (e.g. "token", "ing") or exact ID (e.g. 15339)...'
                  aria-label="Search vocabulary tokens or IDs"
                  className="w-full rounded-lg border border-slate-700/80 bg-[#070B14] pl-9 pr-3.5 py-2 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-300 hover:border-slate-500 whitespace-nowrap shrink-0 cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Category Filter Buttons */}
            <div className="flex flex-wrap gap-1.5 mb-4">
              {CATEGORY_FILTERS.map((cat) => {
                const isActive = categoryFilter === cat.id;
                return (
                  <button
                    key={cat.id}
                    type="button"
                    onClick={() => setCategoryFilter(cat.id)}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      isActive
                        ? 'bg-blue-500/20 text-blue-200 border border-blue-500/50'
                        : 'bg-[#070B14] text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {cat.label}
                  </button>
                );
              })}
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto max-h-[420px] overflow-y-auto rounded-lg border border-slate-800 bg-[#070B14]">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-[#0B111F] border-b border-slate-800 text-[11px] font-mono text-slate-400">
                  <tr>
                    <th className="py-2.5 px-3 font-medium">Token ID</th>
                    <th className="py-2.5 px-3 font-medium">Token String</th>
                    <th className="py-2.5 px-3 font-medium">Category</th>
                    <th className="py-2.5 px-3 font-medium text-right">Bytes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-xs">
                  {searchResults.items.length === 0 ? (
                    <tr>
                      <td
                        colSpan={4}
                        className="py-10 text-center text-slate-400"
                      >
                        No vocabulary entries matched "{searchQuery}".
                      </td>
                    </tr>
                  ) : (
                    searchResults.items.map((item) => {
                      const isSelected = selectedItem?.id === item.id;
                      return (
                        <tr
                          key={item.id}
                          onClick={() => setSelectedItem(item)}
                          className={`cursor-pointer transition-colors ${
                            isSelected
                              ? 'bg-blue-500/15 text-slate-50'
                              : 'hover:bg-slate-800/40 text-slate-300'
                          }`}
                        >
                          <td className="py-2 px-3 font-mono tabular-nums text-blue-300">
                            {item.id}
                          </td>
                          <td className="py-2 px-3 font-mono text-slate-100">
                            "{item.displayText}"
                          </td>
                          <td className="py-2 px-3 text-slate-400">
                            {item.category}
                          </td>
                          <td className="py-2 px-3 font-mono tabular-nums text-right text-slate-400">
                            {item.byteLength}B
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right: Token Definition & Neighboring Ranks */}
          <aside className="lg:col-span-5 rounded-xl border border-slate-800 bg-[#0D1322] p-5 space-y-5">
            {selectedItem ? (
              <>
                <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-800/80">
                  <div>
                    <div className="text-xs font-mono text-blue-400">
                      Vocabulary Entry · {encodingMeta.label}
                    </div>
                    <div className="mt-1 font-mono text-2xl font-bold text-slate-50 break-all">
                      "{selectedItem.displayText}"
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs text-slate-400">Token ID</div>
                    <div className="font-mono text-2xl font-bold tabular-nums text-blue-300">
                      {selectedItem.id}
                    </div>
                  </div>
                </div>

                <div className="space-y-2.5 text-xs">
                  <div className="text-slate-400 font-medium">
                    Technical Definition
                  </div>
                  <p className="text-sm text-slate-200 leading-relaxed">
                    {selectedItem.definition}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="rounded-lg border border-slate-800 bg-[#070B14] p-3">
                    <div className="text-[11px] text-slate-400">BPE Merge Rank</div>
                    <div className="mt-1 font-mono text-sm tabular-nums text-slate-100">
                      #{selectedItem.id.toLocaleString()} /{' '}
                      {encodingMeta.vocabSize.toLocaleString()}
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-[#070B14] p-3">
                    <div className="text-[11px] text-slate-400">UTF-8 Hex</div>
                    <div className="mt-1 font-mono text-sm text-cyan-300 truncate">
                      {selectedItem.utf8Hex.map((h) => `0x${h}`).join(' ')}
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-800/80">
                  <div className="text-xs font-medium text-slate-300 mb-2">
                    Adjacent Vocabulary IDs (Nearby BPE Merge Priority)
                  </div>
                  <div className="space-y-1.5">
                    {neighbors.map((nb) => {
                      const isCurrent = nb.id === selectedItem.id;
                      return (
                        <button
                          key={nb.id}
                          type="button"
                          onClick={() => setSelectedItem(nb)}
                          className={`w-full flex items-center justify-between rounded-md border px-3 py-1.5 text-xs font-mono transition-colors cursor-pointer ${
                            isCurrent
                              ? 'border-blue-400 bg-blue-500/18 text-slate-50'
                              : 'border-slate-800 bg-[#070B14] text-slate-300 hover:border-slate-600'
                          }`}
                        >
                          <span className="tabular-nums text-blue-300">
                            ID {nb.id}
                          </span>
                          <span>"{nb.displayText}"</span>
                          <span className="text-slate-500">{nb.category}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </>
            ) : (
              <div className="py-12 text-center text-sm text-slate-400">
                Select any row from the vocabulary table to inspect its definition and
                byte encoding.
              </div>
            )}
          </aside>
        </div>
      </div>
    </section>
  );
}
