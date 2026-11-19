import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { motion, useReducedMotion } from 'motion/react';
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from 'lucide-react';
import {
  ENCODING_METADATA,
  type TokenizerEncodingId,
  tokenizeTextSync,
} from '../lib/tokenizer';
import {
  buildTokenFeatureVector,
  computeSelfAttentionMatrix,
  cosineSimilarity,
} from '../lib/math';

interface PipelineSectionProps {
  text: string;
  onTextChange: (next: string) => void;
  encodingId: TokenizerEncodingId;
  onInspectInTokenizer: (text: string) => void;
}

interface StageMeta {
  id: number;
  shortTitle: string;
  title: string;
  summary: string;
  accent: string;
}

const PIPELINE_STAGES: StageMeta[] = [
  {
    id: 0,
    shortTitle: 'Raw Text',
    title: '01. Raw Text & UTF-8 Bytes',
    summary:
      'Input strings enter as Unicode characters and are encoded into UTF-8 byte sequences before regex pre-tokenization splits boundaries.',
    accent: 'cyan',
  },
  {
    id: 1,
    shortTitle: 'Tokenization',
    title: '02. Byte-Pair Encoding (BPE)',
    summary:
      'Frequent byte sequences are iteratively merged into subword tokens from the model’s learned BPE vocabulary.',
    accent: 'blue',
  },
  {
    id: 2,
    shortTitle: 'Token IDs',
    title: '03. Integer Vocabulary Lookup',
    summary:
      'Each subword token maps deterministically to a unique integer index inside the fixed vocabulary table.',
    accent: 'indigo',
  },
  {
    id: 3,
    shortTitle: 'Embeddings',
    title: '04. Token + Positional Embeddings',
    summary:
      'Token IDs index rows in the embedding matrix E and sum with positional encodings so order information enters the vector.',
    accent: 'violet',
  },
  {
    id: 4,
    shortTitle: 'Transformer',
    title: '05. Multi-Head Self-Attention',
    summary:
      'Tokens exchange information via Query-Key dot-product attention weights followed by feed-forward MLP layers.',
    accent: 'cyan',
  },
  {
    id: 5,
    shortTitle: 'Contextual Reps',
    title: '06. Contextual Representations',
    summary:
      'Output hidden states blend each token’s intrinsic meaning with weighted context from neighboring tokens, ready for next-token logits.',
    accent: 'blue',
  },
];

const PIPELINE_PRESETS = [
  {
    label: 'Polysemy Example',
    text: 'I deposited cash at the river bank before sunset.',
  },
  {
    label: 'Technical Prose',
    text: 'Transformers compute attention across subword tokens.',
  },
  {
    label: 'Code & Numbers',
    text: 'const loss = crossEntropy(logits, 2048);',
  },
];

export function PipelineSection({
  text,
  onTextChange,
  encodingId,
  onInspectInTokenizer,
}: PipelineSectionProps) {
  const [activeStage, setActiveStage] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(0);
  const [causalMask, setCausalMask] = useState<boolean>(true);
  const shouldReduceMotion = useReducedMotion();
  const attentionSvgRef = useRef<SVGSVGElement | null>(null);

  const tokenization = useMemo(
    () => tokenizeTextSync(text, encodingId),
    [text, encodingId],
  );

  // Limit attention matrix display to first 14 tokens for legibility
  const attentionTokens = useMemo(
    () => tokenization.tokens.slice(0, 14),
    [tokenization.tokens],
  );

  const attentionData = useMemo(() => {
    const ids = attentionTokens.map((t) => t.id);
    return computeSelfAttentionMatrix(ids, causalMask);
  }, [attentionTokens, causalMask]);

  const contextualVectors = useMemo(() => {
    const { vectors, weights } = attentionData;
    return vectors.map((_, i) => {
      const dim = vectors[0]?.length ?? 16;
      const out = new Array<number>(dim).fill(0);
      for (let j = 0; j < vectors.length; j++) {
        const w = weights[i][j];
        for (let d = 0; d < dim; d++) {
          out[d] += w * vectors[j][d];
        }
      }
      return out;
    });
  }, [attentionData]);

  useEffect(() => {
    if (selectedTokenIdx >= attentionTokens.length) {
      setSelectedTokenIdx(Math.max(0, attentionTokens.length - 1));
    }
  }, [attentionTokens.length, selectedTokenIdx]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setActiveStage((prev) => {
        if (prev >= PIPELINE_STAGES.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1800);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  // Render D3 Self-Attention Heatmap when stage === 4
  useEffect(() => {
    if (activeStage !== 4 || !attentionSvgRef.current) return;
    const svg = d3.select(attentionSvgRef.current);
    svg.selectAll('*').remove();

    const n = attentionTokens.length;
    if (n === 0) return;

    const cellSize = Math.min(36, Math.max(22, Math.floor(380 / n)));
    const margin = { top: 68, right: 20, bottom: 20, left: 96 };
    const width = margin.left + n * cellSize + margin.right;
    const height = margin.top + n * cellSize + margin.bottom;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colorScale = d3
      .scaleSequential(d3.interpolateRgb('#0B1325', '#06B6D4'))
      .domain([0, 0.65]);

    const rows = g
      .selectAll('g.att-row')
      .data(attentionData.weights)
      .enter()
      .append('g')
      .attr('class', 'att-row')
      .attr('transform', (_, i) => `translate(0, ${i * cellSize})`);

    rows
      .selectAll('rect')
      .data((row, i) => row.map((val, j) => ({ val, i, j })))
      .enter()
      .append('rect')
      .attr('x', (d) => d.j * cellSize + 1)
      .attr('y', 1)
      .attr('width', cellSize - 2)
      .attr('height', cellSize - 2)
      .attr('rx', 3)
      .attr('fill', (d) =>
        causalMask && d.j > d.i ? '#090E1A' : colorScale(d.val),
      )
      .attr('stroke', (d) =>
        d.i === selectedTokenIdx ? '#38BDF8' : 'rgba(30,41,59,0.7)',
      )
      .attr('stroke-width', (d) => (d.i === selectedTokenIdx ? 1.5 : 0.75))
      .style('cursor', 'pointer')
      .on('click', (_, d) => setSelectedTokenIdx(d.i))
      .append('title')
      .text(
        (d) =>
          `Query "${attentionTokens[d.i]?.displayText}" → Key "${attentionTokens[d.j]?.displayText}": ${(d.val * 100).toFixed(1)}%`,
      );

    // Row labels (Query tokens)
    g.selectAll('text.row-label')
      .data(attentionTokens)
      .enter()
      .append('text')
      .attr('x', -10)
      .attr('y', (_, i) => i * cellSize + cellSize / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', '11px')
      .attr('fill', (_, i) => (i === selectedTokenIdx ? '#22D3EE' : '#94A3B8'))
      .style('cursor', 'pointer')
      .text((d) => d.displayText.slice(0, 10))
      .on('click', (_, d) => setSelectedTokenIdx(d.index));

    // Column labels (Key tokens)
    g.selectAll('text.col-label')
      .data(attentionTokens)
      .enter()
      .append('text')
      .attr(
        'transform',
        (_, j) =>
          `translate(${j * cellSize + cellSize / 2 + 3}, -10) rotate(-42)`,
      )
      .attr('text-anchor', 'start')
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', '11px')
      .attr('fill', '#94A3B8')
      .text((d) => d.displayText.slice(0, 10));
  }, [activeStage, attentionTokens, attentionData, selectedTokenIdx, causalMask]);

  const currentStageMeta = PIPELINE_STAGES[activeStage];
  const utf8Encoder = useMemo(() => new TextEncoder(), []);
  const rawBytes = useMemo(() => Array.from(utf8Encoder.encode(text)), [text, utf8Encoder]);
  const encodingMeta = ENCODING_METADATA[encodingId];

  const selectedToken = attentionTokens[selectedTokenIdx] ?? tokenization.tokens[0];
  const selectedVector = selectedToken
    ? buildTokenFeatureVector(selectedToken.id, selectedToken.index, 16)
    : [];

  return (
    <section
      id="pipeline"
      className="relative border-b border-slate-800/80 lab-grid-bg"
    >
      <div className="mx-auto max-w-[1360px] px-6 py-14 lg:py-20">
        {/* Hero Header */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-end mb-12">
          <div className="lg:col-span-7">
            <p className="text-xs font-mono text-cyan-400 tracking-wide mb-3">
              Interactive Transformer Laboratory · {encodingMeta.label} · V ={' '}
              {encodingMeta.vocabSize.toLocaleString()}
            </p>
            <h1 className="text-3xl sm:text-4xl lg:text-[42px] font-bold tracking-tight text-slate-50 leading-[1.15] [text-wrap:balance]">
              Trace how Large Language Models turn raw text into contextual meaning
            </h1>
            <p className="mt-4 max-w-2xl text-[15px] sm:text-base text-slate-300 leading-relaxed">
              Enter any sentence below and step through each transformation inside a
              modern decoder or encoder LLM—from UTF-8 bytes and real BPE subword
              tokenization to vocabulary lookup, positional embeddings, and
              self-attention.
            </p>
          </div>

          {/* Live Text Input Deck */}
          <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-[#0D1322] p-5">
            <div className="flex items-center justify-between gap-4 mb-2.5">
              <label
                htmlFor="pipeline-input"
                className="text-xs font-medium text-slate-300"
              >
                Pipeline Input Sequence
              </label>
              <span className="text-xs font-mono tabular-nums text-slate-400">
                {tokenization.totalTokens} tokens · {tokenization.totalChars} chars
              </span>
            </div>
            <textarea
              id="pipeline-input"
              rows={2}
              value={text}
              onChange={(e) => onTextChange(e.target.value)}
              placeholder="Enter text to trace across all 6 LLM pipeline stages..."
              className="w-full rounded-lg border border-slate-700/80 bg-[#070B14] px-3.5 py-2.5 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
            />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                {PIPELINE_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => onTextChange(preset.text)}
                    className="rounded-md border border-slate-700/80 bg-slate-900/70 px-2.5 py-1 text-xs text-slate-300 transition-colors hover:border-cyan-500/60 hover:text-cyan-200 whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => onInspectInTokenizer(text)}
                className="text-xs font-medium text-cyan-300 hover:text-cyan-200 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Open in Tokenizer ↓
              </button>
            </div>
          </div>
        </div>

        {/* Interactive 6-Stage Flow Diagram */}
        <div className="rounded-xl border border-slate-800 bg-[#0D1322] p-5 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-slate-800/80">
            <div>
              <h2 className="text-sm font-semibold text-slate-100">
                01. End-to-End LLM Execution Pipeline
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Click any node or use step controls to inspect live representations at
                that stage
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStage((s) => Math.max(0, s - 1));
                }}
                disabled={activeStage === 0}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-slate-500 disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap shrink-0 cursor-pointer"
              >
                <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
                Prev
              </button>

              <button
                type="button"
                onClick={() => {
                  if (activeStage === PIPELINE_STAGES.length - 1 && !isPlaying) {
                    setActiveStage(0);
                  }
                  setIsPlaying((p) => !p);
                }}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/50 bg-cyan-500/15 px-3.5 py-1.5 text-xs font-medium text-cyan-200 transition-colors hover:bg-cyan-500/25 whitespace-nowrap shrink-0 cursor-pointer"
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-3.5 w-3.5" aria-hidden="true" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5" aria-hidden="true" />
                    Step Auto-Play
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStage((s) =>
                    Math.min(PIPELINE_STAGES.length - 1, s + 1),
                  );
                }}
                disabled={activeStage === PIPELINE_STAGES.length - 1}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-slate-500 disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap shrink-0 cursor-pointer"
              >
                Next
                <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
              </button>

              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStage(0);
                }}
                aria-label="Reset pipeline stage"
                className="inline-flex items-center justify-center rounded-lg border border-slate-700 bg-slate-900 p-1.5 text-slate-300 transition-colors hover:border-slate-500 hover:text-slate-100 cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            </div>
          </div>

          {/* Horizontal Flow Nodes */}
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 items-stretch">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isSelected = activeStage === idx;
              const isPassed = idx < activeStage;

              let previewValue = '';
              if (idx === 0) previewValue = `${text.length} chars`;
              if (idx === 1) previewValue = `${tokenization.totalTokens} tokens`;
              if (idx === 2)
                previewValue = `[${tokenization.tokenIds.slice(0, 2).join(', ')}${tokenization.tokenIds.length > 2 ? ', …' : ''}]`;
              if (idx === 3) previewValue = `${tokenization.totalTokens} × d`;
              if (idx === 4) previewValue = `${attentionTokens.length}×${attentionTokens.length} attn`;
              if (idx === 5) previewValue = `H^(L) states`;

              return (
                <div key={stage.id} className="relative flex items-center">
                  <button
                    type="button"
                    onClick={() => {
                      setIsPlaying(false);
                      setActiveStage(idx);
                    }}
                    className={`group relative w-full h-full rounded-lg border p-3.5 text-left transition-all duration-150 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
                      isSelected
                        ? 'border-cyan-400 bg-cyan-500/12 shadow-[0_0_24px_rgba(6,182,212,0.12)]'
                        : isPassed
                          ? 'border-blue-500/40 bg-blue-500/5 hover:border-blue-400/70'
                          : 'border-slate-800 bg-[#0A101D] hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span
                        className={`text-[11px] font-mono tabular-nums ${
                          isSelected
                            ? 'text-cyan-300'
                            : isPassed
                              ? 'text-blue-300'
                              : 'text-slate-500'
                        }`}
                      >
                        Stage 0{idx + 1}
                      </span>
                      {idx < PIPELINE_STAGES.length - 1 && (
                        <ArrowRight
                          className={`hidden lg:block h-3.5 w-3.5 transition-transform ${
                            isSelected
                              ? 'text-cyan-300 translate-x-0.5'
                              : 'text-slate-600'
                          }`}
                          aria-hidden="true"
                        />
                      )}
                    </div>
                    <div className="text-sm font-semibold text-slate-100 leading-snug">
                      {stage.shortTitle}
                    </div>
                    <div className="mt-1.5 font-mono text-xs tabular-nums text-slate-400 truncate">
                      {previewValue}
                    </div>

                    {isSelected && !shouldReduceMotion && (
                      <motion.div
                        layoutId="pipeline-active-bar"
                        className="absolute inset-x-3 bottom-0 h-0.5 bg-cyan-400 rounded-full"
                        transition={{ duration: 0.2 }}
                      />
                    )}
                  </button>
                </div>
              );
            })}
          </div>

          {/* Stage Inspector Canvas */}
          <div className="mt-6 rounded-lg border border-slate-800 bg-[#070B14] p-5 sm:p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div>
                <h3 className="text-lg font-semibold text-slate-100">
                  {currentStageMeta.title}
                </h3>
                <p className="text-sm text-slate-300 mt-1 max-w-3xl">
                  {currentStageMeta.summary}
                </p>
              </div>
              <div className="text-xs font-mono text-slate-400 whitespace-nowrap shrink-0">
                Encoding: {encodingMeta.label}
              </div>
            </div>

            {/* Stage 0: Raw Text & UTF-8 Bytes */}
            {activeStage === 0 && (
              <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-7">
                  <div className="text-xs font-medium text-slate-400 mb-2">
                    Character Sequence ({text.length} UTF-16 code units)
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {Array.from(text)
                      .slice(0, 64)
                      .map((ch, idx) => (
                        <div
                          key={`${idx}-${ch}`}
                          className="flex flex-col items-center rounded border border-slate-800 bg-[#0D1322] px-2 py-1.5 min-w-[34px]"
                        >
                          <span className="font-mono text-sm text-slate-100">
                            {ch === ' ' ? '·' : ch === '\n' ? '↵' : ch}
                          </span>
                          <span className="font-mono text-[10px] tabular-nums text-slate-500 mt-0.5">
                            U+{ch.codePointAt(0)?.toString(16).toUpperCase()}
                          </span>
                        </div>
                      ))}
                    {Array.from(text).length > 64 && (
                      <span className="self-center font-mono text-xs text-slate-400 px-2">
                        +{Array.from(text).length - 64} more
                      </span>
                    )}
                  </div>
                </div>

                <div className="lg:col-span-5">
                  <div className="text-xs font-medium text-slate-400 mb-2">
                    UTF-8 Byte Stream ({rawBytes.length} bytes)
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-3.5 font-mono text-xs text-cyan-300 leading-relaxed break-all">
                    {rawBytes.length > 0
                      ? rawBytes
                          .slice(0, 72)
                          .map((b) => b.toString(16).padStart(2, '0').toUpperCase())
                          .join(' ')
                      : 'Empty sequence'}
                    {rawBytes.length > 72 ? ' …' : ''}
                  </div>
                  <p className="mt-2.5 text-xs text-slate-400 leading-relaxed">
                    Byte-level BPE operates directly on UTF-8 octets (0x00–0xFF), so
                    any language, symbol, or emoji can be encoded without out-of-vocabulary
                    errors.
                  </p>
                </div>
              </div>
            )}

            {/* Stage 1: Tokenization */}
            {activeStage === 1 && (
              <div className="mt-5">
                <div className="flex items-center justify-between gap-4 mb-3">
                  <div className="text-xs font-medium text-slate-400">
                    Real BPE Subword Segmentation ({tokenization.totalTokens} tokens)
                  </div>
                  <div className="text-xs font-mono tabular-nums text-slate-400">
                    Compression: {tokenization.charsPerToken.toFixed(2)} chars / token
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {tokenization.tokens.map((tok, i) => {
                    const tones = [
                      'border-cyan-500/40 bg-cyan-500/12 text-cyan-100',
                      'border-blue-500/40 bg-blue-500/12 text-blue-100',
                      'border-violet-500/40 bg-violet-500/12 text-violet-100',
                      'border-teal-500/40 bg-teal-500/12 text-teal-100',
                    ];
                    return (
                      <div
                        key={`${tok.id}-${i}`}
                        className={`rounded-md border px-2.5 py-1.5 font-mono text-sm ${tones[i % tones.length]}`}
                      >
                        <span>{tok.displayText}</span>
                        <span className="ml-2 text-[11px] opacity-70">
                          {tok.category === 'leading-space-word'
                            ? 'space+word'
                            : tok.category}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Stage 2: Token IDs */}
            {activeStage === 2 && (
              <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-8">
                  <div className="text-xs font-medium text-slate-400 mb-2.5">
                    Token-to-Integer Mapping ({encodingMeta.vocabSize.toLocaleString()} total entries)
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                    {tokenization.tokens.map((tok) => (
                      <div
                        key={tok.index}
                        className="flex items-center justify-between gap-2 rounded-md border border-slate-800 bg-[#0D1322] px-3 py-2"
                      >
                        <span className="font-mono text-xs text-slate-200 truncate">
                          {tok.displayText}
                        </span>
                        <span className="font-mono text-xs font-semibold tabular-nums text-indigo-300">
                          {tok.id}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="lg:col-span-4 rounded-lg border border-slate-800 bg-[#0D1322] p-4">
                  <div className="text-xs font-medium text-slate-300 mb-2">
                    Model Input Tensor (1D Int64)
                  </div>
                  <code className="block rounded bg-[#070B14] p-3 font-mono text-xs text-indigo-300 break-all">
                    input_ids = [{tokenization.tokenIds.join(', ')}]
                  </code>
                  <p className="mt-3 text-xs text-slate-400 leading-relaxed">
                    Neural networks cannot multiply text strings. From this stage
                    onward, the model sees only this integer vector of shape{' '}
                    <span className="font-mono text-slate-200">
                      [1, {tokenization.totalTokens}]
                    </span>
                    .
                  </p>
                </div>
              </div>
            )}

            {/* Stage 3: Embeddings + Positional Encoding */}
            {activeStage === 3 && (
              <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-5">
                  <div className="text-xs font-medium text-slate-400 mb-2.5">
                    Select Token to Inspect Embedding Vector
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {attentionTokens.map((tok) => (
                      <button
                        key={tok.index}
                        type="button"
                        onClick={() => setSelectedTokenIdx(tok.index)}
                        className={`rounded-md border px-2.5 py-1.5 font-mono text-xs transition-colors cursor-pointer ${
                          tok.index === selectedTokenIdx
                            ? 'border-violet-400 bg-violet-500/20 text-violet-100'
                            : 'border-slate-800 bg-[#0D1322] text-slate-300 hover:border-slate-600'
                        }`}
                      >
                        {tok.displayText}{' '}
                        <span className="text-slate-400">#{tok.index}</span>
                      </button>
                    ))}
                  </div>
                  {selectedToken && (
                    <div className="mt-4 rounded-lg border border-slate-800 bg-[#0D1322] p-3.5 text-xs text-slate-300 space-y-1.5">
                      <div>
                        Token:{' '}
                        <span className="font-mono text-violet-300">
                          "{selectedToken.displayText}"
                        </span>{' '}
                        · ID{' '}
                        <span className="font-mono tabular-nums text-slate-100">
                          {selectedToken.id}
                        </span>
                      </div>
                      <div>
                        Position Index:{' '}
                        <span className="font-mono tabular-nums text-cyan-300">
                          pos = {selectedToken.index}
                        </span>
                      </div>
                      <p className="text-slate-400 pt-1 leading-relaxed">
                        Output shape after lookup:{' '}
                        <span className="font-mono text-slate-200">
                          [{tokenization.totalTokens}, d_model]
                        </span>
                        . Below is an illustrative 16-dimension projection combining
                        token identity with sinusoidal positional encoding.
                      </p>
                    </div>
                  )}
                </div>

                <div className="lg:col-span-7 rounded-lg border border-slate-800 bg-[#0D1322] p-4">
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-xs font-medium text-slate-300">
                      Illustrative 16D Vector Signature (Token + Sinusoidal Position)
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      dim 0 → 15
                    </span>
                  </div>
                  <div className="grid grid-cols-8 sm:grid-cols-16 gap-1.5 items-end h-28 pt-2">
                    {selectedVector.map((val, dIdx) => {
                      const pct = Math.min(100, Math.round(Math.abs(val) * 62));
                      const pos = val >= 0;
                      return (
                        <div
                          key={dIdx}
                          className="flex flex-col items-center justify-end h-full"
                          title={`d[${dIdx}] = ${val.toFixed(3)}`}
                        >
                          <div
                            className={`w-full rounded-sm transition-all duration-150 ${
                              pos ? 'bg-cyan-400/80' : 'bg-violet-400/80'
                            }`}
                            style={{ height: `${Math.max(10, pct)}%` }}
                          />
                          <span className="font-mono text-[10px] tabular-nums text-slate-500 mt-1">
                            {dIdx}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>Cyan: positive activation · Violet: negative activation</span>
                    <span>
                      [{selectedVector.slice(0, 4).map((v) => v.toFixed(2)).join(', ')}, …]
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Stage 4: Transformer Self-Attention */}
            {activeStage === 4 && (
              <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                <div className="lg:col-span-7">
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                    <div className="text-xs font-medium text-slate-300">
                      Illustrative Self-Attention Matrix softmax(QKᵀ / √d)
                    </div>
                    <div className="inline-flex rounded-lg border border-slate-800 bg-[#0D1322] p-0.5">
                      <button
                        type="button"
                        onClick={() => setCausalMask(true)}
                        className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer ${
                          causalMask
                            ? 'bg-cyan-500/20 text-cyan-200'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Causal (GPT)
                      </button>
                      <button
                        type="button"
                        onClick={() => setCausalMask(false)}
                        className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer ${
                          !causalMask
                            ? 'bg-violet-500/20 text-violet-200'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Bidirectional (BERT)
                      </button>
                    </div>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-slate-800 bg-[#0D1322] p-3">
                    <svg ref={attentionSvgRef} className="w-full max-w-[490px] mx-auto" />
                  </div>
                </div>

                <div className="lg:col-span-5 rounded-lg border border-slate-800 bg-[#0D1322] p-4 space-y-3">
                  <div className="text-xs font-medium text-slate-300">
                    Attention Focus for Query Token{' '}
                    <span className="font-mono text-cyan-300">
                      "{selectedToken?.displayText ?? ''}"
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">
                    Click any row in the matrix to inspect how much context weight
                    it pulls from other tokens:
                  </p>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                    {(attentionData.weights[selectedTokenIdx] ?? []).map((w, j) => {
                      const targetTok = attentionTokens[j];
                      if (!targetTok) return null;
                      const pct = Math.round(w * 100);
                      return (
                        <div key={j} className="space-y-1">
                          <div className="flex items-center justify-between text-xs font-mono">
                            <span className="text-slate-200">
                              {targetTok.displayText}
                            </span>
                            <span className="tabular-nums text-cyan-300">
                              {(w * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                            <div
                              className="h-full bg-cyan-400 transition-all duration-150"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Stage 5: Contextual Representations */}
            {activeStage === 5 && (
              <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-7">
                  <div className="text-xs font-medium text-slate-400 mb-3">
                    Contextualized Output States & Context Drift (Illustrative Hidden States)
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {attentionTokens.map((tok, i) => {
                      const inputVec = attentionData.vectors[i];
                      const outVec = contextualVectors[i];
                      const retention =
                        inputVec && outVec ? cosineSimilarity(inputVec, outVec) : 1;
                      const contextAbsorbed = Math.max(0, 1 - retention);
                      return (
                        <div
                          key={tok.index}
                          className="rounded-lg border border-slate-800 bg-[#0D1322] p-3"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono text-sm font-medium text-slate-100">
                              {tok.displayText}
                            </span>
                            <span className="font-mono text-xs tabular-nums text-blue-300">
                              ID {tok.id}
                            </span>
                          </div>
                          <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-slate-400">
                            <span>Self-retention: {(retention * 100).toFixed(1)}%</span>
                            <span className="text-cyan-300">
                              Context shift: {(contextAbsorbed * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="lg:col-span-5 rounded-lg border border-slate-800 bg-[#0D1322] p-4 space-y-3">
                  <div className="text-sm font-semibold text-slate-100">
                    Why Contextual Representations Matter
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Before the Transformer, token{' '}
                    <span className="font-mono text-cyan-300">bank</span> had the
                    exact same static vector in every sentence. After self-attention
                    mixes weighted Value vectors across the sequence, its hidden
                    state moves into the semantic neighborhood of{' '}
                    <span className="font-mono text-slate-100">river</span> or{' '}
                    <span className="font-mono text-slate-100">cash</span>.
                  </p>
                  <div className="pt-2 border-t border-slate-800">
                    <a
                      href="#embeddings"
                      className="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-300 hover:text-cyan-200 transition-colors"
                    >
                      Compare Static vs. Contextual Vectors in PCA Space
                      <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                    </a>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
