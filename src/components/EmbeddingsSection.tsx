import { useEffect, useMemo, useState } from 'react';
import {
  ANALOGY_PRESETS,
  CONTEXTUAL_EXPERIMENTS,
  STATIC_EMBEDDINGS,
  type SemanticCluster,
} from '../data/embeddingsData';
import {
  computePCA,
  cosineSimilarity,
  euclideanDistance,
  vectorAdd,
  vectorSub,
} from '../lib/math';
import { EmbeddingSpace3D, type PlotPoint3D } from './EmbeddingSpace3D';

const CLUSTER_COLORS: Record<SemanticCluster, string> = {
  royalty: '#A78BFA',
  'ai-tech': '#22D3EE',
  finance: '#34D399',
  nature: '#38BDF8',
  animals: '#FBBF24',
  geography: '#60A5FA',
  'static-anchor': '#F43F5E',
  'context-a': '#10B981',
  'context-b': '#06B6D4',
};

const CLUSTER_LABELS: Record<SemanticCluster, string> = {
  royalty: 'Royalty & People',
  'ai-tech': 'AI & Computing',
  finance: 'Finance',
  nature: 'Nature & Water',
  animals: 'Fauna & Physical',
  geography: 'Geography',
  'static-anchor': 'Static Anchor',
  'context-a': 'Context A',
  'context-b': 'Context B',
};

const DEFAULT_LIVE_SENTENCES = [
  'I deposited cash into my checking account at the bank.',
  'We walked along the muddy river bank at sunset.',
  'Commercial banks raised interest rates on business loans.',
  'Clear water flowed over rocks beside the stream.',
  'Transformer neural networks compute self-attention over tokens.',
  'Writing Python software for machine learning models.',
];

export function EmbeddingsSection() {
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>('bank');
  const [dataSourceMode, setDataSourceMode] = useState<'reference' | 'live'>('reference');

  // Vector Arithmetic state
  const [analogyA, setAnalogyA] = useState<string>('king');
  const [analogyB, setAnalogyB] = useState<string>('man');
  const [analogyC, setAnalogyC] = useState<string>('woman');

  // Graph controls & inspection
  const [showCorpusWords, setShowCorpusWords] = useState<boolean>(true);
  const [showTrajectories, setShowTrajectories] = useState<boolean>(true);
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [resetSignal, setResetSignal] = useState<number>(0);
  const [hoveredPoint, setHoveredPoint] = useState<PlotPoint3D | null>(null);
  const [selectedPair, setSelectedPair] = useState<[string, string]>(['ctx-a', 'ctx-b']);

  // Live Gemini Embeddings state
  const [liveSentencesText, setLiveSentencesText] = useState<string>(
    DEFAULT_LIVE_SENTENCES.join('\n'),
  );
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [livePayload, setLivePayload] = useState<{
    model: string;
    dimensions: number;
    texts: string[];
    embeddings: number[][];
  } | null>(null);

  const activeExperiment = useMemo(
    () =>
      CONTEXTUAL_EXPERIMENTS.find((e) => e.id === selectedExperimentId) ??
      CONTEXTUAL_EXPERIMENTS[0],
    [selectedExperimentId],
  );

  // Reset selected pair nicely when switching polysemy experiment
  useEffect(() => {
    if (dataSourceMode === 'reference') {
      setSelectedPair(['ctx-a', 'ctx-b']);
    }
  }, [selectedExperimentId, dataSourceMode]);

  // Compute Vector Arithmetic: A - B + C
  const analogyResults = useMemo(() => {
    const wordMap = new Map(STATIC_EMBEDDINGS.map((w) => [w.word, w]));
    const vA = wordMap.get(analogyA)?.vector;
    const vB = wordMap.get(analogyB)?.vector;
    const vC = wordMap.get(analogyC)?.vector;
    if (!vA || !vB || !vC) return [];

    const target = vectorAdd(vectorSub(vA, vB), vC);
    return STATIC_EMBEDDINGS.filter(
      (item) =>
        item.word !== analogyA && item.word !== analogyB && item.word !== analogyC,
    )
      .map((item) => ({
        word: item.word,
        description: item.description,
        similarity: cosineSimilarity(target, item.vector),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
  }, [analogyA, analogyB, analogyC]);

  // Compute Real 3D PCA Projection
  const pcaData = useMemo(() => {
    if (dataSourceMode === 'live' && livePayload && livePayload.embeddings.length >= 2) {
      const pca = computePCA(livePayload.embeddings);
      const points: PlotPoint3D[] = livePayload.texts.map((sentence, idx) => ({
        id: `live-${idx}`,
        label: `S${idx + 1}`,
        sublabel: sentence,
        cluster: idx % 2 === 0 ? 'context-b' : 'ai-tech',
        vector: livePayload.embeddings[idx],
        x: pca.projected[idx][0],
        y: pca.projected[idx][1],
        z: pca.projected[idx][2],
        kind: 'live',
      }));
      return { pca, points };
    }

    // Reference 24D Dataset: corpus words + active contextual experiment points
    const rawEntries: Array<Omit<PlotPoint3D, 'x' | 'y' | 'z'>> = [];

    if (showCorpusWords) {
      for (const item of STATIC_EMBEDDINGS) {
        if (item.word === activeExperiment.targetWord) continue;
        rawEntries.push({
          id: `word-${item.word}`,
          label: item.word,
          sublabel: item.description,
          cluster: item.cluster,
          vector: item.vector,
          kind: 'static',
        });
      }
    }

    rawEntries.push({
      id: 'static-anchor',
      label: `${activeExperiment.targetWord} (static)`,
      sublabel: 'Fixed Word2Vec/GloVe representation averaging all meanings',
      cluster: 'static-anchor',
      vector: activeExperiment.staticAnchorVector,
      kind: 'static-anchor',
    });

    rawEntries.push({
      id: 'ctx-a',
      label: `${activeExperiment.targetWord} [${activeExperiment.senseALabel}]`,
      sublabel: activeExperiment.sentenceA,
      cluster: 'context-a',
      vector: activeExperiment.vectorA,
      kind: 'context-a',
    });

    rawEntries.push({
      id: 'ctx-b',
      label: `${activeExperiment.targetWord} [${activeExperiment.senseBLabel}]`,
      sublabel: activeExperiment.sentenceB,
      cluster: 'context-b',
      vector: activeExperiment.vectorB,
      kind: 'context-b',
    });

    const pca = computePCA(rawEntries.map((r) => r.vector));
    const points: PlotPoint3D[] = rawEntries.map((entry, idx) => ({
      ...entry,
      x: pca.projected[idx][0],
      y: pca.projected[idx][1],
      z: pca.projected[idx][2],
    }));

    return { pca, points };
  }, [dataSourceMode, livePayload, showCorpusWords, activeExperiment]);

  // Pairwise metrics for selected two points
  const pairMetrics = useMemo(() => {
    const p1 = pcaData.points.find((p) => p.id === selectedPair[0]);
    const p2 = pcaData.points.find((p) => p.id === selectedPair[1]);
    if (!p1 || !p2) return null;
    return {
      p1,
      p2,
      cosine: cosineSimilarity(p1.vector, p2.vector),
      euclidean: euclideanDistance(p1.vector, p2.vector),
    };
  }, [pcaData.points, selectedPair]);

  const handleSelectPoint = (pointId: string) => {
    setSelectedPair((prev) =>
      prev[1] === pointId ? prev : [prev[1], pointId],
    );
  };

  const handleComputeLiveEmbeddings = async () => {
    const lines = liveSentencesText
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
      .slice(0, 12);

    if (lines.length < 2) {
      setLiveError('Enter at least 2 sentences (one per line) to compute 3D PCA.');
      return;
    }

    setLiveLoading(true);
    setLiveError(null);

    try {
      const response = await fetch('/api/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: lines }),
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json?.error || `HTTP ${response.status}`);
      }
      setLivePayload(json);
      setDataSourceMode('live');
      setSelectedPair(['live-0', 'live-1']);
    } catch (err) {
      setLiveError(
        err instanceof Error
          ? err.message
          : 'Failed to retrieve live embeddings.',
      );
    } finally {
      setLiveLoading(false);
    }
  };

  // Real cosine similarities for the active contextual experiment
  const simAToNeighbors = useMemo(() => {
    const map = new Map(STATIC_EMBEDDINGS.map((w) => [w.word, w.vector]));
    return activeExperiment.senseANeighbors.map((w) => ({
      word: w,
      simA: cosineSimilarity(activeExperiment.vectorA, map.get(w) ?? []),
      simB: cosineSimilarity(activeExperiment.vectorB, map.get(w) ?? []),
    }));
  }, [activeExperiment]);

  const staticWordsList = useMemo(
    () => STATIC_EMBEDDINGS.map((w) => w.word),
    [],
  );

  return (
    <section id="embeddings" className="bg-[#070B14]">
      <div className="mx-auto max-w-[1360px] px-6 py-16 lg:py-20">
        {/* Section Header */}
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-10">
          <div>
            <p className="text-xs font-mono text-violet-400 mb-2">
              Vector Geometry &amp; Real Mathematical PCA
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-50">
              04. Embeddings Explorer
            </h2>
            <p className="mt-2 max-w-2xl text-[15px] text-slate-300 leading-relaxed">
              Compare how fixed word embeddings (Word2Vec / GloVe) encode global
              analogies versus how contextual Transformer embeddings (BERT / modern
              LLMs) dynamically shift word representations based on surrounding context.
            </p>
          </div>
          <div className="font-mono text-xs text-slate-400">
            {dataSourceMode === 'live' && livePayload
              ? `Live Neural Model: ${livePayload.model} (${livePayload.dimensions}D → 3D PCA)`
              : 'Reference Dataset: 24D Semantic Vectors · Live 3D TypeScript PCA'}
          </div>
        </div>

        {/* Two Separate Sections: Static vs Contextual */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* Sub-section A: Static Embeddings */}
          <article className="rounded-xl border border-slate-800 bg-[#0D1322] p-6 flex flex-col justify-between space-y-5">
            <div>
              <div className="text-xs font-mono text-cyan-400 mb-1.5">
                Part A · Fixed Word Representations
              </div>
              <h3 className="text-xl font-bold text-slate-50">
                Static Embeddings (Word2Vec &amp; GloVe)
              </h3>
              <p className="mt-2.5 text-sm text-slate-300 leading-relaxed">
                <strong>Word2Vec</strong> (Skip-gram &amp; CBOW) predicts nearby
                context words to learn dense vectors, while <strong>GloVe</strong>{' '}
                factorizes a global word-to-word co-occurrence matrix. Both produce a
                single static lookup table: every word receives{' '}
                <strong>one fixed vector</strong> regardless of sentence context.
              </p>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                Because semantic relationships correspond to linear directions in vector
                space, static embeddings support algebraic analogies such as{' '}
                <span className="font-mono text-cyan-300">
                  king − man + woman ≈ queen
                </span>
                .
              </p>
            </div>

            {/* Interactive Vector Arithmetic Sandbox */}
            <div className="rounded-lg border border-slate-800 bg-[#070B14] p-4 space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-200">
                  Vector Arithmetic Lab (24D Cosine Search)
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {ANALOGY_PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => {
                        setAnalogyA(preset.a);
                        setAnalogyB(preset.b);
                        setAnalogyC(preset.c);
                      }}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-0.5 font-mono text-[11px] text-slate-300 hover:border-cyan-400 hover:text-cyan-200 cursor-pointer"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 items-center">
                <div>
                  <label
                    htmlFor="analogy-a"
                    className="block text-[11px] font-mono text-slate-400 mb-1"
                  >
                    Base (+)
                  </label>
                  <select
                    id="analogy-a"
                    value={analogyA}
                    onChange={(e) => setAnalogyA(e.target.value)}
                    className="w-full rounded border border-slate-700 bg-[#0D1322] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  >
                    {staticWordsList.map((w) => (
                      <option key={w} value={w}>
                        {w}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="analogy-b"
                    className="block text-[11px] font-mono text-slate-400 mb-1"
                  >
                    Subtract (−)
                  </label>
                  <select
                    id="analogy-b"
                    value={analogyB}
                    onChange={(e) => setAnalogyB(e.target.value)}
                    className="w-full rounded border border-slate-700 bg-[#0D1322] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  >
                    {staticWordsList.map((w) => (
                      <option key={w} value={w}>
                        {w}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="analogy-c"
                    className="block text-[11px] font-mono text-slate-400 mb-1"
                  >
                    Add (+)
                  </label>
                  <select
                    id="analogy-c"
                    value={analogyC}
                    onChange={(e) => setAnalogyC(e.target.value)}
                    className="w-full rounded border border-slate-700 bg-[#0D1322] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  >
                    {staticWordsList.map((w) => (
                      <option key={w} value={w}>
                        {w}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] font-mono text-slate-400">
                  Nearest Neighbors to ({analogyA} − {analogyB} + {analogyC}):
                </div>
                {analogyResults.map((res, idx) => (
                  <div
                    key={res.word}
                    className={`flex items-center justify-between rounded px-2.5 py-1.5 text-xs font-mono ${
                      idx === 0
                        ? 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-100'
                        : 'bg-[#0D1322] text-slate-300'
                    }`}
                  >
                    <span>
                      #{idx + 1} {res.word}{' '}
                      <span className="font-sans text-[11px] text-slate-400">
                        · {res.description}
                      </span>
                    </span>
                    <span className="tabular-nums text-cyan-300">
                      cos {(res.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </article>

          {/* Sub-section B: Contextual Embeddings */}
          <article className="rounded-xl border border-slate-800 bg-[#0D1322] p-6 flex flex-col justify-between space-y-5">
            <div>
              <div className="text-xs font-mono text-violet-400 mb-1.5">
                Part B · Context-Conditioned Representations
              </div>
              <h3 className="text-xl font-bold text-slate-50">
                Contextual Embeddings (BERT &amp; Transformers)
              </h3>
              <p className="mt-2.5 text-sm text-slate-300 leading-relaxed">
                Models like <strong>BERT</strong> pass tokens through stacked
                bidirectional self-attention layers. Instead of a fixed dictionary
                vector, a word’s output representation is recomputed dynamically for
                every sentence—allowing polysemous words to occupy completely different
                regions of embedding space.
              </p>
            </div>

            {/* Polysemy Context Inspector */}
            <div className="rounded-lg border border-slate-800 bg-[#070B14] p-4 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-200">
                  Polysemy Experiment Target Word
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {CONTEXTUAL_EXPERIMENTS.map((exp) => (
                    <button
                      key={exp.id}
                      type="button"
                      onClick={() => {
                        setDataSourceMode('reference');
                        setSelectedExperimentId(exp.id);
                      }}
                      className={`rounded-md px-2.5 py-1 font-mono text-xs transition-colors cursor-pointer ${
                        selectedExperimentId === exp.id && dataSourceMode === 'reference'
                          ? 'bg-violet-500/20 border border-violet-400 text-violet-200'
                          : 'bg-slate-900 border border-slate-700 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      "{exp.targetWord}"
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="rounded-md border border-emerald-500/35 bg-emerald-500/8 p-3">
                  <div className="text-[11px] font-mono text-emerald-300">
                    Context A · {activeExperiment.senseALabel}
                  </div>
                  <p className="mt-1 text-xs text-slate-100 leading-relaxed">
                    "{activeExperiment.sentenceA}"
                  </p>
                </div>
                <div className="rounded-md border border-cyan-500/35 bg-cyan-500/8 p-3">
                  <div className="text-[11px] font-mono text-cyan-300">
                    Context B · {activeExperiment.senseBLabel}
                  </div>
                  <p className="mt-1 text-xs text-slate-100 leading-relaxed">
                    "{activeExperiment.sentenceB}"
                  </p>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>
                    Cosine Similarity Shift Toward "{activeExperiment.senseALabel}" Neighbors
                  </span>
                  <span>
                    Context A vs Context B sim:{' '}
                    {(
                      cosineSimilarity(
                        activeExperiment.vectorA,
                        activeExperiment.vectorB,
                      ) * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                {simAToNeighbors.map((row) => (
                  <div
                    key={row.word}
                    className="grid grid-cols-12 items-center gap-2 text-xs font-mono"
                  >
                    <span className="col-span-3 text-slate-300 truncate">
                      {row.word}
                    </span>
                    <div className="col-span-9 flex items-center gap-3">
                      <span className="text-emerald-300 tabular-nums w-20">
                        A: {(row.simA * 100).toFixed(0)}%
                      </span>
                      <span className="text-cyan-300 tabular-nums w-20">
                        B: {(row.simB * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-800 pt-2.5">
                {activeExperiment.explanation}
              </p>
            </div>
          </article>
        </div>

        {/* Interactive 3D PCA Comparison Graph + Controls */}
        <div className="rounded-xl border border-slate-800 bg-[#0D1322] p-5 sm:p-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
            <div>
              <h3 className="text-lg font-semibold text-slate-100">
                Interactive 3D PCA Projection: Static vs. Contextual Space
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Real 3-component Principal Component Analysis computed live in
                TypeScript · Orbit in 3D, zoom, and click any two nodes to measure
                high-dimensional cosine similarity
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex rounded-lg border border-slate-800 bg-[#070B14] p-0.5">
                <button
                  type="button"
                  onClick={() => setDataSourceMode('reference')}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                    dataSourceMode === 'reference'
                      ? 'bg-cyan-500/20 text-cyan-200'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  24D Reference PCA
                </button>
                <button
                  type="button"
                  onClick={() => setDataSourceMode('live')}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                    dataSourceMode === 'live'
                      ? 'bg-violet-500/20 text-violet-200'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Live Gemini Model PCA
                </button>
              </div>

              {dataSourceMode === 'reference' && (
                <>
                  <button
                    type="button"
                    onClick={() => setShowCorpusWords((v) => !v)}
                    className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                      showCorpusWords
                        ? 'border-blue-500/50 bg-blue-500/15 text-blue-200'
                        : 'border-slate-700 bg-[#070B14] text-slate-400'
                    }`}
                  >
                    {showCorpusWords ? 'Corpus Clusters: On' : 'Corpus Clusters: Off'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowTrajectories((v) => !v)}
                    className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                      showTrajectories
                        ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-200'
                        : 'border-slate-700 bg-[#070B14] text-slate-400'
                    }`}
                  >
                    {showTrajectories ? 'Shift Vectors: On' : 'Shift Vectors: Off'}
                  </button>
                </>
              )}

              <button
                type="button"
                onClick={() => setAutoRotate((v) => !v)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                  autoRotate
                    ? 'border-violet-500/50 bg-violet-500/15 text-violet-200'
                    : 'border-slate-700 bg-[#070B14] text-slate-300 hover:border-slate-500'
                }`}
              >
                {autoRotate ? 'Auto-Orbit: On' : 'Auto-Orbit: Off'}
              </button>

              <button
                type="button"
                onClick={() => setResetSignal((s) => s + 1)}
                className="rounded-lg border border-slate-700 bg-[#070B14] px-3 py-1.5 text-xs text-slate-300 hover:border-slate-500 cursor-pointer"
              >
                Reset Camera
              </button>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* 3D WebGL Plot Canvas */}
            <div className="lg:col-span-8 relative rounded-lg border border-slate-800 bg-[#070B14] p-3">
              <EmbeddingSpace3D
                points={pcaData.points}
                clusterColors={CLUSTER_COLORS}
                explainedVarianceRatio={pcaData.pca.explainedVarianceRatio}
                dataSourceMode={dataSourceMode}
                showTrajectories={showTrajectories}
                selectedPair={selectedPair}
                onSelectPoint={handleSelectPoint}
                onHoverPoint={setHoveredPoint}
                resetSignal={resetSignal}
                autoRotate={autoRotate}
              />

              {/* Hover Coordinate Readout */}
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-2 py-1 border-t border-slate-800/80 text-xs font-mono text-slate-400">
                {hoveredPoint ? (
                  <span className="text-slate-200">
                    {hoveredPoint.label} · PC1: {hoveredPoint.x.toFixed(3)} · PC2:{' '}
                    {hoveredPoint.y.toFixed(3)} · PC3: {hoveredPoint.z.toFixed(3)}{' '}
                    —{' '}
                    <span className="font-sans text-slate-400">
                      {hoveredPoint.sublabel}
                    </span>
                  </span>
                ) : (
                  <span>
                    Hover any 3D node to inspect (PC1, PC2, PC3) coordinates · Total
                    variance captured:{' '}
                    {(pcaData.pca.totalVarianceCaptured * 100).toFixed(1)}% (
                    {pcaData.pca.dimensions}D → 3D)
                  </span>
                )}
              </div>
            </div>

            {/* Right Column: Pairwise Distance Inspector + Live Gemini Embeddings Launcher */}
            <div className="lg:col-span-4 space-y-4">
              {/* Pairwise Vector Comparison */}
              <div className="rounded-lg border border-slate-800 bg-[#070B14] p-4 space-y-3">
                <div className="text-xs font-semibold text-slate-200">
                  Selected Pair Measurement ({pcaData.pca.dimensions}D Full Space)
                </div>

                {pairMetrics ? (
                  <>
                    <div className="space-y-1.5 text-xs">
                      <div className="rounded bg-[#0D1322] px-2.5 py-1.5 font-mono text-slate-100 truncate">
                        Point 1: {pairMetrics.p1.label}
                      </div>
                      <div className="rounded bg-[#0D1322] px-2.5 py-1.5 font-mono text-slate-100 truncate">
                        Point 2: {pairMetrics.p2.label}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 pt-1">
                      <div>
                        <div className="text-[11px] text-slate-400">
                          Cosine Similarity
                        </div>
                        <div className="font-mono text-xl font-semibold tabular-nums text-cyan-300">
                          {(pairMetrics.cosine * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] text-slate-400">
                          Euclidean Dist
                        </div>
                        <div className="font-mono text-xl font-semibold tabular-nums text-violet-300">
                          {pairMetrics.euclidean.toFixed(3)}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-slate-400">
                    Click two nodes on the scatter plot to measure similarity.
                  </p>
                )}
              </div>

              {/* Legend */}
              {dataSourceMode === 'reference' && (
                <div className="rounded-lg border border-slate-800 bg-[#070B14] p-4">
                  <div className="text-xs font-semibold text-slate-200 mb-2.5">
                    Cluster Legend
                  </div>
                  <div className="grid grid-cols-2 gap-y-1.5 gap-x-2 text-xs text-slate-300">
                    {(
                      [
                        'static-anchor',
                        'context-a',
                        'context-b',
                        'royalty',
                        'ai-tech',
                        'finance',
                        'nature',
                        'geography',
                      ] as SemanticCluster[]
                    ).map((key) => (
                      <div key={key} className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: CLUSTER_COLORS[key] }}
                          aria-hidden="true"
                        />
                        <span className="truncate">{CLUSTER_LABELS[key]}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Live Gemini Neural Embedding Sandbox */}
              <div className="rounded-lg border border-slate-800 bg-[#070B14] p-4 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-200">
                    Live Neural Embeddings (gemini-embedding-2)
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Enter 2–12 sentences (one per line) to embed them with the real{' '}
                  <span className="font-mono text-slate-300">gemini-embedding-2</span>{' '}
                  model and project the resulting high-dimensional vectors via PCA:
                </p>
                <textarea
                  rows={5}
                  value={liveSentencesText}
                  onChange={(e) => setLiveSentencesText(e.target.value)}
                  aria-label="Custom sentences for live neural embedding PCA"
                  className="w-full rounded border border-slate-700 bg-[#0D1322] p-2.5 font-mono text-[11px] text-slate-100 focus:border-violet-400 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleComputeLiveEmbeddings}
                  disabled={liveLoading}
                  className="w-full rounded-lg border border-violet-500/50 bg-violet-500/15 px-3 py-2 text-xs font-medium text-violet-200 transition-colors hover:bg-violet-500/25 disabled:opacity-50 cursor-pointer"
                >
                  {liveLoading
                    ? 'Computing Real Embeddings & PCA…'
                    : 'Compute Live Neural PCA'}
                </button>

                {liveError && (
                  <div
                    role="alert"
                    className="rounded border border-rose-500/40 bg-rose-500/10 p-2.5 text-xs text-rose-200 leading-relaxed"
                  >
                    {liveError}
                  </div>
                )}

                {livePayload && (
                  <div className="space-y-1 pt-1 border-t border-slate-800 text-[11px] font-mono text-slate-300">
                    <div className="text-violet-300">
                      Returned {livePayload.embeddings.length} ×{' '}
                      {livePayload.dimensions}D vectors
                    </div>
                    {livePayload.texts.map((t, i) => (
                      <div key={i} className="truncate text-slate-400">
                        S{i + 1}: {t}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
