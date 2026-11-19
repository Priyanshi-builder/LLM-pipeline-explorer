/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { PipelineSection } from './components/PipelineSection';
import { TokenizerSection } from './components/TokenizerSection';
import { VocabularySection } from './components/VocabularySection';
import { EmbeddingsSection } from './components/EmbeddingsSection';
import type { TokenizerEncodingId } from './lib/tokenizer';

const DEFAULT_LAB_TEXT = 'I deposited cash at the river bank before sunset.';

export default function App() {
  const [labText, setLabText] = useState<string>(DEFAULT_LAB_TEXT);
  const [encodingId, setEncodingId] = useState<TokenizerEncodingId>('cl100k_base');
  const [focusedVocabId, setFocusedVocabId] = useState<number | null>(null);
  const [activeSection, setActiveSection] = useState<string>('pipeline');

  useEffect(() => {
    const ids = ['pipeline', 'tokenizer', 'vocabulary', 'embeddings'];
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);

    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]?.target?.id) {
          setActiveSection(visible[0].target.id);
        }
      },
      {
        rootMargin: '-20% 0px -55% 0px',
        threshold: [0.1, 0.25, 0.5],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const handleResetLab = () => {
    setLabText(DEFAULT_LAB_TEXT);
    setEncodingId('cl100k_base');
    setFocusedVocabId(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleInspectInTokenizer = (text: string) => {
    setLabText(text);
    const el = document.getElementById('tokenizer');
    el?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleLookupVocabToken = (tokenId: number) => {
    setFocusedVocabId(tokenId);
    const el = document.getElementById('vocabulary');
    el?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 flex flex-col">
      <Navbar activeSection={activeSection} onResetLab={handleResetLab} />

      <main className="flex-1">
        <PipelineSection
          text={labText}
          onTextChange={setLabText}
          encodingId={encodingId}
          onInspectInTokenizer={handleInspectInTokenizer}
        />

        <TokenizerSection
          text={labText}
          onTextChange={setLabText}
          encodingId={encodingId}
          onEncodingChange={setEncodingId}
          onLookupVocabToken={handleLookupVocabToken}
        />

        <VocabularySection
          encodingId={encodingId}
          focusedTokenId={focusedVocabId}
        />

        <EmbeddingsSection />
      </main>

      <footer className="border-t border-slate-800/80 bg-[#070B14] py-8">
        <div className="mx-auto max-w-[1360px] px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <div>
            LLM Pipeline Explorer · Interactive Tokenization, Vocabulary &amp; PCA
            Embedding Lab
          </div>
          <div className="flex items-center gap-5">
            <a href="#pipeline" className="hover:text-slate-200 transition-colors">
              Pipeline
            </a>
            <a href="#tokenizer" className="hover:text-slate-200 transition-colors">
              Tokenizer
            </a>
            <a href="#vocabulary" className="hover:text-slate-200 transition-colors">
              Vocabulary
            </a>
            <a href="#embeddings" className="hover:text-slate-200 transition-colors">
              Embeddings
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
