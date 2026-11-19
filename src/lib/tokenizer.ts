import cl100kApi from 'gpt-tokenizer/encoding/cl100k_base';
import type { GptEncoding } from 'gpt-tokenizer/GptEncoding';

export type TokenizerEncodingId = 'cl100k_base' | 'o200k_base' | 'r50k_base';

export interface EncodingMeta {
  id: TokenizerEncodingId;
  label: string;
  models: string;
  vocabSize: number;
  description: string;
}

export const ENCODING_METADATA: Record<TokenizerEncodingId, EncodingMeta> = {
  cl100k_base: {
    id: 'cl100k_base',
    label: 'cl100k_base',
    models: 'GPT-4 · GPT-3.5-Turbo · text-embedding-ada-002',
    vocabSize: 100277,
    description:
      '100K-token Byte-Pair Encoding (BPE) vocabulary with improved whitespace, code, and multilingual compression.',
  },
  o200k_base: {
    id: 'o200k_base',
    label: 'o200k_base',
    models: 'GPT-4o · o1 · o3',
    vocabSize: 200019,
    description:
      '200K-token BPE vocabulary doubling merge capacity for fewer subword splits across non-English scripts and code.',
  },
  r50k_base: {
    id: 'r50k_base',
    label: 'r50k_base',
    models: 'GPT-2 · GPT-3 (davinci)',
    vocabSize: 50257,
    description:
      'Classic 50K-token byte-level BPE vocabulary introduced with GPT-2, illustrating denser subword fragmentation.',
  },
};

export type TokenCategory =
  | 'word'
  | 'leading-space-word'
  | 'subword'
  | 'number'
  | 'punctuation'
  | 'whitespace'
  | 'special';

export interface TokenInfo {
  index: number;
  id: number;
  text: string;
  displayText: string;
  byteLength: number;
  utf8Hex: string[];
  category: TokenCategory;
  startsWithSpace: boolean;
  isMultiByte: boolean;
}

export interface TokenizationResult {
  encoding: TokenizerEncodingId;
  text: string;
  tokens: TokenInfo[];
  tokenIds: number[];
  totalTokens: number;
  totalChars: number;
  totalBytes: number;
  charsPerToken: number;
}

export interface VocabItem {
  id: number;
  text: string;
  displayText: string;
  category: TokenCategory;
  byteLength: number;
  utf8Hex: string[];
  definition: string;
}

const encodingCache = new Map<TokenizerEncodingId, GptEncoding>();
encodingCache.set('cl100k_base', cl100kApi);

export async function loadEncoding(id: TokenizerEncodingId): Promise<GptEncoding> {
  const cached = encodingCache.get(id);
  if (cached) return cached;

  if (id === 'o200k_base') {
    const mod = await import('gpt-tokenizer/encoding/o200k_base');
    encodingCache.set('o200k_base', mod.default);
    return mod.default;
  }

  if (id === 'r50k_base') {
    const mod = await import('gpt-tokenizer/encoding/r50k_base');
    encodingCache.set('r50k_base', mod.default);
    return mod.default;
  }

  return cl100kApi;
}

export function getLoadedEncoding(id: TokenizerEncodingId): GptEncoding {
  return encodingCache.get(id) ?? cl100kApi;
}

const textEncoder = new TextEncoder();

export function toDisplayToken(text: string): string {
  if (text.length === 0) return '∅';
  return text
    .replace(/ /g, '·')
    .replace(/\n/g, '↵\n')
    .replace(/\t/g, '→');
}

export function classifyToken(text: string, id: number, specialIds?: Set<number>): TokenCategory {
  if (specialIds?.has(id) || (text.startsWith('<|') && text.endsWith('|>')) || text === '<|endoftext|>') {
    return 'special';
  }
  if (/^\s+$/.test(text)) {
    return 'whitespace';
  }
  if (/^\s?[0-9]+$/.test(text)) {
    return 'number';
  }
  if (/^[^\w\s]+$/.test(text)) {
    return 'punctuation';
  }
  if (/^ [A-Za-z]{2,}$/.test(text)) {
    return 'leading-space-word';
  }
  if (/^[A-Za-z]{3,}$/.test(text)) {
    return 'word';
  }
  return 'subword';
}

export function describeTokenCategory(category: TokenCategory, text: string, id: number): string {
  switch (category) {
    case 'leading-space-word':
      return `Space-prefixed word token ("${text.trim()}"). BPE merges the preceding ASCII space (0x20) directly into common words so sentences take half as many tokens.`;
    case 'word':
      return `Unprefixed whole-word or root token ("${text}"). Commonly appears at sequence start, after punctuation, or as a capitalized stem.`;
    case 'subword':
      return `Subword fragment ("${text}"). Used when a word is rare, compound, cased unusually, or split into morphological pieces by BPE merge rank #${id.toLocaleString()}.`;
    case 'number':
      return `Numeric chunk ("${text.trim()}"). Modern BPE vocabularies pre-chunk digits into 1–3 digit spans for consistent arithmetic tokenization.`;
    case 'punctuation':
      return `Punctuation / symbol token ("${text}"). Encoded separately or grouped into common multi-character operators in code.`;
    case 'whitespace':
      return `Whitespace or newline control sequence (${text.length} char${text.length === 1 ? '' : 's'}). Preserves indentation and paragraph structure.`;
    case 'special':
      return `Reserved model control token (${text}) outside standard text merges, signaling boundaries like end-of-text or fill-in-the-middle.`;
  }
}

export function tokenizeTextSync(
  text: string,
  encodingId: TokenizerEncodingId,
): TokenizationResult {
  const api = getLoadedEncoding(encodingId);
  const tokenIds = api.encode(text, { allowedSpecial: 'all' });
  const totalBytes = textEncoder.encode(text).length;

  const tokens: TokenInfo[] = [];
  let index = 0;

  // Decode token by token cleanly via decodeGenerator or individual decode
  for (const id of tokenIds) {
    let piece = '';
    try {
      piece = api.decode([id]);
    } catch {
      piece = '';
    }
    const bytes = Array.from(textEncoder.encode(piece));
    const utf8Hex = bytes.map((b) => b.toString(16).padStart(2, '0').toUpperCase());
    const category = classifyToken(piece, id);

    tokens.push({
      index,
      id,
      text: piece,
      displayText: toDisplayToken(piece),
      byteLength: bytes.length,
      utf8Hex,
      category,
      startsWithSpace: piece.startsWith(' '),
      isMultiByte: bytes.length > piece.length,
    });
    index++;
  }

  return {
    encoding: encodingId,
    text,
    tokens,
    tokenIds,
    totalTokens: tokenIds.length,
    totalChars: text.length,
    totalBytes,
    charsPerToken: tokenIds.length > 0 ? text.length / tokenIds.length : 0,
  };
}

// Vocabulary index cache per encoding
const vocabIndexCache = new Map<TokenizerEncodingId, VocabItem[]>();

const SPECIAL_TOKEN_OVERRIDES: Record<TokenizerEncodingId, Array<[number, string]>> = {
  cl100k_base: [
    [100257, '<|endoftext|>'],
    [100258, '<|fim_prefix|>'],
    [100259, '<|fim_middle|>'],
    [100260, '<|fim_suffix|>'],
    [100276, '<|endofprompt|>'],
  ],
  o200k_base: [
    [199999, '<|endoftext|>'],
    [200018, '<|endofprompt|>'],
  ],
  r50k_base: [[50256, '<|endoftext|>']],
};

export function getVocabularyIndex(encodingId: TokenizerEncodingId): VocabItem[] {
  const existing = vocabIndexCache.get(encodingId);
  if (existing) return existing;

  const api = getLoadedEncoding(encodingId);
  const meta = ENCODING_METADATA[encodingId];
  const specialMap = new Map<number, string>(SPECIAL_TOKEN_OVERRIDES[encodingId] ?? []);
  const specialIds = new Set<number>(specialMap.keys());

  const items: VocabItem[] = [];
  // Regular BPE merge IDs run from 0 up to baseCount
  const regularLimit =
    encodingId === 'cl100k_base' ? 100256 : encodingId === 'o200k_base' ? 199999 : 50256;

  for (let id = 0; id < regularLimit; id++) {
    let text = '';
    try {
      text = api.decode([id]);
    } catch {
      continue;
    }
    const bytes = Array.from(textEncoder.encode(text));
    const utf8Hex = bytes.map((b) => b.toString(16).padStart(2, '0').toUpperCase());
    const category = classifyToken(text, id, specialIds);
    items.push({
      id,
      text,
      displayText: toDisplayToken(text),
      category,
      byteLength: bytes.length,
      utf8Hex,
      definition: describeTokenCategory(category, text, id),
    });
  }

  for (const [id, text] of specialMap.entries()) {
    const bytes = Array.from(textEncoder.encode(text));
    items.push({
      id,
      text,
      displayText: text,
      category: 'special',
      byteLength: bytes.length,
      utf8Hex: bytes.map((b) => b.toString(16).padStart(2, '0').toUpperCase()),
      definition: describeTokenCategory('special', text, id),
    });
  }

  if (items.length > meta.vocabSize) {
    items.length = meta.vocabSize;
  }

  vocabIndexCache.set(encodingId, items);
  return items;
}

export function searchVocabulary(
  query: string,
  encodingId: TokenizerEncodingId,
  category: TokenCategory | 'all' = 'all',
  limit = 60,
): {
  items: VocabItem[];
  totalMatches: number;
  totalVocabSize: number;
} {
  const index = getVocabularyIndex(encodingId);
  const trimmed = query.trim();
  const isNumericId = /^\d+$/.test(trimmed);
  const numericQuery = isNumericId ? Number(trimmed) : -1;
  const lower = trimmed.toLowerCase();

  const matches: VocabItem[] = [];

  for (let i = 0; i < index.length; i++) {
    const item = index[i];
    if (category !== 'all' && item.category !== category) continue;

    if (!trimmed) {
      matches.push(item);
      continue;
    }

    if (isNumericId && item.id === numericQuery) {
      matches.unshift(item);
      continue;
    }

    const itemLower = item.text.toLowerCase();
    if (itemLower.includes(lower) || String(item.id).includes(trimmed)) {
      matches.push(item);
    }
  }

  // Sort exact/prefix matches higher when searching text
  if (trimmed && !isNumericId) {
    matches.sort((a, b) => {
      const aClean = a.text.trim().toLowerCase();
      const bClean = b.text.trim().toLowerCase();
      const aExact = aClean === lower ? 0 : aClean.startsWith(lower) ? 1 : 2;
      const bExact = bClean === lower ? 0 : bClean.startsWith(lower) ? 1 : 2;
      if (aExact !== bExact) return aExact - bExact;
      return a.id - b.id;
    });
  }

  return {
    items: matches.slice(0, limit),
    totalMatches: matches.length,
    totalVocabSize: ENCODING_METADATA[encodingId].vocabSize,
  };
}

export function getNeighboringVocabItems(
  id: number,
  encodingId: TokenizerEncodingId,
  radius = 3,
): VocabItem[] {
  const index = getVocabularyIndex(encodingId);
  const pos = index.findIndex((item) => item.id === id);
  if (pos === -1) return [];
  const start = Math.max(0, pos - radius);
  const end = Math.min(index.length, pos + radius + 1);
  return index.slice(start, end);
}
