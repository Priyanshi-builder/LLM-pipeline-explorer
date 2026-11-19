export type SemanticCluster =
  | 'royalty'
  | 'ai-tech'
  | 'finance'
  | 'nature'
  | 'animals'
  | 'geography'
  | 'static-anchor'
  | 'context-a'
  | 'context-b';

export interface StaticEmbeddingWord {
  word: string;
  cluster: SemanticCluster;
  description: string;
  vector: number[];
}

export interface ContextualSenseExample {
  id: string;
  targetWord: string;
  senseALabel: string;
  sentenceA: string;
  senseANeighbors: string[];
  vectorA: number[];
  senseBLabel: string;
  sentenceB: string;
  senseBNeighbors: string[];
  vectorB: number[];
  staticAnchorVector: number[];
  explanation: string;
}

/**
 * 24-Dimensional Semantic Reference Vectors (aligned with GloVe/Word2Vec semantic axes).
 * Axes:
 * 0: Royalty/Authority   1: Male(+)/Female(-)  2: Human/Agent      3: Adult(+)/Youth(-)
 * 4: Computing/Software  5: AI/Math/Model      6: Finance/Economy  7: Water/Nature
 * 8: Botanical/Food      9: Animal/Fauna      10: Flight/Avian    11: Mechanical/Artifact
 * 12: Optics/Light      13: Mass/Lightness    14: Capital/City    15: Europe(+)/Asia(-)
 * 16: Concreteness      17: Abstractness      18: Scale/Size      19: Motion/Action
 * 20: Hardware/Device   21: Reptile/Wild      22: Sport/Recreation 23: Institution/Org
 */
export const STATIC_EMBEDDINGS: StaticEmbeddingWord[] = [
  // Royalty & People
  {
    word: 'king',
    cluster: 'royalty',
    description: 'Monarch (male, sovereign)',
    vector: [0.92, 0.85, 0.82, 0.72, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.65, 0.35, 0.4, 0.1, 0.0, 0.0, 0.0, 0.45],
  },
  {
    word: 'queen',
    cluster: 'royalty',
    description: 'Monarch (female, sovereign)',
    vector: [0.92, -0.85, 0.82, 0.72, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.15, 0.2, 0.65, 0.35, 0.38, 0.1, 0.0, 0.0, 0.0, 0.45],
  },
  {
    word: 'man',
    cluster: 'royalty',
    description: 'Adult male person',
    vector: [0.05, 0.88, 0.9, 0.75, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0, 0.75, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.05],
  },
  {
    word: 'woman',
    cluster: 'royalty',
    description: 'Adult female person',
    vector: [0.05, -0.88, 0.9, 0.75, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.75, 0.1, 0.18, 0.2, 0.0, 0.0, 0.0, 0.05],
  },
  {
    word: 'prince',
    cluster: 'royalty',
    description: 'Royal heir (male, youth)',
    vector: [0.84, 0.8, 0.8, -0.45, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.18, 0.65, 0.25, 0.25, 0.15, 0.0, 0.0, 0.0, 0.35],
  },
  {
    word: 'princess',
    cluster: 'royalty',
    description: 'Royal heir (female, youth)',
    vector: [0.84, -0.8, 0.8, -0.45, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.1, 0.18, 0.65, 0.25, 0.24, 0.15, 0.0, 0.0, 0.0, 0.35],
  },
  {
    word: 'crown',
    cluster: 'royalty',
    description: 'Royal emblem / headgear',
    vector: [0.88, 0.0, 0.15, 0.1, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.35, 0.15, 0.1, 0.05, 0.1, 0.82, 0.3, 0.15, 0.0, 0.0, 0.0, 0.0, 0.35],
  },

  // Geography (Capitals & Countries)
  {
    word: 'paris',
    cluster: 'geography',
    description: 'Capital city of France',
    vector: [0.1, 0.0, 0.15, 0.0, 0.0, 0.0, 0.2, 0.08, 0.05, 0.0, 0.0, 0.1, 0.05, 0.0, 0.94, 0.88, 0.72, 0.2, 0.85, 0.05, 0.0, 0.0, 0.0, 0.45],
  },
  {
    word: 'france',
    cluster: 'geography',
    description: 'Western European nation',
    vector: [0.12, 0.0, 0.1, 0.0, 0.0, 0.0, 0.22, 0.1, 0.08, 0.0, 0.0, 0.05, 0.0, 0.0, -0.4, 0.9, 0.55, 0.42, 0.95, 0.0, 0.0, 0.0, 0.0, 0.65],
  },
  {
    word: 'tokyo',
    cluster: 'geography',
    description: 'Capital city of Japan',
    vector: [0.1, 0.0, 0.15, 0.0, 0.25, 0.15, 0.25, 0.05, 0.02, 0.0, 0.0, 0.2, 0.05, 0.0, 0.94, -0.88, 0.72, 0.2, 0.88, 0.05, 0.12, 0.0, 0.0, 0.45],
  },
  {
    word: 'japan',
    cluster: 'geography',
    description: 'East Asian island nation',
    vector: [0.12, 0.0, 0.1, 0.0, 0.22, 0.12, 0.25, 0.12, 0.05, 0.0, 0.0, 0.18, 0.0, 0.0, -0.4, -0.9, 0.55, 0.42, 0.92, 0.0, 0.12, 0.0, 0.0, 0.65],
  },
  {
    word: 'london',
    cluster: 'geography',
    description: 'Capital city of the UK',
    vector: [0.22, 0.0, 0.15, 0.0, 0.05, 0.0, 0.38, 0.08, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.92, 0.85, 0.72, 0.2, 0.86, 0.05, 0.0, 0.0, 0.0, 0.5],
  },

  // Computing & AI
  {
    word: 'transformer',
    cluster: 'ai-tech',
    description: 'Self-attention neural architecture',
    vector: [0.0, 0.0, 0.0, 0.0, 0.86, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.2, 0.88, 0.45, 0.2, 0.42, 0.0, 0.0, 0.1],
  },
  {
    word: 'attention',
    cluster: 'ai-tech',
    description: 'Query-Key weighted context mechanism',
    vector: [0.0, 0.0, 0.15, 0.0, 0.72, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.1, 0.94, 0.3, 0.25, 0.2, 0.0, 0.0, 0.0],
  },
  {
    word: 'vector',
    cluster: 'ai-tech',
    description: 'Ordered numerical coordinates in space',
    vector: [0.0, 0.0, 0.0, 0.0, 0.78, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.15, 0.92, 0.25, 0.15, 0.15, 0.0, 0.0, 0.0],
  },
  {
    word: 'token',
    cluster: 'ai-tech',
    description: 'Subword unit indexed by vocabulary ID',
    vector: [0.0, 0.0, 0.0, 0.0, 0.84, 0.85, 0.12, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.25, 0.82, 0.12, 0.1, 0.2, 0.0, 0.0, 0.0],
  },
  {
    word: 'neural',
    cluster: 'ai-tech',
    description: 'Layered connectionist network',
    vector: [0.0, 0.0, 0.08, 0.0, 0.76, 0.94, 0.0, 0.0, 0.05, 0.08, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.18, 0.88, 0.35, 0.18, 0.3, 0.0, 0.0, 0.05],
  },
  {
    word: 'software',
    cluster: 'ai-tech',
    description: 'Executable code and applications',
    vector: [0.0, 0.0, 0.05, 0.0, 0.95, 0.55, 0.18, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.4, 0.15, 0.65, 0.0, 0.0, 0.35],
  },
  {
    word: 'computer',
    cluster: 'ai-tech',
    description: 'Programmable electronic hardware',
    vector: [0.0, 0.0, 0.0, 0.0, 0.92, 0.52, 0.15, 0.0, 0.0, 0.0, 0.0, 0.45, 0.08, 0.18, 0.0, 0.0, 0.78, 0.35, 0.35, 0.1, 0.94, 0.0, 0.0, 0.2],
  },
  {
    word: 'code',
    cluster: 'ai-tech',
    description: 'Structured programming instructions',
    vector: [0.0, 0.0, 0.0, 0.0, 0.96, 0.6, 0.05, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.3, 0.78, 0.2, 0.12, 0.42, 0.0, 0.0, 0.1],
  },

  // Finance
  {
    word: 'money',
    cluster: 'finance',
    description: 'Medium of exchange & currency',
    vector: [0.1, 0.0, 0.05, 0.0, 0.05, 0.0, 0.96, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.58, 0.62, 0.2, 0.15, 0.0, 0.0, 0.0, 0.55],
  },
  {
    word: 'loan',
    cluster: 'finance',
    description: 'Borrowed capital with interest',
    vector: [0.05, 0.0, 0.12, 0.0, 0.02, 0.05, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.78, 0.3, 0.18, 0.0, 0.0, 0.0, 0.72],
  },
  {
    word: 'deposit',
    cluster: 'finance',
    description: 'Funds placed into a financial account',
    vector: [0.0, 0.0, 0.1, 0.0, 0.05, 0.0, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.32, 0.64, 0.22, 0.3, 0.0, 0.0, 0.0, 0.68],
  },
  {
    word: 'account',
    cluster: 'finance',
    description: 'Ledger record of financial balance',
    vector: [0.0, 0.0, 0.1, 0.0, 0.22, 0.05, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.76, 0.25, 0.1, 0.15, 0.0, 0.0, 0.68],
  },
  {
    word: 'credit',
    cluster: 'finance',
    description: 'Borrowing capacity & trust rating',
    vector: [0.0, 0.0, 0.1, 0.0, 0.1, 0.05, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.82, 0.28, 0.1, 0.05, 0.0, 0.0, 0.65],
  },

  // Nature, Water & Fruit
  {
    word: 'river',
    cluster: 'nature',
    description: 'Flowing natural watercourse',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.96, 0.22, 0.08, 0.0, 0.0, 0.08, 0.1, 0.12, 0.0, 0.88, 0.12, 0.78, 0.55, 0.0, 0.05, 0.05, 0.0],
  },
  {
    word: 'stream',
    cluster: 'nature',
    description: 'Small flowing body of water',
    vector: [0.0, 0.0, 0.0, 0.0, 0.15, 0.05, 0.0, 0.9, 0.2, 0.08, 0.0, 0.0, 0.06, 0.05, 0.05, 0.0, 0.82, 0.18, 0.45, 0.6, 0.05, 0.05, 0.05, 0.0],
  },
  {
    word: 'shore',
    cluster: 'nature',
    description: 'Land edge along water',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.18, 0.05, 0.08, 0.0, 0.05, 0.08, 0.1, 0.0, 0.88, 0.1, 0.65, 0.05, 0.0, 0.02, 0.05, 0.0],
  },
  {
    word: 'water',
    cluster: 'nature',
    description: 'Clear liquid H2O element',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.95, 0.25, 0.05, 0.0, 0.0, 0.1, 0.12, 0.0, 0.0, 0.9, 0.12, 0.5, 0.45, 0.0, 0.0, 0.0, 0.0],
  },
  {
    word: 'fruit',
    cluster: 'nature',
    description: 'Edible sweet botanical produce',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.35, 0.95, 0.05, 0.0, 0.0, 0.05, 0.08, 0.0, 0.0, 0.9, 0.08, 0.18, 0.0, 0.0, 0.0, 0.0, 0.0],
  },
  {
    word: 'orchard',
    cluster: 'nature',
    description: 'Cultivated grove of fruit trees',
    vector: [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.12, 0.45, 0.9, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.86, 0.1, 0.68, 0.0, 0.0, 0.0, 0.0, 0.05],
  },
  {
    word: 'tree',
    cluster: 'nature',
    description: 'Perennial woody plant',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.55, 0.88, 0.1, 0.1, 0.0, 0.0, 0.25, 0.0, 0.0, 0.92, 0.12, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0],
  },

  // Animals & Sports/Tools
  {
    word: 'bird',
    cluster: 'animals',
    description: 'Feathered winged vertebrate',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.1, 0.88, 0.94, 0.0, 0.0, -0.25, 0.0, 0.0, 0.88, 0.05, 0.18, 0.65, 0.0, 0.05, 0.0, 0.0],
  },
  {
    word: 'wing',
    cluster: 'animals',
    description: 'Aerodynamic appendage for flight',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.68, 0.92, 0.22, 0.0, -0.2, 0.0, 0.0, 0.85, 0.08, 0.22, 0.6, 0.0, 0.0, 0.0, 0.0],
  },
  {
    word: 'cave',
    cluster: 'animals',
    description: 'Natural underground rock chamber',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.58, 0.05, 0.35, 0.2, 0.0, -0.45, 0.35, 0.08, 0.0, 0.9, 0.1, 0.72, 0.0, 0.0, 0.25, 0.0, 0.0],
  },
  {
    word: 'snake',
    cluster: 'animals',
    description: 'Elongated legless reptile',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.08, 0.86, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.88, 0.05, 0.22, 0.52, 0.0, 0.95, 0.0, 0.0],
  },
  {
    word: 'baseball',
    cluster: 'animals',
    description: 'Bat-and-ball team sport',
    vector: [0.0, 0.1, 0.25, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.15, 0.58, 0.0, 0.1, 0.0, 0.1, 0.82, 0.2, 0.35, 0.68, 0.0, 0.0, 0.95, 0.25],
  },
  {
    word: 'machine',
    cluster: 'animals',
    description: 'Engineered mechanical apparatus',
    vector: [0.0, 0.0, 0.0, 0.0, 0.45, 0.35, 0.1, 0.0, 0.0, 0.0, 0.0, 0.94, 0.0, 0.55, 0.0, 0.0, 0.86, 0.2, 0.65, 0.4, 0.72, 0.0, 0.0, 0.15],
  },
  {
    word: 'lamp',
    cluster: 'animals',
    description: 'Electric illumination fixture',
    vector: [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.94, 0.12, 0.0, 0.0, 0.88, 0.08, 0.2, 0.0, 0.45, 0.0, 0.0, 0.0],
  },
  {
    word: 'feather',
    cluster: 'animals',
    description: 'Ultra-light avian plumage',
    vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.05, 0.72, 0.82, 0.0, 0.05, -0.88, 0.0, 0.0, 0.9, 0.08, 0.08, 0.2, 0.0, 0.0, 0.0, 0.0],
  },
];

export const CONTEXTUAL_EXPERIMENTS: ContextualSenseExample[] = [
  {
    id: 'bank',
    targetWord: 'bank',
    senseALabel: 'Financial Institution',
    sentenceA: 'I deposited my paycheck at the bank.',
    senseANeighbors: ['money', 'loan', 'deposit', 'account'],
    vectorA: [0.05, 0.0, 0.12, 0.0, 0.08, 0.0, 0.94, 0.02, 0.0, 0.0, 0.0, 0.04, 0.0, 0.05, 0.05, 0.0, 0.45, 0.65, 0.42, 0.12, 0.05, 0.0, 0.0, 0.85],
    senseBLabel: 'River Shoreline',
    sentenceB: 'We rested on the grassy river bank.',
    senseBNeighbors: ['river', 'stream', 'shore', 'water'],
    vectorB: [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02, 0.94, 0.35, 0.08, 0.02, 0.0, 0.05, 0.08, 0.08, 0.0, 0.88, 0.1, 0.62, 0.2, 0.0, 0.02, 0.05, 0.0],
    staticAnchorVector: [0.02, 0.0, 0.07, 0.0, 0.04, 0.0, 0.5, 0.48, 0.15, 0.04, 0.01, 0.02, 0.02, 0.06, 0.06, 0.0, 0.66, 0.38, 0.5, 0.15, 0.02, 0.01, 0.02, 0.42],
    explanation:
      'In static Word2Vec/GloVe, "bank" collapses financial and riverside usages into one averaged vector. In BERT, self-attention attends to "deposited"/"paycheck" vs. "grassy"/"river", pushing the contextual vector into two distant neighborhoods.',
  },
  {
    id: 'apple',
    targetWord: 'apple',
    senseALabel: 'Botanical Fruit',
    sentenceA: 'She picked a crisp red apple from the orchard.',
    senseANeighbors: ['fruit', 'orchard', 'tree', 'water'],
    vectorA: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.38, 0.96, 0.04, 0.0, 0.0, 0.06, 0.08, 0.0, 0.0, 0.92, 0.06, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0],
    senseBLabel: 'Technology Company',
    sentenceB: 'Apple unveiled new silicon inside its flagship computer.',
    senseBNeighbors: ['computer', 'software', 'code', 'machine'],
    vectorB: [0.0, 0.0, 0.08, 0.0, 0.86, 0.5, 0.38, 0.0, 0.02, 0.0, 0.0, 0.38, 0.05, 0.1, 0.05, 0.05, 0.45, 0.6, 0.55, 0.1, 0.88, 0.0, 0.0, 0.8],
    staticAnchorVector: [0.0, 0.0, 0.04, 0.0, 0.42, 0.25, 0.2, 0.18, 0.48, 0.02, 0.0, 0.18, 0.05, 0.09, 0.02, 0.02, 0.68, 0.32, 0.35, 0.05, 0.44, 0.0, 0.0, 0.4],
    explanation:
      'Static embeddings average culinary and Silicon Valley corpora, placing "apple" awkwardly between "fruit" and "software". Contextual embeddings resolve the ambiguity using surrounding nouns like "orchard" vs. "silicon" and "computer".',
  },
  {
    id: 'bat',
    targetWord: 'bat',
    senseALabel: 'Nocturnal Mammal',
    sentenceA: 'A bat glided silently out of the dark cave.',
    senseANeighbors: ['bird', 'wing', 'cave', 'feather'],
    vectorA: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.05, 0.88, 0.9, 0.0, -0.3, -0.2, 0.0, 0.0, 0.88, 0.05, 0.18, 0.65, 0.0, 0.15, 0.0, 0.0],
    senseBLabel: 'Sports Equipment',
    sentenceB: 'He gripped the wooden bat and hit a baseball.',
    senseBNeighbors: ['baseball', 'machine', 'wing', 'man'],
    vectorB: [0.0, 0.05, 0.12, 0.0, 0.0, 0.0, 0.04, 0.02, 0.12, 0.0, 0.08, 0.68, 0.0, 0.15, 0.0, 0.05, 0.88, 0.12, 0.28, 0.62, 0.0, 0.0, 0.94, 0.12],
    staticAnchorVector: [0.0, 0.02, 0.06, 0.0, 0.0, 0.0, 0.02, 0.15, 0.08, 0.44, 0.48, 0.34, -0.15, -0.02, 0.0, 0.02, 0.88, 0.08, 0.23, 0.63, 0.0, 0.08, 0.46, 0.06],
    explanation:
      'The word "bat" shares zero semantic overlap between biology and baseball. Transformer layers re-weight its vector using "cave" and "glided" (sense A) versus "wooden" and "baseball" (sense B).',
  },
  {
    id: 'python',
    targetWord: 'python',
    senseALabel: 'Constrictor Reptile',
    sentenceA: 'The giant python coiled around a tree branch.',
    senseANeighbors: ['snake', 'tree', 'bird', 'cave'],
    vectorA: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.18, 0.88, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0, 0.9, 0.05, 0.35, 0.48, 0.0, 0.95, 0.0, 0.0],
    senseBLabel: 'Programming Language',
    sentenceB: 'We wrote Python code to train the neural transformer.',
    senseBNeighbors: ['code', 'software', 'neural', 'transformer'],
    vectorB: [0.0, 0.0, 0.02, 0.0, 0.95, 0.82, 0.05, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.25, 0.82, 0.28, 0.14, 0.48, 0.0, 0.0, 0.15],
    staticAnchorVector: [0.0, 0.0, 0.01, 0.0, 0.5, 0.42, 0.02, 0.14, 0.09, 0.42, 0.0, 0.08, 0.0, 0.06, 0.0, 0.0, 0.56, 0.44, 0.32, 0.3, 0.24, 0.45, 0.0, 0.08],
    explanation:
      'In modern developer corpora, static embeddings pull "python" heavily toward software, losing its zoological meaning. Contextual representations cleanly separate the reptile from the programming language.',
  },
  {
    id: 'light',
    targetWord: 'light',
    senseALabel: 'Electromagnetic Illumination',
    sentenceA: 'The warm lamp cast bright light across the room.',
    senseANeighbors: ['lamp', 'feather', 'water', 'stream'],
    vectorA: [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.05, 0.02, 0.0, 0.0, 0.32, 0.96, 0.0, 0.0, 0.0, 0.55, 0.45, 0.25, 0.18, 0.35, 0.0, 0.0, 0.0],
    senseBLabel: 'Low Physical Mass',
    sentenceB: 'A bird feather is remarkably light to carry.',
    senseBNeighbors: ['feather', 'bird', 'wing', 'lamp'],
    vectorB: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.48, 0.58, 0.05, 0.08, -0.92, 0.0, 0.0, 0.65, 0.32, 0.12, 0.22, 0.0, 0.0, 0.0, 0.0],
    staticAnchorVector: [0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.0, 0.08, 0.04, 0.24, 0.28, 0.18, 0.52, -0.46, 0.0, 0.0, 0.6, 0.38, 0.18, 0.2, 0.18, 0.0, 0.0, 0.0],
    explanation:
      '"Light" illustrates subtle adjective/noun polysemy: optical brightness ("lamp", "bright") versus low gravitational weight ("feather", "carry").',
  },
];

export interface VectorAnalogyPreset {
  id: string;
  label: string;
  a: string;
  b: string;
  c: string;
  expected: string;
  explanation: string;
}

export const ANALOGY_PRESETS: VectorAnalogyPreset[] = [
  {
    id: 'royalty',
    label: 'king − man + woman',
    a: 'king',
    b: 'man',
    c: 'woman',
    expected: 'queen',
    explanation: 'Subtracting "man" removes the male gender direction while preserving royalty; adding "woman" shifts to the female monarch.',
  },
  {
    id: 'heir',
    label: 'prince − man + woman',
    a: 'prince',
    b: 'man',
    c: 'woman',
    expected: 'princess',
    explanation: 'Youth + royalty remain invariant while the gender axis flips cleanly from male to female.',
  },
  {
    id: 'capitals',
    label: 'paris − france + japan',
    a: 'paris',
    b: 'france',
    c: 'japan',
    expected: 'tokyo',
    explanation: 'The difference vector (paris − france) encodes "capital city of"; adding "japan" lands directly on "tokyo".',
  },
];
