export interface PCAResult {
  projected: [number, number, number][];
  explainedVarianceRatio: [number, number, number];
  totalVarianceCaptured: number;
  dimensions: number;
  sampleCount: number;
}

export function dotProduct(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function vectorNorm(a: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  if (normA === 0 || normB === 0) return 0;
  return dotProduct(a, b) / (normA * normB);
}

export function euclideanDistance(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export function vectorAdd(a: number[], b: number[]): number[] {
  const len = Math.min(a.length, b.length);
  const out = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    out[i] = a[i] + b[i];
  }
  return out;
}

export function vectorSub(a: number[], b: number[]): number[] {
  const len = Math.min(a.length, b.length);
  const out = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    out[i] = a[i] - b[i];
  }
  return out;
}

export function normalizeVector(a: number[]): number[] {
  const n = vectorNorm(a);
  if (n === 0) return a.map(() => 0);
  return a.map((v) => v / n);
}

/**
 * Jacobi eigenvalue algorithm for a real symmetric N x N matrix.
 * Returns eigenvalues sorted descending along with corresponding eigenvectors (columns).
 */
function jacobiEigenSymmetric(matrix: number[][], maxIter = 120): {
  eigenvalues: number[];
  eigenvectors: number[][];
} {
  const n = matrix.length;
  const A = matrix.map((row) => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let iter = 0; iter < maxIter * n; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const absVal = Math.abs(A[i][j]);
        if (absVal > maxVal) {
          maxVal = absVal;
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < 1e-10) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];

    const tau = (aqq - app) / (2 * apq);
    const t =
      tau >= 0
        ? 1 / (tau + Math.sqrt(1 + tau * tau))
        : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let r = 0; r < n; r++) {
      if (r !== p && r !== q) {
        const arp = A[r][p];
        const arq = A[r][q];
        A[r][p] = c * arp - s * arq;
        A[p][r] = A[r][p];
        A[r][q] = s * arp + c * arq;
        A[q][r] = A[r][q];
      }
    }

    for (let r = 0; r < n; r++) {
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  const pairs = Array.from({ length: n }, (_, i) => ({
    value: Math.max(0, A[i][i]),
    vector: V.map((row) => row[i]),
  })).sort((a, b) => b.value - a.value);

  return {
    eigenvalues: pairs.map((p) => p.value),
    eigenvectors: pairs.map((p) => p.vector),
  };
}

/**
 * Computes exact 3D Principal Component Analysis (PCA) from arbitrary-dimensional vectors.
 * Automatically uses the dual Gram matrix (N x N) when D > N, yielding exact projections
 * and explained variance ratios in milliseconds even for 3072-D Gemini embeddings.
 */
export function computePCA(vectors: number[][]): PCAResult {
  const n = vectors.length;
  if (n === 0) {
    return {
      projected: [],
      explainedVarianceRatio: [0, 0, 0],
      totalVarianceCaptured: 0,
      dimensions: 0,
      sampleCount: 0,
    };
  }

  const d = vectors[0].length;
  if (n === 1) {
    return {
      projected: [[0, 0, 0]],
      explainedVarianceRatio: [1, 0, 0],
      totalVarianceCaptured: 1,
      dimensions: d,
      sampleCount: 1,
    };
  }

  // 1. Mean-center the matrix Z (N x D)
  const mean = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += vectors[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    mean[j] /= n;
  }

  const Z = vectors.map((vec) => vec.map((val, j) => val - mean[j]));

  const denom = Math.max(1, n - 1);

  if (d <= n) {
    // Standard D x D covariance matrix: C = (Z^T Z) / (n - 1)
    const cov: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    for (let r = 0; r < d; r++) {
      for (let c = r; c < d; c++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          sum += Z[i][r] * Z[i][c];
        }
        const val = sum / denom;
        cov[r][c] = val;
        cov[c][r] = val;
      }
    }

    const { eigenvalues, eigenvectors } = jacobiEigenSymmetric(cov);
    const totalVar = eigenvalues.reduce((acc, v) => acc + v, 0) || 1;
    const pc1 = eigenvectors[0] ?? new Array<number>(d).fill(0);
    const pc2 = eigenvectors[1] ?? new Array<number>(d).fill(0);
    const pc3 = eigenvectors[2] ?? new Array<number>(d).fill(0);

    const projected: [number, number, number][] = Z.map((row) => [
      dotProduct(row, pc1),
      dotProduct(row, pc2),
      dotProduct(row, pc3),
    ]);

    const r1 = (eigenvalues[0] ?? 0) / totalVar;
    const r2 = (eigenvalues[1] ?? 0) / totalVar;
    const r3 = (eigenvalues[2] ?? 0) / totalVar;

    return {
      projected,
      explainedVarianceRatio: [r1, r2, r3],
      totalVarianceCaptured: r1 + r2 + r3,
      dimensions: d,
      sampleCount: n,
    };
  } else {
    // Dual Gram matrix: G = (Z Z^T) / (n - 1), size N x N
    const gram: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < d; k++) {
          sum += Z[i][k] * Z[j][k];
        }
        const val = sum / denom;
        gram[i][j] = val;
        gram[j][i] = val;
      }
    }

    const { eigenvalues, eigenvectors } = jacobiEigenSymmetric(gram);
    const totalVar = eigenvalues.reduce((acc, v) => acc + v, 0) || 1;
    const u1 = eigenvectors[0] ?? new Array<number>(n).fill(0);
    const u2 = eigenvectors[1] ?? new Array<number>(n).fill(0);
    const u3 = eigenvectors[2] ?? new Array<number>(n).fill(0);
    const scale1 = Math.sqrt(Math.max(0, (eigenvalues[0] ?? 0) * denom));
    const scale2 = Math.sqrt(Math.max(0, (eigenvalues[1] ?? 0) * denom));
    const scale3 = Math.sqrt(Math.max(0, (eigenvalues[2] ?? 0) * denom));

    // Projection along principal component k is Z v_k = sqrt(lambda_k * (n - 1)) * u_k
    const projected: [number, number, number][] = Array.from({ length: n }, (_, i) => [
      u1[i] * scale1,
      u2[i] * scale2,
      u3[i] * scale3,
    ]);

    const r1 = (eigenvalues[0] ?? 0) / totalVar;
    const r2 = (eigenvalues[1] ?? 0) / totalVar;
    const r3 = (eigenvalues[2] ?? 0) / totalVar;

    return {
      projected,
      explainedVarianceRatio: [r1, r2, r3],
      totalVarianceCaptured: r1 + r2 + r3,
      dimensions: d,
      sampleCount: n,
    };
  }
}

/**
 * Generates deterministic positional + semantic token representation and
 * computes scaled dot-product self-attention weights softmax(QK^T / sqrt(d)).
 */
export function buildTokenFeatureVector(tokenId: number, position: number, dim = 16): number[] {
  const vec = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    // Deterministic hash from tokenId + sinusoidal positional encoding
    const angle = (tokenId * 0.0173 + 1.31) * (i + 1) + Math.sin(tokenId * 0.0041 * (i + 3));
    const tokenFeat = Math.sin(angle) * 0.75 + Math.cos(tokenId * 0.0091 * (i + 2)) * 0.45;
    const posAngle = position / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
    const posFeat = i % 2 === 0 ? Math.sin(posAngle) : Math.cos(posAngle);
    vec[i] = Number((tokenFeat + posFeat * 0.5).toFixed(4));
  }
  return vec;
}

export function computeSelfAttentionMatrix(
  tokenIds: number[],
  causal = true,
): { vectors: number[][]; weights: number[][] } {
  const n = tokenIds.length;
  const dim = 16;
  const vectors = tokenIds.map((id, idx) => buildTokenFeatureVector(id, idx, dim));
  const scale = Math.sqrt(dim);

  const weights: number[][] = Array.from({ length: n }, (_, i) => {
    const logits = new Array<number>(n).fill(-Infinity);
    let maxLogit = -Infinity;
    for (let j = 0; j < n; j++) {
      if (causal && j > i) continue;
      // Add subtle recency / self-affinity bias + dot-product similarity
      const raw = dotProduct(vectors[i], vectors[j]) / scale + (i === j ? 0.6 : 0);
      logits[j] = raw;
      if (raw > maxLogit) maxLogit = raw;
    }

    const exps = logits.map((l) => (l === -Infinity ? 0 : Math.exp(l - maxLogit)));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sumExp);
  });

  return { vectors, weights };
}
