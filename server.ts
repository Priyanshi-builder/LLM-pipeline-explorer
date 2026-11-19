import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '1mb' }));

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

app.get('/api/health', (_req, res) => {
  const hasKey = Boolean(
    process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY',
  );
  res.json({
    ok: true,
    liveEmbeddingsConfigured: hasKey,
  });
});

interface EmbedRequestBody {
  texts?: unknown;
}

app.post('/api/embed', async (req, res) => {
  try {
    const body = req.body as EmbedRequestBody;
    if (!Array.isArray(body?.texts) || body.texts.length === 0) {
      res.status(400).json({
        error: 'Request body must include a non-empty array of strings in "texts".',
      });
      return;
    }

    const texts = body.texts
      .filter((t): t is string => typeof t === 'string')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)
      .slice(0, 16);

    if (texts.length === 0) {
      res.status(400).json({
        error: 'Provide at least one non-empty text string to embed.',
      });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'GEMINI_API_KEY is not configured in the server environment. Set it in Settings > Secrets to compute live Gemini neural embeddings.',
      });
      return;
    }

    const result = await ai.models.embedContent({
      model: 'gemini-embedding-2',
      contents: texts,
    });

    const vectors = (result.embeddings ?? []).map((entry) => entry.values ?? []);
    if (vectors.length !== texts.length || vectors.some((v) => v.length === 0)) {
      res.status(502).json({
        error: 'Embedding service returned an unexpected vector payload.',
      });
      return;
    }

    res.json({
      model: 'gemini-embedding-2',
      dimensions: vectors[0].length,
      texts,
      embeddings: vectors,
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Failed to compute embeddings via Gemini API.';
    res.status(500).json({ error: message });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`LLM Pipeline Explorer server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
