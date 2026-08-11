require('dotenv').config();

// Sentry must be required and initialized before express (and anything express pulls in,
// like http) so its automatic instrumentation can actually wrap them. Fully optional and
// silent when SENTRY_DSN isn't set — nothing about this can block startup.
const Sentry = require('@sentry/node');
const SENTRY_DSN = String(process.env.SENTRY_DSN || '').trim();
if (SENTRY_DSN) {
  Sentry.init({ dsn: SENTRY_DSN, environment: process.env.NODE_ENV || 'development' });
} else {
  console.warn('Warning: SENTRY_DSN missing. Unhandled errors will only be visible in local/platform logs.');
}

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { transcribeWithHF, analyzeWithHF, analyzeHeuristically } = require('./hf.service');
const { analyzeWithGroq, generateDailyPrompt } = require('./groq.service');

const upload = multer();
const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on('finish', () => {
    const durationMs = Number(process.hrtime.bigint() - start) / 1e6;
    const line = `${new Date().toISOString()} ${req.method} ${req.originalUrl} -> ${res.statusCode} (${durationMs.toFixed(1)}ms)`;
    if (res.statusCode >= 500) console.error(line);
    else console.log(line);
  });
  next();
});

const FRONTEND_DIR = path.resolve(__dirname, '..', '..', 'frontend');
app.use(express.static(FRONTEND_DIR));

const HF_API_KEY = String(process.env.HF_API_KEY || '').trim();
const HF_WHISPER_MODEL = process.env.HF_WHISPER_MODEL || process.env.HF_STT_MODEL || 'openai/whisper-large-v3-turbo';
const HF_WHISPER_FALLBACK_MODEL = process.env.HF_STT_FALLBACK_MODEL || 'openai/whisper-large-v3';
const HF_EMOTION_MODEL = process.env.HF_EMOTION_MODEL || process.env.HF_ANALYSIS_MODEL || 'j-hartmann/emotion-english-distilroberta-base';
const HF_SENTIMENT_MODEL = process.env.HF_SENTIMENT_MODEL || 'cardiffnlp/twitter-roberta-base-sentiment-latest';
const HF_EMOTION_FALLBACK_MODEL = process.env.HF_EMOTION_FALLBACK_MODEL || '';
const HF_SENTIMENT_FALLBACK_MODEL = process.env.HF_SENTIMENT_FALLBACK_MODEL || 'distilbert/distilbert-base-uncased-finetuned-sst-2-english';
const GROQ_API_KEY = String(process.env.GROQ_API_KEY || '').trim();
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const PORT = process.env.PORT || 3000;
const DEMO_TRANSCRIPTION = 'Demo transcription: I am sharing a short sample speech for local testing without an API key.';

const DAILY_PROMPT_FALLBACK_POOL = [
  "What's something that took up space in your mind today?",
  'Tell me about a moment today that stuck with you, big or small.',
  "How did your energy shift over the course of today?",
  'Describe a conversation you had recently that you keep thinking about.',
  "What's something you're looking forward to, and why?",
  'Walk me through how you started your day this morning.',
  "What's been on your mind about work or your daily routine lately?",
  'Tell me about a small win or frustration from the past day or two.',
  'How has your sleep or energy felt this week?',
  "What's something you did today that felt effortless, or something that felt hard?",
  'If you had to describe your mood right now in a short story, what would it be?',
  "What's something you've been putting off, and what's stopping you?"
];

function hasUsableApiKey(key) {
  if (!key) return false;
  // Treat common placeholder values as "no key"
  if (/^hf_[x*]+$/i.test(key)) return false;
  if (/^(your_|replace_|example|demo)/i.test(key)) return false;
  return key.startsWith('hf_') && key.length > 10;
}

const HAS_HF_API_KEY = hasUsableApiKey(HF_API_KEY);
const HAS_GROQ_API_KEY = GROQ_API_KEY.length > 10;

function demoTranscribeResponse(reason) {
  return {
    transcription: DEMO_TRANSCRIPTION,
    mode: 'heuristic_no_api_key',
    model: null,
    continueFlow: true,
    demo: true,
    message: reason || 'Using demo transcription because HF_API_KEY is not configured.'
  };
}

function demoAnalyzeResponse(text, reason) {
  const sourceText = text || DEMO_TRANSCRIPTION;
  const out = analyzeHeuristically(sourceText);
  out.demo = true;
  out.message = reason || (
    !text
      ? 'Using demo analysis text because HF_API_KEY is not configured and no text was provided.'
      : 'Using local heuristic analysis because HF_API_KEY is not configured.'
  );
  return out;
}

if (!HAS_HF_API_KEY) {
  console.warn('Warning: HF_API_KEY missing/invalid. Demo mode is active for /transcribe and /analyze.');
}
if (!HAS_GROQ_API_KEY) {
  console.warn('Warning: GROQ_API_KEY missing/invalid. /analyze will run without the Groq semantic pass.');
}

function makeSilentWavBuffer(durationMs = 1000, sampleRate = 16000) {
  const samples = Math.max(1, Math.floor((durationMs / 1000) * sampleRate));
  const dataBytes = samples * 2; // 16-bit mono PCM
  const buffer = Buffer.alloc(44 + dataBytes);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16); // PCM chunk size
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buffer.writeUInt16LE(2, 32); // block align
  buffer.writeUInt16LE(16, 34); // bits per sample
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataBytes, 40);
  // remaining bytes are already zeros (silence)
  return buffer;
}

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    models: {
      whisper: HF_WHISPER_MODEL,
      whisperFallback: HF_WHISPER_FALLBACK_MODEL,
      emotion: HF_EMOTION_MODEL,
      emotionFallback: HF_EMOTION_FALLBACK_MODEL || null,
      sentiment: HF_SENTIMENT_MODEL,
      sentimentFallback: HF_SENTIMENT_FALLBACK_MODEL,
      groqSemantic: HAS_GROQ_API_KEY ? GROQ_MODEL : null
    }
  });
});

// Whisper debug probe: verifies HF key/model reachability with a generated silent WAV.
app.get('/debug/whisper', async (_req, res) => {
  if (!HAS_HF_API_KEY) {
    return res.status(200).json({
      ok: true,
      demo: true,
      message: 'Skipping Whisper probe because HF_API_KEY is not configured. Demo mode is active.'
    });
  }

  const wav = makeSilentWavBuffer(1000, 16000);
  const result = {
    ok: false,
    tested: {
      primary: HF_WHISPER_MODEL,
      fallback: HF_WHISPER_FALLBACK_MODEL
    },
    primary: null,
    fallback: null
  };

  try {
    const out = await transcribeWithHF({
      apiKey: HF_API_KEY,
      model: HF_WHISPER_MODEL,
      fileBuffer: wav,
      contentType: 'audio/wav'
    });
    result.primary = {
      ok: true,
      transcription: out.transcription || null
    };
    result.ok = true;
    return res.json(result);
  } catch (err) {
    result.primary = {
      ok: false,
      details: err.response
        ? { status: err.response.status, data: err.response.data }
        : { message: err.message }
    };
  }

  if (!HF_WHISPER_FALLBACK_MODEL || HF_WHISPER_FALLBACK_MODEL === HF_WHISPER_MODEL) {
    return res.status(502).json(result);
  }

  try {
    const out = await transcribeWithHF({
      apiKey: HF_API_KEY,
      model: HF_WHISPER_FALLBACK_MODEL,
      fileBuffer: wav,
      contentType: 'audio/wav'
    });
    result.fallback = {
      ok: true,
      transcription: out.transcription || null
    };
    result.ok = true;
    return res.json(result);
  } catch (err) {
    result.fallback = {
      ok: false,
      details: err.response
        ? { status: err.response.status, data: err.response.data }
        : { message: err.message }
    };
    return res.status(502).json(result);
  }
});

app.post('/transcribe', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded (form field `audio`)' });
  }
  if (!HAS_HF_API_KEY) {
    return res.status(200).json(
      demoTranscribeResponse('Using demo transcription because HF_API_KEY is missing or invalid.')
    );
  }

  try {
    let out = null;
    let usedModel = HF_WHISPER_MODEL;
    let primaryErr = null;

    try {
      out = await transcribeWithHF({
        apiKey: HF_API_KEY,
        model: HF_WHISPER_MODEL,
        fileBuffer: req.file.buffer,
        contentType: req.file.mimetype
      });
    } catch (err) {
      primaryErr = err;
      if (HF_WHISPER_FALLBACK_MODEL && HF_WHISPER_FALLBACK_MODEL !== HF_WHISPER_MODEL) {
        out = await transcribeWithHF({
          apiKey: HF_API_KEY,
          model: HF_WHISPER_FALLBACK_MODEL,
          fileBuffer: req.file.buffer,
          contentType: req.file.mimetype
        });
        usedModel = HF_WHISPER_FALLBACK_MODEL;
      } else {
        throw err;
      }
    }

    if (!out.transcription) {
      return res.status(502).json({
        error: 'HF transcription response did not include text',
        raw: out.raw,
        model: usedModel
      });
    }

    return res.json({
      raw: out.raw,
      transcription: out.transcription,
      model: usedModel,
      retried: !!primaryErr
    });
  } catch (err) {
    console.warn(`[transcribe] Whisper API failed, falling back to demo transcription: ${err.message}`);
    return res.status(200).json(
      demoTranscribeResponse('Whisper API failed, so demo transcription was returned.')
    );
  }
});

function getSemanticAnalysis(text) {
  if (!HAS_GROQ_API_KEY) {
    return Promise.resolve({ error: 'GROQ_API_KEY is missing or invalid' });
  }
  if (!text) {
    return Promise.resolve(null);
  }
  return analyzeWithGroq({ apiKey: GROQ_API_KEY, text, model: GROQ_MODEL }).catch((err) => {
    console.warn(`[analyze] Groq semantic analysis failed: ${err.message}`);
    return { error: err.message || 'Groq analysis failed' };
  });
}

app.post('/analyze', async (req, res) => {
  const { text } = req.body || {};
  const semanticPromise = getSemanticAnalysis(text);

  if (!HAS_HF_API_KEY) {
    const out = demoAnalyzeResponse(text, 'Using local heuristic analysis because HF_API_KEY is missing or invalid.');
    out.semantic = await semanticPromise;
    return res.json(out);
  }

  if (!text) {
    return res.status(400).json({ error: 'Request body must include `text`' });
  }

  try {
    const [out, semantic] = await Promise.all([
      analyzeWithHF({
        apiKey: HF_API_KEY,
        text,
        emotionModel: HF_EMOTION_MODEL,
        sentimentModel: HF_SENTIMENT_MODEL,
        emotionFallbackModel: HF_EMOTION_FALLBACK_MODEL,
        sentimentFallbackModel: HF_SENTIMENT_FALLBACK_MODEL
      }),
      semanticPromise
    ]);
    out.semantic = semantic;

    return res.json(out);
  } catch (err) {
    console.warn(`[analyze] HF analysis failed, falling back to local heuristic: ${err.message}`);
    const out = demoAnalyzeResponse(text, 'Analysis API failed, so local heuristic demo analysis was returned.');
    out.semantic = await semanticPromise;
    return res.status(200).json(out);
  }
});

function fallbackPrompt(reason) {
  const prompt = DAILY_PROMPT_FALLBACK_POOL[Math.floor(Math.random() * DAILY_PROMPT_FALLBACK_POOL.length)];
  return { prompt, model: null, personalized: false, demo: true, message: reason };
}

app.post('/api/daily-prompt', async (req, res) => {
  const { recentTranscripts } = req.body || {};

  if (!HAS_GROQ_API_KEY) {
    return res.json(fallbackPrompt('Using a curated prompt because GROQ_API_KEY is missing or invalid.'));
  }

  try {
    const out = await generateDailyPrompt({ apiKey: GROQ_API_KEY, model: GROQ_MODEL, recentTranscripts });
    return res.json(out);
  } catch (err) {
    console.warn(`[daily-prompt] Groq prompt generation failed, falling back to curated pool: ${err.message}`);
    return res.status(200).json(fallbackPrompt('Prompt generation failed, so a curated prompt was returned.'));
  }
});

// Must be registered after all routes, before any other error-handling middleware, so
// unhandled exceptions (anything that isn't already caught and logged locally) are actually
// reported instead of only ever showing up in whichever log nobody happened to be watching.
if (SENTRY_DSN) {
  Sentry.setupExpressErrorHandler(app);
}

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
