const axios = require('axios');
const { postWithRetry } = require('./http_retry');

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

const SYSTEM_PROMPT = [
  'You are analyzing a short daily voice-journal transcript for a personal cognitive-wellness tracking app.',
  'Read the transcript literally AND figuratively: identify idioms, jokes, sarcasm, and other non-literal phrasing that a simple keyword-based sentiment classifier would misread.',
  'Respond with ONLY a JSON object (no markdown, no commentary) matching exactly this shape:',
  '{',
  '  "sentiment_valence": <integer 0-100, 0 = very negative, 100 = very positive>,',
  '  "emotional_tone": <one of "joyful","calm","neutral","anxious","sad","angry","frustrated","mixed">,',
  '  "figurative_language_detected": <boolean>,',
  '  "figurative_language_examples": <array of short quoted strings, empty array if none>,',
  '  "sarcasm_likelihood": <integer 0-100>,',
  '  "note": <one short sentence summarizing the emotional/semantic read>',
  '}'
].join('\n');

function coerceResult(parsed) {
  return {
    sentiment_valence: Number.isFinite(parsed.sentiment_valence)
      ? Math.max(0, Math.min(100, Math.round(parsed.sentiment_valence)))
      : null,
    emotional_tone: typeof parsed.emotional_tone === 'string' ? parsed.emotional_tone : 'unknown',
    figurative_language_detected: !!parsed.figurative_language_detected,
    figurative_language_examples: Array.isArray(parsed.figurative_language_examples)
      ? parsed.figurative_language_examples.slice(0, 10)
      : [],
    sarcasm_likelihood: Number.isFinite(parsed.sarcasm_likelihood)
      ? Math.max(0, Math.min(100, Math.round(parsed.sarcasm_likelihood)))
      : null,
    note: typeof parsed.note === 'string' ? parsed.note : ''
  };
}

async function analyzeWithGroq({ apiKey, text, model }) {
  const resp = await postWithRetry(
    axios,
    GROQ_API_URL,
    {
      model,
      temperature: 0,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: `Transcript: """${text}"""` }
      ]
    },
    {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      timeout: 60000
    },
    { retries: 2 }
  );

  const content = resp.data?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error('Groq response did not include content');
  }

  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch (err) {
    throw new Error('Groq response was not valid JSON');
  }

  return {
    model,
    ...coerceResult(parsed)
  };
}

const PROMPT_SYSTEM_PROMPT = [
  'You generate a single short daily voice-journal prompt for a personal cognitive-wellness tracking app.',
  'The user will speak their answer out loud for roughly 30-60 seconds, so the prompt should invite natural, spontaneous speech - never a yes/no question, never something answerable in one word.',
  'Respond with ONLY a JSON object (no markdown, no commentary) matching exactly this shape:',
  '{ "prompt": <string, one short sentence phrased as a direct question or invitation to speak> }'
].join('\n');

function buildPromptUserMessage(recentTranscripts) {
  const entries = Array.isArray(recentTranscripts)
    ? recentTranscripts.filter((t) => typeof t === 'string' && t.trim()).slice(0, 5)
    : [];

  if (entries.length === 0) {
    return 'This user has no prior entries yet. Generate a warm, general opening prompt.';
  }

  const formatted = entries
    .map((t, i) => `Entry ${i + 1} (most recent first): """${t.slice(0, 500)}"""`)
    .join('\n');

  return [
    "Here are the user's most recent voice-journal entries:",
    formatted,
    '',
    "Generate a NEW prompt informed by what they've shared - you may gently follow up on a topic, person, or theme they mentioned - but do not repeat a previous prompt verbatim and do not fixate on the same topic every time."
  ].join('\n');
}

async function generateDailyPrompt({ apiKey, model, recentTranscripts }) {
  const resp = await postWithRetry(
    axios,
    GROQ_API_URL,
    {
      model,
      temperature: 0.9,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: PROMPT_SYSTEM_PROMPT },
        { role: 'user', content: buildPromptUserMessage(recentTranscripts) }
      ]
    },
    {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      timeout: 30000
    },
    { retries: 2 }
  );

  const content = resp.data?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error('Groq response did not include content');
  }

  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch (err) {
    throw new Error('Groq response was not valid JSON');
  }

  if (typeof parsed.prompt !== 'string' || !parsed.prompt.trim()) {
    throw new Error('Groq response did not include a usable prompt');
  }

  return {
    prompt: parsed.prompt.trim(),
    model,
    personalized: Array.isArray(recentTranscripts) && recentTranscripts.length > 0
  };
}

module.exports = { analyzeWithGroq, generateDailyPrompt };
