// Shared retry/backoff helper for outbound calls to Hugging Face and Groq. A single
// transient network blip (dropped connection, a momentary 502/503 from the upstream)
// otherwise fails the whole request immediately, with no resilience at all.

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isRetryableError(err) {
  const status = err?.response?.status;
  if ([408, 409, 425, 429, 500, 502, 503, 504].includes(status)) return true;
  const msg = String(err?.message || '').toLowerCase();
  return (
    msg.includes('timeout') ||
    msg.includes('econnreset') ||
    msg.includes('socket hang up') ||
    msg.includes('network')
  );
}

// Only retries transient failures (network errors, request timeouts, 5xx/408/409/425/429);
// a genuine 4xx like a bad request or invalid model name fails immediately since retrying
// it would just waste time reproducing the same failure.
async function postWithRetry(axios, url, data, config = {}, { retries = 2, backoffMs = 350 } = {}) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await axios.post(url, data, config);
    } catch (err) {
      lastErr = err;
      if (attempt === retries || !isRetryableError(err)) throw err;
      await delay(backoffMs * (attempt + 1));
    }
  }
  throw lastErr;
}

module.exports = { isRetryableError, postWithRetry, delay };
