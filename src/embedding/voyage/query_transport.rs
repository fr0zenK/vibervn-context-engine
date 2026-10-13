use std::sync::atomic::Ordering;

use anyhow::Result;
use tracing::warn;

use super::retry::{EmbedError, backoff_with_jitter};
use crate::embedding::InputType;

use super::VoyageClient;

impl VoyageClient {
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let key_count = self.inner.api_keys.len();
        let start = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % key_count;

        let mut last_transient = None;
        for offset in 0..key_count {
            let key_index = (start + offset) % key_count;
            match self.try_query_key_retrying(key_index, &texts).await {
                Ok(mut embeddings) => return pop_query_embedding(&mut embeddings),
                Err(EmbedError::RateLimited) => {
                    warn!(key_index, "VoyageAI 429 on query embed; trying next key");
                }
                // Still failing after the retries: another key may route elsewhere.
                Err(EmbedError::Transient(error)) => last_transient = Some(error),
                Err(EmbedError::Other(error)) => return Err(error),
            }
        }
        if let Some(error) = last_transient {
            return Err(error.context("VoyageAI transient error on query embed"));
        }

        let cursor = self.inner.key_cursor.load(Ordering::Relaxed);
        let delay = backoff_with_jitter(2, cursor);
        warn!(
            delay_ms = delay.as_millis() as u64,
            "all VoyageAI keys rate-limited on query embed; backing off"
        );
        tokio::time::sleep(delay).await;

        for offset in 0..key_count {
            let key_index = (start + offset) % key_count;
            match self
                .try_embed_query_with_key(&self.inner.api_keys[key_index], &texts, InputType::Query)
                .await
            {
                Ok(mut embeddings) => return pop_query_embedding(&mut embeddings),
                Err(EmbedError::RateLimited) => continue,
                Err(EmbedError::Transient(error)) => {
                    return Err(error.context("VoyageAI transient error on query embed"));
                }
                Err(EmbedError::Other(error)) => return Err(error),
            }
        }
        anyhow::bail!("VoyageAI query embed still rate-limited after backoff")
    }

    /// One key, retrying transient failures (connection reset, 5xx, timeout) a
    /// couple of times on a short schedule. A single network blip used to fail
    /// the whole MCP call; the query path is latency-sensitive, so the delays stay
    /// sub-second (document embedding has its own, longer retry policy).
    async fn try_query_key_retrying(
        &self,
        key_index: usize,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let key = &self.inner.api_keys[key_index];
        let mut attempt = 0;
        loop {
            match self
                .try_embed_query_with_key(key, texts, InputType::Query)
                .await
            {
                Err(EmbedError::Transient(error)) if attempt < QUERY_TRANSIENT_RETRY_MS.len() => {
                    let delay = QUERY_TRANSIENT_RETRY_MS[attempt];
                    warn!(key_index, attempt, delay_ms = delay, error = %format!("{error:#}"), "VoyageAI transient error on query embed; retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }
}

/// Delays before each retry of a transient query-embed failure.
const QUERY_TRANSIENT_RETRY_MS: [u64; 2] = [200, 500];

fn pop_query_embedding(embeddings: &mut Vec<Vec<f32>>) -> Result<Vec<f32>> {
    embeddings
        .pop()
        .ok_or_else(|| anyhow::anyhow!("VoyageAI returned empty embeddings"))
}
