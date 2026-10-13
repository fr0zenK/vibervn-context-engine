use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;

mod batch_transport;
mod batching;
mod http_transport;
mod query_transport;
mod retry;

#[cfg(test)]
pub use retry::TRANSIENT_RETRY_LIMIT_FOR_TEST;
pub use retry::TransientEmbedExhausted;

const VOYAGE_ENDPOINT: &str = "https://api.voyageai.com/v1/embeddings";
const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
pub const MAX_BATCH_SIZE: usize = 128;
const MAX_BATCH_BYTES: usize = 1_500_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Voyage,
    OpenAI,
}

impl Provider {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Self::OpenAI,
            _ => Self::Voyage,
        }
    }

    /// Canonical stable name used in persisted embedding identities.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Voyage => "voyage",
            Self::OpenAI => "openai",
        }
    }

    fn default_endpoint(self) -> &'static str {
        match self {
            Self::Voyage => VOYAGE_ENDPOINT,
            Self::OpenAI => OPENAI_ENDPOINT,
        }
    }
}

pub fn embedding_url(provider: Provider, base: Option<&str>) -> String {
    let raw = base.unwrap_or_default().trim();
    if raw.is_empty() {
        return provider.default_endpoint().to_owned();
    }
    let trimmed = raw.trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/embeddings")
    }
}

/// Backward-compatible URL resolver using the Voyage default endpoint.
pub fn voyage_url(base: Option<&str>) -> String {
    embedding_url(Provider::Voyage, base)
}

#[derive(Clone)]
pub struct VoyageClient {
    pub(super) inner: Arc<VoyageInner>,
}

pub(super) struct VoyageInner {
    pub(super) http: Client,
    pub(super) query_http: Client,
    pub(super) provider: Provider,
    pub(super) model: String,
    pub(super) api_keys: Vec<String>,
    pub(super) endpoint: String,
    pub(super) dimensions: Option<u32>,
    pub(super) key_cursor: AtomicUsize,
}

impl VoyageClient {
    pub fn new(model: String, api_keys: Vec<String>, base_url: Option<&str>) -> Result<Self> {
        Self::new_for_provider(Provider::Voyage, model, api_keys, base_url, None)
    }

    pub fn new_for_provider(
        provider: Provider,
        model: String,
        api_keys: Vec<String>,
        base_url: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<Self> {
        if api_keys.is_empty() {
            bail!("embedding client requires at least one API key");
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("build reqwest client")?;
        let query_http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("build query reqwest client")?;
        Ok(Self {
            inner: Arc::new(VoyageInner {
                http,
                query_http,
                provider,
                model,
                api_keys,
                endpoint: embedding_url(provider, base_url),
                dimensions,
                key_cursor: AtomicUsize::new(0),
            }),
        })
    }

    /// [`new_for_provider`](Self::new_for_provider), but reusing one process-wide
    /// client while the embedding settings are unchanged. Query paths used to build
    /// a client per request: a new connection pool, so a new TCP+TLS handshake per
    /// query and one socket left in TIME_WAIT each — at agent query rates that
    /// exhausted Windows' 16K ephemeral ports and failed queries outright. Bounded
    /// to a few entries (normally one live config); a settings change adds a new
    /// entry and the oldest falls out.
    pub fn shared_for_provider(
        provider: Provider,
        model: String,
        api_keys: Vec<String>,
        base_url: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<Self> {
        type Key = (Provider, String, Vec<String>, Option<String>, Option<u32>);
        const MAX_SHARED: usize = 4;
        static SHARED: std::sync::Mutex<Vec<(Key, VoyageClient)>> =
            std::sync::Mutex::new(Vec::new());
        let key: Key = (
            provider,
            model,
            api_keys,
            base_url.map(str::to_owned),
            dimensions,
        );
        let mut shared = SHARED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, client)) = shared.iter().find(|(cached, _)| *cached == key) {
            return Ok(client.clone());
        }
        let client =
            Self::new_for_provider(key.0, key.1.clone(), key.2.clone(), key.3.as_deref(), key.4)?;
        if shared.len() >= MAX_SHARED {
            shared.remove(0);
        }
        shared.push((key, client.clone()));
        Ok(client)
    }

    pub fn provider(&self) -> Provider {
        self.inner.provider
    }

    pub fn model(&self) -> &str {
        &self.inner.model
    }

    pub fn dimensions(&self) -> Option<u32> {
        self.inner.dimensions
    }
}

#[cfg(test)]
mod tests;
