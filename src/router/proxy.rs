//! Router → worker REVERSE PROXY with streaming passthrough (SSE-safe).
//!
//! Forwards an incoming request to a repo's worker on `127.0.0.1:<port>` and
//! streams the response back BODY-AND-ALL without buffering — so Server-Sent
//! Events (`/api/repos/:id/index-events`) and any chunked response flow through
//! live. Uses reqwest (already a dep, `stream` feature) as the upstream client
//! and converts its streaming body into an axum body via `Body::from_stream`.
//!
//! Single-flight + spawn-on-miss live in [`acquire_and_proxy`]: it asks the
//! [`Registry`] for a ready worker; on a miss it either performs the spawn (if
//! elected) or awaits the in-flight spawn, then proxies. A spawn that exceeds
//! its budget degrades to a `503` "warming, retry" — NEVER a connection-refused
//! or a hang (window-A is owned here, see `spawn::SPAWN_READY_TIMEOUT`).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use tracing::warn;

use super::jobobject::JobObject;
use super::registry::{Acquire, ReadyWorker, Registry};
use super::spawn::spawn_worker;

/// Shared proxy context: the upstream HTTP client + the worker registry + spawn
/// inputs. Cloned into the router's per-repo handlers.
#[derive(Clone)]
pub struct ProxyCtx {
    pub client: reqwest::Client,
    pub registry: Registry,
    pub job: Arc<JobObject>,
    /// Path to the current executable (workers run the same binary).
    pub exe: std::path::PathBuf,
    /// Passthrough args so workers resolve the same dirs/bind as the router
    /// (e.g. `--data-dir X --bind 127.0.0.1`).
    pub worker_args: Vec<String>,
}

impl ProxyCtx {
    pub fn new(job: Arc<JobObject>, exe: std::path::PathBuf, worker_args: Vec<String>) -> Self {
        Self {
            // No global timeout on the client: long index/query requests stream
            // for a while. Per-attempt budgets are enforced upstream (MCP wait /
            // spawn timeout), not here.
            client: reqwest::Client::builder()
                .pool_idle_timeout(Duration::from_secs(90))
                .build()
                .expect("reqwest client"),
            registry: Registry::new(),
            job,
            exe,
            worker_args,
        }
    }
}

/// Acquire (or spawn) the worker for `repo`, then proxy `req` to it. Degrades to
/// a 503 on spawn timeout/failure. The degrade signal distinguishes two cases so
/// the UI can react correctly (see `warming_response` / `error_response`):
/// - the spawn is still in flight / coalescing → `warming` (`retry:true`),
/// - the spawn genuinely failed (timeout, process error) → `error`
///   (`retry:false`).
///
/// `obtain_worker` returns `Err(msg)` for BOTH today; we map a spawn that never
/// converged to `error` and everything else to `warming`. The distinction is
/// carried in the message prefix `spawn failed:` set by `obtain_worker`.
pub async fn acquire_and_proxy(ctx: &ProxyCtx, repo: &str, req: Request) -> Response {
    let buffered = match buffer_request(req).await {
        Ok(b) => b,
        Err(msg) => return bad_body(msg),
    };
    let mut attempt = 0;
    loop {
        // Single-flight acquire loop: a SpawnElected caller spawns; AwaitSpawn
        // waits then re-acquires; Ready proxies. Bounded retries so a thrashing
        // spawn can't loop forever.
        let worker = match obtain_worker(ctx, repo).await {
            Ok(w) => w,
            Err(msg) => {
                // A genuine spawn failure (process couldn't start / readiness
                // timeout) is an ERROR the user must see + manually retry; a spawn
                // that simply hasn't converged yet is WARMING (auto-retry). Map by
                // the `spawn failed:` prefix obtain_worker sets on a hard failure.
                if msg.starts_with("spawn failed:") {
                    warn!(repo = %repo, error = %msg, "worker spawn failed; surfacing error");
                    return error_response(&msg);
                }
                warn!(repo = %repo, error = %msg, "worker warming; signalling retry");
                return warming_response(&msg);
            }
        };
        match send_buffered(ctx, &worker, &buffered).await {
            Ok(resp) => return resp,
            Err(e) if attempt < WORKER_LOST_RETRIES => {
                warn!(repo = %repo, port = worker.port, error = %e, "worker connection lost before a response; re-acquiring and retrying");
                attempt += 1;
                tokio::time::sleep(WORKER_LOST_RETRY_DELAY).await;
            }
            Err(e) => {
                warn!(repo = %repo, port = worker.port, error = %e, "proxy upstream connect failed; degrading");
                return warming_response("worker connection lost; retry");
            }
        }
    }
}

/// Retries when the worker vanished before answering — it crashed, or the idle
/// reaper took it just as a request arrived. Only a request that got NO response
/// is retried, after re-acquiring (which respawns a dead worker), so the
/// caller sees an answer instead of a "connection lost; retry" 503.
const WORKER_LOST_RETRIES: u32 = 1;
const WORKER_LOST_RETRY_DELAY: Duration = Duration::from_millis(300);

/// SPAWN-BLOCKED proxy: proxy `req` to the repo's worker ONLY IF one is already
/// live. Returns `None` (without spawning) when no worker is resident, so the
/// caller can serve a cold/native response instead. This is the mechanism that
/// keeps "view" routes (status/graph/files/index-events/cancel/chat-DELETE) from
/// ever starting a worker — they call THIS, never `acquire_and_proxy` (which is
/// the spawn path). A worker present but mid-request still proxies (fresh data);
/// absent → cold.
pub async fn proxy_if_live(ctx: &ProxyCtx, repo: &str, req: Request) -> Option<Response> {
    // NON-MUTATING liveness check. `peek_ready` returns a live worker or `None`,
    // reaping a dead `Ready` entry to absent as a side effect — but NEVER
    // electing a spawn. (The old `ready_repos()` + `acquire` pair was the
    // "stuck Indexing… after idle" bug: `acquire` on a dead `Ready` transitions
    // the entry to `Spawning` and returns `SpawnElected`, which this spawn-free
    // route discarded — orphaning the entry in `Spawning` forever.)
    match ctx.registry.peek_ready(repo).await {
        Some(w) => Some(proxy_to(ctx, &w, req).await),
        // No live worker (absent, mid-spawn, or just-reaped dead) → serve cold.
        None => None,
    }
}

/// Acquire (or spawn) the worker for `repo`, POST `body` as JSON to `path` on
/// it, and return the worker's `{"result": "..."}` string unwrapped. Used by the
/// global `/mcp` proxy handler (which can't stream — it needs the funnel's text
/// back to wrap in a `CallToolResult`) and by any router-native handler that
/// must call a single worker JSON endpoint and read the body.
///
/// Unlike `acquire_and_proxy` (which streams an arbitrary HTTP response), this
/// buffers the worker's JSON response and extracts the `result` field, mirroring
/// what the monolith's in-process MCP tool returned. On spawn/connection failure
/// it returns an `Err(String)` so the caller can surface a tool-level error
/// instead of a transport panic.
pub async fn forward_json_to_worker(
    ctx: &ProxyCtx,
    repo: &str,
    path: &str,
    body: serde_json::Value,
) -> Result<String, String> {
    let mut attempt = 0;
    loop {
        let worker = obtain_worker(ctx, repo).await?;
        let url = format!("http://127.0.0.1:{}{}", worker.port, path);
        let resp = match ctx.client.post(&url).json(&body).send().await {
            Ok(resp) => resp,
            // No response at all: see WORKER_LOST_RETRIES.
            Err(e) if attempt < WORKER_LOST_RETRIES => {
                warn!(repo = %repo, port = worker.port, error = %e, "worker connection lost before a response; re-acquiring and retrying");
                attempt += 1;
                tokio::time::sleep(WORKER_LOST_RETRY_DELAY).await;
                continue;
            }
            Err(e) => return Err(format!("worker request failed: {e}")),
        };
        let raw = resp
            .text()
            .await
            .map_err(|e| format!("worker response read failed: {e}"))?;
        return Ok(super::mcp_proxy::unwrap_mcp_tool_result(&raw));
    }
}

/// GET `path` on an ALREADY-LIVE worker for `repo` and return the parsed JSON
/// body verbatim (NO MCP `.result` unwrap, NO spawn). Returns `None` if no
/// worker is live or the request/parse fails — the caller then serves a cold
/// view. This is the read-only sibling of `forward_json_to_worker`: that one
/// POSTs to action/MCP endpoints and unwraps `{result}`, which is WRONG for a
/// plain GET status endpoint (a POST would 405, and `/status` returns the status
/// object directly, not an MCP `{result}` envelope). Used by `get_index_status`
/// to read a live worker's authoritative status without spawning.
pub async fn get_json_if_live(ctx: &ProxyCtx, repo: &str, path: &str) -> Option<serde_json::Value> {
    // Spawn-free liveness via `peek_ready` (reaps a dead `Ready` to absent, never
    // elects a spawn). Replaces the old `ready_repos()` + `acquire` pair that
    // orphaned dead entries into `Spawning` on a status poll.
    let worker = ctx.registry.peek_ready(repo).await?;
    let url = format!("http://127.0.0.1:{}{}", worker.port, path);
    let resp = ctx.client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<serde_json::Value>().await.ok()
}

/// Resolve a ready worker for `repo`, spawning under single-flight if needed.
pub(crate) async fn obtain_worker(ctx: &ProxyCtx, repo: &str) -> Result<Arc<ReadyWorker>, String> {
    // At most a couple of acquire rounds: Ready, or elect→spawn, or await→re-acquire.
    for _ in 0..3 {
        match ctx.registry.acquire(repo).await {
            Acquire::Ready(w) => return Ok(w),
            Acquire::SpawnElected { ready } => {
                // We own the spawn for this repo. On success publish + wake
                // awaiters; on failure abandon so another request can retry.
                match spawn_worker(&ctx.exe, repo, &ctx.worker_args, &ctx.job).await {
                    Ok(worker) => {
                        // publish_ready returns false if this spawn was SUPERSEDED
                        // — the orphan watchdog reaped our `Spawning` entry (we
                        // were slow) and a waiter may have re-elected a fresh
                        // spawn. Our worker is now reachable by nobody yet holds
                        // the RocksDB LOCK, so we MUST kill it (else it would
                        // deadlock the re-elected spawn's open_db), then loop to
                        // re-acquire the new state (Ready, or a fresh Spawning).
                        if ctx
                            .registry
                            .publish_ready(repo, worker.clone(), &ready)
                            .await
                        {
                            return Ok(worker);
                        }
                        warn!(repo = %repo, "spawn superseded (entry reaped/re-elected); killing orphan worker and re-acquiring");
                        worker.kill_and_wait().await;
                        continue;
                    }
                    Err(e) => {
                        ctx.registry.abandon_spawn(repo, &ready).await;
                        return Err(format!("spawn failed: {e:#}"));
                    }
                }
            }
            Acquire::AwaitSpawn { ready } => {
                // Another request is spawning. Wait for it (bounded), then loop
                // to re-acquire (it should now be Ready, or absent on failure).
                let _ = tokio::time::timeout(
                    super::spawn::SPAWN_READY_TIMEOUT + Duration::from_secs(2),
                    ready.notified(),
                )
                .await;
                continue;
            }
        }
    }
    Err("worker spawn did not converge after retries".to_string())
}

/// A proxied request with its body read into memory, so it can be re-sent.
struct BufferedRequest {
    method: reqwest::Method,
    path_and_query: String,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
}

/// Read `req` into a [`BufferedRequest`] (`Err`: why the body was unreadable,
/// answered as a 400 by [`bad_body`]). Most proxied requests are small JSON;
/// buffering the request body (not the response) costs nothing here and is
/// what makes a retry possible.
async fn buffer_request(req: Request) -> Result<BufferedRequest, String> {
    let (parts, body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_owned();
    let body = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b,
        Err(e) => return Err(format!("could not read request body: {e}")),
    };
    Ok(BufferedRequest {
        method: reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
            .unwrap_or(reqwest::Method::GET),
        path_and_query,
        headers: parts.headers,
        body,
    })
}

fn bad_body(msg: String) -> Response {
    (StatusCode::BAD_REQUEST, msg).into_response()
}

/// Forward `req` to the worker and stream the response back.
async fn proxy_to(ctx: &ProxyCtx, worker: &ReadyWorker, req: Request) -> Response {
    let buffered = match buffer_request(req).await {
        Ok(b) => b,
        Err(msg) => return bad_body(msg),
    };
    match send_buffered(ctx, worker, &buffered).await {
        Ok(resp) => resp,
        Err(e) => {
            warn!(port = worker.port, error = %e, "proxy upstream connect failed; degrading");
            warming_response("worker connection lost; retry")
        }
    }
}

/// Send one buffered request to `worker` and stream its response back. `Err`
/// only when no response arrived at all (the worker died or was reaped between
/// acquire and send) — the one case a caller may safely retry.
async fn send_buffered(
    ctx: &ProxyCtx,
    worker: &ReadyWorker,
    req: &BufferedRequest,
) -> Result<Response, reqwest::Error> {
    // Rebuild the upstream URL: same path+query, worker's loopback port.
    let upstream_uri = format!("http://127.0.0.1:{}{}", worker.port, req.path_and_query);
    let _ = upstream_uri.parse::<Uri>(); // validate shape; reqwest re-parses

    let mut upstream_req = ctx
        .client
        .request(req.method.clone(), &upstream_uri)
        .body(req.body.to_vec());
    for (name, value) in req.headers.iter() {
        // Skip hop-by-hop / host headers the upstream sets itself.
        let n = name.as_str().to_ascii_lowercase();
        if matches!(n.as_str(), "host" | "content-length" | "connection") {
            continue;
        }
        if let Ok(v) = value.to_str() {
            upstream_req = upstream_req.header(name.as_str(), v);
        }
    }

    let upstream_resp = upstream_req.send().await?;

    // Build the downstream response, STREAMING the body (SSE-safe).
    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream_resp.headers().iter() {
        let n = name.as_str().to_ascii_lowercase();
        if matches!(
            n.as_str(),
            "connection" | "transfer-encoding" | "content-length"
        ) {
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            axum::http::HeaderName::from_bytes(name.as_str().as_bytes()),
            axum::http::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(hn, hv);
        }
    }

    // Stream the upstream body bytes through without buffering — this is what
    // keeps SSE (`/index-events`) live end-to-end.
    let stream = upstream_resp.bytes_stream();
    let body = Body::from_stream(stream);
    Ok(builder
        .body(body)
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response()))
}

/// Max proxied REQUEST body size (queries / index triggers / config are small).
/// 16 MiB is generous; responses are streamed and not bounded here.
const MAX_REQUEST_BODY: usize = 16 * 1024 * 1024;

/// A uniform "worker warming, retry shortly" degrade response. 503 with a short
/// JSON body + Retry-After. Mirrors the query-warm-path "warming, retry" signal:
/// the client retries and the worker will be ready, instead of a refused
/// connection or a hang.
fn warming_response(detail: &str) -> Response {
    let body = serde_json::json!({
        "status": "warming",
        "detail": detail,
        "retry": true,
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("retry-after", "2")],
        axum::Json(body),
    )
        .into_response()
}

/// A "worker failed to start" degrade response. 503 with `retry:false` so the UI
/// shows a manual-retry error banner (NOT the auto-retry warming skeleton). The
/// distinction lets the frontend render the three states correctly: warming
/// (auto-retry skeleton) vs error (red banner + Retry button).
fn error_response(detail: &str) -> Response {
    let body = serde_json::json!({
        "status": "error",
        "detail": detail,
        "retry": false,
    });
    (StatusCode::SERVICE_UNAVAILABLE, axum::Json(body)).into_response()
}
