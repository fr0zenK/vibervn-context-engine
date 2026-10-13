//! MCP availability: `codebase-retrieval` must answer with code whenever a
//! recent index of the workspace exists, also while that index is being rebuilt
//! or warmed, or after a run failed. Each test parks a REAL index run (engine
//! consumer + pipeline) inside its destructive window by holding its document
//! embedding requests at a local embeddings endpoint, then calls the real MCP
//! funnel. Query embeddings are always answered.

use super::*;
use crate::indexing::{IndexPhase, IndexState};
use crate::store::RepoDbMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tempfile::TempDir;

/// The answer carries at least one `path#Lstart-end` code block and no
/// "use grep" / "retry" degrade.
fn has_results(out: &str) -> bool {
    out.contains("#L")
        && !out.contains("use grep")
        && !out.contains("still warming")
        && !out.starts_with("Error")
}

/// Holds document-embedding requests while armed, so an index run stays parked
/// inside its destructive window (after its deletes, before its publish).
#[derive(Default)]
struct EmbedGate {
    armed: AtomicBool,
    parked: AtomicUsize,
    release: tokio::sync::Notify,
    /// Latency added to every document request, widening each run's window.
    doc_delay_ms: std::sync::atomic::AtomicU64,
}

/// A local embeddings endpoint that maps every input to the same vector, so any
/// stored chunk is a candidate for any query.
async fn gated_embeddings(gate: Arc<EmbedGate>) -> String {
    use axum::{Json, Router, routing::post};
    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<serde_json::Value>| {
            let gate = gate.clone();
            async move {
                let inputs = body["input"].as_array().cloned().unwrap_or_default();
                let preflight = inputs.len() == 1 && inputs[0] == "context-engine preflight";
                let delay = gate.doc_delay_ms.load(Ordering::SeqCst);
                if body["input_type"] == "document" && delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                if body["input_type"] == "document"
                    && !preflight
                    && gate.armed.load(Ordering::SeqCst)
                {
                    let released = gate.release.notified();
                    gate.parked.fetch_add(1, Ordering::SeqCst);
                    released.await;
                }
                let data: Vec<serde_json::Value> = (0..inputs.len())
                    .map(|i| serde_json::json!({"index": i, "embedding": [0.1, 0.2, 0.3, 0.4]}))
                    .collect();
                Json(serde_json::json!({ "data": data }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/v1")
}

struct Harness {
    _home: TempDir,
    dir: TempDir,
    home: std::path::PathBuf,
    repo: String,
    settings: Settings,
    engine: Arc<IndexEngine>,
    repo_dbs: RepoDbMap,
    gate: Arc<EmbedGate>,
}

impl Harness {
    /// A real repo, indexed to completion through the engine's consumer.
    async fn indexed() -> Self {
        let home_dir = TempDir::new().unwrap();
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn parse_config() -> u32 {\n    42\n}\n\npub fn load() { parse_config(); }\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {\n    load();\n}\n").unwrap();
        let repo = crate::store::normalize_repo_path(dir.path().to_str().unwrap());
        let home = home_dir.path().to_path_buf();
        let gate = Arc::new(EmbedGate::default());

        let mut settings = Settings {
            repos: vec![repo.clone()],
            mcp_index_wait_secs: 3,
            ..Default::default()
        };
        settings.embedding.api_keys = vec!["test-key".to_string()];
        settings.embedding.model = "test-model".to_string();
        settings.embedding.voyage_base_url = Some(gated_embeddings(gate.clone()).await);

        let repo_dbs: RepoDbMap = Arc::new(RwLock::new(HashMap::new()));
        let engine = IndexEngine::start(
            home.clone(),
            home.join("embeddings"),
            &settings,
            repo_dbs.clone(),
            Arc::new(RwLock::new(settings.clone())),
            true,
        )
        .await;
        let h = Self {
            _home: home_dir,
            dir,
            home,
            repo,
            settings,
            engine,
            repo_dbs,
            gate,
        };
        h.engine.trigger_index(&h.repo).await.unwrap();
        h.wait_idle().await;
        h
    }

    async fn wait_idle(&self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(s) = self.engine.repo_status(&self.repo).await
                && s.state == IndexState::Idle
                && s.last_indexed_at.is_some()
            {
                return;
            }
            assert!(std::time::Instant::now() < deadline, "index never finished");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Idle and unfenced across several consecutive checks: no queued run is left.
    async fn wait_settled(&self) {
        let mut quiet = 0;
        while quiet < 4 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let idle = self
                .engine
                .repo_status(&self.repo)
                .await
                .is_some_and(|s| s.state == IndexState::Idle);
            if idle && !self.engine.repo_is_mutating(&self.repo).await {
                quiet += 1;
            } else {
                quiet = 0;
            }
        }
    }

    async fn query(&self) -> String {
        run_codebase_retrieval(
            &self.home,
            &self.home,
            &self.engine,
            &self.repo_dbs,
            &self.settings,
            "where is the config parsed",
            &self.repo,
            crate::vector::Scope::Code,
        )
        .await
    }

    /// The worker restarted (or LRU evicted the shard): nothing resident.
    async fn go_cold(&self) {
        self.engine
            .vector_index
            .write()
            .await
            .evict_repo(&self.repo);
    }

    /// Edit a file with content no cache has seen, so the run must embed it.
    fn touch_new_content(&self, name: &str) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::fs::write(
            self.dir.path().join(name),
            format!("fn main() {{\n    load(); // {nonce}\n}}\n"),
        )
        .unwrap();
    }

    /// Start a run and return once it is parked inside its destructive window.
    async fn park_run(&self, rebuild: bool) {
        self.gate.armed.store(true, Ordering::SeqCst);
        self.touch_new_content("main.rs");
        if rebuild {
            self.engine.trigger_rebuild(&self.repo).await.unwrap();
        } else {
            self.engine.trigger_index(&self.repo).await.unwrap();
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while self.gate.parked.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "run never reached its embed"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            self.engine.repo_is_mutating(&self.repo).await,
            "the parked run must be inside its destructive window"
        );
    }

    /// Let the parked run finish and confirm it completes.
    async fn release(&self) {
        self.gate.armed.store(false, Ordering::SeqCst);
        self.gate.release.notify_waiters();
        self.wait_idle().await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_index_answers() {
    let h = Harness::indexed().await;
    let out = h.query().await;
    assert!(has_results(&out), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cold_shard_answers() {
    let h = Harness::indexed().await;
    h.go_cold().await;
    let out = h.query().await;
    assert!(has_results(&out), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn incremental_window_answers_from_previous_index() {
    let h = Harness::indexed().await;
    h.query().await; // resident, as during normal use
    h.park_run(false).await;
    let out = h.query().await;
    h.release().await;
    assert!(has_results(&out), "{out}");
    assert!(out.starts_with(crate::prompts::MCP_STALE_RESULTS), "{out}");
}

/// Worker restart, then the boot catch-up run starts before any query.
#[tokio::test(flavor = "multi_thread")]
async fn cold_incremental_window_answers_from_previous_index() {
    let h = Harness::indexed().await;
    h.go_cold().await;
    h.park_run(false).await;
    let out = h.query().await;
    h.release().await;
    assert!(has_results(&out), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn full_rebuild_window_answers_from_previous_index() {
    let h = Harness::indexed().await;
    h.query().await;
    h.park_run(true).await;
    let out = h.query().await;
    h.release().await;
    assert!(has_results(&out), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cold_full_rebuild_window_answers_from_previous_index() {
    let h = Harness::indexed().await;
    h.go_cold().await;
    h.park_run(true).await;
    let out = h.query().await;
    h.release().await;
    assert!(has_results(&out), "{out}");
}

/// A run that failed inside its destructive window leaves the repo fenced with
/// no run owning it (`abort_update_fail_closed`, `IndexState::Error`).
#[tokio::test(flavor = "multi_thread")]
async fn failed_run_answers_from_previous_index() {
    let h = Harness::indexed().await;
    h.query().await;
    h.park_run(false).await;
    h.engine
        .vector_index
        .write()
        .await
        .abort_update_fail_closed(&h.repo);
    {
        let mut statuses = h.engine.statuses.write().await;
        let s = statuses.entry(h.repo.clone()).or_default();
        s.state = IndexState::Error;
        s.phase = IndexPhase::Idle;
    }
    let out = h.query().await;
    h.gate.armed.store(false, Ordering::SeqCst);
    h.gate.release.notify_waiters();
    assert!(has_results(&out), "{out}");
}

/// After the run publishes, answers are live again (no stale banner).
#[tokio::test(flavor = "multi_thread")]
async fn answers_are_live_again_after_the_run() {
    let h = Harness::indexed().await;
    h.query().await;
    h.park_run(false).await;
    h.release().await;
    let out = h.query().await;
    assert!(has_results(&out), "{out}");
    assert!(!out.starts_with(crate::prompts::MCP_STALE_RESULTS), "{out}");
}

#[derive(Default)]
struct Seen {
    total: usize,
    stale: usize,
    graph_pending: usize,
    failures: Vec<String>,
}

/// SOAK: queries hammer the MCP funnel while the real pipeline churns through
/// incremental runs, full rebuilds and simulated worker restarts. Every single
/// answer must carry code. `AVAIL_SOAK_SECS` lengthens it (default 6 s).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_every_query_answers_during_index_churn() {
    let secs: u64 = std::env::var("AVAIL_SOAK_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let h = Arc::new(Harness::indexed().await);
    h.gate.doc_delay_ms.store(40, Ordering::SeqCst);
    let stop = Arc::new(AtomicBool::new(false));

    let mut queriers = Vec::new();
    for _ in 0..4 {
        let h = h.clone();
        let stop = stop.clone();
        queriers.push(tokio::spawn(async move {
            let mut seen = Seen::default();
            while !stop.load(Ordering::SeqCst) {
                let out = h.query().await;
                seen.total += 1;
                if out.starts_with(crate::prompts::MCP_STALE_RESULTS) {
                    seen.stale += 1;
                } else if out.starts_with(crate::prompts::MCP_GRAPH_PENDING) {
                    seen.graph_pending += 1;
                }
                if !has_results(&out) {
                    seen.failures
                        .push(out.chars().take(200).collect::<String>());
                }
            }
            seen
        }));
    }

    let start = std::time::Instant::now();
    let mut runs = 0u32;
    while start.elapsed() < Duration::from_secs(secs) {
        runs += 1;
        h.touch_new_content("main.rs");
        if runs.is_multiple_of(3) {
            // A new symbol: Phase 2 has real work.
            std::fs::write(
                h.dir.path().join("lib.rs"),
                format!(
                    "pub fn parse_config() -> u32 {{\n    42\n}}\n\npub fn load() {{ parse_config(); }}\n\npub fn extra_{runs}() {{ load(); }}\n"
                ),
            )
            .unwrap();
        }
        match runs % 7 {
            0 => h.engine.trigger_rebuild(&h.repo).await.unwrap(),
            4 => {
                h.go_cold().await;
                h.engine.trigger_index(&h.repo).await.unwrap();
            }
            _ => h.engine.trigger_index(&h.repo).await.unwrap(),
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    h.gate.doc_delay_ms.store(0, Ordering::SeqCst);
    h.wait_idle().await;
    stop.store(true, Ordering::SeqCst);

    let mut seen = Seen::default();
    for q in queriers {
        let s = q.await.unwrap();
        seen.total += s.total;
        seen.stale += s.stale;
        seen.graph_pending += s.graph_pending;
        seen.failures.extend(s.failures);
    }
    let Seen {
        total,
        stale,
        graph_pending,
        failures,
    } = seen;
    eprintln!(
        "SOAK: {runs} triggered runs, {total} queries ({stale} stale, {graph_pending} \
         graph-pending, {} live), {} without results",
        total - stale - graph_pending,
        failures.len()
    );
    assert!(total >= 20, "too few queries to prove anything: {total}");
    // Coverage: the soak only proves something if queries landed in the windows.
    assert!(stale > 0, "no query landed inside a destructive window");
    assert!(graph_pending > 0, "no query landed inside Phase 2");
    assert!(
        failures.is_empty(),
        "{} of {total} answers had no code: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );

    // Triggers queued during the churn may still run after a momentary Idle:
    // only once the queue has drained must answers be live again.
    h.wait_settled().await;
    let out = h.query().await;
    assert!(
        has_results(&out) && !out.starts_with(crate::prompts::MCP_STALE_RESULTS),
        "{out}"
    );
}

/// SOAK, the user's symptom: every index run lasts LONGER than the MCP wait
/// budget (embedding latency > `mcp_index_wait_secs`). Waiting the run out can
/// no longer save a query, so only the stale snapshot keeps answers flowing.
/// Cycles incremental / cold incremental / full rebuild / cold full rebuild.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_long_runs_still_answer_every_query() {
    let h = Arc::new(Harness::indexed().await);
    let budget_ms = h.settings.mcp_index_wait_secs * 1000;
    h.gate.doc_delay_ms.store(budget_ms + 500, Ordering::SeqCst);
    let stop = Arc::new(AtomicBool::new(false));

    let mut queriers = Vec::new();
    for _ in 0..4 {
        let h = h.clone();
        let stop = stop.clone();
        queriers.push(tokio::spawn(async move {
            let mut seen = Seen::default();
            while !stop.load(Ordering::SeqCst) {
                let out = h.query().await;
                seen.total += 1;
                if out.starts_with(crate::prompts::MCP_STALE_RESULTS) {
                    seen.stale += 1;
                }
                if !has_results(&out) {
                    seen.failures
                        .push(out.chars().take(200).collect::<String>());
                }
            }
            seen
        }));
    }

    for kind in 0..4 {
        if kind % 2 == 1 {
            h.go_cold().await;
        }
        h.touch_new_content("main.rs");
        if kind < 2 {
            h.engine.trigger_index(&h.repo).await.unwrap();
        } else {
            h.engine.trigger_rebuild(&h.repo).await.unwrap();
        }
        // Let the run enter its window, then wait it out.
        tokio::time::sleep(Duration::from_millis(300)).await;
        h.wait_idle().await;
    }
    stop.store(true, Ordering::SeqCst);

    let mut seen = Seen::default();
    for q in queriers {
        let s = q.await.unwrap();
        seen.total += s.total;
        seen.stale += s.stale;
        seen.failures.extend(s.failures);
    }
    eprintln!(
        "SOAK-LONG: 4 runs of > {budget_ms} ms, {} queries ({} stale), {} without results",
        seen.total,
        seen.stale,
        seen.failures.len()
    );
    assert!(
        seen.stale > 0,
        "no query landed inside a destructive window"
    );
    assert!(
        seen.failures.is_empty(),
        "{} of {} answers had no code: {:?}",
        seen.failures.len(),
        seen.total,
        &seen.failures[..seen.failures.len().min(5)]
    );
}

/// UPGRADE (measured on notepad-ade): builds before the identity marker moved
/// ahead of Phase 2 lost it on every interrupted run. The new build read the
/// absent marker as a mismatch: forced a full re-embed and answered "warming" /
/// "use grep" throughout it. An absent marker is now proved by re-embedding a few
/// stored chunks: the marker is restored, nothing is re-embedded, and the very
/// first query (cold worker) already answers with code.
#[tokio::test(flavor = "multi_thread")]
async fn legacy_db_without_identity_marker_answers_without_full_rebuild() {
    let h = Harness::indexed().await;
    let db = crate::store::get_or_open(&h.repo_dbs, &h.home, &h.repo, 0)
        .await
        .unwrap();
    let epoch_before = crate::store::ops::get_index_epoch(&db).await.unwrap();
    db.query("DELETE FROM index_meta WHERE key IN $keys")
        .bind(("keys", vec!["embedding_identity", "edges_resolved"]))
        .await
        .unwrap();
    h.go_cold().await;

    let out = h.query().await;
    assert!(has_results(&out), "{out}");

    h.wait_settled().await;
    let marker = crate::store::ops::get_meta(&db, "embedding_identity")
        .await
        .unwrap();
    assert!(marker.is_some(), "the verified marker is restored");
    assert!(
        crate::store::ops::get_meta(&db, "needs_rebuild")
            .await
            .unwrap()
            .is_none(),
        "an absent marker must not schedule a rebuild"
    );
    assert_eq!(
        crate::store::ops::get_index_epoch(&db).await.unwrap(),
        epoch_before,
        "no destructive (re-embedding) run happened"
    );
    assert!(has_results(&h.query().await));
}
