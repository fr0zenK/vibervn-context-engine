//! Whether an MCP query may be answered from the repo's currently-stored index,
//! must wait for an in-flight run, or must trigger one first.
//!
//! ## Why this is one function over one input struct
//! The decision depends on four independent facts, and every earlier bug in this
//! area came from collapsing them too early:
//!
//! 1. `run_state_is_indexing` — a run owns the repo (`RepoStatus::Indexing`).
//!    On its own this says NOTHING about whether stored data is safe to serve: a
//!    queued or just-started run (e.g. the boot catch-up incremental) has not
//!    touched a single row yet.
//! 2. `repo_is_mutating` — the pipeline has entered its destructive window
//!    (`ShardedVectorIndex::is_mutating`, set by `begin_full_update` /
//!    `begin_incremental_update` BEFORE any delete). This is the only signal that
//!    means "the LIVE index is not safe to return". The previous index's snapshot
//!    still is, when its completion stamp (4) is recent: an MCP call must answer
//!    with code, and a slightly old answer beats "use grep" (`ServeStale`).
//! 3. `chunk_count` — whether anything was ever indexed.
//! 4. `last_indexed_ts` — durable completion stamp, for staleness.
//!
//! Gating on (1) instead of (2) made the FIRST query after every worker start
//! return empty while the boot catch-up run was merely queued; the same query
//! retried a moment later succeeded. Gating on (2) alone (ignoring 3/4) would
//! serve a never-indexed or long-stale repo. Both regressions are covered by
//! tests in this module.
//!
//! The returned [`QueryReadiness`] carries the resulting ACTION, so the caller
//! does not re-derive booleans and cannot drift from the decision made here.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;

use crate::config::Settings;
use crate::indexing::{IndexEngine, IndexPhase, IndexState, RepoStatus};
use crate::store::{self, RepoDbMap};

/// How often the shared index wait loop re-checks run status while blocked.
///
/// Bounded well below `Settings::mcp_index_wait_secs` (the total budget) so the
/// loop observes a state transition promptly without busy-spinning on the shared
/// status map.
pub(crate) const INDEX_READY_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// The single decision both query paths act on.
///
/// Carrying the graph mode AND the remaining warm budget in one value is what
/// keeps REST and MCP from re-deriving the policy: a caller cannot accidentally
/// query with the call graph while phase 2 is still resolving it, nor start a
/// second full warm-wait window after the index wait already burned budget.
#[derive(Debug)]
pub(crate) enum IndexReadiness {
    /// Complete index — query with the call graph. `warm_budget` is what remains
    /// of `Settings::mcp_index_wait_secs` after any index wait.
    Ready { warm_budget: Duration },
    /// Chunks and vectors are durable but phase 2 is still resolving call edges.
    /// Vector results are valid NOW; callers/callees are not yet available, so the
    /// query must run in `QueryGraphMode::VectorOnly` and the surface must tell
    /// the user the graph is pending.
    ReadyVectorOnly { warm_budget: Duration },
    /// A run has the live index fenced (or a failed run left it so) and the last
    /// completed index is recent: query its stale snapshot vector-only, now, and
    /// tell the user. If no snapshot turns out to exist the caller falls back to
    /// [`wait_for_run`].
    ReadyStale { warm_budget: Duration },
    /// The wait budget expired with a run still mid-flight. Nothing safe to serve.
    Timeout,
    /// The index run failed, or the repo DB could not be opened.
    Failed(anyhow::Error),
}

impl IndexReadiness {
    /// True when the query must skip call-graph expansion. Also the value the
    /// REST layer reports as `graph_pending` so the UI can badge it.
    #[cfg(test)]
    pub(crate) fn graph_pending(&self) -> bool {
        matches!(self, Self::ReadyVectorOnly { .. } | Self::ReadyStale { .. })
    }

    /// Graph mode selected by the readiness decision.
    #[cfg(test)]
    pub(crate) fn graph_mode(&self) -> crate::query::engine::QueryGraphMode {
        match self {
            Self::Ready { .. } => crate::query::engine::QueryGraphMode::Full,
            Self::ReadyVectorOnly { .. } | Self::ReadyStale { .. } => {
                crate::query::engine::QueryGraphMode::VectorOnly
            }
            Self::Timeout | Self::Failed(_) => {
                panic!("a non-queryable readiness has no graph mode")
            }
        }
    }

    /// Remaining budget for a cold-shard warm, or `None` when not queryable.
    #[cfg(test)]
    pub(crate) fn warm_budget(&self) -> Option<Duration> {
        match self {
            Self::Ready { warm_budget }
            | Self::ReadyVectorOnly { warm_budget }
            | Self::ReadyStale { warm_budget } => Some(*warm_budget),
            Self::Timeout | Self::Failed(_) => None,
        }
    }
}

/// True only while a run owns the repo AND is in phase 2 (edge resolution).
///
/// Phase 2 rewrites the `calls` table, never the chunks or the vector shard, so
/// the shard is query-ready even though `RepoStatus.state` is still `Indexing`.
/// `IndexState::Idle` with a stale `phase` is NOT phase 2 — the state field is
/// what proves a run is live.
pub(crate) fn is_resolve_edges_status(status: Option<&RepoStatus>) -> bool {
    status.is_some_and(|s| s.state == IndexState::Indexing && s.phase == IndexPhase::ResolveEdges)
}

/// Ensure the repository index is queryable before a query starts, and say HOW
/// it may be queried.
///
/// Both normal search and MCP call this exact helper, so the trigger/wait/degrade
/// decision has one source of truth. It evaluates durable index state, triggers a
/// run when necessary, and waits for the owning run to leave the destructive
/// window — short-circuiting to [`IndexReadiness::ReadyVectorOnly`] whenever the
/// run is merely resolving the call graph.
pub(crate) async fn await_index_ready(
    settings: &Settings,
    index_engine: &Arc<IndexEngine>,
    repo_dbs: &RepoDbMap,
    data_dir: &std::path::Path,
    repo: &str,
) -> IndexReadiness {
    let db = match store::get_or_open(repo_dbs, data_dir, repo, settings.repo_generation(repo))
        .await
        .context("failed to open repository index")
    {
        Ok(db) => db,
        Err(error) => return IndexReadiness::Failed(error),
    };
    let chunk_count = store::ops::count_chunks(&db).await.unwrap_or(0);
    let last_indexed_ts = store::ops::get_meta(&db, "last_indexed_at")
        .await
        .unwrap_or(None);
    let status = index_engine.repo_status(repo).await;

    // Phase 2 fast path, checked BEFORE the freshness policy: the vectors are
    // final for this run, so waiting for edge resolution would burn the whole
    // budget for results that are already correct minus callers/callees.
    //
    // The shard is NOT necessarily resident, though: a worker that started (or
    // evicted it) before this run stays cold, since an incremental publish only
    // patches a resident shard. The full budget goes to warming it — a zero
    // budget made every query during a long Phase 2 answer "still warming".
    let full_budget = Duration::from_secs(settings.mcp_index_wait_secs);
    if is_resolve_edges_status(status.as_ref()) {
        return IndexReadiness::ReadyVectorOnly {
            warm_budget: full_budget,
        };
    }

    let readiness = evaluate(
        &RepoIndexState {
            run_state_is_indexing: status
                .as_ref()
                .is_some_and(|s| s.state == IndexState::Indexing),
            repo_is_mutating: index_engine.repo_is_mutating(repo).await,
            chunk_count,
            last_indexed_ts: last_indexed_ts.as_deref(),
        },
        chrono::Duration::days(settings.mcp_stale_after_days as i64),
    );

    if readiness.requires_trigger()
        && let Err(error) = index_engine.trigger_index(repo).await
    {
        return IndexReadiness::Failed(error);
    }
    if let QueryReadiness::ServeStale { .. } = readiness {
        return IndexReadiness::ReadyStale {
            warm_budget: full_budget,
        };
    }
    if !readiness.requires_wait() {
        return IndexReadiness::Ready {
            warm_budget: full_budget,
        };
    }
    wait_for_run(index_engine, repo, full_budget).await
}

/// Block (bounded by `budget`) until the run owning `repo` leaves its destructive
/// window, and say how the repo may then be queried.
pub(crate) async fn wait_for_run(
    index_engine: &Arc<IndexEngine>,
    repo: &str,
    budget: Duration,
) -> IndexReadiness {
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        tokio::time::sleep(INDEX_READY_POLL_INTERVAL).await;
        let status = index_engine.repo_status(repo).await;
        // The run may reach phase 2 while we are blocked — take the fast path
        // rather than waiting out edge resolution, with whatever budget is left.
        if is_resolve_edges_status(status.as_ref()) {
            return IndexReadiness::ReadyVectorOnly {
                warm_budget: deadline.saturating_duration_since(tokio::time::Instant::now()),
            };
        }
        match status.as_ref().map(|s| &s.state) {
            Some(IndexState::Idle) => {
                return IndexReadiness::Ready {
                    warm_budget: deadline.saturating_duration_since(tokio::time::Instant::now()),
                };
            }
            Some(IndexState::Error) => {
                let error = status
                    .and_then(|s| s.error)
                    .unwrap_or_else(|| "unknown indexing error".to_string());
                return IndexReadiness::Failed(anyhow::anyhow!("indexing failed: {error}"));
            }
            _ if tokio::time::Instant::now() >= deadline => return IndexReadiness::Timeout,
            _ => {}
        }
    }
}

/// The four observed facts about a repo's index at query time. Borrowed rather
/// than owned so the caller passes what it already read from the DB/engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RepoIndexState<'a> {
    /// A run owns the repo (`RepoStatus.state == Indexing`). NOT sufficient on
    /// its own to withhold results — see the module docs.
    pub run_state_is_indexing: bool,
    /// The pipeline is inside its destructive publish window. This is the
    /// authoritative "nothing safe to serve" signal.
    pub repo_is_mutating: bool,
    /// Chunk rows currently present in the repo DB.
    pub chunk_count: u64,
    /// Durable `last_indexed_at`. The in-memory `RepoStatus.last_indexed_at` is
    /// intentionally NOT consulted: only the committed stamp proves completion.
    pub last_indexed_ts: Option<&'a str>,
}

/// The action the MCP query path must take, derived from [`RepoIndexState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryReadiness {
    /// The stored index is complete and safe to query right now.
    ///
    /// `refresh_in_background` is set when the durable stamp is missing or
    /// unparseable (legacy/corrupt) AND no run already owns the repo, so a
    /// non-blocking refresh can write a real stamp for next time. Results are
    /// served immediately either way.
    ServeExisting { refresh_in_background: bool },
    /// Nothing safe to serve, but a run already owns the repo — join its wait
    /// instead of queueing a duplicate trigger.
    WaitForRun,
    /// Nothing safe to serve and no run owns the repo (never indexed, genuinely
    /// stale, or a failed destructive run whose fence is still set) — trigger a
    /// run, then wait.
    TriggerThenWait,
    /// The live index is being replaced (or a failed run left it fenced/emptied),
    /// but the last COMPLETED index is within the staleness threshold: answer now
    /// from that index's snapshot instead of waiting out the run. A recent
    /// slightly-old answer beats "use grep". `trigger_repair` is set when no run
    /// owns the repo, so the fenced state is not left in place.
    ServeStale { trigger_repair: bool },
}

impl QueryReadiness {
    /// True when the stored index may be queried without waiting. Also the
    /// "had usable data before" predicate for the index-failed degrade path.
    #[cfg(test)]
    pub(crate) fn serves_existing_index(self) -> bool {
        matches!(self, Self::ServeExisting { .. })
    }

    /// True when the caller must run the bounded wait loop before querying.
    pub(crate) fn requires_wait(self) -> bool {
        matches!(self, Self::WaitForRun | Self::TriggerThenWait)
    }

    /// True when the caller must queue an index run.
    pub(crate) fn requires_trigger(self) -> bool {
        matches!(
            self,
            Self::TriggerThenWait
                | Self::ServeExisting {
                    refresh_in_background: true
                }
                | Self::ServeStale {
                    trigger_repair: true
                }
        )
    }
}

/// Decide readiness from the four observed facts.
///
/// Rules, in precedence order:
/// * inside the destructive window → never serve (data is being replaced)
/// * `chunk_count == 0` → never indexed, nothing to serve
/// * stamp missing/unparseable but chunks exist → serve (legacy or corrupt stamp
///   must not punish the user), and backfill a stamp when nothing owns the repo
/// * stamp within `stale_threshold` → serve
/// * stamp older than `stale_threshold` → do not serve
///
/// When the first two rules withhold the live index but the durable stamp of the
/// last completed run is within `stale_threshold`, the answer is
/// [`QueryReadiness::ServeStale`] rather than a wait: `last_indexed_at` is only
/// written after a run completes, so it dates the snapshot being served.
pub(crate) fn evaluate(
    state: &RepoIndexState<'_>,
    stale_threshold: chrono::Duration,
) -> QueryReadiness {
    let parsed_stamp = state
        .last_indexed_ts
        .and_then(|ts| ts.parse::<chrono::DateTime<chrono::Utc>>().ok());

    let serves_existing = if state.repo_is_mutating || state.chunk_count == 0 {
        false
    } else {
        match parsed_stamp {
            // Missing or unparseable stamp with chunks present: usable.
            None => true,
            Some(dt) => (chrono::Utc::now() - dt) <= stale_threshold,
        }
    };

    if serves_existing {
        return QueryReadiness::ServeExisting {
            refresh_in_background: parsed_stamp.is_none() && !state.run_state_is_indexing,
        };
    }
    let last_run_is_recent =
        parsed_stamp.is_some_and(|dt| (chrono::Utc::now() - dt) <= stale_threshold);
    if last_run_is_recent && (state.repo_is_mutating || state.chunk_count == 0) {
        return QueryReadiness::ServeStale {
            trigger_repair: !state.run_state_is_indexing,
        };
    }
    if state.run_state_is_indexing {
        QueryReadiness::WaitForRun
    } else {
        QueryReadiness::TriggerThenWait
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::engine::QueryGraphMode;

    fn default_stale_threshold() -> chrono::Duration {
        chrono::Duration::days(crate::config::DEFAULT_MCP_STALE_AFTER_DAYS as i64)
    }

    fn ts_days_ago(n: i64) -> String {
        (chrono::Utc::now() - chrono::Duration::days(n)).to_rfc3339()
    }

    /// Builder defaulting to "idle repo, freshly indexed" so each test states
    /// only the fact it is exercising.
    fn state(chunk_count: u64, ts: Option<&str>) -> RepoIndexState<'_> {
        RepoIndexState {
            run_state_is_indexing: false,
            repo_is_mutating: false,
            chunk_count,
            last_indexed_ts: ts,
        }
    }

    fn readiness(s: &RepoIndexState<'_>) -> QueryReadiness {
        evaluate(s, default_stale_threshold())
    }

    // 1. chunk_count == 0 → never serve, regardless of timestamp.
    #[test]
    fn no_chunks_is_not_usable() {
        assert!(!readiness(&state(0, None)).serves_existing_index());
        let ts = ts_days_ago(1);
        assert!(!readiness(&state(0, Some(&ts))).serves_existing_index());
    }

    // 2. chunks + no stamp → serve (legacy/pre-timestamp index) and backfill.
    #[test]
    fn legacy_index_no_timestamp_is_usable() {
        let r = readiness(&state(1, None));
        assert!(r.serves_existing_index());
        assert_eq!(
            r,
            QueryReadiness::ServeExisting {
                refresh_in_background: true
            },
            "a missing stamp on an idle repo must schedule a backfill refresh"
        );
    }

    // 3. chunks + stamp inside threshold → serve, no refresh needed.
    #[test]
    fn fresh_timestamp_is_usable() {
        let ts = ts_days_ago(1);
        assert_eq!(
            readiness(&state(100, Some(&ts))),
            QueryReadiness::ServeExisting {
                refresh_in_background: false
            }
        );
    }

    // 4. chunks + stamp past threshold → do not serve.
    #[test]
    fn old_timestamp_is_not_usable() {
        let ts = ts_days_ago(30);
        assert!(!readiness(&state(100, Some(&ts))).serves_existing_index());
    }

    // 5. chunks + unparseable stamp → serve (corrupt stamp, don't punish user).
    #[test]
    fn unparseable_timestamp_is_usable() {
        assert!(readiness(&state(50, Some("not-a-date"))).serves_existing_index());
    }

    // 6a. Boundary: just inside threshold.
    #[test]
    fn just_inside_threshold_is_usable() {
        let ts = ts_days_ago(6);
        assert!(readiness(&state(10, Some(&ts))).serves_existing_index());
    }

    // 6b. Boundary: just outside threshold.
    #[test]
    fn just_outside_threshold_is_not_usable() {
        let ts = ts_days_ago(8);
        assert!(!readiness(&state(10, Some(&ts))).serves_existing_index());
    }

    /// REGRESSION: the first query after a worker start returned empty because a
    /// merely-queued boot catch-up run (`state == Indexing`, fence NOT set) was
    /// treated as "unusable". The complete previous index must still be served,
    /// with no duplicate trigger queued behind the run that already owns the repo.
    #[test]
    fn indexing_before_mutation_keeps_previous_results_usable() {
        let ts = ts_days_ago(1);
        let mut s = state(100, Some(&ts));
        s.run_state_is_indexing = true;

        let r = readiness(&s);
        assert_eq!(
            r,
            QueryReadiness::ServeExisting {
                refresh_in_background: false
            },
            "a queued/early run must not hide the complete previous index"
        );
        assert!(!r.requires_wait(), "must answer immediately, not wait");
        assert!(
            !r.requires_trigger(),
            "a run already owns the repo — must not queue a duplicate trigger"
        );
    }

    /// Inside the destructive window the LIVE index is never served, but a recent
    /// last-completed index is: answered at once from its stale snapshot, with no
    /// wait and no duplicate trigger behind the run that owns the repo.
    #[test]
    fn fresh_timestamp_inside_mutation_fence_serves_stale() {
        let ts = ts_days_ago(1);
        let mut s = state(100, Some(&ts));
        s.run_state_is_indexing = true;
        s.repo_is_mutating = true;

        let r = readiness(&s);
        assert!(!r.serves_existing_index());
        assert_eq!(
            r,
            QueryReadiness::ServeStale {
                trigger_repair: false
            }
        );
        assert!(!r.requires_wait());
        assert!(!r.requires_trigger(), "a run already owns the repo");
    }

    /// "Not too old": a snapshot whose last completed run is past the staleness
    /// threshold is not served — the query waits for the run instead.
    #[test]
    fn old_timestamp_inside_mutation_fence_waits() {
        let ts = ts_days_ago(30);
        let mut s = state(100, Some(&ts));
        s.run_state_is_indexing = true;
        s.repo_is_mutating = true;
        assert_eq!(readiness(&s), QueryReadiness::WaitForRun);
    }

    /// A full rebuild has deleted every chunk: still serve the recent snapshot.
    #[test]
    fn emptied_by_running_rebuild_serves_stale() {
        let ts = ts_days_ago(1);
        let mut s = state(0, Some(&ts));
        s.run_state_is_indexing = true;
        s.repo_is_mutating = true;
        assert_eq!(
            readiness(&s),
            QueryReadiness::ServeStale {
                trigger_repair: false
            }
        );
    }

    /// Partially-written chunks during a destructive run are not usable either.
    #[test]
    fn partial_chunks_are_not_usable_while_mutating() {
        let ts = ts_days_ago(1);
        let mut s = state(50, Some(&ts));
        s.repo_is_mutating = true;
        assert!(!readiness(&s).serves_existing_index());
    }

    /// A failed destructive run leaves the fence set with no run owning the repo:
    /// the live index is unusable and nobody will repair it unless we trigger —
    /// but the recent snapshot still answers meanwhile.
    #[test]
    fn stranded_mutation_fence_serves_stale_and_triggers_repair() {
        let ts = ts_days_ago(1);
        let mut s = state(100, Some(&ts));
        s.run_state_is_indexing = false;
        s.repo_is_mutating = true;

        let r = readiness(&s);
        assert_eq!(
            r,
            QueryReadiness::ServeStale {
                trigger_repair: true
            }
        );
        assert!(r.requires_trigger());
        assert!(!r.requires_wait());
    }

    /// Stranded fence over an index past the threshold: repair, then wait.
    #[test]
    fn stranded_mutation_fence_on_old_index_triggers_then_waits() {
        let ts = ts_days_ago(30);
        let mut s = state(100, Some(&ts));
        s.repo_is_mutating = true;
        assert_eq!(readiness(&s), QueryReadiness::TriggerThenWait);
    }

    /// Never-indexed idle repo: trigger the first build, then wait.
    #[test]
    fn never_indexed_idle_repo_triggers_then_waits() {
        assert_eq!(readiness(&state(0, None)), QueryReadiness::TriggerThenWait);
    }

    /// Stale idle repo: trigger a refresh, then wait.
    #[test]
    fn stale_idle_repo_triggers_then_waits() {
        let ts = ts_days_ago(30);
        assert_eq!(
            readiness(&state(10, Some(&ts))),
            QueryReadiness::TriggerThenWait
        );
    }

    /// Missing stamp while a run already owns the repo: still serve the existing
    /// index, but do NOT queue a duplicate backfill trigger.
    #[test]
    fn missing_stamp_during_run_serves_without_duplicate_trigger() {
        let mut s = state(10, None);
        s.run_state_is_indexing = true;
        let r = readiness(&s);
        assert_eq!(
            r,
            QueryReadiness::ServeExisting {
                refresh_in_background: false
            }
        );
        assert!(!r.requires_trigger());
    }

    #[test]
    fn poll_interval_is_below_the_total_wait_budget() {
        let default_budget = Duration::from_secs(crate::config::DEFAULT_MCP_INDEX_WAIT_SECS);
        assert!(
            INDEX_READY_POLL_INTERVAL < default_budget,
            "poll interval must fit many times inside the default wait budget"
        );
    }

    /// The enum is the contract both surfaces read. `graph_pending` must be set by
    /// exactly one variant, and only the two queryable variants may hand back a
    /// warm budget — a caller that got `Timeout`/`Failed` has nothing to query.
    #[test]
    fn readiness_variants_expose_graph_pending_and_budget() {
        let budget = Duration::from_secs(7);

        let ready = IndexReadiness::Ready {
            warm_budget: budget,
        };
        assert!(!ready.graph_pending());
        assert_eq!(ready.warm_budget(), Some(budget));
        assert_eq!(ready.graph_mode(), QueryGraphMode::Full);

        let vector_only = IndexReadiness::ReadyVectorOnly {
            warm_budget: budget,
        };
        assert!(vector_only.graph_pending());
        assert_eq!(vector_only.warm_budget(), Some(budget));
        assert_eq!(vector_only.graph_mode(), QueryGraphMode::VectorOnly);

        assert!(!IndexReadiness::Timeout.graph_pending());
        assert_eq!(IndexReadiness::Timeout.warm_budget(), None);

        let failed = IndexReadiness::Failed(anyhow::anyhow!("boom"));
        assert!(!failed.graph_pending());
        assert_eq!(failed.warm_budget(), None);
    }
}
