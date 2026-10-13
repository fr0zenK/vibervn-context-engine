//! Translate one [`IndexReadiness`] decision into what the REST search endpoint
//! must do — kept OUT of both `readiness.rs` (which owns the policy) and
//! `server.rs` (which owns HTTP plumbing) so each stays readable.
//!
//! ## Why this is a pure function over an inline match in the handler
//! `post_query` and `run_codebase_retrieval` must degrade identically; the only
//! legitimate difference is the SHAPE of the degrade (JSON vs prose). Expressing
//! the REST shape as a value makes that mapping unit-testable without booting
//! axum or reaching a live index, which is what lets the 503-warming contract and
//! the `graph_pending` flag be covered by tests at all.

use std::time::Duration;

use serde_json::{Value, json};

use super::readiness::IndexReadiness;
use crate::query::engine::QueryGraphMode;

/// HTTP status for "a run is still mid-flight; the same request will succeed
/// shortly". Matches the router's worker-warming degrade so the UI's existing
/// auto-retry path (`classifyWorkerDegrade`) handles both without branching.
pub(crate) const WARMING_STATUS: u16 = 503;

/// HTTP status for "the index run failed" — an upstream failure, not a client
/// error, and not retryable without user action.
pub(crate) const INDEX_FAILED_STATUS: u16 = 502;

/// What `post_query` does with a readiness decision.
#[derive(Debug)]
pub(crate) enum RestGate {
    /// Run the query. `graph_pending` is surfaced verbatim in the response body
    /// so the UI can badge "callers/callees not available yet".
    Query {
        graph_mode: QueryGraphMode,
        warm_budget: Duration,
        graph_pending: bool,
    },
    /// Nothing safe to serve yet; the client should retry.
    Warming,
    /// The index run failed.
    Failed(String),
}

impl RestGate {
    /// Extract the successful query plan without exposing enum fields to the
    /// HTTP module. `None` for a degrade.
    pub(crate) fn query_parts(&self) -> Option<(QueryGraphMode, Duration, bool)> {
        match self {
            Self::Query {
                graph_mode,
                warm_budget,
                graph_pending,
            } => Some((*graph_mode, *warm_budget, *graph_pending)),
            Self::Warming | Self::Failed(_) => None,
        }
    }

    /// HTTP status for a degrade. `None` for [`RestGate::Query`], which keeps its
    /// own 200 path.
    #[cfg(test)]
    pub(crate) fn degrade_status(&self) -> Option<u16> {
        match self {
            Self::Query { .. } => None,
            Self::Warming => Some(WARMING_STATUS),
            Self::Failed(_) => Some(INDEX_FAILED_STATUS),
        }
    }

    /// The JSON body for a degrade. `None` for [`RestGate::Query`].
    ///
    /// The warming body carries `status`/`retry` because the frontend classifies
    /// retryable degrades off those two fields, not off the status code alone.
    pub(crate) fn degrade_body(&self) -> Option<Value> {
        match self {
            Self::Query { .. } => None,
            Self::Warming => Some(json!({
                "status": "warming",
                "error": "indexing is still in progress; retry shortly",
                "retry": true,
            })),
            Self::Failed(message) => Some(json!({
                "status": "error",
                "error": message,
                "retry": false,
            })),
        }
    }
}

/// Map a readiness decision to the REST action. Total over the enum: a new
/// readiness variant cannot silently fall through to "just query".
pub(crate) fn rest_gate(readiness: IndexReadiness) -> RestGate {
    match readiness {
        IndexReadiness::Ready { warm_budget } => RestGate::Query {
            graph_mode: QueryGraphMode::Full,
            warm_budget,
            graph_pending: false,
        },
        // A stale snapshot answers vector-only too: `calls` may be mid-rewrite.
        IndexReadiness::ReadyVectorOnly { warm_budget }
        | IndexReadiness::ReadyStale { warm_budget } => RestGate::Query {
            graph_mode: QueryGraphMode::VectorOnly,
            warm_budget,
            graph_pending: true,
        },
        IndexReadiness::Timeout => RestGate::Warming,
        IndexReadiness::Failed(error) => RestGate::Failed(format!("{error:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> Duration {
        Duration::from_secs(crate::config::DEFAULT_MCP_INDEX_WAIT_SECS)
    }

    /// A complete index queries WITH the call graph and reports no pending graph,
    /// so the UI shows no badge.
    #[test]
    fn ready_queries_full_graph_and_reports_not_pending() {
        let gate = rest_gate(IndexReadiness::Ready {
            warm_budget: budget(),
        });
        match gate {
            RestGate::Query {
                graph_mode,
                warm_budget,
                graph_pending,
            } => {
                assert_eq!(graph_mode, QueryGraphMode::Full);
                assert_eq!(warm_budget, budget());
                assert!(
                    !graph_pending,
                    "a complete index must not badge the graph as pending"
                );
            }
            other => panic!("expected Query, got {other:?}"),
        }
        assert_eq!(
            rest_gate(IndexReadiness::Ready {
                warm_budget: budget()
            })
            .degrade_status(),
            None
        );
    }

    /// REGRESSION (the ResolveEdges fast path): phase 2 must NOT block the query.
    /// It serves vector-only results immediately and flags `graph_pending` so the
    /// user is told callers/callees are missing rather than silently getting none.
    #[test]
    fn resolve_edges_queries_vector_only_and_reports_pending() {
        let gate = rest_gate(IndexReadiness::ReadyVectorOnly {
            warm_budget: Duration::ZERO,
        });
        match gate {
            RestGate::Query {
                graph_mode,
                graph_pending,
                ..
            } => {
                assert_eq!(graph_mode, QueryGraphMode::VectorOnly);
                assert!(
                    graph_pending,
                    "vector-only results must be flagged so the UI can badge them"
                );
            }
            other => panic!("expected Query, got {other:?}"),
        }
    }

    /// The wait budget expiring is a RETRYABLE degrade: 503 with `retry:true`, so
    /// the UI auto-retries instead of showing a hard error.
    #[test]
    fn timeout_is_a_retryable_503_warming() {
        let gate = rest_gate(IndexReadiness::Timeout);
        assert_eq!(gate.degrade_status(), Some(WARMING_STATUS));
        assert_eq!(WARMING_STATUS, 503, "the UI's warming path keys off 503");

        let body = gate.degrade_body().expect("warming must carry a body");
        assert_eq!(body["status"], "warming");
        assert_eq!(
            body["retry"], true,
            "classifyWorkerDegrade treats retry:true as auto-retry"
        );
    }

    /// A failed run is NOT retryable — the user must act, so the UI shows an error
    /// banner rather than spinning.
    #[test]
    fn failed_is_a_non_retryable_502_carrying_the_cause() {
        let gate = rest_gate(IndexReadiness::Failed(anyhow::anyhow!(
            "embed key rejected"
        )));
        assert_eq!(gate.degrade_status(), Some(INDEX_FAILED_STATUS));

        let body = gate.degrade_body().expect("failure must carry a body");
        assert_eq!(body["retry"], false);
        assert!(
            body["error"]
                .as_str()
                .unwrap_or_default()
                .contains("embed key rejected"),
            "the cause must reach the client, got: {body}"
        );
    }

    /// `Query` never degrades — guards against a caller treating a successful
    /// gate as an error shape.
    #[test]
    fn query_has_no_degrade_shape() {
        let gate = rest_gate(IndexReadiness::ReadyVectorOnly {
            warm_budget: Duration::ZERO,
        });
        assert_eq!(gate.degrade_status(), None);
        assert!(gate.degrade_body().is_none());
    }
}
