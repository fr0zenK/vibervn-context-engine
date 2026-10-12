//! One-shot code↔docs fallback for scoped retrieval.
//!
//! `codebase-retrieval` searches exactly one class (code or docs) per call. When
//! the requested class yields no candidates at all, the OTHER class is searched
//! once and returned, flagged so the caller can tell the agent what it actually
//! got. Kept as a small generic function over a "gather" closure so the control
//! flow (when to retry, when NOT to) is unit-tested without a live index.

use std::collections::HashMap;
use std::future::Future;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::query::engine::QueryResult;
use crate::query::merger::MergeChunk;
use crate::vector::{Scope, ScopeFilter};

/// Candidates for one scan of one scope: vector search → hydrate → post-filters.
pub(crate) struct Gathered {
    pub base_chunks: Vec<MergeChunk>,
    /// The shard was still loading, or a candidate lost the publication race. An
    /// empty `base_chunks` with this set means "retry shortly", never "no results".
    pub warming: bool,
    /// Wall time of the vector scan(s) that produced this, for `QueryTiming`.
    pub search_ms: u64,
    /// DB handles snapshot, reused by graph expansion later in the pipeline.
    pub db_map: HashMap<String, Surreal<Db>>,
}

/// What was actually served and under which scope.
pub(crate) struct Served {
    pub gathered: Gathered,
    /// The scope the returned candidates belong to (`None` = unscoped query).
    pub scope: Option<ScopeFilter>,
    /// True only when the requested scope was empty and the other was served.
    pub fallback: bool,
}

/// Gather candidates for `requested`; if that is empty (and not merely warming),
/// retry ONCE with the opposite scope.
///
/// Fallback is deliberately narrow:
/// - only for a scoped request (`None` = unscoped, nothing to fall back from);
/// - never when the first scan reported `warming` — that would mask a retryable
///   state as a different answer;
/// - at most one retry, so two empty classes cannot ping-pong;
/// - "empty" is "no candidates survived", not a relevance threshold: cosine
///   always returns the nearest rows, so a weak match is a result, not an empty.
///
/// `allow_fallback = false` scans the requested scope once and never switches —
/// used by the forced retry after a reranker rejection, which must not bounce
/// back into the scope it just came from.
pub(crate) async fn gather_with_fallback<F, Fut>(
    requested: Option<ScopeFilter>,
    allow_fallback: bool,
    mut gather: F,
) -> Served
where
    F: FnMut(Option<ScopeFilter>) -> Fut,
    Fut: Future<Output = Gathered>,
{
    let mut first = gather(requested.clone()).await;
    let Some(requested_filter) = requested else {
        return Served {
            gathered: first,
            scope: None,
            fallback: false,
        };
    };
    if !allow_fallback || !first.base_chunks.is_empty() || first.warming {
        return Served {
            gathered: first,
            scope: Some(requested_filter),
            fallback: false,
        };
    }

    let other = requested_filter.opposite();
    let mut second = gather(Some(other.clone())).await;
    if !second.base_chunks.is_empty() {
        second.search_ms += first.search_ms;
        return Served {
            gathered: second,
            scope: Some(other),
            fallback: true,
        };
    }

    // Both empty. Report the requested scope, but if the retry saw an incomplete
    // index surface that instead of a misleading "no results".
    first.search_ms += second.search_ms;
    first.warming = second.warming;
    Served {
        gathered: first,
        scope: Some(requested_filter),
        fallback: false,
    }
}

/// Second kind of fallback: the requested scope HAD candidates but the reranker
/// rejected every one of them. The candidates are the nearest rows by cosine, so
/// "all rejected" means the requested class holds nothing relevant — try the
/// other class once, with the same query.
///
/// `run(forced_scope, allow_gather_fallback)` executes the whole pipeline once:
/// `(None, true)` honors the request (and may use the gather-level fallback);
/// `(Some(other), false)` is the single forced retry.
///
/// Deliberately narrow:
/// - only for a scoped query whose result is empty because the reranker rejected
///   everything (not "no candidates" — the gather-level fallback owns that);
/// - never while `warming` (retry-later, not a verdict);
/// - never when the gather-level fallback already switched scope: the original
///   scope had no candidates, so bouncing back would only repeat that;
/// - at most one retry; a failing or empty retry returns the ORIGINAL answer, so
///   this can only ever improve on "No relevant code found", never worsen it.
///
/// Cost: one extra full pipeline pass (query embedding — normally an embedding
/// cache hit —, one scan, graph expand and one rerank call), paid only on this
/// rare path.
pub(crate) async fn retry_on_rerank_rejection<F, Fut>(mut run: F) -> anyhow::Result<QueryResult>
where
    F: FnMut(Option<Scope>, bool) -> Fut,
    Fut: Future<Output = anyhow::Result<QueryResult>>,
{
    let first = run(None, true).await?;
    let Some(served) = first.scope.as_deref().and_then(Scope::parse) else {
        return Ok(first);
    };
    if first.scope_fallback || first.warming || !first.rerank_rejected_all() {
        return Ok(first);
    }

    let other = served.opposite();
    match run(Some(other), false).await {
        Ok(mut second) if !second.results.is_empty() => {
            second.scope = Some(other.as_str().to_owned());
            second.scope_fallback = true;
            Ok(second)
        }
        Ok(_) => Ok(first),
        Err(e) => {
            tracing::warn!(error = %e, "scope retry after reranker rejection failed; keeping the original answer");
            Ok(first)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::*;
    use crate::query::merger::MergeChunk;
    use crate::vector::{Scope, ScopeFilter};

    fn chunk(file: &str) -> MergeChunk {
        MergeChunk {
            file: file.to_string(),
            line_start: 1,
            line_end: 2,
            score: 0.5,
            content: String::new(),
            symbol: None,
            symbol_fqn: None,
            symbol_kind: None,
        }
    }

    fn gathered(files: &[&str], warming: bool, search_ms: u64) -> Gathered {
        Gathered {
            base_chunks: files.iter().map(|f| chunk(f)).collect(),
            warming,
            search_ms,
            db_map: HashMap::new(),
        }
    }

    fn filter(scope: Scope) -> Option<ScopeFilter> {
        Some(ScopeFilter::new(scope, ["md"]))
    }

    /// Drive `gather_with_fallback` with canned per-scope data and record which
    /// scopes were actually searched, in order.
    async fn run(
        requested: Option<ScopeFilter>,
        code: Gathered,
        docs: Gathered,
    ) -> (Served, Vec<Option<Scope>>) {
        let calls: RefCell<Vec<Option<Scope>>> = RefCell::new(Vec::new());
        let code = RefCell::new(Some(code));
        let docs = RefCell::new(Some(docs));
        let served = gather_with_fallback(requested, true, |scope: Option<ScopeFilter>| {
            calls.borrow_mut().push(scope.as_ref().map(|s| s.scope()));
            let out = match scope.as_ref().map(|s| s.scope()) {
                Some(Scope::Docs) => docs.borrow_mut().take(),
                _ => code.borrow_mut().take(),
            }
            .expect("each scope is searched at most once");
            async move { out }
        })
        .await;
        (served, calls.into_inner())
    }

    /// The forced retry after a reranker rejection must not bounce back into the
    /// scope that was just rejected: with fallback disabled an empty scope is
    /// simply empty, after exactly one scan.
    #[tokio::test]
    async fn disabled_fallback_searches_once_even_when_empty() {
        let scans = RefCell::new(0u32);
        let served = gather_with_fallback(filter(Scope::Docs), false, |_scope| {
            *scans.borrow_mut() += 1;
            async { gathered(&[], false, 3) }
        })
        .await;
        assert_eq!(*scans.borrow(), 1);
        assert!(!served.fallback);
        assert_eq!(served.scope.as_ref().map(|f| f.scope()), Some(Scope::Docs));
        assert!(served.gathered.base_chunks.is_empty());
    }

    fn served_files(s: &Served) -> Vec<&str> {
        s.gathered
            .base_chunks
            .iter()
            .map(|c| c.file.as_str())
            .collect()
    }

    #[tokio::test]
    async fn non_empty_requested_scope_is_served_without_a_second_search() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&["a.rs"], false, 5),
            gathered(&["README.md"], false, 7),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code)]);
        assert!(!s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert_eq!(served_files(&s), vec!["a.rs"]);
    }

    #[tokio::test]
    async fn empty_code_falls_back_to_docs_once_and_says_so() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], false, 5),
            gathered(&["README.md"], false, 7),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code), Some(Scope::Docs)]);
        assert!(s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Docs));
        assert_eq!(served_files(&s), vec!["README.md"]);
        assert_eq!(
            s.gathered.search_ms, 12,
            "search time covers both scans, not just the fallback"
        );
    }

    #[tokio::test]
    async fn empty_docs_falls_back_to_code() {
        let (s, calls) = run(
            filter(Scope::Docs),
            gathered(&["a.rs"], false, 1),
            gathered(&[], false, 1),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Docs), Some(Scope::Code)]);
        assert!(s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert_eq!(served_files(&s), vec!["a.rs"]);
    }

    #[tokio::test]
    async fn both_empty_reports_the_requested_scope_and_no_fallback() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], false, 1),
            gathered(&[], false, 1),
        )
        .await;
        assert_eq!(calls.len(), 2, "exactly one retry, never a loop");
        assert!(!s.fallback);
        assert_eq!(s.scope.as_ref().map(|f| f.scope()), Some(Scope::Code));
        assert!(s.gathered.base_chunks.is_empty());
        assert!(!s.gathered.warming);
    }

    /// A shard that is still loading is "retry shortly", not "this class is
    /// empty": falling back would mask the warming outcome.
    #[tokio::test]
    async fn warming_empty_result_never_triggers_fallback() {
        let (s, calls) = run(
            filter(Scope::Code),
            gathered(&[], true, 1),
            gathered(&["README.md"], false, 1),
        )
        .await;
        assert_eq!(calls, vec![Some(Scope::Code)]);
        assert!(!s.fallback);
        assert!(s.gathered.warming);
        assert!(s.gathered.base_chunks.is_empty());
    }

    /// If the fallback scan itself reports an incomplete index, surface the
    /// warming signal instead of a misleading "no results".
    #[tokio::test]
    async fn warming_during_fallback_scan_is_surfaced() {
        let (s, _) = run(
            filter(Scope::Code),
            gathered(&[], false, 1),
            gathered(&[], true, 1),
        )
        .await;
        assert!(!s.fallback);
        assert!(s.gathered.warming);
    }

    #[tokio::test]
    async fn no_requested_scope_means_no_fallback_even_when_empty() {
        let (s, calls) = run(
            None,
            gathered(&[], false, 1),
            gathered(&["README.md"], false, 1),
        )
        .await;
        assert_eq!(calls, vec![None]);
        assert!(!s.fallback);
        assert!(s.scope.is_none());
        assert!(s.gathered.base_chunks.is_empty());
    }
}

#[cfg(test)]
mod retry_tests {
    use std::cell::RefCell;

    use super::*;
    use crate::query::engine::{CodeResult, QueryResult, QueryTiming, RerankInfo};
    use crate::vector::Scope;

    fn code_result(file: &str) -> CodeResult {
        CodeResult {
            file: file.to_string(),
            line_start: 1,
            line_end: 2,
            score: 0.5,
            content: String::new(),
            symbol: None,
            symbol_kind: None,
            callers: None,
            caller_files: None,
            caller_names: vec![],
            callee_names: vec![],
            callees: None,
        }
    }

    /// `rejected`: the reranker actively rejected every candidate (a real
    /// response, no fallback, no skip reason) — the exact shape `mcp.rs` already
    /// reports as "No relevant code found".
    fn qr(files: &[&str], rejected: bool, scope: Option<Scope>, warming: bool) -> QueryResult {
        QueryResult {
            results: files.iter().map(|f| code_result(f)).collect(),
            pre_rerank_results: vec![],
            timing: QueryTiming {
                embed_ms: 0,
                search_ms: 0,
                graph_ms: 0,
                merge_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
            rerank: rejected.then(|| RerankInfo {
                raw_request: String::new(),
                raw_response: "{\"chunks\":[]}".to_string(),
                fallback_used: false,
                skip_reason: None,
            }),
            warming,
            graph_pending: false,
            scope: scope.map(|s| s.as_str().to_string()),
            scope_fallback: false,
        }
    }

    type Calls = RefCell<Vec<(Option<Scope>, bool)>>;

    async fn run(
        first: QueryResult,
        second: anyhow::Result<QueryResult>,
    ) -> (anyhow::Result<QueryResult>, Vec<(Option<Scope>, bool)>) {
        let calls: Calls = RefCell::new(Vec::new());
        let first = RefCell::new(Some(first));
        let second = RefCell::new(Some(second));
        let out = retry_on_rerank_rejection(|forced, allow_gather_fallback| {
            calls.borrow_mut().push((forced, allow_gather_fallback));
            let r = if calls.borrow().len() == 1 {
                Ok(first.borrow_mut().take().expect("first"))
            } else {
                second.borrow_mut().take().expect("at most one retry")
            };
            async move { r }
        })
        .await;
        (out, calls.into_inner())
    }

    #[tokio::test]
    async fn rejected_everything_retries_the_other_scope_and_labels_it() {
        let (out, calls) = run(
            qr(&[], true, Some(Scope::Code), false),
            Ok(qr(&["README.md"], false, Some(Scope::Docs), false)),
        )
        .await;
        let out = out.unwrap();
        assert_eq!(
            calls,
            vec![(None, true), (Some(Scope::Docs), false)],
            "retry is forced to the OTHER scope and must not fall back again"
        );
        assert_eq!(out.results.len(), 1);
        assert!(out.scope_fallback);
        assert_eq!(out.scope.as_deref(), Some("docs"));
    }

    #[tokio::test]
    async fn rejected_in_docs_retries_code() {
        let (out, calls) = run(
            qr(&[], true, Some(Scope::Docs), false),
            Ok(qr(&["a.rs"], false, Some(Scope::Code), false)),
        )
        .await;
        let out = out.unwrap();
        assert_eq!(calls[1], (Some(Scope::Code), false));
        assert!(out.scope_fallback);
        assert_eq!(out.scope.as_deref(), Some("code"));
    }

    #[tokio::test]
    async fn both_scopes_rejected_returns_the_original_answer() {
        let (out, calls) = run(
            qr(&[], true, Some(Scope::Code), false),
            Ok(qr(&[], true, Some(Scope::Docs), false)),
        )
        .await;
        let out = out.unwrap();
        assert_eq!(calls.len(), 2, "exactly one retry");
        assert!(!out.scope_fallback);
        assert_eq!(out.scope.as_deref(), Some("code"));
        assert!(out.results.is_empty());
        assert!(
            out.rerank.is_some(),
            "still the rejected-by-reranker answer"
        );
    }

    #[tokio::test]
    async fn a_failing_retry_never_turns_a_valid_answer_into_an_error() {
        let (out, _) = run(
            qr(&[], true, Some(Scope::Code), false),
            Err(anyhow::anyhow!("embed failed")),
        )
        .await;
        let out = out.unwrap();
        assert!(!out.scope_fallback);
        assert_eq!(out.scope.as_deref(), Some("code"));
    }

    #[tokio::test]
    async fn no_retry_when_the_first_attempt_has_results() {
        let (out, calls) = run(
            qr(&["a.rs"], false, Some(Scope::Code), false),
            Ok(qr(&["README.md"], false, Some(Scope::Docs), false)),
        )
        .await;
        assert_eq!(calls.len(), 1);
        assert_eq!(out.unwrap().results.len(), 1);
    }

    /// No candidates at all is the gather-level fallback's job, not this one.
    #[tokio::test]
    async fn no_retry_for_an_empty_result_the_reranker_never_saw() {
        let (_, calls) = run(
            qr(&[], false, Some(Scope::Code), false),
            Ok(qr(&["README.md"], false, Some(Scope::Docs), false)),
        )
        .await;
        assert_eq!(calls.len(), 1);
    }

    /// If the gather-level fallback already served the other scope (the requested
    /// one had nothing), bouncing back to it would only repeat an empty search.
    #[tokio::test]
    async fn no_retry_after_the_gather_level_fallback_already_switched_scope() {
        let mut first = qr(&[], true, Some(Scope::Docs), false);
        first.scope_fallback = true;
        let (out, calls) = run(first, Ok(qr(&["a.rs"], false, Some(Scope::Code), false))).await;
        assert_eq!(calls.len(), 1);
        assert!(out.unwrap().scope_fallback);
    }

    #[tokio::test]
    async fn no_retry_while_warming_or_for_an_unscoped_query() {
        let (_, calls) = run(
            qr(&[], true, Some(Scope::Code), true),
            Ok(qr(&["README.md"], false, Some(Scope::Docs), false)),
        )
        .await;
        assert_eq!(
            calls.len(),
            1,
            "warming is a retry-later state, not a reject"
        );

        let (_, calls) = run(
            qr(&[], true, None, false),
            Ok(qr(&["README.md"], false, Some(Scope::Docs), false)),
        )
        .await;
        assert_eq!(calls.len(), 1, "unscoped queries have no other scope");
    }

    #[tokio::test]
    async fn rerank_rejection_is_not_claimed_for_skipped_or_fallback_rerank() {
        let mut skipped = qr(&[], true, Some(Scope::Code), false);
        skipped.rerank.as_mut().unwrap().skip_reason = Some("no keys".into());
        assert!(!skipped.rerank_rejected_all());
        let mut used_fallback = qr(&[], true, Some(Scope::Code), false);
        used_fallback.rerank.as_mut().unwrap().fallback_used = true;
        assert!(!used_fallback.rerank_rejected_all());
        assert!(!qr(&["a.rs"], true, Some(Scope::Code), false).rerank_rejected_all());
        assert!(qr(&[], true, Some(Scope::Code), false).rerank_rejected_all());
    }
}
