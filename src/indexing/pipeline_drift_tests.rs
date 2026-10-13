//! Incremental-vs-full equivalence oracle on the REAL pipeline.
//!
//! WHY this file exists: the older oracle (`incremental_calls` in pipeline.rs)
//! seeds `raw_edge` rows by hand, so it never exercised the RAM-path full rebuild,
//! which did not persist `raw_edge`. That gap let three call-graph defects ship:
//! a no-change run after an incremental wiped `calls` down to recently edited
//! files, unchanged callers of a renamed symbol lost every outgoing edge, and
//! callers of a newly added symbol never got an edge. Every test here drives the
//! production `run()` end to end (full rebuild → edits → incremental → no-change
//! run) and requires the resulting `calls` set to equal a fresh full rebuild of
//! the final tree.

use super::*;
use crate::store::open_db;
use serde::Deserialize;
use tempfile::TempDir;

type CallSet = Vec<(String, String, String, String)>;

async fn calls_snapshot(db: &Surreal<Db>) -> CallSet {
    #[derive(Deserialize)]
    struct Row {
        in_file: String,
        out_file: String,
        in_name: Option<String>,
        out_name: Option<String>,
    }
    let rows: Vec<Row> = db
        .query("SELECT in_file, out_file, in_name, out_name FROM calls")
        .await
        .unwrap()
        .take(0)
        .unwrap();
    let mut v: CallSet = rows
        .into_iter()
        .map(|r| {
            (
                r.in_file,
                r.out_file,
                r.in_name.unwrap_or_default(),
                r.out_name.unwrap_or_default(),
            )
        })
        .collect();
    v.sort();
    v
}

fn repo_path(dir: &TempDir) -> String {
    dir.path().to_str().unwrap().replace('\\', "/")
}

fn write(dir: &TempDir, rel: &str, content: &str) {
    std::fs::write(dir.path().join(rel), content).unwrap();
}

/// The exact path key the walker stored for `rel`. A watcher change set must use
/// the same string, or deletes silently match nothing.
async fn stored_key(db: &Surreal<Db>, repo: &str, rel: &str) -> String {
    get_all_file_meta(db, repo)
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.path)
        .find(|p| p.replace('\\', "/").ends_with(&format!("/{rel}")))
        .unwrap_or_else(|| panic!("no file_meta for {rel}"))
}

fn pipeline_for(repo: &str) -> IndexPipeline {
    IndexPipeline::new(repo.to_string(), None).allow_no_client_mutation_for_test()
}

/// Fresh full rebuild of whatever is on disk now, in a throwaway DB.
async fn fresh_full_calls(repo: &str) -> CallSet {
    let home = TempDir::new().unwrap();
    let db = open_db(home.path(), repo, 0).await.unwrap();
    pipeline_for(repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("oracle full rebuild");
    calls_snapshot(&db).await
}

/// Watcher-path incremental for `changed`, then a no-change walk run (the boot
/// catch-up / MCP-readiness shape), then compare with a fresh full rebuild.
async fn assert_incremental_matches_full(
    db: &Surreal<Db>,
    repo: &str,
    changed: Vec<FileChange>,
    label: &str,
) {
    let p = pipeline_for(repo);
    p.run(db, Some(changed), false, None, None, None, &[], None)
        .await
        .expect("incremental run");
    let after_incremental = calls_snapshot(db).await;
    p.run(db, None, false, None, None, None, &[], None)
        .await
        .expect("no-change run");
    let after_noop = calls_snapshot(db).await;
    let expected = fresh_full_calls(repo).await;
    assert_eq!(
        after_incremental, expected,
        "[{label}] calls after incremental must equal a fresh full rebuild"
    );
    assert_eq!(
        after_noop, expected,
        "[{label}] a no-change run after an incremental must not change calls"
    );
}

/// Primary defect: full rebuild → body-only edit → no-change run used to replay
/// Phase 2 from the partial `raw_edge` table and drop every edge of files that
/// were not edited since the rebuild.
#[tokio::test]
async fn no_change_run_after_incremental_keeps_full_call_graph() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");
    write(&dir, "edit.rs", "fn qux() { foo(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let initial = calls_snapshot(&db).await;
    assert!(
        initial.len() >= 4,
        "fixture must produce cross-file edges (got {initial:?})"
    );

    let edit = stored_key(&db, &repo, "edit.rs").await;
    write(&dir, "edit.rs", "fn qux() { foo(); bar(); }\n");
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: edit,
            kind: ChangeKind::Modified,
        }],
        "body edit",
    )
    .await;
}

/// S1: renaming a symbol in lib.rs puts unchanged user.rs in the blast radius;
/// its calls were deleted and re-resolved from `raw_edge` rows it did not have,
/// losing even the edges to symbols that did not change.
#[tokio::test]
async fn rename_keeps_unchanged_callers_other_edges() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() {}\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let lib = stored_key(&db, &repo, "lib.rs").await;
    write(&dir, "lib.rs", "fn foo() {}\nfn bar_renamed() {}\n");
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: lib,
            kind: ChangeKind::Modified,
        }],
        "rename",
    )
    .await;
}

/// S2: unchanged user.rs already calls `later()`; adding `later` to lib.rs must
/// create the user.rs → lib.rs edge, as a full rebuild does.
#[tokio::test]
async fn added_symbol_gets_edges_from_unchanged_callers() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    write(&dir, "user.rs", "fn baz() { foo(); later(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let lib = stored_key(&db, &repo, "lib.rs").await;
    write(&dir, "lib.rs", "fn foo() {}\nfn later() {}\n");
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: lib,
            kind: ChangeKind::Modified,
        }],
        "add symbol",
    )
    .await;
}

/// Deleting the callee file: callers keep their remaining edges.
#[tokio::test]
async fn deleted_file_keeps_unchanged_callers_other_edges() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    write(&dir, "gone.rs", "fn bar() {}\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let gone = stored_key(&db, &repo, "gone.rs").await;
    std::fs::remove_file(dir.path().join("gone.rs")).unwrap();
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: gone,
            kind: ChangeKind::Deleted,
        }],
        "delete file",
    )
    .await;
}

async fn count_rows(db: &Surreal<Db>, table: &str) -> i64 {
    #[derive(Deserialize)]
    struct C {
        count: i64,
    }
    let rows: Vec<C> = db
        .query(format!("SELECT count() AS count FROM {table} GROUP ALL"))
        .await
        .unwrap()
        .take(0)
        .unwrap();
    rows.first().map(|r| r.count).unwrap_or(0)
}

/// D3/D4: a DB written by the old code — raw_edge holding only one file's rows,
/// `calls` already truncated by the lossy replay, `edges_resolved` stamped over
/// it, no `raw_edge_complete` — heals on the next no-change run WITHOUT an
/// embedding client, leaving chunks and file_meta untouched.
#[tokio::test]
async fn damaged_legacy_db_heals_without_embedding_client() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");
    write(&dir, "edit.rs", "fn qux() { foo(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let expected = calls_snapshot(&db).await;
    let chunks_before = count_rows(&db, "chunk").await;
    let meta_before = get_all_file_meta(&db, &repo).await.unwrap().len();

    // Reproduce the legacy damage.
    let edit = stored_key(&db, &repo, "edit.rs").await;
    db.query("DELETE FROM raw_edge WHERE from_file != $f")
        .bind(("f", edit.clone()))
        .await
        .unwrap();
    db.query("DELETE FROM calls WHERE in_file != $f")
        .bind(("f", edit))
        .await
        .unwrap();
    db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", RAW_EDGE_COMPLETE_KEY))
        .await
        .unwrap();
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_some());
    assert_ne!(
        calls_snapshot(&db).await,
        expected,
        "fixture must be damaged"
    );

    // No client, no test override: the heal must not need to embed anything.
    IndexPipeline::new(repo.clone(), None)
        .run(&db, Some(vec![]), false, None, None, None, &[], None)
        .await
        .expect("no-change run heals");

    assert_eq!(calls_snapshot(&db).await, expected, "calls healed");
    assert!(
        get_meta(&db, RAW_EDGE_COMPLETE_KEY)
            .await
            .unwrap()
            .is_some()
    );
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_some());
    assert_eq!(
        count_rows(&db, "chunk").await,
        chunks_before,
        "chunks untouched"
    );
    assert_eq!(
        get_all_file_meta(&db, &repo).await.unwrap().len(),
        meta_before,
        "file_meta untouched"
    );
}

/// Same legacy state, but the next trigger carries an edit: the incremental
/// must heal instead of re-resolving its blast radius from the partial table.
#[tokio::test]
async fn incremental_on_legacy_db_heals_instead_of_partial_resolve() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() {}\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    db.query("DELETE FROM raw_edge").await.unwrap();
    db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", RAW_EDGE_COMPLETE_KEY))
        .await
        .unwrap();

    let lib = stored_key(&db, &repo, "lib.rs").await;
    write(&dir, "lib.rs", "fn foo() {}\nfn bar_renamed() {}\n");
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: lib,
            kind: ChangeKind::Modified,
        }],
        "legacy rename",
    )
    .await;
    assert!(
        get_meta(&db, RAW_EDGE_COMPLETE_KEY)
            .await
            .unwrap()
            .is_some()
    );
}

/// D4: the v6→v7 migration clears both edge markers (stamping v7), and the next
/// no-change run heals `calls` — even when the v6 DB had them all stamped.
#[tokio::test]
async fn v6_to_v7_migration_routes_to_edge_heal() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let expected = calls_snapshot(&db).await;
    // A damaged v6 DB: truncated calls, markers stamped.
    db.query("DELETE FROM calls").await.unwrap();
    set_meta(&db, crate::store::DB_SCHEMA_VERSION_KEY, "6")
        .await
        .unwrap();

    crate::store::run_migration_v6_to_v7(&db).await.unwrap();
    assert_eq!(crate::store::read_db_schema_version(&db).await, 7);
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_none());
    assert!(
        get_meta(&db, RAW_EDGE_COMPLETE_KEY)
            .await
            .unwrap()
            .is_none()
    );

    IndexPipeline::new(repo.clone(), None)
        .run(&db, Some(vec![]), false, None, None, None, &[], None)
        .await
        .expect("post-migration no-change run heals");
    assert_eq!(calls_snapshot(&db).await, expected);
}

/// S3/D6: framework detection keys on package.json, which an edit to App.tsx
/// alone does not include. The incremental must keep App's framework (JSX)
/// edge, as a full rebuild produces it.
#[tokio::test]
async fn framework_edges_survive_single_file_incremental() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(
        &dir,
        "package.json",
        "{\"dependencies\": {\"react\": \"18.0.0\"}}\n",
    );
    write(
        &dir,
        "Button.tsx",
        "export function Button() {\n  return null;\n}\n",
    );
    write(
        &dir,
        "App.tsx",
        "export function App() {\n  return <Button />;\n}\n",
    );

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let initial = calls_snapshot(&db).await;
    assert!(
        initial
            .iter()
            .any(|(_, _, from, to)| from.ends_with("::App") && to.ends_with("::Button")),
        "fixture must produce the JSX framework edge (got {initial:?})"
    );

    let app = stored_key(&db, &repo, "App.tsx").await;
    write(
        &dir,
        "App.tsx",
        "// edited\nexport function App() {\n  return <Button />;\n}\n",
    );
    assert_incremental_matches_full(
        &db,
        &repo,
        vec![FileChange {
            path: app,
            kind: ChangeKind::Modified,
        }],
        "framework edit",
    )
    .await;
}

/// D5: a brand-new DB is stamped with the current schema version on open, so no
/// v1→current migration chain runs in the background against its first index.
#[tokio::test]
async fn fresh_db_is_stamped_and_spawns_no_migration() {
    let home = TempDir::new().unwrap();
    let repo = "/proj/fresh-stamp".to_string();
    let repo_dbs: crate::store::RepoDbMap =
        std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new()));
    let db = crate::store::get_or_open(&repo_dbs, home.path(), &repo, 0)
        .await
        .unwrap();
    assert_eq!(
        crate::store::read_db_schema_version(&db).await,
        crate::store::DB_SCHEMA_VERSION
    );
}

/// D5: a DB holding data but no version key is a genuine legacy DB and keeps
/// migrating; `wait_migration` returns only once that chain is done.
#[tokio::test]
async fn legacy_unversioned_db_migrates_and_wait_blocks_until_done() {
    let home = TempDir::new().unwrap();
    let repo = "/proj/legacy-stamp".to_string();
    let repo_dbs: crate::store::RepoDbMap =
        std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new()));
    let db = crate::store::get_or_open(&repo_dbs, home.path(), &repo, 0)
        .await
        .unwrap();
    // Turn it into an unversioned legacy DB with data (one handle: RocksDB's
    // per-directory LOCK is not released in-process on Windows).
    db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", crate::store::DB_SCHEMA_VERSION_KEY))
        .await
        .unwrap();
    db.query(
        "CREATE chunk SET file = '/proj/legacy-stamp/a.rs', line_start = 1,          line_end = 2, content = 'x', embedding = [0.1, 0.2], symbol_ref = NONE;",
    )
    .await
    .unwrap();
    let version = crate::store::stamp_if_fresh(&db).await.unwrap();
    assert_eq!(version, 1, "a DB with data is not fresh");

    crate::store::maybe_spawn_migration(
        repo_dbs.clone(),
        crate::store::normalize_repo_path(&repo),
        version,
    );
    crate::store::wait_migration(&repo).await;
    assert_eq!(
        crate::store::read_db_schema_version(&db).await,
        crate::store::DB_SCHEMA_VERSION,
        "wait_migration must return only after the chain stamped the version"
    );
}

/// A local embeddings endpoint answering every request with `status`.
async fn rejecting_voyage(
    status: axum::http::StatusCode,
) -> crate::embedding::voyage::VoyageClient {
    use axum::{Json, Router, routing::post};
    let app = Router::new().route(
        "/v1/embeddings",
        post(move || async move { (status, Json(serde_json::json!({"error": "key expired"}))) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    crate::embedding::voyage::VoyageClient::new(
        "test-model".to_string(),
        vec!["test-key".to_string()],
        Some(&format!("http://{addr}/v1")),
    )
    .unwrap()
}

/// S18/D19: a full rebuild whose embedding client fails fatally (e.g. 403 key
/// expired) must leave the existing index intact — it used to delete everything
/// first and then fail on the first uncached chunk, emptying the repo.
#[tokio::test]
async fn full_rebuild_with_dead_key_keeps_existing_index() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "user.rs", "fn baz() { foo(); bar(); }\n");

    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let calls_before = calls_snapshot(&db).await;
    let chunks_before = count_rows(&db, "chunk").await;
    let metas_before = get_all_file_meta(&db, &repo).await.unwrap().len();
    assert!(chunks_before > 0 && metas_before == 2);

    let dead = rejecting_voyage(axum::http::StatusCode::FORBIDDEN).await;
    let result = IndexPipeline::new(repo.clone(), Some(dead))
        .run(&db, None, true, None, None, None, &[], None)
        .await;
    let err = result.err().expect("a dead key must fail the rebuild");
    assert!(
        crate::indexing::retry_backoff::is_fatal_index_error(&err),
        "a dead key is fatal: no automatic retry may hammer it ({err:#})"
    );

    assert_eq!(
        count_rows(&db, "chunk").await,
        chunks_before,
        "chunks intact"
    );
    assert_eq!(
        get_all_file_meta(&db, &repo).await.unwrap().len(),
        metas_before,
        "file_meta intact"
    );
    assert_eq!(calls_snapshot(&db).await, calls_before, "calls intact");
}

/// A local embeddings endpoint whose reply is scripted per request:
/// `reply(request_index, n_inputs) -> (status, body)`.
async fn scripted_voyage<F>(reply: F) -> crate::embedding::voyage::VoyageClient
where
    F: Fn(usize, usize) -> (axum::http::StatusCode, serde_json::Value) + Send + Sync + 'static,
{
    use axum::{Json, Router, routing::post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = std::sync::Arc::new(AtomicUsize::new(0));
    let reply = std::sync::Arc::new(reply);
    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<serde_json::Value>| {
            let counter = counter.clone();
            let reply = reply.clone();
            async move {
                let n = body["input"].as_array().map(|a| a.len()).unwrap_or(0);
                let i = counter.fetch_add(1, Ordering::SeqCst);
                let (status, value) = reply(i, n);
                (status, Json(value))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    crate::embedding::voyage::VoyageClient::new(
        "test-model".to_string(),
        vec!["test-key".to_string()],
        Some(&format!("http://{addr}/v1")),
    )
    .unwrap()
}

/// `n` valid 4-d embeddings in Voyage's response shape.
fn ok_embeddings(n: usize) -> serde_json::Value {
    let data: Vec<serde_json::Value> = (0..n)
        .map(|i| serde_json::json!({"index": i, "embedding": [0.1, 0.2, 0.3, 0.4 + i as f32]}))
        .collect();
    serde_json::json!({"data": data})
}

/// A fatal embed error mid-run must fail the run even when the caller passed no
/// cancel token — Stage 2 used to record the error, find no token to cancel,
/// and let the run report success over the files it had just dropped.
#[tokio::test]
async fn fatal_embed_error_without_cancel_token_fails_the_run() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");

    // Request 0 (the preflight) succeeds; every later request is a 403.
    let client = scripted_voyage(|i, n| {
        if i == 0 {
            (axum::http::StatusCode::OK, ok_embeddings(n))
        } else {
            (
                axum::http::StatusCode::FORBIDDEN,
                serde_json::json!({"error": "key expired"}),
            )
        }
    })
    .await;
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let result = IndexPipeline::new(repo.clone(), Some(client))
        .run(&db, None, true, None, None, None, &[], None)
        .await;
    assert!(
        result.is_err(),
        "a fatal embed error must fail the run, got {:?}",
        result.map(|s| s.indexed_files)
    );
}

/// S11/D16: an out-of-order response is put back in input order by `index`.
#[tokio::test]
async fn voyage_response_is_reordered_by_index() {
    let client = scripted_voyage(|_, n| {
        let data: Vec<serde_json::Value> = (0..n)
            .rev()
            .map(|i| serde_json::json!({"index": i, "embedding": [i as f32 + 1.0, 0.0]}))
            .collect();
        (
            axum::http::StatusCode::OK,
            serde_json::json!({"data": data}),
        )
    })
    .await;
    let texts: Vec<String> = (0..3).map(|i| format!("t{i}")).collect();
    let embs = client
        .embed(&texts, crate::embedding::InputType::Document)
        .await
        .expect("embed");
    let firsts: Vec<f32> = embs.iter().map(|e| e[0]).collect();
    assert_eq!(
        firsts,
        vec![1.0, 2.0, 3.0],
        "vectors must follow input order"
    );
}

/// The stored-key spelling of `rel` under the repo (same prefix format as the
/// walker), for paths that are not files in file_meta (directories, new files).
async fn key_like(db: &Surreal<Db>, repo: &str, existing_rel: &str, rel: &str) -> String {
    let k = stored_key(db, repo, existing_rel).await;
    let base = &k[..k.len() - existing_rel.len()];
    format!("{base}{}", rel.replace('/', std::path::MAIN_SEPARATOR_STR))
}

async fn indexed_rels(db: &Surreal<Db>, repo: &str) -> Vec<String> {
    let mut v: Vec<String> = get_all_file_meta(db, repo)
        .await
        .unwrap()
        .into_iter()
        .map(|m| {
            let p = m.path.replace('\\', "/");
            let r = repo.replace('\\', "/");
            p.strip_prefix(&format!("{r}/")).unwrap_or(&p).to_string()
        })
        .collect();
    v.sort();
    v
}

/// S6/D11: a directory created (or moved in) arrives as ONE event for the
/// directory; its files must be indexed.
#[tokio::test]
async fn directory_create_event_indexes_its_files() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    std::fs::create_dir_all(dir.path().join("sub/deeper")).unwrap();
    write(&dir, "sub/a.rs", "fn a() { foo(); }\n");
    write(&dir, "sub/deeper/b.rs", "fn b() { a(); }\n");
    let sub = key_like(&db, &repo, "lib.rs", "sub").await;
    pipeline_for(&repo)
        .run(
            &db,
            Some(vec![FileChange {
                path: sub,
                kind: ChangeKind::Added,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("dir create run");
    assert_eq!(
        indexed_rels(&db, &repo).await,
        vec!["lib.rs", "sub/a.rs", "sub/deeper/b.rs"]
    );
}

/// S6/D11: deleting (or moving out) a directory arrives as ONE event for the
/// directory; every indexed file under it must be removed.
#[tokio::test]
async fn directory_delete_event_removes_its_files() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    std::fs::create_dir_all(dir.path().join("old/inner")).unwrap();
    write(&dir, "old/b.rs", "fn b() { foo(); }\n");
    write(&dir, "old/inner/c.rs", "fn c() {}\n");
    write(&dir, "older.rs", "fn older() {}\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let old = key_like(&db, &repo, "lib.rs", "old").await;
    std::fs::remove_dir_all(dir.path().join("old")).unwrap();
    pipeline_for(&repo)
        .run(
            &db,
            Some(vec![FileChange {
                path: old,
                kind: ChangeKind::Deleted,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("dir delete run");
    assert_eq!(
        indexed_rels(&db, &repo).await,
        vec!["lib.rs", "older.rs"],
        "files under the deleted dir are gone; the sibling sharing its prefix stays"
    );
    assert_eq!(count_rows(&db, "chunk").await, 2);
}

/// D11: Windows emits `Modify` on every parent of a changed child. A Modify on a
/// directory must not expand it (that would walk up to the repo root on every
/// batch); only create / rename-to does.
#[tokio::test]
async fn directory_modify_event_does_not_expand() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    std::fs::create_dir_all(dir.path().join("pkg")).unwrap();
    write(&dir, "pkg/x.rs", "fn x() {}\n");
    let pkg = key_like(&db, &repo, "lib.rs", "pkg").await;
    pipeline_for(&repo)
        .run(
            &db,
            Some(vec![FileChange {
                path: pkg,
                kind: ChangeKind::Modified,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("dir modify run");
    assert_eq!(indexed_rels(&db, &repo).await, vec!["lib.rs"]);
}

async fn pending_retry_rels(db: &Surreal<Db>, repo: &str) -> Vec<String> {
    let r = repo.replace('\\', "/");
    let mut v: Vec<String> = crate::store::ops::get_pending_retry(db)
        .await
        .unwrap()
        .into_iter()
        .map(|p| {
            let p = p.replace('\\', "/");
            p.strip_prefix(&format!("{r}/")).unwrap_or(&p).to_string()
        })
        .collect();
    v.sort();
    v
}

/// S7/D13: every incremental run folds the durable retry set into its change
/// set — so a file skipped earlier is retried even by a watcher run that never
/// mentions it — and drops entries whose file no longer exists.
#[tokio::test]
async fn pending_retry_is_folded_into_the_next_run_and_cleaned() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    // A file whose earlier attempt was skipped (no file_meta, queued for retry),
    // plus a queued file that has since been deleted.
    write(&dir, "skipped.rs", "fn skipped() { foo(); }\n");
    let skipped = key_like(&db, &repo, "lib.rs", "skipped.rs").await;
    let gone = key_like(&db, &repo, "lib.rs", "gone.rs").await;
    crate::store::ops::add_pending_retry(&db, &[skipped, gone])
        .await
        .unwrap();

    pipeline_for(&repo)
        .run(&db, Some(vec![]), false, None, None, None, &[], None)
        .await
        .expect("watcher run with an empty change set");
    assert_eq!(indexed_rels(&db, &repo).await, vec!["lib.rs", "skipped.rs"]);
    assert!(
        pending_retry_rels(&db, &repo).await.is_empty(),
        "retried and vanished entries are both cleared"
    );
}

/// S7/D13: a file that cannot be read mid-run (locked by another process, e.g.
/// an editor mid-save on Windows) used to lose its old data AND drop out of the
/// index silently — a watcher-path run never revisits it. It must be queued and
/// picked up by the next run once readable.
#[cfg(windows)]
#[tokio::test]
async fn unreadable_file_is_queued_and_indexed_once_readable() {
    use std::os::windows::fs::OpenOptionsExt;
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    write(&dir, "busy.rs", "fn busy() {}\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let busy = stored_key(&db, &repo, "busy.rs").await;

    write(&dir, "busy.rs", "fn busy() { foo(); }\n");
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0) // exclusive: every other open fails with a sharing violation
        .open(dir.path().join("busy.rs"))
        .unwrap();
    pipeline_for(&repo)
        .run(
            &db,
            Some(vec![FileChange {
                path: busy.clone(),
                kind: ChangeKind::Modified,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("run with the file locked");
    assert_eq!(pending_retry_rels(&db, &repo).await, vec!["busy.rs"]);
    drop(lock);

    pipeline_for(&repo)
        .run(&db, Some(vec![]), false, None, None, None, &[], None)
        .await
        .expect("next run");
    assert_eq!(indexed_rels(&db, &repo).await, vec!["busy.rs", "lib.rs"]);
    assert!(pending_retry_rels(&db, &repo).await.is_empty());
}

/// S20: a file with an indexable extension but binary content (e.g. a literal
/// NUL inside index.html's JS) is skipped — and used to get no file_meta, so
/// EVERY walk re-detected it as Added. No catch-up / poll was ever a no-change
/// run: each one was an incremental that hid the shard, invalidated the
/// persisted shard file and bumped the epoch. Once recorded, an unchanged binary
/// file must not count as a change; when it later becomes text it is indexed.
#[tokio::test]
async fn binary_file_does_not_turn_every_walk_into_an_incremental() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    std::fs::write(dir.path().join("page.html"), b"<script>'\x00'</script>\n").unwrap();
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let epoch_before = crate::store::ops::get_index_epoch(&db).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, false, None, None, None, &[], None)
        .await
        .expect("walk run");
    assert_eq!(
        crate::store::ops::get_index_epoch(&db).await.unwrap(),
        epoch_before,
        "an unchanged binary file must not trigger an incremental"
    );

    // Becomes text later → detected (mtime/size changed) and indexed.
    std::fs::write(dir.path().join("page.html"), "<p>hello</p>\n").unwrap();
    pipeline_for(&repo)
        .run(&db, None, false, None, None, None, &[], None)
        .await
        .expect("walk run after the file became text");
    let page = stored_key(&db, &repo, "page.html").await;
    let meta = get_all_file_meta(&db, &repo)
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.path == page)
        .unwrap();
    assert!(meta.chunk_count > 0, "now-text file is chunked");
}

/// Watcher run touching only `rel` (already stored) as Modified.
async fn watcher_touch(db: &Surreal<Db>, repo: &str, rel: &str) {
    let path = key_like(db, repo, "lib.rs", rel).await;
    pipeline_for(repo)
        .run(
            db,
            Some(vec![FileChange {
                path,
                kind: ChangeKind::Modified,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("watcher run");
}

/// Editing a NESTED ignore file must drop the already-stored files it now
/// excludes: no watcher event will ever arrive for those files themselves.
#[tokio::test]
async fn nested_ignore_edit_prunes_stored_files() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    std::fs::create_dir_all(dir.path().join(".git")).unwrap();
    std::fs::create_dir_all(dir.path().join("app/build-tdd")).unwrap();
    write(&dir, "lib.rs", "fn foo() {}\n");
    write(&dir, "app/main.rs", "fn main() { foo(); }\n");
    write(&dir, "app/build-tdd/gen.rs", "fn gen() { foo(); }\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    assert_eq!(
        indexed_rels(&db, &repo).await,
        vec!["app/build-tdd/gen.rs", "app/main.rs", "lib.rs"]
    );

    write(&dir, "app/.gitignore", "*build-*\n");
    watcher_touch(&db, &repo, "app/.gitignore").await;
    assert_eq!(
        indexed_rels(&db, &repo).await,
        vec!["app/main.rs", "lib.rs"],
        "the nested .gitignore edit must prune the artifact"
    );
}

/// A DB written before nested-ignore support (no prune marker) holds artifacts
/// the old root-only watcher filter leaked. The first watcher run prunes them
/// even though its batch never mentions an ignore file, then records the marker.
#[tokio::test]
async fn legacy_db_without_prune_marker_is_pruned_once() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    std::fs::create_dir_all(dir.path().join(".git")).unwrap();
    std::fs::create_dir_all(dir.path().join("app/build-tdd")).unwrap();
    write(&dir, "lib.rs", "fn foo() {}\n");
    write(&dir, "app/build-tdd/gen.rs", "fn gen() { foo(); }\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    pipeline_for(&repo)
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    assert!(get_meta(&db, IGNORE_PRUNE_KEY).await.unwrap().is_some());

    // Simulate the legacy state: artifact stored, nested rule present, no marker.
    write(&dir, "app/.gitignore", "*build-*\n");
    db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", IGNORE_PRUNE_KEY))
        .await
        .unwrap();

    write(&dir, "lib.rs", "fn foo() { }\n");
    watcher_touch(&db, &repo, "lib.rs").await;
    assert_eq!(indexed_rels(&db, &repo).await, vec!["lib.rs"]);
    assert!(
        get_meta(&db, IGNORE_PRUNE_KEY).await.unwrap().is_some(),
        "a successful run records the prune so later batches skip the O(stored) check"
    );
}

#[test]
fn incremental_phase2_budget_needs_both_size_and_share() {
    let min = INCREMENTAL_PHASE2_FULL_FALLBACK_MIN_FILES;
    assert!(
        !exceeds_incremental_phase2_budget(min - 1, min),
        "small sets stay scoped"
    );
    assert!(exceeds_incremental_phase2_budget(min, min * 2));
    assert!(
        !exceeds_incremental_phase2_budget(min, min * 100),
        "a large set in a much larger repo stays scoped"
    );
}

/// Defining a name that hundreds of files call pulls them all into the blast
/// radius (direction 2). Past the budget the run replays the full Phase 2 —
/// and the call graph must still equal a fresh full rebuild.
#[tokio::test]
async fn oversized_blast_radius_falls_back_to_full_phase2() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\n");
    let callers = INCREMENTAL_PHASE2_FULL_FALLBACK_MIN_FILES + 4;
    for i in 0..callers {
        write(
            &dir,
            &format!("c{i}.rs"),
            &format!("fn c{i}() {{ target(); foo(); }}\n"),
        );
    }
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let p = pipeline_for(&repo);
    p.run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    write(&dir, "lib.rs", "fn foo() {}\nfn target() {}\n");
    let lib = stored_key(&db, &repo, "lib.rs").await;
    let stats = p
        .run(
            &db,
            Some(vec![FileChange {
                path: lib,
                kind: ChangeKind::Modified,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("incremental run");
    assert!(
        stats.incr_p2_full_fallback,
        "resolve set {} must trigger the fallback",
        stats.incr_resolve_set_size
    );
    assert_eq!(calls_snapshot(&db).await, fresh_full_calls(&repo).await);
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_some());
}

/// Cancel used to be ignored for the whole incremental Phase 2 (the UI's Cancel
/// did nothing for minutes). It must abort between batches.
#[tokio::test]
async fn incremental_phase2_honors_cancel() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let p = pipeline_for(&repo);
    p.run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");

    let lib = stored_key(&db, &repo, "lib.rs").await;
    let token = CancellationToken::new();
    token.cancel();
    let err = p
        .resolve_edges_incremental_with(&db, &[lib], &[], &[], None, None, Some(&token))
        .await
        .expect_err("a cancelled token must abort");
    assert!(matches!(
        err.downcast_ref::<PipelineAbort>(),
        Some(PipelineAbort::Cancelled)
    ));
}

/// The state a crash or cancel mid-Phase-2 leaves since the identity marker
/// commits before Phase 2: identity present, `edges_resolved` absent, `calls`
/// partial. The next WATCHER run (with changes, not the no-change replay path)
/// must replay the full Phase 2 — re-resolving only its own blast radius would
/// stamp `edges_resolved` over a graph still missing the earlier edges — and
/// must not re-embed anything.
#[tokio::test]
async fn run_after_interrupted_phase2_replays_full_graph_without_reembedding() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "a.rs", "fn a() { foo(); bar(); }\n");
    write(&dir, "b.rs", "fn b() { a(); }\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let p = pipeline_for(&repo);
    p.run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    let expected = fresh_full_calls(&repo).await;
    assert_eq!(calls_snapshot(&db).await, expected);

    // Interrupted Phase 2: the graph of a.rs is gone and the marker is unset.
    let a = stored_key(&db, &repo, "a.rs").await;
    db.query("DELETE FROM calls WHERE in_file = $f")
        .bind(("f", a))
        .await
        .unwrap();
    db.query("DELETE FROM index_meta WHERE key = $k")
        .bind(("k", EDGES_RESOLVED_KEY))
        .await
        .unwrap();
    let chunks_before = count_rows(&db, "chunk").await;

    write(&dir, "b.rs", "fn b() { a(); }\n// edited\n");
    let b = stored_key(&db, &repo, "b.rs").await;
    let stats = p
        .run(
            &db,
            Some(vec![FileChange {
                path: b,
                kind: ChangeKind::Modified,
            }]),
            false,
            None,
            None,
            None,
            &[],
            None,
        )
        .await
        .expect("watcher run");
    assert!(
        stats.incr_p2_full_fallback,
        "a partial prior graph must be replayed in full"
    );
    assert_eq!(calls_snapshot(&db).await, expected);
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_some());
    assert_eq!(
        count_rows(&db, "chunk").await,
        chunks_before,
        "no full rebuild"
    );
}

/// The identity marker commits once the vectors are durable, BEFORE Phase 2. A
/// cancel (or crash) inside Phase 2 used to roll the published vectors back,
/// delete the marker and set `needs_rebuild` — a full re-embed for a graph-only
/// interruption, and meanwhile every warm read "no identity" as a mismatch. Now
/// only `edges_resolved` stays unset and the next run replays Phase 2.
#[tokio::test]
async fn phase2_cancel_keeps_identity_and_vectors() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let repo = repo_path(&dir);
    write(&dir, "lib.rs", "fn foo() {}\nfn bar() { foo(); }\n");
    write(&dir, "a.rs", "fn a() { foo(); }\n");
    write(&dir, "b.rs", "fn b() { bar(); }\n");
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let client = scripted_voyage(|_, n| (axum::http::StatusCode::OK, ok_embeddings(n))).await;
    let identity = EmbeddingIdentity::from_client(&client).as_key_string();
    let p = IndexPipeline::new(repo.clone(), Some(client));
    p.run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("initial full rebuild");
    assert_eq!(
        get_meta(&db, EMBEDDING_IDENTITY_KEY)
            .await
            .unwrap()
            .as_deref(),
        Some(identity.as_str())
    );

    // A delete-only batch embeds nothing, so the pre-cancelled token is first
    // observed inside Phase 2.
    let b = stored_key(&db, &repo, "b.rs").await;
    std::fs::remove_file(dir.path().join("b.rs")).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let err = p
        .run(
            &db,
            Some(vec![FileChange {
                path: b,
                kind: ChangeKind::Deleted,
            }]),
            false,
            None,
            None,
            None,
            &[],
            Some(token),
        )
        .await
        .err()
        .expect("cancelled inside Phase 2");
    assert!(matches!(
        err.downcast_ref::<PipelineAbort>(),
        Some(PipelineAbort::Cancelled)
    ));
    assert_eq!(
        get_meta(&db, EMBEDDING_IDENTITY_KEY)
            .await
            .unwrap()
            .as_deref(),
        Some(identity.as_str()),
        "the committed vectors keep their identity"
    );
    assert!(get_meta(&db, "needs_rebuild").await.unwrap().is_none());
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_none());

    p.run(&db, None, false, None, None, None, &[], None)
        .await
        .expect("next run replays Phase 2");
    assert_eq!(calls_snapshot(&db).await, fresh_full_calls(&repo).await);
    assert!(get_meta(&db, EDGES_RESOLVED_KEY).await.unwrap().is_some());
}

/// Mirror every indexable file of `src` into `dst` (same relative paths), so a
/// benchmark can edit files without touching the real repository.
fn mirror_indexable(src: &str, dst: &std::path::Path) -> usize {
    let files = crate::indexing::walker::walk_repo(src);
    for f in &files {
        let rel = std::path::Path::new(f)
            .strip_prefix(src)
            .expect("walked path under root");
        let to = dst.join(rel);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(f, &to).unwrap();
    }
    files.len()
}

/// ACCEPTANCE BENCH (notepad-ade, ~7 min of incremental Phase 2 for a 23-file
/// change before the per-file read + to_name index): mirror the repo, full
/// rebuild, then time incremental Phase 2 for 40 edited C/C++ files in two
/// shapes — body-only edits (resolve set = the 40 files) and edits that ADD a
/// commonly-called name (direction-2 expansion). Targets: < 5 s each, and never
/// slower than the full Phase 2 of the same repo. Run with
/// `cargo test --release --lib notepad_incremental_phase2_bench -- --ignored --nocapture`
/// (`NOTEPAD_ADE_REPO` overrides the source path).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn notepad_incremental_phase2_bench() {
    // Phase-level PERF lines on stderr (RUST_LOG overrides).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();
    let src = std::env::var("NOTEPAD_ADE_REPO")
        .unwrap_or_else(|_| "D:/projects/cpp/notepad-ade".to_string());
    if !std::path::Path::new(&src).exists() {
        eprintln!("SKIP: {src} not present");
        return;
    }
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let mirrored = mirror_indexable(&src, dir.path());
    let repo = repo_path(&dir);
    let db = open_db(home.path(), &repo, 0).await.unwrap();
    let p = pipeline_for(&repo);

    let full = p
        .run(&db, None, true, None, None, None, &[], None)
        .await
        .expect("full rebuild");
    eprintln!(
        "BENCH full rebuild: files={mirrored} phase2_ms={} edges={}",
        full.phase2_ms, full.phase2_edges_written
    );

    let mut targets: Vec<String> = get_all_file_meta(&db, &repo)
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.path)
        .filter(|p| {
            let p = p.to_ascii_lowercase();
            p.ends_with(".cpp") || p.ends_with(".h") || p.ends_with(".c")
        })
        .collect();
    targets.sort();
    let step = (targets.len() / 40).max(1);
    let targets: Vec<String> = targets.into_iter().step_by(step).take(40).collect();
    assert_eq!(targets.len(), 40, "need 40 C/C++ files");

    let edit_and_run = |label: &'static str, suffix: &'static str| {
        let targets = targets.clone();
        let db = db.clone();
        let p = pipeline_for(&repo);
        async move {
            let changes: Vec<FileChange> = targets
                .iter()
                .map(|t| {
                    let mut f = std::fs::OpenOptions::new().append(true).open(t).unwrap();
                    std::io::Write::write_all(&mut f, suffix.as_bytes()).unwrap();
                    FileChange {
                        path: t.clone(),
                        kind: ChangeKind::Modified,
                    }
                })
                .collect();
            let s = p
                .run(&db, Some(changes), false, None, None, None, &[], None)
                .await
                .expect("incremental run");
            eprintln!(
                "BENCH {label}: phase2_ms={} resolve_set={} dir2_ms={} delete_ms={} reresolve_ms={} full_fallback={}",
                s.incr_phase2_total_ms,
                s.incr_resolve_set_size,
                s.incr_p2_dir2_scan_ms,
                s.incr_p2_delete_calls_ms,
                s.incr_p2_reresolve_ms,
                s.incr_p2_full_fallback
            );
            s.incr_phase2_total_ms
        }
    };

    let body_only = edit_and_run("body-only x40", "\n// bench edit\n").await;
    let added = edit_and_run(
        "added common names x40",
        "\nstatic int size() { return 0; }\nstatic void init() {}\n",
    )
    .await;

    // Correctness first, so a timing miss never hides a wrong graph.
    assert_eq!(
        calls_snapshot(&db).await,
        fresh_full_calls(&repo).await,
        "the incremental graph must equal a fresh full rebuild"
    );
    assert!(
        body_only < 5_000,
        "body-only incremental Phase 2 took {body_only} ms"
    );
    // Allow run-to-run noise on top of the measured full pass.
    let budget = 5_000u64.max(full.phase2_ms * 5 / 4);
    assert!(
        added <= budget,
        "added-name incremental Phase 2 took {added} ms (full Phase 2: {} ms)",
        full.phase2_ms
    );
}
