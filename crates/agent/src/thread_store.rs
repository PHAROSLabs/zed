use crate::{DbThread, DbThreadMetadata, ThreadsDatabase};
use agent_client_protocol::v2::schema as acp_v2;
use anyhow::{Result, anyhow};
use collections::HashMap;
use futures::{FutureExt, future::Shared};
use gpui::{App, Context, Entity, Global, Task, prelude::*};
use util::path_list::PathList;

struct GlobalThreadStore(Entity<ThreadStore>);

impl Global for GlobalThreadStore {}

pub struct ThreadStore {
    threads: Vec<DbThreadMetadata>,
    self_costs: HashMap<acp_v2::SessionId, f64>,
    child_costs_by_parent: HashMap<acp_v2::SessionId, HashMap<acp_v2::SessionId, f64>>,
    reload_task: Shared<Task<()>>,
}

impl ThreadStore {
    pub fn init_global(cx: &mut App) {
        let thread_store = cx.new(|cx| Self::new(cx));
        cx.set_global(GlobalThreadStore(thread_store));
    }

    pub fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalThreadStore>().0.clone()
    }

    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalThreadStore>().map(|g| g.0.clone())
    }

    pub fn new(cx: &mut Context<Self>) -> Self {
        let reload_task = Self::spawn_reload(cx);
        Self {
            threads: Vec::new(),
            self_costs: HashMap::default(),
            child_costs_by_parent: HashMap::default(),
            reload_task,
        }
    }

    pub fn thread_cost_usd(&self, session_id: &acp_v2::SessionId) -> Option<f64> {
        let self_cost = self.self_costs.get(session_id).copied().unwrap_or(0.0);
        let sub_cost: f64 = self
            .child_costs_by_parent
            .get(session_id)
            .map(|children| children.values().copied().sum())
            .unwrap_or(0.0);
        let total = self_cost + sub_cost;
        (total > 0.0).then_some(total)
    }

    pub fn live_thread_cost_usd(
        &self,
        session_id: &acp_v2::SessionId,
        live_self_cost: f64,
        running_subagents: &[(acp_v2::SessionId, f64)],
    ) -> Option<f64> {
        let mut children = self
            .child_costs_by_parent
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        for (sub_id, sub_cost) in running_subagents {
            children.insert(sub_id.clone(), *sub_cost);
        }
        let sub_cost: f64 = children.values().copied().sum();
        let total = live_self_cost + sub_cost;
        (total > 0.0).then_some(total)
    }

    /// Resolves when the most recently initiated reload has completed.
    /// Callers that need to read `entries()` and can't tolerate the initial
    /// empty state must await this before reading.
    pub fn reload_task(&self) -> Shared<Task<()>> {
        self.reload_task.clone()
    }

    pub fn thread_from_session_id(
        &self,
        session_id: &acp_v2::SessionId,
    ) -> Option<&DbThreadMetadata> {
        self.threads.iter().find(|thread| &thread.id == session_id)
    }

    pub fn load_thread(
        &mut self,
        id: acp_v2::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<Option<DbThread>>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.background_spawn(async move {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            database.load_thread(id).await
        })
    }

    pub fn save_thread(
        &mut self,
        id: acp_v2::SessionId,
        thread: crate::DbThread,
        folder_paths: PathList,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            database.save_thread(id, thread, folder_paths).await?;
            this.update(cx, |this, cx| this.reload(cx))
        })
    }

    pub fn delete_thread(
        &mut self,
        id: acp_v2::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            database.delete_thread(id.clone()).await?;
            this.update(cx, |this, cx| this.reload(cx))
        })
    }

    pub fn delete_threads(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            database.delete_threads().await?;
            this.update(cx, |this, cx| this.reload(cx))
        })
    }

    pub fn reload(&mut self, cx: &mut Context<Self>) {
        self.reload_task = Self::spawn_reload(cx);
    }

    fn spawn_reload(cx: &mut Context<Self>) -> Shared<Task<()>> {
        let database_connection = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let Ok(database) = database_connection.await.map_err(|err| anyhow!(err)) else {
                return;
            };
            let Ok(all_threads) = database.list_threads().await else {
                return;
            };
            let (self_costs, child_costs_by_parent) =
                database.list_thread_costs().await.unwrap_or_default();
            this.update(cx, |this, cx| {
                this.threads.clear();
                for thread in all_threads {
                    if thread.parent_session_id.is_some() {
                        continue;
                    }
                    this.threads.push(thread);
                }
                this.self_costs = self_costs;
                this.child_costs_by_parent = child_costs_by_parent;
                cx.notify();
            })
            .ok();
        })
        .shared()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn entries(&self) -> impl Iterator<Item = DbThreadMetadata> + '_ {
        self.threads.iter().cloned()
    }

    pub fn entry_ids(&self) -> impl Iterator<Item = acp_v2::SessionId> + '_ {
        self.threads.iter().map(|t| t.id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};
    use collections::HashMap;
    use gpui::TestAppContext;
    use std::sync::Arc;

    fn session_id(value: &str) -> acp_v2::SessionId {
        acp_v2::SessionId::new(Arc::<str>::from(value))
    }

    fn make_thread(title: &str, updated_at: DateTime<Utc>) -> DbThread {
        DbThread {
            title: title.to_string().into(),
            messages: Vec::new(),
            updated_at,
            detailed_summary: None,
            initial_project_snapshot: None,
            cumulative_token_usage: Default::default(),
            request_token_usage: HashMap::default(),
            model: None,
            profile: None,
            subagent_context: None,
            speed: None,
            thinking_enabled: false,
            thinking_effort: None,
            draft_prompt: None,
            ui_scroll_position: None,
            sandboxed_terminal_temp_dir: None,
            sandbox_grants: Default::default(),
        }
    }

    #[gpui::test]
    async fn test_entries_are_sorted_by_updated_at(cx: &mut TestAppContext) {
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        cx.run_until_parked();

        let older_id = session_id("thread-a");
        let newer_id = session_id("thread-b");

        let older_thread = make_thread(
            "Thread A",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );
        let newer_thread = make_thread(
            "Thread B",
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        );

        let save_older = thread_store.update(cx, |store, cx| {
            store.save_thread(older_id.clone(), older_thread, PathList::default(), cx)
        });
        save_older.await.unwrap();

        let save_newer = thread_store.update(cx, |store, cx| {
            store.save_thread(newer_id.clone(), newer_thread, PathList::default(), cx)
        });
        save_newer.await.unwrap();

        cx.run_until_parked();

        let entries: Vec<_> = thread_store.read_with(cx, |store, _cx| store.entries().collect());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, newer_id);
        assert_eq!(entries[1].id, older_id);
    }

    #[gpui::test]
    async fn test_delete_threads_clears_entries(cx: &mut TestAppContext) {
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        cx.run_until_parked();

        let thread_id = session_id("thread-a");
        let thread = make_thread(
            "Thread A",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );

        let save_task = thread_store.update(cx, |store, cx| {
            store.save_thread(thread_id, thread, PathList::default(), cx)
        });
        save_task.await.unwrap();

        cx.run_until_parked();
        assert!(!thread_store.read_with(cx, |store, _cx| store.is_empty()));

        let delete_task = thread_store.update(cx, |store, cx| store.delete_threads(cx));
        delete_task.await.unwrap();
        cx.run_until_parked();

        assert!(thread_store.read_with(cx, |store, _cx| store.is_empty()));
    }

    #[gpui::test]
    async fn test_delete_thread_removes_only_target(cx: &mut TestAppContext) {
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        cx.run_until_parked();

        let first_id = session_id("thread-a");
        let second_id = session_id("thread-b");

        let first_thread = make_thread(
            "Thread A",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );
        let second_thread = make_thread(
            "Thread B",
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        );

        let save_first = thread_store.update(cx, |store, cx| {
            store.save_thread(first_id.clone(), first_thread, PathList::default(), cx)
        });
        save_first.await.unwrap();
        let save_second = thread_store.update(cx, |store, cx| {
            store.save_thread(second_id.clone(), second_thread, PathList::default(), cx)
        });
        save_second.await.unwrap();
        cx.run_until_parked();

        let delete_task =
            thread_store.update(cx, |store, cx| store.delete_thread(first_id.clone(), cx));
        delete_task.await.unwrap();
        cx.run_until_parked();

        let entries: Vec<_> = thread_store.read_with(cx, |store, _cx| store.entries().collect());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, second_id);
    }

    #[gpui::test]
    async fn test_save_thread_refreshes_ordering(cx: &mut TestAppContext) {
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        cx.run_until_parked();

        let first_id = session_id("thread-a");
        let second_id = session_id("thread-b");

        let first_thread = make_thread(
            "Thread A",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );
        let second_thread = make_thread(
            "Thread B",
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        );

        let save_first = thread_store.update(cx, |store, cx| {
            store.save_thread(first_id.clone(), first_thread, PathList::default(), cx)
        });
        save_first.await.unwrap();
        let save_second = thread_store.update(cx, |store, cx| {
            store.save_thread(second_id.clone(), second_thread, PathList::default(), cx)
        });
        save_second.await.unwrap();
        cx.run_until_parked();

        let updated_first = make_thread(
            "Thread A",
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        );
        let update_task = thread_store.update(cx, |store, cx| {
            store.save_thread(first_id.clone(), updated_first, PathList::default(), cx)
        });
        update_task.await.unwrap();
        cx.run_until_parked();

        let entries: Vec<_> = thread_store.read_with(cx, |store, _cx| store.entries().collect());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, first_id);
        assert_eq!(entries[1].id, second_id);
    }

    #[gpui::test]
    async fn test_thread_cost_aggregation_with_subagents(cx: &mut TestAppContext) {
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        cx.run_until_parked();

        let parent_id = session_id("parent-thread");
        let sub_id = session_id("subagent-thread");

        let mut parent_thread = make_thread(
            "Parent Thread",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );
        parent_thread.model = Some(crate::DbLanguageModel {
            provider: "Regulaido Gateway".into(),
            model: "gemini-4-argon".into(),
        });
        parent_thread.cumulative_token_usage = language_model::TokenUsage {
            input_tokens: 100_000,
            cache_read_input_tokens: 500_000,
            cache_creation_input_tokens: 0,
            output_tokens: 10_000,
        };
        // gemini-4-argon: (2.00, 0.10, 10.00) -> 0.20 + 0.05 + 0.10 = $0.35

        let mut sub_thread = make_thread(
            "Subagent Thread",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 1, 0).unwrap(),
        );
        sub_thread.model = Some(crate::DbLanguageModel {
            provider: "Regulaido Gateway".into(),
            model: "gemini-3.8-flash".into(),
        });
        sub_thread.subagent_context = Some(crate::SubagentContext {
            parent_thread_id: parent_id.clone(),
            depth: 1,
            outcomes: Vec::new(),
        });
        sub_thread.cumulative_token_usage = language_model::TokenUsage {
            input_tokens: 100_000,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            output_tokens: 20_000,
        };
        // gemini-3.8-flash: (0.75, 0.075, 3.75) -> 0.075 + 0.075 = $0.15

        let save_parent = thread_store.update(cx, |store, cx| {
            store.save_thread(parent_id.clone(), parent_thread, PathList::default(), cx)
        });
        save_parent.await.unwrap();

        let save_sub = thread_store.update(cx, |store, cx| {
            store.save_thread(sub_id.clone(), sub_thread, PathList::default(), cx)
        });
        save_sub.await.unwrap();
        cx.run_until_parked();

        let total_cost = thread_store
            .read_with(cx, |store, _cx| store.thread_cost_usd(&parent_id))
            .expect("parent thread should have cost");
        assert!((total_cost - 0.50).abs() < 1e-6, "got {total_cost}");
        assert_eq!(
            crate::format_cost_usd(total_cost).as_deref(),
            Some("$0.50")
        );
    }
}
