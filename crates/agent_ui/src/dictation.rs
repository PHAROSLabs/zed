#![cfg_attr(not(any(feature = "audio", test)), allow(dead_code))]

use std::path::PathBuf;
use std::sync::Arc;

use acp_thread::{AgentThreadEntry, MessageContent};
use agent_client_protocol::v2::schema as acp_v2;
use anyhow::{Context as _, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::Deserialize;
use serde_json::json;

const DEFAULT_GATEWAY_ENDPOINT: &str = "https://llm.regulaido.ai";
const DICTATION_MODEL: &str = "gemini-4-hydrogen-priority";
const MAX_TURNS_IN_CONTEXT: usize = 4;
const MAX_CHARS_PER_TURN: usize = 320;
const MAX_TOTAL_TURN_CONTEXT_CHARS: usize = 1_400;
const MAX_DRAFT_CONTEXT_CHARS: usize = 400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    pub endpoint: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DictationContext {
    pub workspace_roots: Vec<String>,
    pub active_file: Option<String>,
    pub thread_title: Option<String>,
    pub recent_turns: Vec<(String, String)>,
    pub draft_text: String,
}

#[derive(Debug, Deserialize)]
struct GatewayConfigFile {
    endpoint: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Option<Vec<ChatCompletionChoice>>,
    error: Option<ChatCompletionError>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    finish_reason: Option<String>,
    message: Option<ChatCompletionMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionError {
    message: Option<String>,
}

pub fn resolve_gateway_config() -> Result<GatewayConfig> {
    if let Ok(api_key) = std::env::var("REGULAIDO_GATEWAY_API_KEY") {
        let trimmed_key = api_key.trim().to_string();
        if !trimmed_key.is_empty() {
            let endpoint = std::env::var("REGULAIDO_GATEWAY_ENDPOINT")
                .ok()
                .map(|value| normalize_endpoint(&value))
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| DEFAULT_GATEWAY_ENDPOINT.to_string());
            return Ok(GatewayConfig {
                endpoint,
                api_key: trimmed_key,
            });
        }
    }

    let Some(config_path) = gateway_config_file_path() else {
        bail!(
            "Regulaido Gateway API key not configured. Set REGULAIDO_GATEWAY_API_KEY or run scripts/llm-gateway/setup.sh."
        );
    };

    let raw = std::fs::read_to_string(&config_path).with_context(|| {
        format!(
            "Failed to read Regulaido Gateway config at {}",
            config_path.display()
        )
    })?;
    parse_gateway_config_json(&raw)
}

fn gateway_config_file_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("regulaido")
            .join("llm_gateway.json"),
    )
}

fn normalize_endpoint(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_string()
}

pub fn parse_gateway_config_json(raw: &str) -> Result<GatewayConfig> {
    let parsed: GatewayConfigFile =
        serde_json::from_str(raw).context("Invalid Regulaido Gateway config JSON")?;
    let api_key = parsed
        .api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .context("Regulaido Gateway config is missing api_key")?;
    let endpoint = parsed
        .endpoint
        .map(|value| normalize_endpoint(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_GATEWAY_ENDPOINT.to_string());
    Ok(GatewayConfig { endpoint, api_key })
}

pub fn extract_recent_turns_from_entries(entries: &[AgentThreadEntry]) -> Vec<(String, String)> {
    let mut collected_reversed: Vec<(String, String)> = Vec::new();
    let mut total_chars = 0usize;

    for entry in entries.iter().rev() {
        if collected_reversed.len() >= MAX_TURNS_IN_CONTEXT
            || total_chars >= MAX_TOTAL_TURN_CONTEXT_CHARS
        {
            break;
        }

        let (role, raw_text) = match entry {
            AgentThreadEntry::UserMessage(message) => {
                ("User", extract_plain_text(&message.content))
            }
            AgentThreadEntry::AssistantMessage(message) => {
                let mut combined = String::new();
                for chunk in &message.chunks {
                    if let acp_thread::AssistantMessageChunk::Message { block, .. } = chunk {
                        let text = extract_plain_text(block);
                        if !text.is_empty() {
                            if !combined.is_empty() {
                                combined.push(' ');
                            }
                            combined.push_str(&text);
                        }
                    }
                }
                ("Assistant", combined)
            }
            AgentThreadEntry::SessionMessage(_)
            | AgentThreadEntry::ToolCall(_)
            | AgentThreadEntry::Elicitation(_)
            | AgentThreadEntry::ContextCompaction(_) => continue,
        };

        let normalized = collapse_whitespace(&raw_text);
        if normalized.is_empty() {
            continue;
        }

        let remaining_budget = MAX_TOTAL_TURN_CONTEXT_CHARS.saturating_sub(total_chars);
        let limit = MAX_CHARS_PER_TURN.min(remaining_budget);
        if limit == 0 {
            break;
        }
        let excerpt = truncate_head_chars(&normalized, limit);
        total_chars = total_chars.saturating_add(excerpt.chars().count());
        collected_reversed.push((role.to_string(), excerpt));
    }

    collected_reversed.reverse();
    collected_reversed
}

fn extract_plain_text(content: &MessageContent) -> String {
    let mut text = String::new();
    for block in content.source_blocks() {
        if let acp_v2::ContentBlock::Text(text_content) = block {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&text_content.text);
        }
    }
    text
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_head_chars(input: &str, max_chars: usize) -> String {
    let count = input.chars().count();
    if count <= max_chars {
        return input.to_string();
    }
    if max_chars <= 3 {
        return input.chars().take(max_chars).collect();
    }
    let mut truncated: String = input.chars().take(max_chars - 3).collect();
    truncated.push_str("...");
    truncated
}

pub fn truncate_tail_chars(input: &str, max_chars: usize) -> String {
    let count = input.chars().count();
    if count <= max_chars {
        return input.to_string();
    }
    if max_chars <= 3 {
        return input
            .chars()
            .skip(count.saturating_sub(max_chars))
            .collect();
    }
    let keep = max_chars - 3;
    let tail: String = input.chars().skip(count.saturating_sub(keep)).collect();
    format!("...{tail}")
}

pub fn format_dictation_system_prompt(context: &DictationContext) -> String {
    let mut prompt = String::from(
        "You are a speech-to-text transcription engine for software developers. \
         Transcribe technical vocabulary, file paths, symbol names, package names, programming terms, syntax, and acronyms precisely. \
         Do NOT transcribe filler sounds, hesitation noises, or vocal pauses such as 'um', 'uh', 'ähm', 'äh', 'mhm', 'hmm', or 'er'. \
         Do NOT transcribe non-verbal sounds or acoustic descriptions like (clears throat), (coughs), or [laughter]. \
         Do not summarize, explain, or answer questions. Output ONLY the clean spoken words.",
    );

    let mut context_lines = Vec::new();
    if !context.workspace_roots.is_empty() {
        context_lines.push(format!("Workspace: {}", context.workspace_roots.join(", ")));
    }
    if let Some(active_file) = context.active_file.as_deref()
        && !active_file.trim().is_empty()
    {
        context_lines.push(format!("Active file: {}", active_file.trim()));
    }
    if let Some(title) = context.thread_title.as_deref()
        && !title.trim().is_empty()
    {
        context_lines.push(format!("Thread title: {}", title.trim()));
    }
    if !context.recent_turns.is_empty() {
        context_lines.push("Recent conversation:".to_string());
        for (role, excerpt) in &context.recent_turns {
            context_lines.push(format!("- {role}: {excerpt}"));
        }
    }
    let trimmed_draft = context.draft_text.trim();
    if !trimmed_draft.is_empty() {
        context_lines.push(format!(
            "Current draft before cursor: {}",
            truncate_tail_chars(trimmed_draft, MAX_DRAFT_CONTEXT_CHARS)
        ));
    }

    if !context_lines.is_empty() {
        prompt.push_str(
            "\n\nUse the following rough session context ONLY to disambiguate spelling of domain terms, project names, file paths, symbols, and proper nouns. \
             Do NOT repeat, continue, summarize, or answer anything from this context.\n<conversation_context>\n",
        );
        prompt.push_str(&context_lines.join("\n"));
        prompt.push_str("\n</conversation_context>");
    }

    prompt
}

pub fn clean_transcription_text(input: &str) -> String {
    let trimmed = input.trim();
    let trimmed =
        trimmed.trim_start_matches(|ch: char| ch == ',' || ch == '.' || ch.is_whitespace());
    let trimmed = trimmed.trim_end_matches(|ch: char| ch == ',' || ch.is_whitespace());
    trimmed.trim().to_string()
}

pub fn needs_leading_space(existing_text_before_cursor: &str, transcript: &str) -> bool {
    let Some(last_char) = existing_text_before_cursor.chars().last() else {
        return false;
    };
    if last_char.is_whitespace() || matches!(last_char, '(' | '[' | '{' | '"' | '\'' | '/') {
        return false;
    }
    let Some(first_char) = transcript.chars().next() else {
        return false;
    };
    !matches!(
        first_char,
        ',' | '.' | '!' | '?' | ':' | ';' | ')' | ']' | '}'
    )
}

pub fn parse_transcription_response(status_code: u16, raw_body: &str) -> Result<String> {
    let parsed: ChatCompletionResponse = serde_json::from_str(raw_body).with_context(|| {
        format!(
            "Invalid JSON from dictation gateway (HTTP {status_code}): {}",
            truncate_head_chars(raw_body.trim(), 200)
        )
    })?;

    if let Some(error) = parsed.error {
        let message = error
            .message
            .unwrap_or_else(|| format!("Gateway returned HTTP {status_code}"));
        bail!("{message}");
    }

    if status_code >= 400 {
        bail!("Gateway returned HTTP {status_code}");
    }

    let Some(choice) = parsed
        .choices
        .and_then(|choices| choices.into_iter().next())
    else {
        bail!("Gateway returned no transcription choices");
    };

    if choice.finish_reason.as_deref() == Some("content_filter") {
        bail!("Transcription blocked by content filter");
    }

    let content = choice
        .message
        .and_then(|message| message.content)
        .unwrap_or_default();
    let cleaned = clean_transcription_text(&content);
    if cleaned.is_empty() {
        bail!("No speech recognized");
    }
    Ok(cleaned)
}

pub async fn transcribe_wav_with_gateway(
    http_client: Arc<dyn HttpClient>,
    config: GatewayConfig,
    wav_bytes: Vec<u8>,
    context: DictationContext,
) -> Result<String> {
    let base64_audio = BASE64_STANDARD.encode(&wav_bytes);
    let system_prompt = format_dictation_system_prompt(&context);

    let payload = json!({
        "model": DICTATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this audio verbatim."
                    }
                ]
            }
        ],
        "reasoning_effort": "low",
        "max_tokens": 16384
    });

    let body_bytes =
        serde_json::to_vec(&payload).context("Failed to serialize dictation payload")?;
    let uri = format!("{}/v1/chat/completions", config.endpoint);

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", config.api_key))
        .body(AsyncBody::from(body_bytes))
        .context("Failed to build dictation HTTP request")?;

    let mut response = http_client
        .send(request)
        .await
        .context("Dictation request to Regulaido Gateway failed")?;
    let status_code = response.status().as_u16();
    let mut response_body = String::new();
    response
        .body_mut()
        .read_to_string(&mut response_body)
        .await
        .context("Failed to read dictation response body")?;

    parse_transcription_response(status_code, &response_body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gateway_config_json() {
        let valid = r#"{"endpoint":"https://llm.regulaido.ai/","api_key":"sk-test-123"}"#;
        let config = parse_gateway_config_json(valid).expect("valid config should parse");
        assert_eq!(config.endpoint, "https://llm.regulaido.ai");
        assert_eq!(config.api_key, "sk-test-123");

        let missing_key = r#"{"endpoint":"https://llm.regulaido.ai","api_key":"   "}"#;
        assert!(parse_gateway_config_json(missing_key).is_err());
    }

    #[test]
    fn test_format_dictation_system_prompt_with_and_without_context() {
        let empty = DictationContext::default();
        let prompt_empty = format_dictation_system_prompt(&empty);
        assert!(!prompt_empty.contains("<conversation_context>"));

        let populated = DictationContext {
            workspace_roots: vec!["compass".to_string()],
            active_file: Some("crates/agent_ui/src/dictation.rs".to_string()),
            thread_title: Some("Add native dictation".to_string()),
            recent_turns: vec![
                (
                    "User".to_string(),
                    "Please check PgBouncer port 6432".to_string(),
                ),
                (
                    "Assistant".to_string(),
                    "Use port 5432 for Drizzle.".to_string(),
                ),
            ],
            draft_text: "Also check".to_string(),
        };
        let prompt_populated = format_dictation_system_prompt(&populated);
        assert!(prompt_populated.contains("<conversation_context>"));
        assert!(prompt_populated.contains("Workspace: compass"));
        assert!(prompt_populated.contains("Active file: crates/agent_ui/src/dictation.rs"));
        assert!(prompt_populated.contains("Thread title: Add native dictation"));
        assert!(prompt_populated.contains("- User: Please check PgBouncer port 6432"));
        assert!(prompt_populated.contains("Current draft before cursor: Also check"));
    }

    #[test]
    fn test_parse_transcription_response() {
        let ok_json = r#"{"choices":[{"finish_reason":"stop","message":{"content":"  Use FastAPI and pgvector.  "}}]}"#;
        let text = parse_transcription_response(200, ok_json).expect("valid response should parse");
        assert_eq!(text, "Use FastAPI and pgvector.");

        let filtered_json =
            r#"{"choices":[{"finish_reason":"content_filter","message":{"content":null}}]}"#;
        assert!(parse_transcription_response(200, filtered_json).is_err());

        let error_json = r#"{"error":{"message":"Invalid key"}}"#;
        let err = parse_transcription_response(401, error_json).unwrap_err();
        assert!(err.to_string().contains("Invalid key"));
    }

    #[test]
    fn test_needs_leading_space() {
        assert!(!needs_leading_space("", "hello"));
        assert!(!needs_leading_space("hello ", "world"));
        assert!(needs_leading_space("hello", "world"));
        assert!(!needs_leading_space("hello", ". Also"));
    }
}
