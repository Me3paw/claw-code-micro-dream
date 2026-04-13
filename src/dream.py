#!/usr/bin/env python3
import os
import json
import re
import pathlib
import requests
import sys
import datetime

# Constants
# Respect environment variables for OpenAI-compatible APIs (like Ollama via /v1)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-token")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")

# Use CLAW_MODEL if set, otherwise default to a 'smart' model alias
MODEL_NAME = os.getenv("CLAW_MODEL", "smart")

# Regex Ground Truth Extractors
RE_TOOL_USE_NAME = re.compile(r'"name":\s*"(?P<name>[^"]+)"')
RE_TOOL_NAME = re.compile(r'"tool_name":\s*"(?P<name>[^"]+)"')
RE_IS_ERROR = re.compile(r'"is_error":\s*(?P<val>true|false)')

def format_timestamp(ms):
    if ms is not None:
        return datetime.datetime.fromtimestamp(ms / 1000.0).strftime('%d %m %Y %H:%M:%S')
    return "00 00 0000 00:00:00"

def flush_vram(model=MODEL_NAME):
    try:
        # Try Ollama native first for flush
        requests.post(OLLAMA_API_URL, json={"model": model, "keep_alive": 0}, timeout=10)
    except:
        pass

def get_turn_tool_stats(raw_lines):
    tools, errors = [], []
    for line in raw_lines:
        if '"type":"tool_use"' in line:
            m = RE_TOOL_USE_NAME.search(line)
            if m: tools.append(m.group('name'))
        if '"type":"tool_result"' in line:
            m_name = RE_TOOL_NAME.search(line)
            m_err = RE_IS_ERROR.search(line)
            if m_name and m_name.group('name') not in tools: tools.append(m_name.group('name'))
            if m_err: errors.append(m_err.group('val'))
    return ", ".join(sorted(set(tools))) if tools else "N/A", ", ".join(sorted(set(errors))) if errors else "N/A"

def call_llm(prompt, model, system_prompt=None, json_mode=False):
    """Generic wrapper to call Ollama or OpenAI-compatible API."""
    # If OPENAI_BASE_URL looks like a standard /v1 endpoint, use ChatCompletions
    if "/v1" in OPENAI_BASE_URL:
        url = f"{OPENAI_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=300)
            res.raise_for_status()
            content = res.json()['choices'][0]['message']['content']
            return json.loads(content) if json_mode else content
        except Exception as e:
            if json_mode: return {"action": f"Error: {e}", "changes": "N/A", "durable_memories": []}
            return f"Error: {e}"
    else:
        # Fallback to Ollama native generate API
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "stream": False
        }
        if json_mode:
            payload["format"] = "json"
        
        try:
            res = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
            res.raise_for_status()
            content = res.json()['response']
            return json.loads(content) if json_mode else content
        except Exception as e:
            if json_mode: return {"action": f"Error: {e}", "changes": "N/A", "durable_memories": []}
            return f"Error: {e}"

def summarize_turn(turn_content, ts_str, model=MODEL_NAME):
    system_prompt = (
        "You are a technical knowledge extraction engine. Output ONLY valid JSON with keys: "
        "'action' (What the AI did in one short sentence), "
        "'changes' (Concise list of files modified/created/deleted), "
        "'durable_memories' (List of technical facts, architectural decisions, or bug fixes discovered. "
        "Focus on things that stay true across sessions. Return [] if none)."
    )
    processed = []
    for msg in turn_content:
        role = msg.get('role')
        blocks = msg.get('blocks', [])
        content = []
        for b in blocks:
            if b.get('type') == 'text': content.append(b.get('text', ''))
            elif b.get('type') == 'tool_use': content.append(f"[Tool: {b.get('name')} Input: {b.get('input')}]")
            elif b.get('type') == 'tool_result': content.append(f"[Result for {b.get('tool_name')} Success: {not b.get('is_error')}]")
        processed.append({"role": role, "content": "\n".join(content)})

    prompt = f"Timestamp: {ts_str}\n\nTurn:\n{json.dumps(processed, indent=2)}"
    return call_llm(prompt, model, system_prompt=system_prompt, json_mode=True)

def consolidate_timeline(timeline_file, last_jsonl_name, model=MODEL_NAME):
    """PRUNING LOGIC: Aggressively merges logically related turns into milestones."""
    print(f"\n--- Pruning Timeline (Aggressive) for {last_jsonl_name} ---")
    if not timeline_file.exists(): return
    
    with open(timeline_file, 'r') as f: content = f.read()

    lines = content.splitlines()
    logs_to_compress, bookkeeping = [], []
    for line in lines:
        if line.startswith("Processed: ") or line.startswith("# ") or line.startswith("Consolidated"):
            bookkeeping.append(line)
        elif line.strip():
            logs_to_compress.append(line)

    system_prompt = (
        "You are a timeline compression expert. Your goal is to rewrite the ENTIRE timeline into the fewest milestones possible. "
        "MERGE consecutive turns if they belong to the same logical task (e.g., Turn 3 modifies a file, Turn 4 creates its companion). "
        "If a sequence of turns has zero net effect (created then deleted), REMOVE it entirely. "
        "CRITICAL: Re-index to sequential [Turn X] numbering. "
        "Format: '[DD MM YYYY HH:MM:SS] [Turn X] {Unified Action} | changes: {Aggregated Files} | tools: {Used Tools} | err: {Err}'."
        "Output ONLY the rewritten logs."
    )
    
    prompt = "Timeline Content:\n" + "\n".join(logs_to_compress)
    consolidated = call_llm(prompt, model, system_prompt=system_prompt)
    
    try:
        with open(timeline_file, 'w') as f:
            f.write(f"# Session Timeline for {timeline_file.stem.replace('timeline_', '')}\n\n")
            f.write(consolidated.strip() + "\n\n")
            processed_markers = sorted(list(set([l for l in bookkeeping if l.startswith("Processed:")])))
            for m in processed_markers: f.write(m + "\n")
            f.write(f"\nConsolidated up to {last_jsonl_name}\n")
        print("Timeline pruning successful.")
    except Exception as e:
        print(f"Timeline pruning failed: {e}")

def consolidate_knowledge(kb_file, last_jsonl_name, model=MODEL_NAME):
    """PRUNING LOGIC: Synthesizes facts and removes outdated info/contradictions."""
    print(f"\n--- Pruning Knowledge Base for {last_jsonl_name} ---")
    if not kb_file.exists(): return
    
    with open(kb_file, 'r') as f: content = f.read()

    system_prompt = (
        "You are a Knowledge Base Architect. Synthesize the provided raw facts into a clean technical reference. "
        "Group by topics (## Architecture, ## Setup, ## Bugs, etc.). "
        "MERGE related details into cohesive bullet points. "
        "DELETE facts that have been rendered obsolete or contradicted by newer information. "
        "Remove transient workspace notes. Output ONLY Markdown."
    )
    
    prompt = f"Raw Facts:\n{content}"
    organized = call_llm(prompt, model, system_prompt=system_prompt)
    
    try:
        with open(kb_file, 'w') as f:
            f.write(f"# Knowledge Base for {kb_file.stem.replace('knowledge_base_', '')}\n\n")
            f.write(organized.strip() + "\n\n")
            f.write(f"\nConsolidated up to {last_jsonl_name}\n")
        print("KB pruning successful.")
    except Exception as e:
        print(f"KB pruning failed: {e}")

def process_file(file_path, timeline_file, kb_file):
    print(f"\n--- Dreaming session: {file_path.name} ---")
    prompt_timestamps = []
    session_model = None
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'prompt_history': prompt_timestamps.append(data.get('timestamp_ms'))
                if data.get('type') == 'session_meta' and data.get('model'):
                    session_model = data.get('model')
            except: continue

    # Use the model from the session if found, otherwise fallback to global MODEL_NAME
    active_model = session_model if session_model else MODEL_NAME
    print(f"Using model: {active_model}")

    turns, turn_raw_lines, turn_ts_values = [], [], []
    current_turn, current_raw, ts_idx = [], [], 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                if data.get('type') == 'message':
                    msg = data.get('message', {})
                    if msg.get('role') == 'user':
                        if current_turn:
                            turns.append(current_turn); turn_raw_lines.append(current_raw)
                        current_turn, current_raw = [msg], [line]
                        turn_ts_values.append(prompt_timestamps[ts_idx] if ts_idx < len(prompt_timestamps) else None)
                        ts_idx += 1
                    elif msg.get('role') in ['assistant', 'tool']:
                        if current_turn:
                            current_turn.append(msg); current_raw.append(line)
            except: continue
    if current_turn:
        turns.append(current_turn); turn_raw_lines.append(current_raw)

    results_tl, results_kb = [], []
    for i, turn in enumerate(turns):
        ts_str = format_timestamp(turn_ts_values[i]) if i < len(turn_ts_values) else "00 00 0000 00:00:00"
        print(f"Summarizing Turn {i+1}/{len(turns)}...", end="\r")
        summary = summarize_turn(turn, ts_str, model=active_model)
        tool_name, is_error = get_turn_tool_stats(turn_raw_lines[i])
        
        results_tl.append(f"[{ts_str}] [Turn {i+1}] {summary.get('action', 'N/A')} | changes: {summary.get('changes', 'N/A')} | tools: {tool_name} | err: {is_error}")
        
        mems = summary.get('durable_memories', [])
        if mems: results_kb.extend([f"- {m} (Ref: {file_path.name})" for m in mems])
        
    with open(timeline_file, 'a') as tf:
        tf.write(f"\n### {file_path.name}\n")
        for res in results_tl: tf.write(f"- {res}\n")
        tf.write(f"\nProcessed: {file_path.name}\n")
        
    if results_kb:
        with open(kb_file, 'a') as kf:
            kf.write(f"\n<!-- Raw facts from {file_path.name} -->\n")
            for fact in results_kb: kf.write(fact + "\n")
    
    flush_vram(model=active_model)

def main():
    # Prioritize the directory where the user is actually working
    working_dir = pathlib.Path(os.getenv("CLAW_WORKING_DIR", os.getcwd()))
    
    # Still need to find the project root for Python module loading and metadata
    project_root = pathlib.Path.cwd()
    if project_root.name == "rust": project_root = project_root.parent
    if not (project_root / "src").is_dir() and (project_root.parent / "src").is_dir():
        project_root = project_root.parent

    # Metadata files go into the working directory
    timeline_file = working_dir / f"timeline_{working_dir.name}.md"
    kb_file = working_dir / f"knowledge_base_{working_dir.name}.md"
    
    for f, t in [(timeline_file, "Timeline"), (kb_file, "Knowledge Base")]:
        if not f.exists():
            with open(f, 'w') as tf: tf.write(f"# {t} for {working_dir.name}\n\n")

    processed_files, last_tl_con, last_kb_con = set(), None, None
    with open(timeline_file, 'r') as tf:
        for line in tf:
            if line.startswith("Processed: "): processed_files.add(line.replace("Processed: ", "").strip())
            if line.startswith("Consolidated up to "): last_tl_con = line.replace("Consolidated up to ", "").strip()
    
    if kb_file.exists():
        with open(kb_file, 'r') as kf:
            for line in kf:
                if line.startswith("Consolidated up to "): last_kb_con = line.replace("Consolidated up to ", "").strip()
                    
    # Find all possible session locations relative to working_dir
    sessions_bases = [
        working_dir / ".claw" / "sessions",
        working_dir / "rust" / ".claw" / "sessions",
    ]
    
    all_jsonl_files = []
    for base in sessions_bases:
        if base.exists():
            # Support both flat and hex-subdir structures
            all_jsonl_files.extend(list(base.glob("*.jsonl")))
            all_jsonl_files.extend(list(base.glob("*/*.jsonl")))

    if not all_jsonl_files:
        print(f"No session files found in {sessions_bases}")
        return

    all_jsonl_files.sort(key=lambda x: x.stat().st_mtime)
    
    new_files = [f for f in all_jsonl_files if f.name not in processed_files]
    if new_files:
        try:
            for f in new_files: process_file(f, timeline_file, kb_file)
            consolidate_timeline(timeline_file, new_files[-1].name, model=MODEL_NAME)
            consolidate_knowledge(kb_file, new_files[-1].name, model=MODEL_NAME)
        finally:
            flush_vram()
    else:
        sorted_processed = [f.name for f in all_jsonl_files if f.name in processed_files]
        if sorted_processed:
            latest = sorted_processed[-1]
            if latest != last_tl_con or latest != last_kb_con:
                if latest != last_tl_con: consolidate_timeline(timeline_file, latest, model=MODEL_NAME)
                if latest != last_kb_con: consolidate_knowledge(kb_file, latest, model=MODEL_NAME)
                flush_vram()
            else:
                print("Already consolidated.")

if __name__ == "__main__":
    main()
