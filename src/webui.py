import streamlit as st
import subprocess
from llama_cpp import Llama
import os
import tempfile
import re

# --- LLM Loader ---
@st.cache_resource(show_spinner="Loading LLM model...")
def get_llm(model_path):
    return Llama(model_path=model_path, n_gpu_layers=-1)

# --- Example loading and selection ---
def load_examples(example_file="examples.txt"):
    if not os.path.exists(example_file):
        return []
    with open(example_file, "r", encoding="utf-8") as f:
        content = f.read()
    raw_examples = content.strip().split("\n\n")
    return raw_examples

def select_relevant_examples(user_task, examples, max_examples=4):
    user_words = set(re.findall(r"\w+", user_task.lower()))
    scored = []
    for ex in examples:
        ex_lower = ex.lower()
        score = sum(1 for w in user_words if w in ex_lower)
        scored.append((score, ex))
    scored.sort(reverse=True)
    selected = [ex for score, ex in scored if score > 0][:max_examples]
    if len(selected) < max_examples:
        for score, ex in scored:
            if ex not in selected:
                selected.append(ex)
            if len(selected) >= max_examples:
                break
    return selected

# --- Streamlit UI setup ---
st.set_page_config(page_title="AI Command Prompt", layout="wide")
st.title("AI-Powered Command Prompt (Local LLM)")

# --- Model selection ---
model_dir = st.text_input(
    "LLM model directory",
    value=os.path.expanduser(r"~/.lmstudio/models/lmstudio-community"),
    key="llm_model_dir"
)
models = [os.path.join(root, file)
          for root, _, files in os.walk(model_dir)
          for file in files if file.endswith(".gguf")]
if not models:
    st.error(f"No models found in {model_dir}")
    st.stop()

model_path = st.selectbox(
    "Choose LLM model",
    models,
    format_func=lambda p: os.path.basename(p)
)

llm = None
if model_path:
    if "last_model_path" not in st.session_state:
        st.session_state.last_model_path = model_path
    if model_path != st.session_state.last_model_path:
        get_llm.clear()
        import gc
        gc.collect()
        st.session_state.last_model_path = model_path
    llm = get_llm(model_path)

# --- LLM prompt ---
user_task = st.text_area(
    "Describe what you want to do in the command prompt:",
    placeholder="e.g. List all files in the current directory"
)
if hasattr(st, "copy_to_clipboard"):
    st.copy_to_clipboard(user_task, "Copy Prompt to Clipboard")

if "command_history" not in st.session_state:
    st.session_state.command_history = []

# --- Generate Command ---
if st.button("Generate Command") and llm is not None:
    all_examples = load_examples("examples.txt")
    selected_examples = select_relevant_examples(user_task, all_examples, max_examples=2)
    examples_text = "\n\n".join(selected_examples)
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        instructions = f.read().strip()
    prompt = f"{instructions}\nExamples:\n{examples_text}\n\nUser request: {user_task}\nNote:\nCommand:"
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.2,
        stop=["User request:", "\n\nNote:", "\n\nCommand:"]
    )
    raw_output = output["choices"][0]["text"].strip()
    note_match = re.search(r"Note:(.*?)(?:Command:|$)", raw_output, re.DOTALL | re.IGNORECASE)
    command_match = re.search(r"Command:(.*?)(?:Note:|$)", raw_output, re.DOTALL | re.IGNORECASE)
    note = note_match.group(1).strip() if note_match else ""
    command = command_match.group(1).strip() if command_match else raw_output.strip()
    command = re.sub(r"^```[a-zA-Z]*\n?|```$", "", command, flags=re.MULTILINE).strip()
    if note:
        st.info(note)
    st.code(command, language="powershell")
    st.session_state.generated_command = command
    st.session_state.generated_note = note

# Allow editing the generated command before running
if "generated_command" in st.session_state and st.session_state.generated_command:
    edited_command = st.text_area(
        "Edit the generated PowerShell command before running:",
        value=st.session_state.generated_command,
        key="editable_generated_command"
    )
    st.session_state.generated_command = edited_command

# --- Output display helper ---
def show_output(output, error, label="PowerShell Output"):
    st.markdown(f"**{label}:**")
    st.code(output or "(no output)", language="text")
    if error:
        st.markdown("**PowerShell Error:**")
        st.code(error, language="text")

# --- Command execution ---
def run_powershell_command_capture(command):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ps1", mode="w", encoding="utf-8") as ps1_file:
            ps1_file.write(command)
            ps1_file_path = ps1_file.name
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1_file_path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        return output, error
    except Exception as e:
        return "", str(e)

def run_powershell_command(command):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ps1", mode="w", encoding="utf-8") as ps1_file:
            ps1_file.write(command)
            ps1_file_path = ps1_file.name
        subprocess.Popen(
            f'start powershell -NoExit -ExecutionPolicy Bypass -File "{ps1_file_path}"',
            shell=True
        )
        st.success("Command sent to a new PowerShell window.")
    except Exception as e:
        st.error(f"Error running command: {e}")

# --- Main run buttons (only after command is generated) ---
if st.session_state.get("generated_command"):
    if st.button("Run Command in New PowerShell Window"):
        run_powershell_command(st.session_state.generated_command)
        st.session_state.command_history.append({
            "request": user_task,
            "command": st.session_state.generated_command,
            "note": st.session_state.generated_note,
            "output": "",
            "error": ""
        })
    if st.button("Run and Capture Output"):
        output, error = run_powershell_command_capture(st.session_state.generated_command)
        st.session_state.last_output = output
        st.session_state.last_error = error
        show_output(output, error)
        st.session_state.command_history.append({
            "request": user_task,
            "command": st.session_state.generated_command,
            "note": st.session_state.generated_note,
            "output": output,
            "error": error
        })

# --- Command History Panel ---
def build_command_context(history, idx):
    context = []
    base_idx = idx
    while base_idx > 0 and history[base_idx]['request'].startswith("[Correction]"):
        base_idx -= 1
    context.append(f"User request: {history[base_idx]['request']}")
    context.append(f"Initial command: {history[base_idx]['command']}")
    if history[base_idx].get("note"):
        context.append(f"Note: {history[base_idx]['note']}")
    for i in range(base_idx + 1, idx + 1):
        req = history[i]['request']
        if req.startswith("[Correction]"):
            context.append(f"Correction: {history[i]['command']}")
            if history[i].get("note"):
                context.append(f"Correction Note: {history[i]['note']}")
        elif req.startswith("[Q]"):
            context.append(f"Question: {history[i]['question']}")
            context.append(f"LLM Answer: {history[i]['answer']}")
    current_command = history[idx]['command']
    for h in history[idx+1:]:
        if h.get('request', '').startswith('[Q]') and h.get('command') == current_command:
            context.append(f"Question: {h['question']}")
            context.append(f"LLM Answer: {h['answer']}")
    return "\n".join(context)

st.markdown("## Command History")
for i, entry in enumerate(reversed(st.session_state.command_history)):
    idx = len(st.session_state.command_history) - 1 - i
    st.markdown(f"**Request:** {entry['request']}")
    if hasattr(st, "copy_to_clipboard"):
        st.copy_to_clipboard(entry['request'], f"Copy Prompt #{idx+1}")
    if entry.get("note"):
        st.markdown(f"**Note:** {entry['note']}")
    # Always show output history above the command
    output = entry.get("output", "")
    error = entry.get("error", "")
    show_output(output, error, label="PowerShell Output History")
    st.code(entry["command"], language="powershell")

    col1, col2, col3 = st.columns(3)
    with col1:
        if entry.get("command"):
            if st.button(f"Run Again #{idx+1}", key=f"run_again_{idx}_{i}"):
                run_powershell_command(entry["command"])
            if st.button(f"Run and Capture Output #{idx+1}", key=f"run_capture_{idx}_{i}"):
                output, error = run_powershell_command_capture(entry["command"])
                show_output(output, error)
                st.session_state.command_history[idx]["output"] = output
                st.session_state.command_history[idx]["error"] = error

    with col2:
        if not entry['request'].startswith("[Correction]"):
            if st.button(f"Request Correction #{idx+1}", key=f"correction_{idx}_{i}"):
                output = entry.get("output", st.session_state.get("last_output", ""))
                error = entry.get("error", st.session_state.get("last_error", ""))
                context = build_command_context(st.session_state.command_history, idx)
                correction_prompt = (
                    f"{context}\n"
                    f"PowerShell output: {output}\n"
                    f"PowerShell error: {error}\n"
                    "Please generate a corrected command.\n"
                    "IMPORTANT: Always output your answer in two parts.\n"
                    "First, output Note: (with any important warnings, context, or usage tips). If there is nothing important to note, output 'Note: None'.\n"
                    "Second, output Command: (the PowerShell code).\n"
                    "Only output the Note and the Command, nothing else.\n"
                    "Note:\nCommand:"
                )
                correction_output = llm(correction_prompt, max_tokens=256, temperature=0.2)
                correction_raw = correction_output["choices"][0]["text"].strip()
                note_match = re.search(r"Note:(.*?)(?:Command:|$)", correction_raw, re.DOTALL | re.IGNORECASE)
                command_match = re.search(r"Command:(.*?)(?:Note:|$)", correction_raw, re.DOTALL | re.IGNORECASE)
                note = note_match.group(1).strip() if note_match else ""
                command = command_match.group(1).strip() if command_match else correction_raw.strip()
                command = re.sub(r"^```[a-zA-Z]*\n?|```$", "", command, flags=re.MULTILINE).strip()
                st.info(f"Correction Note: {note}")
                st.code(command, language="powershell")
                st.session_state.command_history.append({
                    "request": f"[Correction] {entry['request']}",
                    "command": command,
                    "note": note
                })
        if entry['request'].startswith("[Correction]"):
            if st.button(f"Run Corrected Command #{idx+1}", key=f"run_correction_{idx}_{i}"):
                run_powershell_command(entry["command"])
            if st.button(f"Run and Capture Correction #{idx+1}", key=f"run_capture_correction_{idx}_{i}"):
                output, error = run_powershell_command_capture(entry["command"])
                show_output(output, error, label="PowerShell Output (Correction)")
                st.session_state.command_history[idx]["output"] = output
                st.session_state.command_history[idx]["error"] = error

    with col3:
        # Unified Q&A for command, output, and error
        qas_all = [
            h for h in st.session_state.command_history
            if h.get('request', '').startswith('[Q-ALL]') and h.get('command') == entry['command']
        ]
        for q_idx, qa in enumerate(qas_all, 1):
            st.markdown(f"**Q-ALL{q_idx}:** {qa['question']}")
            st.markdown(f"**A-ALL{q_idx}:** {qa['answer']}")
        if st.button(f"Ask About Command/Output/Error #{idx+1}", key=f"ask_all_{idx}_{i}"):
            st.session_state[f"show_ask_all_{idx}_{i}"] = True
        if st.session_state.get(f"show_ask_all_{idx}_{i}", False):
            output = entry.get("output", "")
            error = entry.get("error", "")
            show_output(output, error)
            question_count = len(qas_all)
            user_all_question = st.text_input(
                f"Your question about Command/Output/Error for Command #{idx+1}:",
                key=f"user_all_question_{idx}_{i}_{question_count}"
            )
            if user_all_question.strip() and st.button(f"Submit Q #{idx+1}", key=f"submit_all_question_{idx}_{i}_{question_count}"):
                MAX_CONTEXT_CHARS = 1500
                MAX_OUTPUT_CHARS = 1000
                MAX_ERROR_CHARS = 1000

                context = build_command_context(st.session_state.command_history, idx)
                context = context[-MAX_CONTEXT_CHARS:]  # keep last N chars

                output = (output or "")[-MAX_OUTPUT_CHARS:]
                error = (error or "")[-MAX_ERROR_CHARS:]

                ask_all_prompt = (
                    f"{context}\n"
                    f"PowerShell output: {output}\n"
                    f"PowerShell error: {error}\n"
                    f"Question: {user_all_question}\n"
                    "Please answer clearly and concisely."
                )
                ask_all_response = llm(ask_all_prompt, max_tokens=256, temperature=0.2)
                answer = ask_all_response["choices"][0]["text"].strip()
                st.session_state[f"last_llm_answer_{idx}_{i}"] = answer
                st.session_state[f"show_ask_all_{idx}_{i}"] = True

            # Show the last answer if available (INSIDE the loop and col3)
            if st.session_state.get(f"last_llm_answer_{idx}_{i}"):
                st.markdown(f"**LLM Answer:** {st.session_state[f'last_llm_answer_{idx}_{i}']}")