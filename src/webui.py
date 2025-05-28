import streamlit as st
import subprocess
from llama_cpp import Llama
import os
import tempfile
import re

@st.cache_resource(show_spinner="Loading LLM model...")
def get_llm(model_path):
    # Use all layers on GPU if possible; adjust n_gpu_layers as needed for your VRAM
    return Llama(model_path=model_path, n_gpu_layers=-1)

def load_examples(example_file="examples.txt"):
    if not os.path.exists(example_file):
        return []
    with open(example_file, "r", encoding="utf-8") as f:
        content = f.read()
    # Split examples by double newlines
    raw_examples = content.strip().split("\n\n")
    return raw_examples

def select_relevant_examples(user_task, examples, max_examples=4):
    # Simple keyword matching: select examples that share words with the user_task
    user_words = set(re.findall(r"\w+", user_task.lower()))
    scored = []
    for ex in examples:
        ex_lower = ex.lower()
        score = sum(1 for w in user_words if w in ex_lower)
        scored.append((score, ex))
    # Sort by score, highest first, and take top N
    scored.sort(reverse=True)
    selected = [ex for score, ex in scored if score > 0][:max_examples]
    # If not enough, fill with top examples
    if len(selected) < max_examples:
        for score, ex in scored:
            if ex not in selected:
                selected.append(ex)
            if len(selected) >= max_examples:
                break
    return selected

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
        get_llm.clear()  # This will clear the cached model
        import gc
        gc.collect()
        st.session_state.last_model_path = model_path

    llm = get_llm(model_path)

# --- LLM prompt ---
user_task = st.text_area(
    "Describe what you want to do in the command prompt:",
    placeholder="e.g. List all files in the current directory"
)

# Copy prompt to clipboard (Streamlit 1.32+)
if hasattr(st, "copy_to_clipboard"):
    st.copy_to_clipboard(user_task, "Copy Prompt to Clipboard")

# Initialize command history in session state
if "command_history" not in st.session_state:
    st.session_state.command_history = []

if st.button("Generate Command"):
    # --- RAG: Select relevant examples ---
    all_examples = load_examples("examples.txt")
    selected_examples = select_relevant_examples(user_task, all_examples, max_examples=2)  # Fewer examples!
    examples_text = "\n\n".join(selected_examples)
    # --- Instructions ---
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

    # Extract Note and Command
    # Extract only the first Note/Command pair
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
    # Save to history
    st.session_state.command_history.append({
        "request": user_task,
        "command": command,
        "note": note
    })

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

if "generated_command" in st.session_state and st.session_state.generated_command:
    if st.button("Run Command in New PowerShell Window"):
        run_powershell_command(st.session_state.generated_command)
    if st.button("Run and Capture Output"):
        output, error = run_powershell_command_capture(st.session_state.generated_command)
        st.session_state.last_output = output
        st.session_state.last_error = error
        st.markdown("**PowerShell Output:**")
        st.code(output or "(no output)", language="text")
        if error:
            st.markdown("**PowerShell Error:**")
            st.code(error, language="text")

# --- Command History Panel ---
def build_command_context(history, idx):
    """
    Build a context string for the LLM including the original request,
    all corrections, and all questions/answers for this command up to idx,
    PLUS all Q&A for the current command, even if added later.
    """
    context = []
    # Find the original request for this thread
    base_idx = idx
    while base_idx > 0 and history[base_idx]['request'].startswith("[Correction]"):
        base_idx -= 1
    context.append(f"User request: {history[base_idx]['request']}")
    context.append(f"Initial command: {history[base_idx]['command']}")
    if history[base_idx].get("note"):
        context.append(f"Note: {history[base_idx]['note']}")
    # Add all corrections and Q&A after the base up to idx
    for i in range(base_idx + 1, idx + 1):
        req = history[i]['request']
        if req.startswith("[Correction]"):
            context.append(f"Correction: {history[i]['command']}")
            if history[i].get("note"):
                context.append(f"Correction Note: {history[i]['note']}")
        elif req.startswith("[Q]"):
            context.append(f"Question: {history[i]['question']}")
            context.append(f"LLM Answer: {history[i]['answer']}")
    # --- Add: Include all Q&A for the current command, even if added later ---
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
    st.code(entry["command"], language="powershell")
    col1, col2, col3 = st.columns(3)
    with col1:
        if entry.get("command"):
            if st.button(f"Run Again #{idx+1}", key=f"run_again_{idx}_{i}"):
                run_powershell_command(entry["command"])
            if st.button(f"Run and Capture Output #{idx+1}", key=f"run_capture_{idx}_{i}"):
                output, error = run_powershell_command_capture(entry["command"])
                st.markdown("**PowerShell Output:**")
                st.code(output or "(no output)", language="text")
                if error:
                    st.markdown("**PowerShell Error:**")
                    st.code(error, language="text")
                # Store output/error in the history entry
                entry["output"] = output
                entry["error"] = error
    with col2:
        # Only show "Request Correction" for non-correction entries
        if not entry['request'].startswith("[Correction]"):
            if st.button(f"Request Correction #{idx+1}", key=f"correction_{idx}_{i}"):
                # Use output/error from this entry if available, else fallback to last_output/last_error
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

        # Only show "Run Corrected Command" and "Run and Capture Correction" for correction entries
        if entry['request'].startswith("[Correction]"):
            if st.button(f"Run Corrected Command #{idx+1}", key=f"run_correction_{idx}_{i}"):
                run_powershell_command(entry["command"])
            if st.button(f"Run and Capture Correction #{idx+1}", key=f"run_capture_correction_{idx}_{i}"):
                output, error = run_powershell_command_capture(entry["command"])
                st.markdown("**PowerShell Output (Correction):**")
                st.code(output or "(no output)", language="text")
                if error:
                    st.markdown("**PowerShell Error (Correction):**")
                    st.code(error, language="text")
    with col3:
        # Show all previous Q&A for this command
        qas = [
            h for h in st.session_state.command_history
            if h.get('request', '').startswith('[Q]') and h.get('command') == entry['command']
        ]
        for q_idx, qa in enumerate(qas, 1):
            st.markdown(f"**Q{q_idx}:** {qa['question']}")
            st.markdown(f"**A{q_idx}:** {qa['answer']}")

        # Show all previous Q-Output Q&A for this command
        qas_output = [
            h for h in st.session_state.command_history
            if h.get('request', '').startswith('[Q-Output]') and h.get('command') == entry['command']
        ]
        for q_idx, qa in enumerate(qas_output, 1):
            st.markdown(f"**Q-Output{q_idx}:** {qa['question']}")
            st.markdown(f"**A-Output{q_idx}:** {qa['answer']}")

        # Button to open the Q&A input for this entry
        if st.button(f"Ask About This Command #{idx+1}", key=f"ask_{idx}_{i}"):
            st.session_state[f"show_ask_{idx}_{i}"] = True

        # If the Q&A input is open, allow repeated questions
        if st.session_state.get(f"show_ask_{idx}_{i}", False):
            question_count = len(qas)
            user_question = st.text_input(
                f"Your question about Command #{idx+1}:",
                key=f"user_question_{idx}_{i}_{question_count}"
            )
            if user_question.strip() and st.button(f"Submit Question #{idx+1}", key=f"submit_question_{idx}_{i}_{question_count}"):
                context = build_command_context(st.session_state.command_history, idx)
                ask_prompt = (
                    f"{context}\n"
                    f"Question: {user_question}\n"
                    "Please answer clearly and concisely."
                )
                ask_output = llm(ask_prompt, max_tokens=256, temperature=0.2)
                answer = ask_output["choices"][0]["text"].strip()
                st.markdown(f"**LLM Answer:** {answer}")
                # Save Q&A to history for future context
                st.session_state.command_history.append({
                    "request": f"[Q] {entry['request']}",
                    "command": entry["command"],
                    "question": user_question,
                    "answer": answer
                })
                # Keep the Q&A input open for more questions
                st.session_state[f"show_ask_{idx}_{i}"] = True

        # Only show "Ask About Output" if there is output or error
        if entry.get("output") or entry.get("error"):
            if st.button(f"Ask About Output #{idx+1}", key=f"ask_output_{idx}_{i}"):
                st.session_state[f"show_ask_output_{idx}_{i}"] = True

            # If the Ask About Output input is open
            if st.session_state.get(f"show_ask_output_{idx}_{i}", False):
                output = entry.get("output", "")
                error = entry.get("error", "")
                st.markdown("**PowerShell Output:**")
                st.code(output or "(no output)", language="text")
                if error:
                    st.markdown("**PowerShell Error:**")
                    st.code(error, language="text")
                user_output_question = st.text_input(
                    f"Your question about the output for Command #{idx+1}:",
                    key=f"user_output_question_{idx}_{i}"
                )
                if user_output_question.strip() and st.button(f"Submit Output Question #{idx+1}", key=f"submit_output_question_{idx}_{i}"):
                    context = build_command_context(st.session_state.command_history, idx)
                    ask_output_prompt = (
                        f"{context}\n"
                        f"PowerShell output: {output}\n"
                        f"PowerShell error: {error}\n"
                        f"Question about the output: {user_output_question}\n"
                        "Please answer clearly and concisely."
                    )
                    ask_output_response = llm(ask_output_prompt, max_tokens=256, temperature=0.2)
                    answer = ask_output_response["choices"][0]["text"].strip()
                    st.markdown(f"**LLM Answer:** {answer}")
                    # Save Q&A about output to history
                    st.session_state.command_history.append({
                        "request": f"[Q-Output] {entry['request']}",
                        "command": entry["command"],
                        "question": user_output_question,
                        "answer": answer,
                        "output": output,
                        "error": error
                    })
                    st.session_state[f"show_ask_output_{idx}_{i}"] = True