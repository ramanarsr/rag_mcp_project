# streamlit_app/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import uuid
import re
import numpy as np
import smtplib
from email.mime.text import MIMEText
from client.mcp_client import generate as mcp_generate, judge as mcp_judge
from langchain.memory import ConversationBufferMemory
from bert_score import score as bert_score

st.set_page_config(page_title="High School Science RAG (MCP)", layout="wide")

# --- Email utility for learning agent feedback ---
def send_feedback_email(query_text, feedback_text, response_text):
    sender = "ramanachandranrsr@gmail.com"       # <-- Replace with your sender email
    password = st.secrets["APP_PASS"]        # <-- Gmail App Password
    recipient = "ramanachandranrs@gmail.com"    # <-- Replace with your recipient email

    subject = "🧠 RAG Learning Agent User Feedback"
    body = f"""
A user submitted feedback for the RAG model.

🔍 **Query:** 
    {query_text}

🤖 **RAG Response:**
    {response_text}

💬 **Feedback:**
    {feedback_text}
    """

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        st.success("✅ Your feedback was sent successfully! Thank you.")
    except Exception as e:
        st.error(f"⚠️ Failed to send feedback: {e}")

# --- Evaluation dataset ---

evaluation_dataset = [
    { "query": "What is hypermetropia and how does it differ from myopia?",
      "ground_truth": "Hypermetropia, or farsightedness, is an eye defect where a person cannot see nearby objects clearly, though their distant vision is clear. Myopia, or nearsightedness, is the opposite defect, where a person cannot see distant objects clearly." },
    { "query": "What is the heating effect of electric current, also known as Joule's Law of Heating?",
      "ground_truth": "It is the phenomenon where the dissipation of energy in a purely resistive circuit occurs entirely in the form of heat. The heat produced (H) is given by the formula H = I²Rt." },
    { "query": "What is the difference between reactants and products in a chemical reaction?",
      "ground_truth": "Reactants are the substances that exist before a chemical reaction begins and are written on the left side of a chemical equation. Products are the new substances that are formed as a result of the reaction and are written on the right side of the equation." },
    { "query": "Write the overall balanced chemical equation for photosynthesis.",
      "ground_truth": "6CO2 + 6H2 O + Light → C6 H12 O6 + 6O2" },
    { "query": "Distinguish between a concave mirror and a convex mirror based on their reflecting surfaces.",
      "ground_truth": "A concave mirror is a spherical mirror whose reflecting surface is curved inwards, towards the center of the sphere. A convex mirror is a spherical mirror whose reflecting surface is curved outwards, away from the center of the sphere." },
    { "query": "What is a solenoid, and what is the nature of the magnetic field inside it?",
      "ground_truth": "A solenoid is a coil comprising several circular turns of insulated copper wire wrapped tightly in the shape of a cylinder. The magnetic field inside the solenoid is uniform." },
    { "query": "What was the basis for Dobereiner's classification of elements into triads?",
      "ground_truth": "Dobereiner used the physical and chemical characteristics of each element to divide them into triads. In a trio, the elements were organized in ascending order of their atomic masses, and he grouped elements with related qualities." },
    { "query": "What is the typical phenotypic ratio of a dihybrid cross in the F2 generation, according to Mendel?",
      "ground_truth": "The typical phenotypic ratio of a dihybrid cross in the F2 generation, according to Mendel, is 9:3:3:1." },
    { "query": "How does litmus paper indicate pH?",
      "ground_truth": "The color of Litmus paper changes color on pH." },
    { "query": "List two physical properties that are characteristic of metals.",
      "ground_truth": "Two physical properties that are characteristic of metals are:1. Luster: Metals have a shiny appearance.2. Conductivity: They conduct heat and electricity." }
]

# --- session state ---
if "chats" not in st.session_state:
    st.session_state["chats"] = []
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None
if "feedback_mode" not in st.session_state:
    st.session_state["feedback_mode"] = False
if "feedback_text" not in st.session_state:
    st.session_state["feedback_text"] = ""

def new_chat(first_question=None):
    chat_id = str(uuid.uuid4())
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    title = first_question[:40] + "..." if first_question else "New Chat"
    chat = {"id": chat_id, "title": title, "messages": [], "memory": memory}
    st.session_state["chats"].insert(0, chat)
    st.session_state["current_chat"] = chat_id
    return chat

def get_chat_by_id(chat_id):
    for c in st.session_state["chats"]:
        if c["id"] == chat_id:
            return c
    return None

def delete_chat(chat_id):
    st.session_state["chats"] = [c for c in st.session_state["chats"] if c["id"] != chat_id]
    if st.session_state.get("current_chat") == chat_id:
        st.session_state["current_chat"] = st.session_state["chats"][0]["id"] if st.session_state["chats"] else None

def normalize(text):
    return re.sub(r'[^a-z0-9]', ' ', text.lower()).strip()

def compute_metrics(preds, labels):
    preds_clean = [normalize(p) for p in preds]
    labels_clean = [normalize(l) for l in labels]

    precisions, recalls, f1s = [], [], []
    for pred, label in zip(preds_clean, labels_clean):
        pred_tokens = pred.split()
        label_tokens = label.split()
        common = set(pred_tokens) & set(label_tokens)
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(label_tokens) if label_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    P, R, F1 = bert_score(preds, labels, lang='en', verbose=False)
    return {
        "Precision": round(np.mean(precisions), 3),
        "Recall": round(np.mean(recalls), 3),
        "F1-Score": round(np.mean(f1s), 3),
        "BERTScore F1": round(float(F1.mean()), 3)
    }

# UI
with st.sidebar:
    st.markdown("### 💬 Chats")
    if st.button("➕ New Chat"):
        new_chat()
    st.markdown("---")
    for chat in st.session_state["chats"]:
        cols = st.columns([2, 0.3])
        if cols[0].button(chat["title"], key=f"open_{chat['id']}"):
            st.session_state["current_chat"] = chat["id"]
        if cols[1].button("🗑", key=f"del_{chat['id']}"):
            delete_chat(chat["id"])
            st.rerun()

current_chat_id = st.session_state.get("current_chat")
current_chat = get_chat_by_id(current_chat_id) if current_chat_id else None

if not current_chat:
    st.info("Select or create a chat from the sidebar.")
else:
    # --- Display chat history ---
    for i, msg in enumerate(current_chat["messages"]):
        # Prevent duplicate rendering on reruns
        role = "🧑‍🎓 You" if msg["role"] == "user" else "🤖 Tutor"
        st.markdown(f"**{role}:** {msg['content']}")
        st.markdown("---")

        # Only tutors get the Report button
        # --- Inside the loop for assistant messages ---
        if msg["role"] == "assistant":
            key_prefix = f"feedback_{current_chat_id}_{i}"

            # Initialize per-message feedback state
            if f"{key_prefix}_state" not in st.session_state:
                st.session_state[f"{key_prefix}_state"] = "idle"  # "idle", "writing", "sent"

            feedback_state = st.session_state[f"{key_prefix}_state"]

            # --- Idle state: show Report button ---
            if feedback_state == "idle":
                if st.button("📝 Report / Suggest Update", key=f"{key_prefix}_button"):
                    st.session_state[f"{key_prefix}_state"] = "writing"
                    st.rerun()

            # --- Writing state: show feedback box ---
            elif feedback_state == "writing":
                feedback_text = st.text_area(
                    "Enter your feedback or suggestion here:",
                    key=f"{key_prefix}_text",
                    placeholder="Describe what's wrong or suggest improvements...",
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("📨 Submit Feedback", key=f"{key_prefix}_submit"):
                        feedback_text = feedback_text.strip()
                        if feedback_text:
                            user_query = ""
                            assistant_response = msg["content"]
                            if i > 0 and current_chat["messages"][i - 1]["role"] == "user":
                                user_query = current_chat["messages"][i - 1]["content"]

                            send_feedback_email(user_query, feedback_text, assistant_response)
                            st.session_state[f"{key_prefix}_state"] = "sent"
                            st.rerun()
                        else:
                            st.warning("Please enter some feedback before submitting.")

                with col2:
                    if st.button("❌ Cancel", key=f"{key_prefix}_cancel"):
                        st.session_state[f"{key_prefix}_state"] = "idle"
                        st.rerun()

            # --- Sent state: show success message briefly, then reset ---
            elif feedback_state == "sent":
                st.success("✅ Your feedback was sent successfully! Thank you.")
                st.session_state[f"{key_prefix}_state"] = "idle"

    # Prevent duplicate answers across reruns
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None


    # --- Input box and query handler ---
    def handle_query():
        q = st.session_state[f"input_{current_chat_id}"].strip()
        if not q:
            return

        # Skip if this query was already handled (avoids rerun duplication)
        if st.session_state["last_query"] == q:
            return

        st.session_state["last_query"] = q  # mark it processed

        # Update chat title if new
        if current_chat["title"] == "New Chat":
            current_chat["title"] = q[:40] + "..."

        # Save user message
        current_chat["messages"].append({"role": "user", "content": q})
        current_chat["memory"].save_context({"input": q}, {"output": ""})

        # Prepare memory
        mem_msgs = current_chat["memory"].load_memory_variables({}).get("chat_history", [])
        mem_text = "\n".join(
            [str(m.content if hasattr(m, "content") else m)[:200] for m in mem_msgs[-6:]]
        ) if mem_msgs else ""

        # Generate answer
        with st.spinner("Generating answer via MCP..."):
            resp = mcp_generate(q, memory=mem_text, k=5)
            ans = resp.get("answer", "Error: no answer")

        # Append only once
        current_chat["messages"].append({"role": "assistant", "content": ans})
        current_chat["memory"].save_context({"input": q}, {"output": ans})

        # clear input box
        st.session_state[f"input_{current_chat_id}"] = ""

    st.text_input("Ask your science question:", key=f"input_{current_chat_id}", on_change=handle_query)

    # --- Evaluation ---
    if st.button("Run Evaluation"):
        st.subheader("📊 Evaluation Results")
        with st.spinner("Evaluating model via MCP..."):
            queries = [item["query"] for item in evaluation_dataset]
            labels = [item["ground_truth"] for item in evaluation_dataset]
            preds = []
            for q in queries:
                r = mcp_generate(q, memory="", k=5)
                preds.append(r.get("answer", ""))
            scores = compute_metrics(preds, labels)
            judge_scores = []
            for query, pred, label in zip(queries, preds, labels):
                out = mcp_judge(query, gen_ans=pred, ref_ans=label)
                judge_output = out.get("judge_output", "")
                match = re.search(r"Score:\s*(\d)", judge_output)
                judge_scores.append(int(match.group(1)) if match else 0)
            llm_judge_avg = round(np.mean(judge_scores) / 5, 3) if judge_scores else 0.0
            scores["LLM-Judge Score"] = llm_judge_avg
            st.write("### Summary Metrics")

            st.json(scores)




