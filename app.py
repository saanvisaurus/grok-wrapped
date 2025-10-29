# app.py - GROK WRAPPED with GRADIO
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import json
from datetime import datetime
import io
import base64

def analyze_chats(file):
    if file is None:
        return "Please upload your `conversations.json` file."

    try:
        data = json.load(file)
    except:
        return "Invalid file. Please upload `conversations.json` from ChatGPT export."

    # === Parse Messages ===
    def extract_messages(conv):
        messages = []
        mapping = conv.get("mapping", {})
        current = conv.get("current_node")
        while current and current in mapping:
            node = mapping[current]
            msg = node.get("message")
            if msg and msg.get("content", {}).get("content_type") == "text":
                parts = msg["content"].get("parts", [])
                if parts:
                    text = " ".join(str(p) for p in parts)
                    author = "You" if msg["author"]["role"] == "user" else "AI"
                    ts = msg.get("create_time")
                    messages.append({
                        "author": author,
                        "text": text,
                        "timestamp": datetime.fromtimestamp(ts) if ts else None,
                        "title": conv.get("title", "Untitled")
                    })
            current = node.get("parent")
        return messages[::-1]

    all_msgs = []
    for conv in data:
        all_msgs.extend(extract_messages(conv))

    if not all_msgs:
        return "No messages found. Try a different file."

    df = pd.DataFrame(all_msgs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['month'] = df['timestamp'].dt.to_period('M')
    you = df[df['author'] == 'You']

    # === Stats ===
    total_chats = df['title'].nunique()
    your_msgs = len(you)
    total_words = sum(len(re.findall(r'\w+', str(t))) for t in you['text'])
    avg_len = total_words / your_msgs if your_msgs else 0

    monthly = you.groupby('month').size().sort_values(ascending=False).head(3)

    text_lower = " ".join(you['text']).lower()
    topics = {
        'Coding': r'\b(python|code|function|debug|javascript|html|css|api|bug|error)\b',
        'Learning': r'\b(explain|what is|how to|teach|learn|tutorial)\b',
        'Creative': r'\b(story|write|poem|idea|imagine|create)\b',
        'Fun': r'\b(joke|meme|funny|lol|haha|roast)\b'
    }
    topic_counts = {name: len(re.findall(pattern, text_lower)) for name, pattern in topics.items()}

    stop_words = {'the', 'and', 'to', 'a', 'in', 'is', 'you', 'i', 'of', 'it'}
    words = re.findall(r'\w+', text_lower)
    word_freq = Counter(w for w in words if w not in stop_words and len(w) > 3)
    top_words = word_freq.most_common(5)

    # === Generate Charts ===
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"

    # Busiest Months
    fig, ax = plt.subplots(figsize=(6, 4))
    monthly.plot(kind='bar', ax=ax, color='#1DB954')
    ax.set_title("Busiest Months")
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45)
    months_img = plot_to_base64(fig)

    # Top Topics
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(topic_counts.values(), labels=topic_counts.keys(), autopct='%1.0f%%', colors=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_title("Top Topics")
    topics_img = plot_to_base64(fig)

    # === HTML Output ===
    html = f"""
    <div style="text-align:center; font-family:Arial; padding:20px;">
        <h1 style="color:#1DA1F2;">Grok Wrapped 2025</h1>
        <p><b>{total_chats}</b> Conversations • <b>{your_msgs}</b> Messages • <b>{avg_len:.0f}</b> Avg Words</p>
        <div style="display:flex; justify-content:center; gap:20px; margin:20px;">
            <img src="{months_img}" width="45%">
            <img src="{topics_img}" width="45%">
        </div>
        <h3>Top Words</h3>
        <ul>{''.join([f"<li><b>{w}</b>: {c}</li>" for w, c in top_words])}</ul>
        <p>Share: <a href="https://huggingface.co/spaces/saanvisaurus/grok-wrapped" target="_blank">huggingface.co/spaces/saanvisaurus/grok-wrapped</a></p>
    </div>
    """
    return html

# === Gradio Interface ===
with gr.Blocks(title="Grok Wrapped", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Grok Wrapped 2025\n*Your AI year in review — like Spotify*")
    file_input = gr.File(label="Upload `conversations.json` (ChatGPT Export)", file_types=[".json"])
    output = gr.HTML()
    file_input.change(analyze_chats, inputs=file_input, outputs=output)
    gr.Markdown("**How to get your data:** Go to [chat.openai.com](https://chat.openai.com) → Settings → Export Data")

demo.launch()