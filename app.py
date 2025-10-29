import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import json
from datetime import datetime
import tempfile
import os

def analyze_chats(file):
    if file is None:
        return "Please upload your `conversations.json` file."

    # FIX: Read file content properly
    try:
        file_content = file.read().decode('utf-8')
        data = json.loads(file_content)
    except Exception as e:
        return f"Error reading file: {str(e)}. Make sure it's a valid JSON."

    all_msgs = []
    for conv in data:
        mapping = conv.get("mapping", {})
        current = conv.get("current_node")
        messages = []
        while current and current in mapping:
            node = mapping[current]
            msg = node.get("message")
            if msg and msg.get("content", {}).get("content_type") == "text":
                parts = msg["content"].get("parts", [])
                if parts:
                    text = " ".join(str(p) for p in parts if p).strip()
                    author = "You" if msg["author"]["role"] == "user" else "AI"
                    ts = msg.get("create_time")
                    messages.append({
                        "author": author,
                        "text": text,
                        "timestamp": datetime.fromtimestamp(ts) if ts else None,
                        "title": conv.get("title", "Untitled")
                    })
            current = node.get("parent")
        all_msgs.extend(messages[::-1])

    if not all_msgs:
        return "No text messages found in the file."

    df = pd.DataFrame(all_msgs)
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    you = df[df['author'] == 'You']

    if you.empty:
        return "No messages from you found."

    total_chats = df['title'].nunique()
    your_msgs = len(you)
    total_words = sum(len(re.findall(r'\w+', str(t))) for t in you['text'])
    avg_len = total_words / your_msgs if your_msgs > 0 else 0

    monthly = you.groupby('month').size().sort_values(ascending=False).head(5)

    text_lower = " ".join(you['text'].dropna()).lower()
    topics = {
        'Coding': r'\b(python|code|wrapper|api|build)\b',
        'Learning': r'\b(what|how|explain|why|teach)\b',
        'Science': r'\b(turing|machine|molarity|moles)\b',
        'Fun': r'\b(joke|meme|roast|lol)\b'
    }
    topic_counts = {k: len(re.findall(v, text_lower)) for k, v in topics.items()}

    stop_words = {'the', 'and', 'to', 'a', 'in', 'is', 'you', 'i', 'of', 'it'}
    words = re.findall(r'\w+', text_lower)
    top_words = Counter(w for w in words if w not in stop_words and len(w) > 3).most_common(5)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        monthly.plot(kind='bar', ax=ax1, color='#1DB954')
        ax1.set_title('Busiest Months')
        ax1.set_ylabel('Messages')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart1 = os.path.join(tmpdir, 'months.png')
        fig1.savefig(chart1)
        plt.close(fig1)

        # Pie chart
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.pie(topic_counts.values(), labels=topic_counts.keys(), autopct='%1.1f%%',
                colors=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Top Topics')
        chart2 = os.path.join(tmpdir, 'topics.png')
        fig2.savefig(chart2)
        plt.close(fig2)

        html = f"""
        <div style="text-align:center; font-family:Arial; padding:20px;">
            <h1 style="color:#1DA1F2;">Your AI Wrapped 2025</h1>
            <p><strong>{total_chats} Chats • {your_msgs} Messages • {avg_len:.0f} Avg Words</strong></p>
            <img src="file={chart1}" style="max-width:100%; margin:10px;">
            <img src="file={chart2}" style="max-width:100%; margin:10px;">
            <h3>Top Words</h3>
            <ul style="list-style:none; padding:0;">
        """
        for w, c in top_words:
            html += f"<li><strong>{w}</strong>: {c}</li>"
        html += """
            </ul>
            <p><em>Share: <a href="https://huggingface.co/spaces/saanvisaurus/grok-wrapped">huggingface.co/spaces/saanvisaurus/grok-wrapped</a></em></p>
        </div>
        """
        return html

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Your AI Wrapped 2025")
    gr.Markdown("Upload **your** `conversations.json` from ChatGPT or Grok!")
    file_input = gr.File(label="Upload conversations.json", file_types=[".json"])
    output = gr.HTML()
    file_input.change(analyze_chats, inputs=file_input, outputs=output)

demo.launch()