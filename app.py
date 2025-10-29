import gradio as gr
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from datetime import datetime
import tempfile
import os

def analyze_chats(file):
    if file is None:
        return "Upload your `conversations.json`"

    try:
        # FIX: Read file correctly in Gradio
        content = file.read().decode('utf-8')
        data = json.loads(content)
    except Exception as e:
        return f"File error: {e}<br>Make sure it's a valid JSON file."

    messages = []
    for conv in data:
        title = conv.get("title", "Untitled")
        mapping = conv.get("mapping", {})
        current = conv.get("current_node")
        while current and current in mapping:
            node = mapping[current]
            msg = node.get("message")
            if msg and msg.get("content", {}).get("content_type") == "text":
                parts = msg["content"].get("parts", [])
                if parts:
                    text = " ".join(str(p) for p in parts).strip()
                    role = msg["author"]["role"]
                    author = "You" if role == "user" else "AI"
                    ts = msg.get("create_time")
                    if text:
                        messages.append({
                            "author": author,
                            "text": text,
                            "timestamp": datetime.fromtimestamp(ts) if ts else None,
                            "title": title
                        })
            current = node.get("parent")
        messages = messages[::-1]  # Chronological

    if not messages:
        return "No messages found."

    df = pd.DataFrame(messages).dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    you = df[df['author'] == 'You']

    if you.empty:
        return "No messages from you."

    total_chats = df['title'].nunique()
    your_msgs = len(you)
    words = sum(len(re.findall(r'\w+', t)) for t in you['text'])
    avg = words / your_msgs if your_msgs else 0

    monthly = you.groupby('month').size().sort_values(ascending=False).head(5)

    text = " ".join(you['text']).lower()
    topics = {
        'Coding': len(re.findall(r'\b(python|code|wrapper|api|build)\b', text)),
        'Learning': len(re.findall(r'\b(what|how|explain|why|teach)\b', text)),
        'Science': len(re.findall(r'\b(turing|machine|molarity|moles)\b', text))
    }

    stop = {'the','and','to','a','in','is','you','i','of','it'}
    top_words = Counter(w for w in re.findall(r'\w+', text) 
                       if w not in stop and len(w) > 3).most_common(5)

    with tempfile.TemporaryDirectory() as tmp:
        # Bar
        fig, ax = plt.subplots(figsize=(8,5))
        monthly.plot.bar(ax=ax, color='#1DB954')
        ax.set_title('Busiest Months')
        ax.set_ylabel('Messages')
        plt.xticks(rotation=45)
        p1 = f"{tmp}/months.png"
        fig.savefig(p1, bbox_inches='tight')
        plt.close()

        # Pie
        fig, ax = plt.subplots(figsize=(7,7))
        ax.pie(topics.values(), labels=topics.keys(), autopct='%1.1f%%',
               colors=['#FF6B35','#4ECDC4','#45B7D1'])
        ax.set_title('Top Topics')
        p2 = f"{tmp}/topics.png"
        fig.savefig(p2, bbox_inches='tight')
        plt.close()

        html = f"""
        <div style="text-align:center; font-family:Arial;">
            <h1 style="color:#1DA1F2;">Your AI Wrapped 2025</h1>
            <p><b>{total_chats} Chats • {your_msgs} Messages • {avg:.0f} Avg Words</b></p>
            <img src="file={p1}" style="max-width:100%; margin:10px;">
            <img src="file={p2}" style="max-width:100%; margin:10px;">
            <h3>Top Words</h3>
            <ul style="list-style:none;">"""
        for w,c in top_words:
            html += f"<li><b>{w}</b>: {c}</li>"
        html += "</ul></div>"
        return html

with gr.Blocks() as demo:
    gr.Markdown("# Your AI Wrapped 2025")
    gr.Markdown("**Upload `conversations.json` from ChatGPT or Grok**")
    inp = gr.File(label="Upload conversations.json", file_types=[".json"])
    out = gr.HTML()
    inp.change(analyze_chats, inp, out)

demo.launch()