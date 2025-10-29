# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import json
from datetime import datetime

st.set_page_config(page_title="Grok Wrapped", page_icon="🚀", layout="centered")

st.markdown("""
<style>
    .big-font {font-size: 60px !important; font-weight: bold; color: #1DA1F2; text-align: center;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Grok Wrapped 2025</p>', unsafe_allow_html=True)
st.markdown("### *Your AI year in review — like Spotify Wrapped*")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your `conversations.json` (from ChatGPT → Settings → Export Data)",
    type="json"
)

if uploaded_file is not None:
    with st.spinner("🤖 Analyzing your chats..."):
        data = json.load(uploaded_file)
        
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

        df = pd.DataFrame(all_msgs)
        if df.empty:
            st.error("No messages found. Try a different export.")
            st.stop()

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['timestamp'].dt.to_period('M')
        you = df[df['author'] == 'You']

        # === Stats ===
        total_chats = df['title'].nunique()
        your_msgs = len(you)
        total_words = sum(len(re.findall(r'\w+', str(t))) for t in you['text'])
        avg_len = total_words / your_msgs if your_msgs else 0

        monthly = you.groupby('month').size().sort_values(ascending=False).head(3)

        # Topics
        text_lower = " ".join(you['text']).lower()
        topics = {
            'Coding': r'\b(python|code|function|debug|javascript|html|css|api|bug|error)\b',
            'Learning': r'\b(explain|what is|how to|teach|learn|understand|tutorial)\b',
            'Creative': r'\b(story|write|poem|idea|imagine|create|draw|song)\b',
            'Fun': r'\b(joke|meme|funny|lol|haha|roast|silly|weird)\b'
        }
        topic_counts = {name: len(re.findall(pattern, text_lower)) for name, pattern in topics.items()}

        # Top words
        stop_words = {'the', 'and', 'to', 'a', 'in', 'is', 'you', 'i', 'of', 'it', 'that', 'for', 'with', 'on'}
        words = re.findall(r'\w+', text_lower)
        word_freq = Counter(w for w in words if w not in stop_words and len(w) > 3)
        top_words = word_freq.most_common(5)

    # === DISPLAY ===
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>{total_chats}</h3><p>Conversations</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>{your_msgs}</h3><p>Messages Sent</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>{avg_len:.0f}</h3><p>Avg Words</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Busiest Months")
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly.plot(kind='bar', ax=ax, color='#1DB954')
        ax.set_xticklabels([str(m) for m in monthly.index], rotation=45)
        ax.set_ylabel("Messages")
        st.pyplot(fig)

    with col2:
        st.subheader("🧠 Top Topics")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(topic_counts.values(), labels=topic_counts.keys(), autopct='%1.0f%%', startangle=90,
               colors=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.axis('equal')
        st.pyplot(fig)

    st.subheader("🔤 Your Top Words")
    words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    st.table(words_df.style.format({"Count": "{:}"}).set_properties(**{'text-align': 'left'}))

    st.success("🎉 **Your Grok Wrapped is ready!** Share this page with friends.")
    
    st.markdown("""
    <a href="https://x.com/intent/tweet?text=I just got my Grok Wrapped! {your_msgs} messages in {total_chats} chats. Try yours: {url}" target="_blank">
        <button style="background:#1DA1F2; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;">
            Share on X
        </button>
    </a>
    """.format(your_msgs=your_msgs, total_chats=total_chats, url="your-link-here"), unsafe_allow_html=True)

else:
    st.info("👆 Upload your `conversations.json` to see your Wrapped!")
    st.markdown("""
    **How to get your data:**
    1. Go to [chat.openai.com](https://chat.openai.com)
    2. Settings → Data controls → **Export data**
    3. Download the ZIP → extract `conversations.json`
    4. Upload it here!
    """)