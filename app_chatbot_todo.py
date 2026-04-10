# ============================================================
# TODO LIST – app_chatbot.py (Chatbot Phân tích Phản hồi SV)
# ============================================================
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import io
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

try:
    from underthesea import sentiment, word_tokenize
except ImportError:
    st.error("Vui lòng cài underthesea: pip install underthesea")
    sentiment = None
    word_tokenize = None

# ============================================================
# CONSTANTS
# ============================================================
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

# TODO 1: Stopwords tiếng Việt (tải từ GitHub hoặc file)
@st.cache_data
def load_stopwords() -> set:
    """Tải stopwords tiếng Việt (dùng danh sách chuẩn từ GitHub)"""
    try:
        url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
        import requests
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            words = [line.strip() for line in resp.text.splitlines() if line.strip()]
            return set(words)
    except:
        pass
    # Fallback stopwords cơ bản
    return {
        "là", "và", "của", "có", "không", "được", "cho", "với", "trong", "này",
        "để", "rất", "nhưng", "thì", "mà", "cũng", "lại", "hay", "nhiều", "ít",
        "tôi", "bạn", "chúng", "ta", "ông", "bà", "anh", "chị", "em", "học", "sinh"
    }

STOPWORDS = load_stopwords()

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    """TODO 11: Khởi tạo session_state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:           # ← Đây là nguyên nhân lỗi trước đó
        st.session_state.history = []               # list[dict] chứa các phân tích
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False

# ============================================================
# HÀM PHÂN TÍCH
# ============================================================
@st.cache_resource
def get_sentiment_model():
    """TODO 2: Cache model underthesea"""
    return sentiment

@st.cache_data
def analyze_feedback(text: str) -> dict:
    """TODO 2, 8, 13: Phân tích cảm xúc + từ khóa + xử lý edge case"""
    if not text or len(text.strip()) < 3:
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.5,
            "keywords": [],
            "timestamp": datetime.now().isoformat()
        }

    try:
        model = get_sentiment_model()
        sent = model(text)
        # underthesea sentiment trả về 'positive'/'negative'/'neutral'
        label = sent if isinstance(sent, str) else sent[0] if isinstance(sent, (list, tuple)) else "neutral"
        
        # Tính confidence giả (vì underthesea không trả confidence chính thức)
        confidence = 0.85 if label in ["positive", "negative"] else 0.65

        # Trích xuất từ khóa (word_tokenize + loại stopwords)
        tokens = word_tokenize(text.lower()) if word_tokenize else text.lower().split()
        keywords = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
        keyword_counts = Counter(keywords).most_common(10)
        
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(confidence, 2),
            "keywords": [k for k, _ in keyword_counts],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        st.warning(f"Lỗi phân tích: {e}")
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.5,
            "keywords": [],
            "timestamp": datetime.now().isoformat()
        }

def render_analysis(result: dict) -> str:
    """TODO: Tạo markdown cho chat bubble"""
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    conf_pct = int(result["confidence"] * 100)
    return f"""
{emoji} **{result['sentiment'].upper()}** ({conf_pct}%)

**Phản hồi:** {result['text']}

**Từ khóa chính:** {', '.join(result['keywords'][:8]) or 'Không có'}
"""

# ============================================================
# UPLOAD & EXPORT
# ============================================================
def handle_file_upload() -> list[str]:
    """TODO 3: Hỗ trợ upload CSV/Excel"""
    uploaded = st.sidebar.file_uploader("📁 Upload file phản hồi (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            # Giả sử cột chứa phản hồi là "feedback" hoặc cột đầu tiên
            col = "feedback" if "feedback" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")
    return []

def export_history(history: list[dict]) -> bytes:
    """TODO 4: Export lịch sử ra CSV"""
    if not history:
        return b""
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode("utf-8")

# ============================================================
# VISUALIZATION
# ============================================================
def render_wordcloud(keywords: list[str]):
    """TODO 5: Word Cloud"""
    if not keywords:
        st.info("Chưa có từ khóa để vẽ word cloud")
        return
    text = " ".join(keywords)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def render_sentiment_timeline(history: list[dict]):
    """TODO 6: Biểu đồ xu hướng cảm xúc"""
    if len(history) < 2:
        return
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["score"] = df["sentiment"].map(sentiment_map)
    
    fig = px.line(df, x="date", y="score", markers=True,
                  title="Xu hướng cảm xúc theo thời gian",
                  labels={"score": "Cảm xúc (Positive=1, Neutral=0, Negative=-1)"})
    st.plotly_chart(fig, use_container_width=True)

def render_sidebar_stats(history: list[dict]):
    """TODO 5, 12: Thống kê sidebar"""
    st.sidebar.subheader("📊 Thống kê tổng hợp")
    
    if not history:
        st.sidebar.info("Chưa có phản hồi nào.")
        return
    
    df = pd.DataFrame(history)
    total = len(df)
    pos = len(df[df["sentiment"] == "positive"])
    neg = len(df[df["sentiment"] == "negative"])
    neu = total - pos - neg
    
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Tổng", total)
    col2.metric("😊 Tích cực", pos)
    col3.metric("😟 Tiêu cực", neg)
    
    st.sidebar.markdown("### Cảm xúc")
    st.sidebar.bar_chart({"Tích cực": pos, "Trung lập": neu, "Tiêu cực": neg}, use_container_width=True)
    
    # Word cloud
    all_keywords = [kw for item in history for kw in item.get("keywords", [])]
    if all_keywords:
        st.sidebar.markdown("### ☁️ Word Cloud")
        render_wordcloud(all_keywords)
    
    # Timeline
    if len(history) >= 3:
        st.sidebar.markdown("### 📈 Xu hướng")
        render_sentiment_timeline(history)

# ============================================================
# QUẢN LÝ LỊCH SỬ
# ============================================================
def save_history():
    """TODO 11: Lưu lịch sử vào JSON (persist khi reload)"""
    try:
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_history():
    """TODO 11: Load lịch sử từ file"""
    try:
        with open("history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def delete_feedback(index: int):
    """TODO 7: Xóa phản hồi"""
    if 0 <= index < len(st.session_state.history):
        del st.session_state.history[index]
        if index < len(st.session_state.messages):
            del st.session_state.messages[index * 2 : index * 2 + 2]  # xóa cả user + assistant
        save_history()
        st.rerun()

# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(page_title="Phân tích Phản hồi SV", page_icon="🤖", layout="wide")
    init_session_state()
    
    # Load lịch sử cũ nếu có
    if not st.session_state.history:
        st.session_state.history = load_history()
    
    # ── Sidebar ──
    with st.sidebar:
        render_sidebar_stats(st.session_state.history)
        
        # TODO 3: Upload file
        feedbacks = handle_file_upload()
        if feedbacks:
            for fb in feedbacks:
                result = analyze_feedback(fb)
                st.session_state.history.append(result)
                st.session_state.messages.append({"role": "user", "content": fb})
                st.session_state.messages.append({"role": "assistant", "content": render_analysis(result)})
            save_history()
            st.sidebar.success(f"Đã thêm {len(feedbacks)} phản hồi từ file!")
            st.rerun()
        
        # TODO 4: Export
        if st.session_state.history:
            csv_bytes = export_history(st.session_state.history)
            st.sidebar.download_button(
                label="📥 Tải lịch sử CSV",
                data=csv_bytes,
                file_name=f"phan_tich_phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        if st.sidebar.button("🗑️ Xóa toàn bộ lịch sử"):
            st.session_state.history.clear()
            st.session_state.messages.clear()
            save_history()
            st.rerun()

    # ── Main Area ──
    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")
    st.markdown("Nhập phản hồi (có thể nhiều dòng) → AI sẽ phân tích cảm xúc và từ khóa.")

    # Hiển thị lịch sử chat
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Nút xóa cho mỗi cặp
            if msg["role"] == "assistant" and (i // 2) < len(st.session_state.history):
                if st.button("🗑️ Xóa", key=f"del_{i}"):
                    delete_feedback(i // 2)

    # Ô chat input
    if prompt := st.chat_input("Nhập phản hồi của sinh viên tại đây..."):
        # Thêm user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        
        for line in lines:
            result = analyze_feedback(line)
            st.session_state.history.append(result)
            
            analysis_md = render_analysis(result)
            st.session_state.messages.append({"role": "assistant", "content": analysis_md})
        
        save_history()   # TODO 11
        st.rerun()

if __name__ == "__main__":
    main()
