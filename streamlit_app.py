"""
台灣年報下載 + 多策略 RAG 問答系統 × Telegram 推播整合版（Streamlit 版）
pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn requests beautifulsoup4

執行方式：
    streamlit run streamlit_app.py

說明：
- Groq API Token／Telegram Bot Token／Chat ID 一律由使用者於介面輸入，程式碼中不寫死任何金鑰或 ID。
- 提供「取得 Chat ID」按鈕（呼叫 getUpdates）與「測試連線」按鈕（呼叫 getMe）。
- RAG 回答會自動把條列重點轉換成「emoji + 重點」格式，可一鍵推播到 Telegram（超過 4000 字元自動分段）。
- 因 Streamlit 每次互動都會重新執行整個腳本，所有「按鈕結果」都存在 st.session_state 內以便畫面保留。
"""

import itertools
import os
import re
import tempfile
import zipfile

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
#  Part 1｜財報下載
# ─────────────────────────────────────────────


def fetch_annual_report(stock_id: str, year: str) -> tuple[str, str | None]:
    url = "https://doc.twse.com.tw/server-java/t57sb01"

    data1 = {
        "id": "", "key": "", "step": "1",
        "co_id": stock_id, "year": year,
        "seamon": "", "mtype": "F", "dtype": "F04",
    }
    try:
        resp1 = requests.post(url, data=data1, timeout=15)
        soup1 = BeautifulSoup(resp1.text, "html.parser")
        link1 = soup1.find("a").text
    except Exception as e:
        return f"❌ [{stock_id}] 取得檔名失敗：{e}", None

    data2 = {"step": "9", "kind": "F", "co_id": stock_id, "filename": link1}
    try:
        resp2 = requests.post(url, data=data2, timeout=15)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        link2 = soup2.find("a").get("href")
    except Exception as e:
        return f"❌ [{stock_id}] 取得 PDF 連結失敗：{e}", None

    try:
        resp3 = requests.get("https://doc.twse.com.tw" + link2, timeout=30)
        filename = f"{year}_{stock_id}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, "wb") as f:
            f.write(resp3.content)
        return f"✅ [{stock_id}] {year} 年報下載成功", filepath
    except Exception as e:
        return f"❌ [{stock_id}] 下載 PDF 失敗：{e}", None


def download_reports(stock_ids_input: str, year: str) -> tuple[str, str | None]:
    raw = stock_ids_input.replace(",", "\n").replace(" ", "\n")
    stock_ids = [s.strip() for s in raw.splitlines() if s.strip()]
    if not stock_ids:
        return "⚠️ 請輸入至少一個股號", None
    if not year.strip().isdigit():
        return "⚠️ 年份格式錯誤，請輸入民國年（如：112）", None

    logs, pdf_paths = [], []
    for sid in stock_ids:
        msg, path = fetch_annual_report(sid, year.strip())
        logs.append(msg)
        if path:
            pdf_paths.append(path)

    summary = "\n".join(logs)
    if len(pdf_paths) == 1:
        return summary, pdf_paths[0]
    if pdf_paths:
        zip_path = os.path.join(tempfile.gettempdir(), f"annual_reports_{year.strip()}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in pdf_paths:
                zf.write(p, os.path.basename(p))
        return summary, zip_path
    return summary, None


# ─────────────────────────────────────────────
#  Part 2｜Telegram 推播
# ─────────────────────────────────────────────

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"
EMOJI_CYCLE = ["📌", "📈", "💰", "⚠️", "🔍", "✅", "📊", "🧾", "💡", "🏢"]


def telegram_test_connection(bot_token: str) -> str:
    bot_token = (bot_token or "").strip()
    if not bot_token:
        return "⚠️ 請先輸入 Bot Token"
    try:
        url = TELEGRAM_API_BASE.format(token=bot_token, method="getMe")
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get("ok"):
            bot_info = data["result"]
            return f"✅ 連線成功！Bot 名稱：@{bot_info.get('username')}"
        return f"❌ 連線失敗：{data}"
    except Exception as e:
        return f"❌ 連線失敗：{e}"


def telegram_get_updates(bot_token: str) -> str:
    bot_token = (bot_token or "").strip()
    if not bot_token:
        return "⚠️ 請先輸入 Bot Token"
    try:
        url = TELEGRAM_API_BASE.format(token=bot_token, method="getUpdates")
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if not data.get("ok"):
            return f"❌ 取得更新失敗：{data}"

        results = data.get("result", [])
        if not results:
            return "⚠️ 目前沒有訊息紀錄，請先在 Telegram 對你的 Bot 傳一則訊息，再按一次此按鈕。"

        lines = ["📋 最近與此 Bot 互動過的 Chat："]
        seen = set()
        for item in results:
            msg = item.get("message") or item.get("channel_post") or {}
            chat = msg.get("chat", {})
            cid = chat.get("id")
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            name = chat.get("title") or chat.get("username") or chat.get("first_name") or "未知"
            lines.append(f"‧ Chat ID：{cid}　(名稱：{name})")

        return "\n".join(lines) if len(lines) > 1 else "⚠️ 沒有找到有效的 Chat，請先傳訊息給 Bot。"
    except Exception as e:
        return f"❌ 取得更新失敗：{e}"


def format_answer_for_telegram(question: str, strategy: str, answer: str) -> str:
    """把 RAG 回答的條列重點轉成「emoji + 重點」格式，適合 Telegram 閱讀。"""
    header = f"📢 年報 RAG 問答結果\n❓ 問題：{question}\n🧭 策略：{strategy}\n"
    emoji_iter = itertools.cycle(EMOJI_CYCLE)

    lines_out = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = stripped.startswith(("‧", "-", "•")) or bool(re.match(r"^\d+[.)]", stripped))
        cleaned = re.sub(r"^[‧\-•\d.)]+\s*", "", stripped)
        if is_bullet:
            lines_out.append(f"{next(emoji_iter)} {cleaned}")
        else:
            lines_out.append(cleaned)

    return header + "\n" + "\n".join(lines_out)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> str:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token or not chat_id:
        return "⚠️ 請先輸入 Bot Token 與 Chat ID"
    if not text.strip():
        return "⚠️ 沒有可傳送的內容"

    url = TELEGRAM_API_BASE.format(token=bot_token, method="sendMessage")
    # Telegram 單則訊息上限約 4096 字元，超過就自動分段傳送
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [text]

    try:
        for i, chunk in enumerate(chunks, 1):
            resp = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=15)
            data = resp.json()
            if not data.get("ok"):
                return f"❌ 第 {i}/{len(chunks)} 段傳送失敗：{data}"
        return f"✅ 已成功傳送到 Telegram（共 {len(chunks)} 則訊息）"
    except Exception as e:
        return f"❌ 傳送失敗：{e}"


def send_answer_to_telegram(question: str, strategy: str, answer: str, bot_token: str, chat_id: str) -> str:
    if not answer or not answer.strip():
        return "⚠️ 請先產生 RAG 回答，再傳送到 Telegram"
    formatted = format_answer_for_telegram(question, strategy, answer)
    return send_telegram_message(bot_token, chat_id, formatted)


# ─────────────────────────────────────────────
#  Part 3｜Multi-Strategy RAG
# ─────────────────────────────────────────────


@st.cache_resource(show_spinner="🔄 載入嵌入模型中（第一次執行需要一些時間）...")
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class MultiStrategyRAG:
    def __init__(self):
        self.client: Groq | None = None
        self.embedding_model = get_embedding_model()
        self.chunks: list[str] = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def set_api_key(self, api_key: str) -> str:
        api_key = (api_key or "").strip()
        if not api_key:
            return "⚠️ 請貼上有效的 Groq API Token"
        try:
            self.client = Groq(api_key=api_key)
            return "✅ API Token 已設定，可以開始使用囉 (｡•̀ᴗ-)✧"
        except Exception as e:
            return f"❌ 設定失敗：{e}"

    def _require_client(self):
        if self.client is None:
            raise RuntimeError("尚未設定 API Token，請先於上方貼上你的 Groq Token 並點擊設定 (>﹏<)")

    def load_pdf_file(self, filepath: str) -> str:
        try:
            reader = PdfReader(filepath)
            full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
            self.chunks = self._split_text(full_text, chunk_size=800, overlap=150)
            self._build_indices()
            return (
                f"✅ 成功載入 PDF（{os.path.basename(filepath)}）！"
                f"共 {len(reader.pages)} 頁，分割為 {len(self.chunks)} 個片段。"
            )
        except Exception as e:
            return f"❌ 載入失敗：{e}"

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = re.sub(r"\s+", " ", text[start : start + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _build_indices(self):
        self.embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings.astype("float32"))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

    def strategy_1_basic_similarity(self, query, top_k=3):
        qv = self.embedding_model.encode([query])
        _, idx = self.index.search(qv.astype("float32"), top_k)
        return [self.chunks[i] for i in idx[0]]

    def strategy_2_tfidf(self, query, top_k=3):
        qv = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * qv.T).toarray().flatten()
        return [self.chunks[i] for i in scores.argsort()[-top_k:][::-1]]

    def strategy_3_hybrid(self, query, top_k=3):
        qv = self.embedding_model.encode([query])
        _, sem_idx = self.index.search(qv.astype("float32"), top_k * 2)
        qv_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * qv_tfidf.T).toarray().flatten()
        tfidf_idx = tfidf_scores.argsort()[-top_k * 2 :][::-1]
        combined = list(set(sem_idx[0].tolist() + tfidf_idx.tolist()))
        return [self.chunks[i] for i in combined[:top_k]]

    def strategy_4_reranking(self, query, top_k=3):
        self._require_client()
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)
        reranked = []
        for chunk in candidates:
            prompt = f"問題：{query}\n\n文本：{chunk[:200]}...\n\n相關度(0-10)："
            try:
                r = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10, temperature=0,
                )
                raw = r.choices[0].message.content.strip()
                score = float(re.findall(r"\d+", raw)[0]) if re.findall(r"\d+", raw) else 0
            except Exception:
                score = 0
            reranked.append((chunk, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in reranked[:top_k]]

    def strategy_5_multi_query(self, query, top_k=3):
        self._require_client()
        prompt = f"將以下問題改寫成3個相關但不同角度的問題，用換行分隔：\n{query}"
        try:
            r = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.7,
            )
            queries = [query] + r.choices[0].message.content.strip().split("\n")[:3]
        except Exception:
            queries = [query]
        all_chunks = []
        for q in queries:
            all_chunks.extend(self.strategy_1_basic_similarity(q, top_k=2))
        return list(dict.fromkeys(all_chunks))[:top_k]

    def strategy_6_contextual_compression(self, query, top_k=3):
        self._require_client()
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            prompt = f"從以下文本中提取與問題「{query}」最相關的1-2句話：\n\n{chunk}"
            try:
                r = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150, temperature=0,
                )
                compressed.append(r.choices[0].message.content.strip())
            except Exception:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query, top_k=3):
        small_chunks = self._split_text(" ".join(self.chunks), chunk_size=300, overlap=50)
        small_emb = self.embedding_model.encode(small_chunks, convert_to_numpy=True)
        small_idx = faiss.IndexFlatL2(small_emb.shape[1])
        small_idx.add(small_emb.astype("float32"))
        qv = self.embedding_model.encode([query])
        _, indices = small_idx.search(qv.astype("float32"), top_k)
        results = []
        for i in indices[0]:
            for big in self.chunks:
                if small_chunks[i] in big:
                    results.append(big)
                    break
        return list(dict.fromkeys(results))[:top_k]

    def strategy_8_hypothetical_answer(self, query, top_k=3):
        self._require_client()
        prompt = f"請對以下問題給出一個假設性的答案：\n{query}"
        try:
            r = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.7,
            )
            hypo = r.choices[0].message.content
        except Exception:
            hypo = query
        qv = self.embedding_model.encode([hypo])
        _, idx = self.index.search(qv.astype("float32"), top_k)
        return [self.chunks[i] for i in idx[0]]

    STRATEGY_MAP = {
        "1. 基礎語意搜尋": "strategy_1_basic_similarity",
        "2. TF-IDF 關鍵詞": "strategy_2_tfidf",
        "3. 混合搜尋": "strategy_3_hybrid",
        "4. 重新排序": "strategy_4_reranking",
        "5. 多查詢擴展": "strategy_5_multi_query",
        "6. 上下文壓縮": "strategy_6_contextual_compression",
        "7. 父子文檔": "strategy_7_parent_child",
        "8. 假設性答案 (HyDE)": "strategy_8_hypothetical_answer",
    }

    def generate_answer(self, query: str, strategy: str, top_k: int = 3):
        if self.client is None:
            return "⚠️ 請先於上方貼上並設定你的 Groq API Token (｡ŏ_ŏ)", ""
        if not self.chunks:
            return "❌ 請先載入 PDF 檔案！ (´；ω；`)", ""

        fn = getattr(self, self.STRATEGY_MAP.get(strategy, "strategy_1_basic_similarity"))
        try:
            relevant_chunks = fn(query, top_k)
        except Exception as e:
            return f"❌ 檢索失敗：{e}", ""

        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = (
            "你是專業但活潑的財報分析助手。請根據下方【上下文】回答【問題】，規則如下：\n"
            "1. 用條列式（每點一行，前面加「‧」）呈現重點，最多 5～7 點。\n"
            "2. 每一點盡量精簡（一行內講完），避免長篇大論。\n"
            "3. 在回答的開頭或結尾自然地加入 1 個顏文字，讓語氣更親切（例如 (๑•̀ㅂ•́)و✧、(´・ω・`) 等）。\n"
            "4. 若上下文沒有相關資訊，請直接說明「上下文中查無相關資訊」，並附上一個顏文字表示可惜。\n"
            "5. 只根據上下文回答，不要虛構數字。\n\n"
            f"【上下文】\n{context}\n\n【問題】{query}\n\n請用繁體中文回答："
        )

        try:
            r = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "你是專業且用詞精簡、條列清楚、偶爾使用顏文字的財務報告分析助手。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600, temperature=0.3,
            )
            answer = r.choices[0].message.content
            source_info = (
                f"📚 使用策略：{strategy}\n📄 檢索片段數：{len(relevant_chunks)}\n\n"
                + "=" * 50 + "\n相關文本片段：\n" + "=" * 50 + "\n\n" + context
            )
            return answer, source_info
        except Exception as e:
            return f"❌ 生成答案失敗：{e}", ""


def generate_answer_and_maybe_send(rag: "MultiStrategyRAG", query, strategy, top_k, auto_send, bot_token, chat_id):
    """產生 RAG 回答，並依 auto_send 決定是否自動推播到 Telegram。"""
    answer, source_info = rag.generate_answer(query, strategy, top_k)
    if auto_send:
        tg_status = send_answer_to_telegram(query, strategy, answer, bot_token, chat_id)
    else:
        tg_status = "（未勾選自動傳送，如需推播請按下方「傳送到 Telegram」按鈕）"
    return answer, source_info, tg_status


# ─────────────────────────────────────────────
#  Part 4｜Streamlit UI
# ─────────────────────────────────────────────

STRATEGY_CHOICES = [
    "1. 基礎語意搜尋",
    "2. TF-IDF 關鍵詞",
    "3. 混合搜尋",
    "4. 重新排序",
    "5. 多查詢擴展",
    "6. 上下文壓縮",
    "7. 父子文檔",
    "8. 假設性答案 (HyDE)",
]

st.set_page_config(
    page_title="台灣年報 RAG 問答系統 × Telegram 推播",
    page_icon="📊",
    layout="wide",
)

# ── session_state 初始化 ─────────────────────
_defaults = {
    "rag": None,
    "key_status": "",
    "tg_setup_status": "",
    "dl_log": "",
    "dl_file_path": None,
    "upload_status": "",
    "b_status": "",
    "answer": "",
    "source_info": "",
    "tg_send_status": "",
    "last_query": "",
    "last_strategy": STRATEGY_CHOICES[0],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.rag is None:
    st.session_state.rag = MultiStrategyRAG()

rag: MultiStrategyRAG = st.session_state.rag

st.title("📊 台灣上市公司年報 × RAG 智慧問答系統（含 Telegram 推播）")
st.markdown(
    "> 資料來源：[證交所 TWSE](https://doc.twse.com.tw)　｜　年份請填**民國年**（例：112 = 2023年）"
)
st.warning("⚠️ Groq API Token 與 Telegram Bot Token 都請自行於下方輸入，不要寫死在程式碼裡分享給別人喔 (＞人＜;)")

# ── Groq 金鑰設定 ──────────────────────────
gcol1, gcol2 = st.columns([3, 1])
with gcol1:
    api_key_input = st.text_input(
        "🔑 Groq API Token",
        type="password",
        placeholder="貼上你的 Groq API Key（gsk_ 開頭）",
        key="api_key_input",
    )
with gcol2:
    st.write("")
    st.write("")
    if st.button("✅ 設定 Token", type="primary", use_container_width=True):
        st.session_state.key_status = rag.set_api_key(api_key_input)
st.text_input("Groq Token 狀態", value=st.session_state.key_status, disabled=True, key="key_status_display")

# ── Telegram 設定 ──────────────────────────
with st.expander("📮 Telegram 推播設定", expanded=True):
    st.markdown(
        "1. 先跟 [@BotFather](https://t.me/BotFather) 建立 Bot 取得 Token\n"
        "2. 對你的 Bot 傳一則訊息，再按「取得 Chat ID」查出你的 Chat ID\n"
        "3. 按「測試連線」確認 Token 是否有效"
    )
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        tg_token_input = st.text_input(
            "🔑 Telegram Bot Token",
            type="password",
            placeholder="貼上你的 Bot Token（格式如 123456:ABC-...）",
            key="tg_token_input",
        )
    with tcol2:
        tg_chatid_input = st.text_input("💬 Chat ID", placeholder="例：7742797182", key="tg_chatid_input")

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("🔌 測試連線", use_container_width=True):
            st.session_state.tg_setup_status = telegram_test_connection(tg_token_input)
    with bcol2:
        if st.button("🔎 取得 Chat ID", use_container_width=True):
            st.session_state.tg_setup_status = telegram_get_updates(tg_token_input)

    st.text_area(
        "Telegram 設定狀態", value=st.session_state.tg_setup_status, height=120,
        disabled=True, key="tg_setup_status_display",
    )

tab1, tab2, tab3 = st.tabs([
    "📥 Step 1｜下載年報",
    "📂 Step 2｜載入 PDF",
    "💬 Step 3｜RAG 問答 + Telegram 推播",
])

# ── Tab 1: 年報下載 ──────────────────────
with tab1:
    st.markdown("### 輸入股號與年份，自動從證交所抓取 PDF")
    c1, c2 = st.columns(2)
    with c1:
        stock_input = st.text_area(
            "股號（可多筆）",
            placeholder="每行一個，或用逗號/空格分隔\n例：\n2330\n2317",
            height=130,
            key="stock_input",
        )
        year_input = st.text_input("年份（民國年）", value="112", key="year_input")
        if st.button("🔍 下載年報", type="primary", use_container_width=True):
            log, file_path = download_reports(stock_input, year_input)
            st.session_state.dl_log = log
            st.session_state.dl_file_path = file_path
        st.caption("範例：`2330` + `2317`（換行分隔）搭配年份 `112`；或單一股號 `2330` 搭配 `111`")
    with c2:
        st.text_area("下載狀態", value=st.session_state.dl_log, height=160, disabled=True, key="dl_log_display")
        dl_path = st.session_state.dl_file_path
        if dl_path and os.path.exists(dl_path):
            with open(dl_path, "rb") as f:
                st.download_button(
                    label="⬇️ 下載檔案（PDF 或 ZIP）",
                    data=f.read(),
                    file_name=os.path.basename(dl_path),
                    use_container_width=True,
                )

# ── Tab 2: 載入 PDF ──────────────────────
with tab2:
    st.markdown(
        "### 選擇 PDF 來源\n"
        "- **方式 A**：直接上傳本機 PDF\n"
        "- **方式 B**：輸入已下載年報的股號＋年份，自動抓取並載入"
    )
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 方式 A｜上傳 PDF")
        upload_input = st.file_uploader("選擇 PDF", type=["pdf"], key="upload_pdf")
        if st.button("📤 上傳並載入", type="primary", use_container_width=True):
            if upload_input is None:
                st.session_state.upload_status = "⚠️ 請選擇 PDF 檔案"
            else:
                tmp_path = os.path.join(tempfile.gettempdir(), upload_input.name)
                with open(tmp_path, "wb") as f:
                    f.write(upload_input.getbuffer())
                st.session_state.upload_status = rag.load_pdf_file(tmp_path)
        st.text_area("狀態", value=st.session_state.upload_status, height=100, disabled=True, key="upload_status_display")

    with c2:
        st.markdown("#### 方式 B｜指定股號年份自動載入")
        b_stock = st.text_input("單一股號", placeholder="例：2330", key="b_stock")
        b_year = st.text_input("年份（民國年）", value="112", key="b_year")
        if st.button("🚀 下載並載入", type="primary", use_container_width=True):
            msg, path = fetch_annual_report(b_stock.strip(), b_year.strip())
            if path is None:
                st.session_state.b_status = msg
            else:
                st.session_state.b_status = msg + "\n" + rag.load_pdf_file(path)
        st.text_area("狀態", value=st.session_state.b_status, height=100, disabled=True, key="b_status_display")

# ── Tab 3: RAG 問答 + Telegram 推播 ──────────
with tab3:
    st.markdown("### 針對已載入的年報進行智慧問答，並可一鍵推播重點到 Telegram (｡•̀ᴗ-)✧")
    c1, c2 = st.columns([1, 2])

    with c1:
        strategy_dd = st.selectbox("RAG 策略", STRATEGY_CHOICES, key="strategy_dd")
        top_k_slider = st.slider("Top-K 片段數", 1, 10, 3, key="top_k_slider")
        auto_send_checkbox = st.checkbox("✅ 產生答案後自動傳送到 Telegram", value=False, key="auto_send")
        st.markdown(
            """
            **策略說明**
            ‧ 1. 基礎語意搜尋：向量相似度
            ‧ 2. TF-IDF：詞頻統計
            ‧ 3. 混合搜尋：語意 + 關鍵詞
            ‧ 4. 重新排序：LLM 重新評分
            ‧ 5. 多查詢擴展：生成多問題
            ‧ 6. 上下文壓縮：提取精華
            ‧ 7. 父子文檔：小→大上下文
            ‧ 8. HyDE：假設答案再搜尋
            """
        )

    with c2:
        q_input = st.text_area("問題", placeholder="例：這份年報的營收狀況如何？", height=100, key="q_input")

        if st.button("🔍 提問", type="primary", use_container_width=True):
            answer, source_info, tg_status = generate_answer_and_maybe_send(
                rag, q_input, strategy_dd, top_k_slider,
                auto_send_checkbox, tg_token_input, tg_chatid_input,
            )
            st.session_state.answer = answer
            st.session_state.source_info = source_info
            st.session_state.tg_send_status = tg_status
            st.session_state.last_query = q_input
            st.session_state.last_strategy = strategy_dd

        st.text_area(
            "AI 回答（條列式＋顏文字）", value=st.session_state.answer,
            height=260, disabled=True, key="answer_display",
        )

        with st.expander("📚 查看檢索片段", expanded=False):
            st.text_area(
                "相關來源", value=st.session_state.source_info,
                height=300, disabled=True, key="source_display",
            )

        if st.button("📤 傳送到 Telegram（含 emoji 重點格式）", use_container_width=True):
            st.session_state.tg_send_status = send_answer_to_telegram(
                st.session_state.last_query or q_input,
                st.session_state.last_strategy or strategy_dd,
                st.session_state.answer,
                tg_token_input, tg_chatid_input,
            )

        st.text_area(
            "Telegram 傳送狀態", value=st.session_state.tg_send_status,
            height=90, disabled=True, key="tg_send_status_display",
        )

        st.caption(
            "範例問題：公司的主要業務為何？／去年的營業收入與淨利各是多少？／"
            "公司面臨哪些主要風險？／研發費用佔營收的比例是多少？"
        )
