# streamlit_app.py

import streamlit as st
import requests
import json

# --- 页面配置 ---
st.set_page_config(
    page_title="智能金融服务问答助手",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded" #默认展开侧边栏
)

# --- 自定义样式 ---
st.markdown("""
<style>
    /* 全局字体优化 */
    body {
        color: #333333 !important;
    }

    /* 主标题样式 */
    .title-text {
        color: #1e3a8a;
        border-bottom: 2px solid #1e3a8a;
        padding-bottom: 10px;
    }

    /* 侧边栏样式 */
    .sidebar-header {
        color: white !important;
        margin-top: 10px !important;
    }

    /* 聊天消息样式 */
    .user-message {
        background-color: #e0f2fe;
        color: #0c4a6e;
        border-radius: 15px;
        padding: 12px 18px;
        margin: 8px 0;
    }

    .assistant-message {
        background-color: #f0f9ff;
        color: #0c4a6e;
        border-radius: 15px;
        padding: 12px 18px;
        margin: 8px 0;
        border-left: 4px solid #1e3a8a;
    }

    /* 引用来源样式 */
    .source-card {
        background-color: #eff6ff;
        color: #1e3a8a;
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 状态指示器 */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #4b5563;
        font-style: italic;
    }

    /* 修复Streamlit默认样式冲突 */
    .st-emotion-cache-1q7spjk {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 应用常量 ---
#指向本地运行的FastAPI后端
BACKEND_API_URL = "http://127.0.0.1:8000"

# --- 页面标题和描述 ---
st.markdown('<h1 class="title-text">💰 智能金融服务问答助手</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="background-color:#f0f9ff; padding:15px; border-radius:10px; margin-bottom:20px; color:#0c4a6e;">
    <p style="margin:0;">欢迎使用基于RAG（检索增强生成）技术的智能问答系统。您可以就金融服务相关问题进行提问</p>
    <ul style="margin-top:5px; margin-bottom:5px;">
        <li>投资理财/银行产品/金融政策/信贷业务</li>
    </ul>
</div>
""", unsafe_allow_html=True)
st.divider()#添加水平分割线，视觉分隔标题与聊天区

# --- 侧边栏 ---
with st.sidebar:
    st.markdown('<h2 class="sidebar-header">💡 系统信息</h2>', unsafe_allow_html=True)
    st.info(
        "本项目结合了 **检索(Retrieval)**、**重排(Rerank)** 和 **生成(Generation)** "
        "等技术，为您提供专业、准确的金融咨询服务。"
    )

    # 技术栈详情
    with st.expander("### 🛠️ 技术栈详情", expanded=True):#expander折叠面板
        st.markdown("""
        - **前端框架:** Streamlit
        - **后端框架:** FastAPI
        - **向量检索:** FAISS + BGE-Embeddings
        - **精排模型:** BAAI/bge-reranker-v2-m3
        - **生成模型:** deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
        """)

    # 核心功能流程
    with st.expander("### 🔄 核心功能流程", expanded=True):
        st.markdown("""
        1. **混合检索:** 结合BM25关键词检索与FAISS向量检索
        2. **文档重排:** 使用BGE-Reranker对初筛结果精排
        3. **智能生成:** 基于重排后的金融文档生成专业回答
        """)

    # API信息
    st.markdown("### 🌐 API 端点")
    st.code(BACKEND_API_URL, language="bash")

# --- 初始化会话状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "您好！我是智能金融助手，可以为您解答各类金融服务相关问题。请问有什么可以帮您？"
        }
    ]

# --- 显示历史聊天记录 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

            # 显示引用来源
            if "sources" in message and message["sources"]:
                with st.expander("📚 查看引用来源", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
                        st.markdown(f"**来源 {i + 1}** (页码: {source['metadata'].get('page', 'N/A')})")
                        st.text(source['content'])
                        st.markdown('</div>', unsafe_allow_html=True)

# --- 处理用户输入 ---
if prompt := st.chat_input("请输入您关于金融服务的问题..."):
    # 显示用户的问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    # 调用后端API并显示助手的回答
    with st.chat_message("assistant"):
        # 添加思考状态指示器
        with st.status("**正在分析您的问题...**", expanded=False) as status:
            st.markdown('<div class="thinking-indicator">🔍 检索相关金融文档 | ⚖️ 评估最佳答案 | 🤖 生成专业回复</div>',
                        unsafe_allow_html=True)

            try:
                # 准备请求数据
                payload = {"question": prompt}
                headers = {"Content-Type": "application/json"}

                # 发送POST请求
                response = requests.post(
                    f"{BACKEND_API_URL}/rag_query",
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=120
                )

                # 检查响应
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        answer = result.get("answer", "抱歉，未能生成回答。")
                        sources = result.get("source_documents", [])

                        # 更新状态为完成
                        status.update(label="分析完成", state="complete", expanded=False)

                        # 显示最终答案
                        st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)

                        # 存储助手的回答
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                        # 显示引用来源
                        if sources:
                            with st.expander("📚 查看引用来源", expanded=False):
                                for i, source in enumerate(sources):
                                    st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
                                    st.markdown(f"**来源 {i + 1}** (页码: {source['metadata'].get('page', 'N/A')})")
                                    st.text(source['content'])
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.caption("本次回答未引用特定文档")
                    else:
                        status.update(label="处理失败", state="error")
                        error_message = result.get("error", "发生未知错误。")
                        st.error(f"后端处理失败: {error_message}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"后端处理失败: {error_message}"
                        })
                else:
                    status.update(label="请求失败", state="error")
                    error_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    st.error(f"API请求失败，状态码: {response.status_code}\n错误信息: {error_text}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"API请求失败: {response.status_code}"
                    })

            except requests.exceptions.RequestException as e:
                status.update(label="连接失败", state="error")
                st.error(f"连接后端API时发生网络错误: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"网络错误: {str(e)}"
                })
#streamlit run streamlit_app.py