# app.py

import os
import torch
import requests #用于发送 HTTP 请求
import json #用于处理 JSON 数据
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel #用于定义请求和响应模型
from typing import List, Dict, Any

# LangChain 和向量数据库相关的导入
from langchain_community.vectorstores import FAISS #用于存储和检索文档向量
from langchain_huggingface import HuggingFaceEmbeddings #用于生成文档的嵌入向量
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever #用于组合多个检索器
from langchain_community.retrievers import BM25Retriever

# --- 1. 初始化和配置 ---
print("正在初始化 FastAPI 应用和 RAG 系统...")

# 加载环境变量
load_dotenv() #加载.env文件中的环境变量
os.environ['HF_HOME'] = 'D:/my_models_cache'  # 设置HuggingFace缓存目录

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #选择使用的计算设备
EMBEDDING_MODEL_NAME_OR_PATH = "D:\BAAIbge-small-zh-v1.5"
FAISS_DB_PATH = "./faiss_index"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Reranker 和 LLM 模型配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 使用一个高效的指令模型
DEEPSEEK_API_BASE = "https://api.siliconflow.cn/v1"

# --- 2. 加载模型和数据 (在应用启动时执行一次) ---
# 检查API密钥
if not DEEPSEEK_API_KEY:
    raise ValueError("错误：环境变量 DEEPSEEK_API_KEY 未设置！")

# 检查FAISS索引是否存在
if not os.path.exists(FAISS_DB_PATH):
    raise FileNotFoundError(f"错误：FAISS 索引目录 '{FAISS_DB_PATH}' 未找到。请先运行 build_index.py 来创建索引。")

# 加载嵌入模型
print(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME_OR_PATH} 到设备: {DEVICE}")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME_OR_PATH,
    model_kwargs={'device': DEVICE}
)

# 加载FAISS向量数据库
print(f"正在从 '{FAISS_DB_PATH}' 加载FAISS数据库...")
faiss_db = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True
)

# 加载文档块，用于BM25检索器
#加载并处理本地 PDF 文档，将其分割成小块
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
PDF_DIR = r"E:\实训\金融数据集-报表"
CHUNK_SIZE = 500 #每个文本块约 500 个字符
CHUNK_OVERLAP = 100 #相邻块之间有 100 个字符的重叠
all_chunks = [] #列表
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    all_chunks.extend(chunks)

# 初始化 BM25 关键词检索器，基于词频和逆文档频率的传统检索方法
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = 3  # 设置关键词检索召回数量，每次检索返回 3 个最相关的文档块

# 初始化 FAISS 向量检索器，基于语义相似度的向量检索方法
vector_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})  # 设置向量检索召回数量

# 组合成 EnsembleRetriever 混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 可以调整两种检索器的权重
)
#BM25 擅长精确匹配关键词，对专业术语和特定概念检索效果好
#向量检索擅长语义匹配，能捕捉同义词、近义词和上下文相关内容
print("\nBM25 和 FAISS 检索器已组合成 EnsembleRetriever。")

print("RAG系统初始化完成，准备好接收请求。")


# --- 3. Pydantic 模型定义 ---
class QueryRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    source_documents: List[SourceDocument]

class HealthResponse(BaseModel):
    status: str
    message: str


# --- 4. 辅助函数 (Reranker 和 LLM 调用) ---

def rerank_documents(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """使用BGE-Reranker对文档进行重排"""
    #提取文档内容
    doc_contents = [doc.page_content for doc in docs]
    #准备API请求
    payload = {"model": RERANKER_MODEL, "query": query, "documents": doc_contents}
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    try:
        #调用Rerank API，获取相关性分数
        response = requests.post(f"{DEEPSEEK_API_BASE}/rerank", json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        rerank_results = response.json().get("results", [])

        # 将rerank结果与原始文档关联并排序
        reranked_items = [{"document": docs[res['index']], "score": res['relevance_score']} for res in rerank_results]
        #按相关性降序排序
        reranked_items.sort(key=lambda x: x['score'], reverse=True)
        #返回top_n文档
        return [item['document'] for item in reranked_items[:top_n]]
    except requests.RequestException as e:
        print(f"Reranker API 调用失败: {e}")
        return docs[:top_n]  # 如果rerank失败，返回初始检索的前N个文档作为备用


def generate_answer(query: str, context_docs: list[Document]) -> str:
    """使用 LLM 和重排后的文档生成答案，使用prompt工程优化"""
    #拼接上下文文档
    context = "\n\n".join([doc.page_content for doc in context_docs])

    #系统提示词工程
    system_prompt = """
你是一个金融服务专家，擅长提供准确、简洁的解答。请根据以下相关文档，回答用户的问题。
- 保持回答的专业性和客观性
- 仅基于提供的文档信息回答
- 对复杂问题进行分步解释
- 提供具体的操作步骤和参考信息
"""
    #用户提示词构造
    user_prompt = f"""
问题：{query}

相关文档：
{context}

请根据上述文档，提供一个清晰、准确的回答。
"""
    #准备LLM API请求
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "max_tokens": 1024, #限制生成长度，避免冗余
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    try:
        # 调用LLM API
        response = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"LLM API 调用失败: {e}")
        return "抱歉，调用大语言模型生成答案时发生网络错误。"
    except (KeyError, IndexError) as e:
        print(f"LLM API 响应格式错误: {e}, 响应内容: {response.text}")
        return "抱歉，解析大语言模型返回的答案时发生错误。"


# --- 5. FastAPI 应用和 API 路由 ---
app = FastAPI(title="RAG API", description="基于 FastAPI 的 RAG 问答系统") #创建 FastAPI 应用实例


@app.post("/rag_query", response_model=QueryResponse) #定义一个 POST 请求的 API 路由，用于处理用户的问题
async def rag_query(request: QueryRequest):
    """
    接收用户问题，执行 RAG+Rerank 流程，并返回LLM生成的答案。
    """
    question = request.question

    if not question:
        raise HTTPException(status_code=400, detail="请求体中必须包含 'question' 字段")

    print(f"\n收到新请求: {question}")

    try:
        # 步骤 1: 初始检索，使用混合检索器
        print("步骤 1: 正在使用混合检索器进行初始检索...")
        initial_docs = ensemble_retriever.invoke(question)
        print(f"  - 检索到 {len(initial_docs)} 篇初始文档。")

        # 步骤 2: 文档重排
        print("步骤 2: 正在使用Reranker进行重排...")
        reranked_docs = rerank_documents(question, initial_docs, top_n=3)
        print(f"  - 重排后保留 {len(reranked_docs)} 篇文档。")

        # 步骤 3: 生成答案
        print("步骤 3: 正在调用LLM生成最终答案...")
        answer = generate_answer(question, reranked_docs)
        print(f"  - LLM生成答案完成。")

        # 准备返回的源文档信息
        source_documents = [SourceDocument(
            content=doc.page_content,
            metadata=doc.metadata
        ) for doc in reranked_docs]

        return QueryResponse(
            success=True,
            question=question,
            answer=answer,
            source_documents=source_documents
        )

    except Exception as e:
        print(f"处理请求时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

#健康检查端点
@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="RAG API 服务正在运行")


# --- 6. 启动应用 ---
if __name__ == '__main__':
    import uvicorn
    # 在生产环境中，应使用 Gunicorn 或其他 ASGI 服务器，而不是 uvicorn 的开发服务器
    uvicorn.run(app, host='0.0.0.0', port=5000) #启动 FastAPI 应用，监听指定的主机和端口

# .\my_project_env\Scripts\activate
# uvicorn fastapi_app:app