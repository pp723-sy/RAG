# build_index.py

import os
import torch
import json
import glob
import re
import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 配置 ---
# 设置缓存目录
os.environ['HF_HOME'] = 'D:/my_models_cache'
# 加载环境变量
load_dotenv()

# 文件和模型路径配置
PDF_DIR = r"E:\实训\RAG_new\金融数据集-报表"
EMBEDDING_MODEL_NAME_OR_PATH = r"D:\BAAIbge-small-zh-v1.5"
FAISS_DB_PATH = "./faiss_index"
METADATA_FILE_NAME = "documents_metadata.json" #元数据是描述数据的数据，用于解释、定位、管理或理解原始数据的背景信息。

# 切分参数，增大，适应文档长句
CHUNK_SIZE = 500 #每个文本块的最大长度
CHUNK_OVERLAP = 100 #块间重叠长度

# 计算设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DeepSeek API 配置 ---
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.base_url = "https://api.siliconflow.cn"


# 从文件名提取公司名称（适配金融报表命名规则）
def extract_company_name(filename: str) -> str:
    """从PDF文件名中提取公司名称（如“_ACME研发有限公司_report.pdf”→“ACME研发有限公司”）"""
    pattern = r"_(.*?)_report\.pdf"
    match = re.search(pattern, filename)
    return match.group(1) if match else filename

def load_and_split_pdfs(pdf_dir: str) -> list[Document]:
    """加载指定目录下的所有PDF文档并进行文本切分。"""
    print(f"正在扫描 PDF 目录: {pdf_dir}")
    all_chunks = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"警告：目录中没有找到 PDF 文件: {pdf_dir}")
        return all_chunks

    print(f"找到 {len(pdf_files)} 个PDF文件")

    for pdf_path in pdf_files:
        try:
            filename = os.path.basename(pdf_path)
            company_name = extract_company_name(filename)  # 提取公司名称
            loader = PyPDFLoader(pdf_path) #PDF加载器
            documents = loader.load() #加载文档
            print(f"\n处理文件: {filename}（公司：{company_name}，原始页数: {len(documents)}）")

            text_splitter = RecursiveCharacterTextSplitter( #文本切分
                chunk_size=CHUNK_SIZE, # 块大小
                chunk_overlap=CHUNK_OVERLAP, # 重叠大小
                length_function=len,
                add_start_index=True, # 记录每个块在原文中的起始位置
            )
            chunks = text_splitter.split_documents(documents) #文档分块
            print(f"  切分为 {len(chunks)} 个文本块")

            # 添加到元数据(增强元数据，补充公司名和文档类型)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source_file": filename,
                    "company_name": company_name,  #公司名称
                    "doc_type": "financial_report",  #文档类型
                    "chunk_id": f"{company_name}_chunk_{i}"  # 唯一chunk_id
                })
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"  处理文件 {pdf_path} 时出错: {e}")

    print(f"\n总共处理 {len(pdf_files)} 个文件，生成 {len(all_chunks)} 个文本块")
    return all_chunks


def get_embeddings_model(model_name_or_path: str) -> HuggingFaceEmbeddings:
    """获取嵌入模型。"""
    print(f"正在加载嵌入模型: {model_name_or_path} (device: {DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={'device': DEVICE}
    )
    print("嵌入模型加载完成。")
    return embeddings


def create_and_save_faiss_db(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, db_path: str):
    """创建 FAISS 向量数据库并保存到本地。"""
    if not chunks:
        print("错误：没有可处理的文本块，跳过FAISS索引创建")
        return

    print("正在创建 FAISS 向量数据库...")
    faiss_db = FAISS.from_documents(chunks, embeddings_model) #使用嵌入模型将文本块转换为向量
    print(f"FAISS 向量数据库创建完成。正在保存到: {db_path}")
    faiss_db.save_local(db_path)
    print("FAISS 向量数据库保存成功。")


#将文档块的元数据保存为 JSON 文件，与向量索引关联，支持混合检索
def create_and_save_metadata(chunks: list[Document], output_dir: str, metadata_file_name: str):
    """创建并保存文档块的元数据。"""
    if not chunks:
        print("错误：没有可处理的文本块，跳过元数据保存")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_list = []
    for chunk in chunks:
        metadata_list.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "company_name": chunk.metadata.get("company_name"),  # 新增公司名称
            "source_file": chunk.metadata.get("source_file"),
            "page": chunk.metadata.get("page"),
            "start_index": chunk.metadata.get("start_index"),
            "content_preview": chunk.page_content[:200] + "..."  # 预览内容
        })

    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    print(f"元数据已保存到：{metadata_file_path}")


def generate_faqs_for_chunk(text_chunk: str, num_questions: int = 3) -> list[str]:
    """使用 LLM 为文档块生成可能的用户问题"""
    print(f"\n正在为文本块生成 {num_questions} 个 FAQ...")
    print(f"原始文本: '{text_chunk[:50]}...'")

    # 优化提示词：聚焦金融领域（财务指标、投融资、风险管理等）
    prompt = f"""
    你是一名资深金融分析师，擅长从财务报告中提炼用户关心的问题。请基于以下文档片段，生成{num_questions}个最可能的用户问题，需满足：
    1. 聚焦财务指标（如营收、净利润）、投融资事件（如融资、收购）、风险管理（如债务重组、合规）等金融相关内容；
    2. 问题需具体、可回答（避免过于宽泛）；
    3. 与文档片段直接相关，不偏离内容。
文档片段：
"{text_chunk}"

请以 JSON 格式输出，key 为 "questions"，value 为一个字符串列表。
例如: {{"questions": ["问题 1？", "问题 2？"]}}
"""
    try:
        # 调用DeepSeek API
        response = openai.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        # 清理响应（去除可能的代码块标记）
        response_content = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
        result = json.loads(response_content)
        questions = result.get("questions", [])
        print("FAQ 生成成功！")
        for q in questions:
            print(f"  - {q}")
        return questions
    except Exception as e:
        print(f"FAQ 生成失败: {e}")
        return []


def add_faqs_to_db(faiss_db, chunks):
    """将 FAQ 作为独立的文档块添加到向量数据库"""
    #使系统能直接通过用户问题匹配 FAQ，进而关联原始文档块
    for chunk in chunks:# 遍历每个原始文档块
        # 获取公司名称（用于元数据关联）
        company_name = chunk.metadata.get("company_name", "未知公司")
        # 为当前文档块生成FAQ
        generated_faqs = generate_faqs_for_chunk(chunk.page_content)
        faq_documents = []
        # 为每个FAQ创建Document
        for i, faq_question in enumerate(generated_faqs):
            faq_doc = Document(
                page_content=faq_question, # 生成的问题作为内容
                metadata={
                    "source": chunk.metadata.get("source_file"),
                    "company_name": company_name,  # 公司名称
                    "original_chunk_id": chunk.metadata.get("chunk_id"),#关联原始文档块ID
                    "original_chunk_content": chunk.page_content[:300] + "...",  # 缩短预览内容
                    "type": "faq_question", # 标记为FAQ类型
                    "faq_index": i
                }
            )
            faq_documents.append(faq_doc)
        if faq_documents:
            faiss_db.add_documents(faq_documents) # 添加到向量库
            print("FAQ 文档已成功添加到向量数据库。")
    return faiss_db


if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        print(f"错误：PDF 目录未找到，请确保 '{PDF_DIR}' 存在。")
    else:
        try:
            # 1. 加载并切分所有PDF文件
            document_chunks = load_and_split_pdfs(PDF_DIR)

            if document_chunks:
                # 2. 初始化嵌入模型
                embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)
                # 3. 创建并保存 FAISS 向量数据库
                faiss_db = FAISS.from_documents(document_chunks, embeddings_model)
                # 4. 创建并保存元数据
                create_and_save_metadata(document_chunks, FAISS_DB_PATH, METADATA_FILE_NAME)
                # 5. 数据增强，添加 FAQ
                faiss_db = add_faqs_to_db(faiss_db, document_chunks)
                faiss_db.save_local(FAISS_DB_PATH)  # 保存添加了FAQ的数据库
                print("\n索引和元数据构建完成！")
        except Exception as e:
            print(f"\n构建过程中发生错误: {e}")