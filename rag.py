import torch
import faiss
import numpy as np
from llama_index.core.node_parser import SentenceSplitter
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
from typing import Optional
from openai import OpenAI
import gradio as gr
import os
import fitz  # PyMuPDF
import chardet  # 用于自动检测编码
import traceback
from datetime import datetime, date
from config import Config  # 导入配置文件
from text2vec import TextVector  # 导入文本向量化类

# 知识图谱相关导入（条件导入，避免未安装neo4j时报错）
try:
    from kg_construct import KnowledgeGraphBuilder
    from neo4j_loader import Neo4jHandler
    KG_AVAILABLE = True
    print("✓ 知识图谱功能已加载")
except Exception as e:
    print(f"⚠️ 知识图谱功能不可用: {e}")
    import traceback
    traceback.print_exc()
    KG_AVAILABLE = False

# 学科分类器导入（条件导入）
try:
    from subject_classifier import SubjectClassifier
    _classifier = SubjectClassifier(mode="api")
    CLASSIFIER_AVAILABLE = True
    print("✓ 学科分类器已加载")
except Exception as e:
    print(f"⚠️ 学科分类器不可用: {e}")
    _classifier = None
    CLASSIFIER_AVAILABLE = False

# 全局变量：存储当前问题的学科分类
_current_classification = {"subject": "未分类", "confidence": 0.0}

# 创建知识库根目录和临时文件目录
KB_BASE_DIR = Config.kb_base_dir
os.makedirs(KB_BASE_DIR, exist_ok=True)

# 创建默认知识库目录
DEFAULT_KB = Config.default_kb
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 创建临时输出目录
OUTPUT_DIR = Config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 情感分析与语气调节配置
SENTIMENT_LOG_PATH = os.path.join(OUTPUT_DIR, "sentiment_logs.jsonl")
SENTIMENT_STYLE_MAP = {
    "anxious": "学生可能感到焦急，请先给出最关键的答案或步骤，再补充简短解释，语气温和安抚。",
    "impatient": "学生可能有些不耐烦，请先认可需求，语言简洁直奔主题，避免冗长。",
    "confused": "学生可能感到困惑，请用分步骤说明，举1-2个简单例子，保持耐心。",
    "frustrated": "学生可能情绪低落或沮丧，请先给予肯定与鼓励，再给出清晰指引。",
    "happy": "学生心情较好，可保持友好、积极的语气，适当精炼关键结论。",
    "neutral": "保持专业、清晰、友好的语气，直接回答问题。"
}

client = OpenAI(
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url
)

# LLM 客户端封装
class LLMClient:
    def generate_answer(self, system_prompt, user_prompt, model=Config.llm_model):
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()

# 记录情绪日志
def _log_sentiment(label: str, score: float, text: str):
    """将情绪标签写入本地日志，便于后续调优"""
    try:
        log_item = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "score": float(score),
            "text_preview": (text[:200] + "...") if text and len(text) > 200 else text
        }
        os.makedirs(os.path.dirname(SENTIMENT_LOG_PATH), exist_ok=True)
        with open(SENTIMENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
    except Exception:
        # 日志失败不影响主流程
        pass

# 生成语气提示文本
def _build_tone_instruction(label: str, score: float) -> str:
    """根据情绪标签和强度，生成语气提示文本"""
    label = label or "neutral"
    base = SENTIMENT_STYLE_MAP.get(label, SENTIMENT_STYLE_MAP["neutral"])
    if score is None:
        return base
    # 强度调整：分数高则更强调安抚/简洁
    try:
        score = float(score)
        if score >= 0.75 and label in {"anxious", "frustrated"}:
            base += " 请用更简洁的步骤和安抚语气，先给可执行的答案。"
        elif score >= 0.75 and label == "impatient":
            base += " 进一步压缩冗余描述，先回答结论。"
    except Exception:
        pass
    return base

# 读取JSON文件：兼容常见编码并避免UnicodeDecodeError刷屏
def _load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "gb2312", "latin-1"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            print(f"警告: {path} JSON 解析失败 ({encoding}): {e}")
            return default
        except Exception:
            break
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
        return json.loads(content)
    except Exception as e:
        print(f"警告: {path} 读取失败: {e}")
        return default

# 情感分析主函数
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """调用Qwen接口做轻量情感识别，返回 {label, score, style_instruction}"""
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "style_instruction": SENTIMENT_STYLE_MAP["neutral"]}

    system_prompt = (
        "你是一个中文情绪识别器。请阅读用户的提问，判断其情绪类别与强度。"
        "可用的情绪类别：anxious(焦急)、impatient(不耐烦)、confused(困惑)、"
        "frustrated(沮丧/受挫)、happy(愉快)、neutral(中性)。"
        "请输出JSON，包含字段 label(字符串) 和 score(0-1 浮点数，表示强度)。"
        "只返回JSON。"
    )
    user_prompt = f"用户提问：{text}\n请给出情绪JSON。"

    try:
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        label = str(data.get("label", "neutral")).lower()
        score = float(data.get("score", 0))
    except Exception:
        label = "neutral"
        score = 0.0

    style_instruction = _build_tone_instruction(label, score)
    _log_sentiment(label, score, text)
    return {"label": label, "score": score, "style_instruction": style_instruction}

# 调整答案语气函数
def apply_tone_to_answer(answer: str, tone_instruction: str, sentiment: Dict[str, Any]) -> str:
    """使用LLM在不改动事实的前提下重写答案，调整语气"""
    if not answer or not tone_instruction:
        return answer
    try:
        system_prompt = (
            "你是一个回答语气调节器。在保持信息准确与要点完整的前提下，"
            "根据提供的语气提示，重写或微调回答的措辞。"
        )
        user_prompt = (
            f"语气提示：{tone_instruction}\n"
            f"情绪标签：{sentiment.get('label', 'neutral')}, 强度：{sentiment.get('score', 0)}\n"
            f"原回答：{answer}"
        )
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return answer

# 获取知识库列表
def get_knowledge_bases() -> List[str]:
    """获取所有知识库名称"""
    try:
        if not os.path.exists(KB_BASE_DIR):
            os.makedirs(KB_BASE_DIR, exist_ok=True)
            
        kb_dirs = [d for d in os.listdir(KB_BASE_DIR) 
                  if os.path.isdir(os.path.join(KB_BASE_DIR, d))]
        
        # 确保默认知识库存在
        if DEFAULT_KB not in kb_dirs:
            os.makedirs(os.path.join(KB_BASE_DIR, DEFAULT_KB), exist_ok=True)
            kb_dirs.append(DEFAULT_KB)
            
        return sorted(kb_dirs)
    except Exception as e:
        print(f"获取知识库列表失败: {str(e)}")
        return [DEFAULT_KB]

# 创建新知识库(空壳)
def create_knowledge_base(kb_name: str) -> str:
    """创建新的知识库"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：知识库名称不能为空"
            
        # 净化知识库名称，只允许字母、数字、下划线和中文
        kb_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', kb_name.strip())
        
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 已存在"
            
        os.makedirs(kb_path, exist_ok=True)
        return f"知识库 '{kb_name}' 创建成功"
    except Exception as e:
        return f"创建知识库失败: {str(e)}"

# 删除知识库
def delete_knowledge_base(kb_name: str) -> str:
    """删除指定的知识库"""
    try:
        if kb_name == DEFAULT_KB:
            return f"无法删除默认知识库 '{DEFAULT_KB}'"
            
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 不存在"
            
        shutil.rmtree(kb_path)
        return f"知识库 '{kb_name}' 已删除"
    except Exception as e:
        return f"删除知识库失败: {str(e)}"

# 获取知识库文件列表
def get_kb_files(kb_name: str) -> List[str]:
    """获取指定知识库中的文件列表"""
    try:
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return []
            
        # 获取所有文件（排除索引文件和元数据文件）
        files = [f for f in os.listdir(kb_path) 
                if os.path.isfile(os.path.join(kb_path, f)) and 
                not f.endswith(('.index', '.json'))]
        
        return sorted(files)
    except Exception as e:
        print(f"获取知识库文件列表失败: {str(e)}")
        return []

# 语义分块函数(生成 semantic_chunk_output.json)
def semantic_chunk(text: str, chunk_size=800, chunk_overlap=20) -> List[dict]:
    class EnhancedSentenceSplitter(SentenceSplitter):
        def __init__(self, *args, **kwargs):
            custom_seps = ["；", "!", "?", "\n"]
            separators = [kwargs.get("separator", "。")] + custom_seps
            kwargs["separator"] = '|'.join(map(re.escape, separators))
            super().__init__(*args, **kwargs)

        def _split_text(self, text: str, **kwargs) -> List[str]:
            splits = re.split(f'({self.separator})', text)
            chunks = []
            current_chunk = []
            for part in splits:
                part = part.strip()
                if not part:
                    continue
                if re.fullmatch(self.separator, part):
                    if current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                else:
                    current_chunk.append(part)
            if current_chunk:
                chunks.append("".join(current_chunk))
            return [chunk.strip() for chunk in chunks if chunk.strip()]

    text_splitter = EnhancedSentenceSplitter(
        separator="。",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n"
    )

    paragraphs = []
    current_para = []
    current_len = 0

    for para in text.split("\n\n"):
        para = para.strip()
        para_len = len(para)
        if para_len == 0:
            continue
        if current_len + para_len <= chunk_size:
            current_para.append(para)
            current_len += para_len
        else:
            if current_para:
                paragraphs.append("\n".join(current_para))
            current_para = [para]
            current_len = para_len

    if current_para:
        paragraphs.append("\n".join(current_para))

    chunk_data_list = []
    chunk_id = 0
    skipped_count = 0
    skipped_text = []
    
    for para in paragraphs:
        chunks = text_splitter.split_text(para)
        for chunk in chunks:
            # 降低最小长度要求，从20改为5，避免丢失短内容
            if len(chunk) < 5:
                skipped_count += 1
                skipped_text.append(chunk[:50])  # 记录被跳过的内容
                continue
            chunk_data_list.append({
                "id": f'chunk{chunk_id}',
                "chunk": chunk,
                "method": "semantic_chunk"
            })
            chunk_id += 1
    
    if skipped_count > 0:
        print(f"警告: 跳过了 {skipped_count} 个过短的分块（<5字符），示例: {skipped_text[:3]}")
    
    return chunk_data_list

# 构建Faiss索引(生成 semantic_chunk_metadata.json)
def build_faiss_index(vector_file, index_path, metadata_path):
    try:
        try:
            with open(vector_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(vector_file, 'r', encoding='gbk', errors='ignore') as f:
                data = json.load(f)

        if not data:
            raise ValueError("向量数据为空，请检查输入文件。")
            
        # 确认所有数据项都有向量
        valid_data = []
        for item in data:
            if 'vector' in item and item['vector']:
                valid_data.append(item)
            else:
                print(f"警告: 跳过没有向量的数据项 ID: {item.get('id', '未知')}")
                
        if not valid_data:
            raise ValueError("没有找到任何有效的向量数据。")
            
        # 提取向量
        vectors = [item['vector'] for item in valid_data]
        vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.size == 0:
            raise ValueError("向量数组为空，转换失败。")
            
        # 检查向量维度
        dim = vectors.shape[1]
        n_vectors = vectors.shape[0]
        print(f"构建索引: {n_vectors} 个向量，每个向量维度: {dim}")
        
        # 确定索引类型和参数
        max_nlist = n_vectors // 39
        nlist = min(max_nlist, 128) if max_nlist >= 1 else 1

        if nlist >= 1 and n_vectors >= nlist * 39:
            print(f"使用 IndexIVFFlat 索引，nlist={nlist}")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            if not index.is_trained:
                index.train(vectors)
            index.add(vectors)
        else:
            print(f"使用 IndexFlatIP 索引")
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

        faiss.write_index(index, index_path)
        print(f"成功写入索引到 {index_path}")
        
        # 创建元数据
        metadata = [{'id': item['id'], 'chunk': item['chunk'], 'method': item['method']} for item in valid_data]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"成功写入元数据到 {metadata_path}")
        
        return True
    except Exception as e:
        print(f"构建索引失败: {str(e)}")
        traceback.print_exc()
        raise

# 向量化文件内容(生成 semantic_chunk_vector.json)
def vectorize_file(data_list, output_file_path, field_name="chunk"):
    """向量化文件内容，处理长度限制并确保输入有效"""
    if not data_list:
        print("警告: 没有数据需要向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return
        
    # 准备查询文本，确保每个文本有效且长度适中
    valid_data = []
    valid_texts = []
    
    for item in data_list:
        text = item.get(field_name, "")
        if not text or not isinstance(text, str):
            continue
        # 限制文本长度
        if len(text) > 1000:
            text = text[:1000]
        valid_data.append(item)
        valid_texts.append(text)
    
    if not valid_texts:
        print("警告: 没有有效的文本需要向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return
    
    # 向量化处理
    print(f"对 {len(valid_texts)} 个文本进行向量化...")
    try:
        # 创建向量化器实例
        text_vectorizer = TextVector(Config)
        vectors = text_vectorizer.get_vec(valid_texts)  # 使用 text2vec 模块进行向量化
    except Exception as e:
        print(f"向量化失败: {e}")
        traceback.print_exc()
        vectors = []
    
    if not vectors:
        print("警告: 向量化失败，保存空向量文件")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return
    
    # 组合向量和原始数据
    vectorized_data = []
    for i, item in enumerate(valid_data):
        if i < len(vectors):
            vector = vectors[i]
            if hasattr(vector, 'tolist'):
                item["vector"] = vector.tolist()
            else:
                item["vector"] = list(vector) if not isinstance(vector, list) else vector
            vectorized_data.append(item)
    
    # 保存到JSON文件，确保中文正确显示
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(vectorized_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"向量化完成，共 {len(vectorized_data)} 条记录已保存到 {output_file_path}")

# ==================== 错题本功能：OCR + 题目解析与生成 ====================

# 调用LLM对OCR文本进行错题解析、分类并生成新练习题
def analyze_and_generate_exercises(ocr_items: List[Dict[str, Any]], level: str = "auto", count: int = 5) -> Dict[str, Any]:
    """返回字典包含：summary, categories, exercises"""
    if not ocr_items:
        return {"summary": "未提供有效的错题文本", "categories": [], "exercises": []}

    # 汇总文本
    merged = []
    for it in ocr_items:
        if it.get("text"):
            merged.append(f"[题目: {it.get('filename')}]\n{it.get('text')}")
    merged_text = "\n\n".join(merged)

    system_prompt = (
        "你是一名资深初高中教研老师。请阅读学生上传的错题文本，完成：\n"
        "1) 识别题目类型与知识点分类（如：代数、几何、概率/英语语法/物理等）；\n"
        "2) 总结常见错误原因与解题思路；\n"
        f"3) 根据难度水平({level})，生成 {count} 道新的练习题，覆盖上述知识点，题型多样；\n"
        "4) 以JSON格式返回：{summary, categories:[{name, reasons, tips}], exercises:[{question, answer, category}]}。"
    )

    user_prompt = (
        "以下是学生的错题文本（可能来自OCR，存在噪声）：\n\n" + merged_text +
        "\n\n请按要求返回JSON，若文本不足也尽力分类并生成基础题目。"
    )

    try:
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        # 兜底键
        data.setdefault("summary", "")
        data.setdefault("categories", [])
        data.setdefault("exercises", [])
        return data
    except Exception as e:
        return {"summary": f"生成失败：{str(e)}", "categories": [], "exercises": []}

# 错题本主流程函数，调用 OCR 和 LLM 分析与生成
def process_wrong_problems(images: List[str], level: str = "auto", count: int = 5) -> Tuple[str, Dict[str, Any]]:
    """错题本主流程：OCR->LLM分析与生成。返回（展示文本，结构化结果）。"""
    ocr_items = ocr_images_to_texts(images or [])
    result = analyze_and_generate_exercises(ocr_items, level=level, count=count)
    # 生成一个友好的展示文本
    lines = ["### 错题分析总结", result.get("summary", "")]
    cats = result.get("categories", [])
    if cats:
        lines.append("\n### 分类与建议")
        for c in cats[:6]:
            lines.append(f"- 类别: {c.get('name','')}")
            reasons = ", ".join(c.get('reasons', []) or [])
            tips = ", ".join(c.get('tips', []) or [])
            lines.append(f"  常见错误: {reasons}")
            lines.append(f"  建议: {tips}")
    exs = result.get("exercises", [])
    if exs:
        lines.append("\n### 新练习题（部分预览）")
        for i, q in enumerate(exs[:min(len(exs), count)]):
            lines.append(f"{i+1}. {q.get('question','')}")
            if q.get('answer'):
                lines.append(f"   答案: {q.get('answer')}")
            if q.get('category'):
                lines.append(f"   分类: {q.get('category')}")
    display = "\n".join(lines)
    return display, result

# ==================== 错题本功能结束 ====================

# 向量化查询 - 通用函数，被多处使用
def vectorize_query(query, model_name=Config.model_name, batch_size=Config.batch_size) -> np.ndarray:
    """向量化文本查询，返回嵌入向量，改进错误处理和批处理"""
    embedding_client = OpenAI(
        api_key=Config.api_key,
        base_url=Config.base_url
    )
    
    if not query:
        print("警告: 传入向量化的查询为空")
        return np.array([])
        
    if isinstance(query, str):
        query = [query]
    
    # 验证所有查询文本，确保它们符合API要求
    valid_queries = []
    for q in query:
        if not q or not isinstance(q, str):
            print(f"警告: 跳过无效查询: {type(q)}")
            continue
            
        # 清理文本并检查长度
        clean_q = clean_text(q)
        if not clean_q:
            print("警告: 清理后的查询文本为空")
            continue
            
        # 检查长度是否在API限制范围内
        if len(clean_q) > 8000:
            print(f"警告: 查询文本过长 ({len(clean_q)} 字符)，截断至 8000 字符")
            clean_q = clean_q[:8000]
        
        valid_queries.append(clean_q)
    
    if not valid_queries:
        print("错误: 所有查询都无效，无法进行向量化")
        return np.array([])
    
    # 分批处理有效查询
    all_vectors = []
    for i in range(0, len(valid_queries), batch_size):
        batch = valid_queries[i:i + batch_size]
        try:
            # 记录批次信息便于调试
            print(f"正在向量化批次 {i//batch_size + 1}/{(len(valid_queries)-1)//batch_size + 1}, "
                  f"包含 {len(batch)} 个文本，第一个文本长度: {len(batch[0][:50])}...")
                  
            completion = embedding_client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=Config.dimensions,
                encoding_format="float"
            )
            vectors = [embedding.embedding for embedding in completion.data]
            all_vectors.extend(vectors)
            print(f"批次 {i//batch_size + 1} 向量化成功，获得 {len(vectors)} 个向量")
        except Exception as e:
            print(f"向量化批次 {i//batch_size + 1} 失败：{str(e)}")
            print(f"问题批次中的第一个文本: {batch[0][:100]}...")
            traceback.print_exc()
            # 如果是第一批就失败，直接返回空数组
            if i == 0:
                return np.array([])
            # 否则返回已处理的向量
            break
    
    # 检查是否获得了任何向量
    if not all_vectors:
        print("错误: 向量化过程没有产生任何向量")
        return np.array([])
        
    return np.array(all_vectors)

# 简单的向量搜索，用于基本对比(在已经建立的 FAISS 索引 中查找与查询向量最相似的向量，并返回对应的元数据)
def vector_search(query, index_path, metadata_path, limit):
    """基本向量搜索函数"""
    query_vector = vectorize_query(query)
    if query_vector.size == 0:
        return []
        
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    index = faiss.read_index(index_path)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告：{metadata_path} 包含非法字符，使用 UTF-8 忽略错误重新加载")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)

    D, I = index.search(query_vector, limit)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results

# 清理文本函数
def clean_text(text):
    """清理文本中的非法字符，控制文本长度"""
    if not text:
        return ""
    # 移除控制字符，保留换行和制表符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # 移除重复的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保文本长度在合理范围内
    return text.strip()

# PDF文本提取
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            # 清理不可打印字符，尝试用 UTF-8 解码，失败时忽略非法字符
            text += page_text.encode('utf-8', errors='ignore').decode('utf-8')
        if not text.strip():
            print(f"警告：PDF文件 {pdf_path} 提取内容为空")
        return text
    except Exception as e:
        print(f"PDF文本提取失败：{str(e)}")
        return ""

# 处理单个文件，提取其文本内容
def process_single_file(file_path: str) -> str:
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if not text:
                return f"PDF文件 {file_path} 内容为空或无法提取"
        else:
            with open(file_path, "rb") as f:
                content = f.read()
            result = chardet.detect(content)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            # 尝试多种编码方式
            if detected_encoding and confidence > 0.7:
                try:
                    text = content.decode(detected_encoding, errors="ignore")
                    print(f"文件 {file_path} 使用检测到的编码 {detected_encoding} 解码成功")
                except UnicodeDecodeError:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"文件 {file_path} 使用 {detected_encoding} 解码失败，强制使用 UTF-8 忽略非法字符")
            else:
                # 尝试多种常见编码
                encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1', 'utf-16', 'cp936', 'big5']
                text = None
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        print(f"文件 {file_path} 使用 {encoding} 解码成功")
                        break
                    except UnicodeDecodeError:
                        continue
                
                # 如果所有编码都失败，使用忽略错误的方式解码
                if text is None:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"警告：文件 {file_path} 使用 UTF-8 忽略非法字符")
        
        # 确保文本是干净的，移除非法字符
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        traceback.print_exc()
        return f"处理文件 {file_path} 失败：{str(e)}"

# 批量处理并索引文件 - 修改为支持指定知识库(前面所有函数的整体调用)
def process_and_index_files(file_objs: List, kb_name: str = DEFAULT_KB) -> str:
    """处理并索引文件到指定的知识库"""
    # 确保知识库目录存在
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)
    
    # 设置临时处理文件路径
    semantic_chunk_output = os.path.join(OUTPUT_DIR, "semantic_chunk_output.json")
    semantic_chunk_vector = os.path.join(OUTPUT_DIR, "semantic_chunk_vector.json")
    
    # 设置知识库索引文件路径
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")

    all_chunks = []
    error_messages = []
    try:
        if not file_objs or len(file_objs) == 0:
            return "错误：没有选择任何文件"
            
        print(f"开始处理 {len(file_objs)} 个文件，目标知识库: {kb_name}...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, file_obj.name): file_obj for file_obj in file_objs}
            for future in as_completed(future_to_file):
                result = future.result()
                file_obj = future_to_file[future]
                file_name = file_obj.name
                
                if isinstance(result, str) and result.startswith("处理文件"):
                    error_messages.append(result)
                    print(result)
                    continue
                
                # 检查结果是否为有效文本
                if not result or not isinstance(result, str) or len(result.strip()) == 0:
                    error_messages.append(f"文件 {file_name} 处理后内容为空")
                    print(f"警告: 文件 {file_name} 处理后内容为空")
                    continue
                    
                print(f"对文件 {file_name} 进行语义分块...")
                chunks = semantic_chunk(result)
                
                if not chunks or len(chunks) == 0:
                    error_messages.append(f"文件 {file_name} 无法生成任何分块")
                    print(f"警告: 文件 {file_name} 无法生成任何分块")
                    continue
                
                # 将处理后的文件保存到知识库目录
                file_basename = os.path.basename(file_name)
                dest_file_path = os.path.join(kb_dir, file_basename)
                try:
                    shutil.copy2(file_name, dest_file_path)
                    print(f"已将文件 {file_basename} 复制到知识库 {kb_name}")
                except Exception as e:
                    print(f"复制文件到知识库失败: {str(e)}")
                
                all_chunks.extend(chunks)
                print(f"文件 {file_name} 处理完成，生成 {len(chunks)} 个分块")

        if not all_chunks:
            return "所有文件处理失败或内容为空\n" + "\n".join(error_messages)

        # 确保分块内容干净且长度合适
        valid_chunks = []
        for chunk in all_chunks:
            # 深度清理文本
            clean_chunk_text = clean_text(chunk["chunk"])
            
            # 检查清理后的文本是否有效
            if clean_chunk_text and 1 <= len(clean_chunk_text) <= 8000:
                chunk["chunk"] = clean_chunk_text
                valid_chunks.append(chunk)
            elif len(clean_chunk_text) > 8000:
                # 如果文本太长，截断它
                chunk["chunk"] = clean_chunk_text[:8000]
                valid_chunks.append(chunk)
                print(f"警告: 分块 {chunk['id']} 过长已被截断")
            else:
                print(f"警告: 跳过无效分块 {chunk['id']}")

        if not valid_chunks:
            return "所有生成的分块内容无效或为空\n" + "\n".join(error_messages)
            
        print(f"处理了 {len(all_chunks)} 个分块，有效分块数: {len(valid_chunks)}")

        # 保存语义分块（如果已存在则合并追加，避免覆盖）
        try:
            if os.path.exists(semantic_chunk_output):
                try:
                    with open(semantic_chunk_output, 'r', encoding='utf-8') as jf:
                        existing_chunks = json.load(jf)
                except UnicodeDecodeError:
                    with open(semantic_chunk_output, 'r', encoding='gbk', errors='ignore') as jf:
                        existing_chunks = json.load(jf)
                except Exception:
                    existing_chunks = []
            else:
                existing_chunks = []

            # 计算已有id，避免冲突
            existing_ids = []
            for item in existing_chunks:
                iid = item.get('id')
                if isinstance(iid, str) and iid.startswith('chunk') and iid[5:].isdigit():
                    existing_ids.append(int(iid[5:]))
            next_chunk_id = max(existing_ids) + 1 if existing_ids else 0

            existing_id_set = {it.get('id') for it in existing_chunks}
            for ch in valid_chunks:
                if 'id' not in ch or ch.get('id') in existing_id_set:
                    ch['id'] = f'chunk{next_chunk_id}'
                    next_chunk_id += 1

            merged_chunks = existing_chunks + valid_chunks

            with open(semantic_chunk_output, 'w', encoding='utf-8') as json_file:
                json.dump(merged_chunks, json_file, ensure_ascii=False, indent=4)
            print(f"语义分块完成: {semantic_chunk_output} (已合并，新增 {len(valid_chunks)} 条)")
        except Exception as e:
            print(f"保存语义分块失败: {e}")
            traceback.print_exc()
            return f"保存语义分块失败: {str(e)}"

        # 向量化语义分块
        print(f"开始向量化 {len(valid_chunks)} 个分块...")
        vectorize_file(valid_chunks, semantic_chunk_vector)
        print(f"语义分块向量化完成: {semantic_chunk_vector}")

        # 验证并合并向量文件（避免覆盖已有知识库向量）
        try:
            with open(semantic_chunk_vector, 'r', encoding='utf-8') as f:
                new_vector_data = json.load(f)

            if not new_vector_data or len(new_vector_data) == 0:
                return f"向量化失败: 生成的向量文件为空\n" + "\n".join(error_messages)

            # 确保每条记录包含向量
            if 'vector' not in new_vector_data[0]:
                return f"向量化失败: 数据中缺少向量字段\n" + "\n".join(error_messages)

            print(f"成功生成 {len(new_vector_data)} 个向量")
        except Exception as e:
            return f"读取向量文件失败: {str(e)}\n" + "\n".join(error_messages)

        # 将新生成的向量与知识库中已有的向量合并，避免覆盖
        kb_vector_path = os.path.join(kb_dir, "semantic_chunk_vector.json")
        merged_vectors = []
        try:
            # 读取已有向量（如果存在）
            if os.path.exists(kb_vector_path):
                try:
                    with open(kb_vector_path, 'r', encoding='utf-8') as f:
                        existing_vectors = json.load(f)
                except Exception:
                    existing_vectors = []
            else:
                existing_vectors = []

            # 计算已有id的最大索引，确保新分块id不冲突
            existing_ids = []
            for item in existing_vectors:
                iid = item.get('id')
                if isinstance(iid, str) and iid.startswith('chunk') and iid[5:].isdigit():
                    existing_ids.append(int(iid[5:]))
            next_id = max(existing_ids) + 1 if existing_ids else 0

            existing_id_set = {it.get('id') for it in existing_vectors}
            # 为冲突或缺失的id重新分配唯一id
            for it in new_vector_data:
                if 'id' not in it or it.get('id') in existing_id_set:
                    it['id'] = f'chunk{next_id}'
                    next_id += 1

            merged_vectors = existing_vectors + new_vector_data

            # 将合并后的向量保存到知识库目录（持久化）
            with open(kb_vector_path, 'w', encoding='utf-8') as f:
                json.dump(merged_vectors, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"合并向量文件时出错: {e}")
            traceback.print_exc()
            return f"合并向量文件失败: {str(e)}"

        # 构建索引（使用合并后的向量文件）
        print(f"开始为知识库 {kb_name} 构建索引 (使用合并后向量文件)...")
        build_faiss_index(kb_vector_path, semantic_chunk_index, semantic_chunk_metadata)
        print(f"知识库 {kb_name} 索引构建完成: {semantic_chunk_index}")

        status = f"知识库 {kb_name} 更新成功！共处理 {len(valid_chunks)} 个有效分块。\n"
        if error_messages:
            status += "以下文件处理过程中出现问题：\n" + "\n".join(error_messages)
        return status
    except Exception as e:
        error = f"知识库 {kb_name} 索引构建过程中出错：{str(e)}"
        print(error)
        traceback.print_exc()
        return error + "\n" + "\n".join(error_messages)

# 核心联网搜索功能
def get_search_background(query: str, max_length: int = 1500) -> str:
    try:
        from retrievor import q_searching
        search_results = q_searching(query)
        cleaned_results = re.sub(r'\s+', ' ', search_results).strip()
        return cleaned_results[:max_length]
    except Exception as e:
        print(f"联网搜索失败：{str(e)}")
        return ""

# 根据学科分类增强system prompt
def enhance_prompt_by_subject(base_prompt: str, subject: str) -> str:
    """
    根据学科分类增强system prompt，使回答更有针对性
    
    Args:
        base_prompt: 基础prompt
        subject: 学科分类（理科/文科/未分类）
    
    Returns:
        增强后的prompt
    """
    if subject == "理科":
        enhancement = """

        **理科回答准则**：
        1. 强调准确性和逻辑严谨性
        2. 提供精确的公式、定理和计算步骤
        3. 使用专业术语和科学表达
        4. 如有必要，展示推导过程
        5. 注重因果关系和量化分析
        """
        return base_prompt + enhancement
    
    elif subject == "文科":
        enhancement = """

        **文科回答准则**：
        1. 提供完整的历史背景和文化语境
        2. 注意观点的多元性和辩证性
        3. 适当引用经典著作或名人观点
        4. 强调理解深度和思想性
        5. 注重人文关怀和价值判断
        """
        return base_prompt + enhancement
    
    else:
        # 未分类或分类失败，使用通用prompt
        return base_prompt

# 基本的回答生成
def generate_answer_from_llm(question: str, system_prompt: str = "你是一名专业学术学习助手，请根据以下背景信息和用户提问帮助学习者理解问题。", background_info: Optional[str] = None) -> str:
    llm_client = LLMClient()
    user_prompt = f"问题：{question}"
    if background_info:
        user_prompt = f"背景知识：{background_info}\n\n{user_prompt}"
    try:
        answer = llm_client.generate_answer(system_prompt, user_prompt)
        return answer
    except Exception as e:
        return f"生成回答时出错：{str(e)}"

# 多跳推理RAG系统 - 核心创新点
class ReasoningRAG:
    """
    多跳推理RAG系统，通过迭代式的检索和推理过程回答问题，支持流式响应
    """
    
    def __init__(self, 
                 index_path: str, 
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = Config.llm_model,
                 verbose: bool = False):
        """
        初始化推理RAG系统
        
        参数:
            index_path: FAISS索引的路径
            metadata_path: 元数据JSON文件的路径
            max_hops: 最大推理-检索跳数
            initial_candidates: 初始检索候选数量
            refined_candidates: 精炼检索候选数量
            reasoning_model: 用于推理步骤的LLM模型
            verbose: 是否打印详细日志
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_hops = max_hops
        self.initial_candidates = initial_candidates
        self.refined_candidates = refined_candidates
        self.reasoning_model = reasoning_model
        self.verbose = verbose
        
        # 加载索引和元数据
        self._load_resources()
        
    def _load_resources(self):
        """加载FAISS索引和元数据"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except UnicodeDecodeError:
                with open(self.metadata_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    self.metadata = json.loads(content)
        else:
            raise FileNotFoundError(f"Index or metadata not found at {self.index_path} or {self.metadata_path}")
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """将查询转换为向量"""
        return vectorize_query(query).reshape(1, -1)
    
    def _retrieve(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """使用向量相似性检索块"""
        if query_vector.size == 0:
            return []
            
        D, I = self.index.search(query_vector, limit)
        results = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
        return results
    
    def _generate_reasoning(self, 
                           query: str, 
                           retrieved_chunks: List[Dict[str, Any]], 
                           previous_queries: List[str] = None,
                           hop_number: int = 0) -> Dict[str, Any]:
        """
        为检索到的信息生成推理分析并识别信息缺口
        
        返回包含以下字段的字典:
            - analysis: 对当前信息的推理分析
            - missing_info: 已识别的缺失信息
            - follow_up_queries: 填补信息缺口的后续查询列表
            - is_sufficient: 表示信息是否足够的布尔值
        """
        if previous_queries is None:
            previous_queries = []
            
        # 为模型准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i+1}]: {chunk['chunk']}" 
                                 for i, chunk in enumerate(retrieved_chunks)])
        
        previous_queries_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(previous_queries)])
        
        system_prompt = """
        你是一名专业学术学习助手，帮助学生理解学术内容、解答学习问题，并提供学习建议。
        你的任务是分析检索到的学术资料，识别缺失的学习信息，并提出有针对性的后续学习建议。
        
        重点关注学术领域知识，如:
        - 数学公式与定理
        - 编程技巧与算法
        - 科学理论和实验
        - 历史事件和文化背景
        - 语言学习与语法结构
        """
        
        user_prompt = f"""
        ## 原始查询
        {query}
        
        ## 先前查询（如果有）
        {previous_queries_text if previous_queries else "无"}
        
        ## 检索到的信息（跳数 {hop_number}）
        {chunks_text if chunks_text else "未检索到信息。"}
        
        ## 你的任务
        1. 分析已检索到的信息与原始查询的关系
        2. 确定能够更完整回答查询的特定缺失信息
        3. 提出1-3个针对性的后续查询，以检索缺失信息
        4. 确定当前信息是否足够回答原始查询
        
        以JSON格式回答，包含以下字段:
        - analysis: 对当前信息的详细分析
        - missing_info: 特定缺失信息的列表
        - follow_up_queries: 1-3个具体的后续查询
        - is_sufficient: 表示信息是否足够的布尔值
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            reasoning_text = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                reasoning = json.loads(reasoning_text)
                # 确保预期的键存在
                required_keys = ["analysis", "missing_info", "follow_up_queries", "is_sufficient"]
                for key in required_keys:
                    if key not in reasoning:
                        reasoning[key] = [] if key != "is_sufficient" else False
                return reasoning
            except json.JSONDecodeError:
                # 如果JSON解析失败，则回退
                if self.verbose:
                    print(f"无法从模型输出解析JSON: {reasoning_text[:100]}...")
                return {
                    "analysis": "无法分析检索到的信息。",
                    "missing_info": ["无法识别缺失信息"],
                    "follow_up_queries": [],
                    "is_sufficient": False
                }
                
        except Exception as e:
            if self.verbose:
                print(f"推理生成错误: {e}")
                print(traceback.format_exc())
            return {
                "analysis": "分析过程出错。",
                "missing_info": [],
                "follow_up_queries": [],
                "is_sufficient": False
            }
    
    def _synthesize_answer(self, 
                          query: str, 
                          all_chunks: List[Dict[str, Any]],
                          reasoning_steps: List[Dict[str, Any]],
                          use_table_format: bool = False) -> str:
        """从所有检索到的块和推理步骤中合成最终答案"""
        # 合并所有块，去除重复
        unique_chunks = []
        chunk_ids = set()
        for chunk in all_chunks:
            if chunk["id"] not in chunk_ids:
                unique_chunks.append(chunk)
                chunk_ids.add(chunk["id"])
        
        # 准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i+1}]: {chunk['chunk']}" 
                                  for i, chunk in enumerate(unique_chunks)])
        
        # 准备推理跟踪
        reasoning_trace = ""
        for i, step in enumerate(reasoning_steps):
            reasoning_trace += f"\n\n推理步骤 {i+1}:\n"
            reasoning_trace += f"分析: {step['analysis']}\n"
            reasoning_trace += f"缺失信息: {', '.join(step['missing_info'])}\n"
            reasoning_trace += f"后续查询: {', '.join(step['follow_up_queries'])}"
        
        system_prompt = """
        你是一名专业的学术学习助手。基于检索到的信息块，为用户的查询合成一个全面的答案。
        
        重点提供有关学术知识的准确、基于证据的信息，包括公式推导、编程实现、科学理论和实验等方面。
        
        逻辑地组织你的答案，并在适当时引用块中的具体信息。如果信息不完整，请承认限制。
        """
        
        output_format_instruction = ""
        if use_table_format:
            output_format_instruction = """
            请尽可能以Markdown表格格式组织你的回答。如果信息适合表格形式展示，请使用表格；
            如果不适合表格形式，可以先用文本介绍，然后再使用表格总结关键信息。
            
            表格语法示例：
            | 标题1 | 标题2 | 标题3 |
            | ----- | ----- | ----- |
            | 内容1 | 内容2 | 内容3 |
            
            确保表格格式符合Markdown标准，以便正确渲染。
            """
        
        user_prompt = f"""
        ## 原始查询
        {query}
        
        ## 检索到的信息块
        {chunks_text}
        
        ## 推理过程
        {reasoning_trace}
        
        ## 你的任务
        使用提供的信息块为原始查询合成一个全面的答案。你的答案应该:
        
        1. 直接回应查询
        2. 结构清晰，易于理解
        3. 基于检索到的信息
        4. 承认可用信息中的任何重大缺口
        
        {output_format_instruction}
        
        以直接回应提出原始查询的用户的方式呈现你的答案。
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"答案合成错误: {e}")
                print(traceback.format_exc())
            return "由于出错，无法生成答案。"
    
    # 多跳 RAG 的“直播版”，和前端深度绑定
    def stream_retrieve_and_answer(self, query: str, use_table_format: bool = False):
        """
        执行多跳检索和回答生成的流式方法，逐步返回结果
        
        这是一个生成器函数，会在处理的每个阶段产生中间结果
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        
        # 生成状态更新
        yield {
            "status": "正在将查询向量化...",
            "reasoning_display": "",
            "answer": None,
            "all_chunks": [],
            "reasoning_steps": []
        }
        
        # 初始检索
        try:
            query_vector = self._vectorize_query(query)
            if query_vector.size == 0:
                yield {
                    "status": "向量化失败",
                    "reasoning_display": "由于嵌入错误，无法处理查询。",
                    "answer": "由于嵌入错误，无法处理查询。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return
                
            yield {
                "status": "正在执行初始检索...",
                "reasoning_display": "",
                "answer": None,
                "all_chunks": [],
                "reasoning_steps": []
            }
            
            initial_chunks = self._retrieve(query_vector, self.initial_candidates)
            all_chunks.extend(initial_chunks)
            
            if not initial_chunks:
                yield {
                    "status": "未找到相关信息",
                    "reasoning_display": "未找到与您的查询相关的信息。",
                    "answer": "未找到与您的查询相关的信息。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return
            
            # 更新状态，展示找到的初始块
            chunks_preview = "\n".join([f"- {chunk['chunk'][:100]}..." for chunk in initial_chunks[:2]])
            yield {
                "status": f"找到 {len(initial_chunks)} 个相关信息块，正在生成初步分析...",
                "reasoning_display": f"### 检索到的初始信息\n{chunks_preview}\n\n### 正在分析...",
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": []
            }
            
            # 初始推理
            reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
            reasoning_steps.append(reasoning)
            
            # 生成当前的推理显示
            reasoning_display = "### 多跳推理过程\n"
            reasoning_display += f"**推理步骤 1**\n"
            reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
            reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
            if reasoning['follow_up_queries']:
                reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
            reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n\n"
            
            yield {
                "status": "初步分析完成",
                "reasoning_display": reasoning_display,
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
            # 检查是否需要额外的跳数
            hop = 1
            while (hop < self.max_hops and 
                  not reasoning["is_sufficient"] and 
                  reasoning["follow_up_queries"]):
                
                follow_up_status = f"执行跳数 {hop}，正在处理 {len(reasoning['follow_up_queries'])} 个后续查询..."
                yield {
                    "status": follow_up_status,
                    "reasoning_display": reasoning_display + f"\n\n### {follow_up_status}",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                hop_chunks = []
                
                # 处理每个后续查询
                for i, follow_up_query in enumerate(reasoning["follow_up_queries"]):
                    all_queries.append(follow_up_query)
                    
                    query_status = f"处理后续查询 {i+1}/{len(reasoning['follow_up_queries'])}: {follow_up_query}"
                    yield {
                        "status": query_status,
                        "reasoning_display": reasoning_display + f"\n\n### {query_status}",
                        "answer": None,
                        "all_chunks": all_chunks,
                        "reasoning_steps": reasoning_steps
                    }
                    
                    # 为后续查询检索
                    follow_up_vector = self._vectorize_query(follow_up_query)
                    if follow_up_vector.size > 0:
                        follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                        hop_chunks.extend(follow_up_chunks)
                        all_chunks.extend(follow_up_chunks)
                        
                        # 更新状态，显示新找到的块数量
                        yield {
                            "status": f"查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "reasoning_display": reasoning_display + f"\n\n为查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "answer": None,
                            "all_chunks": all_chunks,
                            "reasoning_steps": reasoning_steps
                        }
                
                # 为此跳数生成推理
                yield {
                    "status": f"正在为跳数 {hop} 生成推理分析...",
                    "reasoning_display": reasoning_display + f"\n\n### 正在为跳数 {hop} 生成推理分析...",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                reasoning = self._generate_reasoning(
                    query, 
                    hop_chunks, 
                    previous_queries=all_queries[:-1],
                    hop_number=hop
                )
                reasoning_steps.append(reasoning)
                
                # 更新推理显示
                reasoning_display += f"\n**推理步骤 {hop+1}**\n"
                reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
                reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
                if reasoning['follow_up_queries']:
                    reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
                reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n"
                
                yield {
                    "status": f"跳数 {hop} 完成",
                    "reasoning_display": reasoning_display,
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                hop += 1
            
            # 合成最终答案
            yield {
                "status": "正在合成最终答案...",
                "reasoning_display": reasoning_display + "\n\n### 正在合成最终答案...",
                "answer": "正在处理您的问题，请稍候...",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
            answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)
            
            # 为最终显示准备检索内容汇总
            all_chunks_summary = "\n\n".join([f"**检索块 {i+1}**:\n{chunk['chunk']}" 
                                           for i, chunk in enumerate(all_chunks[:10])])  # 限制显示前10个块
            
            if len(all_chunks) > 10:
                all_chunks_summary += f"\n\n...以及另外 {len(all_chunks) - 10} 个块（总计 {len(all_chunks)} 个）"
                
            enhanced_display = reasoning_display + "\n\n### 检索到的内容\n" + all_chunks_summary + "\n\n### 回答已生成"
            
            yield {
                "status": "回答已生成",
                "reasoning_display": enhanced_display,
                "answer": answer,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            if self.verbose:
                print(error_msg)
                print(traceback.format_exc())
            
            yield {
                "status": "处理出错",
                "reasoning_display": error_msg,
                "answer": f"处理您的问题时遇到错误: {str(e)}",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
    
    # 多跳 RAG 的“非直播版”，一次性返回结果，和后端深度绑定
    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        执行多跳检索和回答生成的主要方法
        
        返回:
            包含以下内容的元组:
            - 最终答案
            - 包含推理步骤和所有检索到的块的调试字典
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        debug_info = {"reasoning_steps": [], "all_chunks": [], "all_queries": all_queries}
        
        # 初始检索
        query_vector = self._vectorize_query(query)
        if query_vector.size == 0:
            return "由于嵌入错误，无法处理查询。", debug_info
            
        initial_chunks = self._retrieve(query_vector, self.initial_candidates)
        all_chunks.extend(initial_chunks)
        debug_info["all_chunks"].extend(initial_chunks)
        
        if not initial_chunks:
            return "未找到与您的查询相关的信息。", debug_info
        
        # 初始推理
        reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
        reasoning_steps.append(reasoning)
        debug_info["reasoning_steps"].append(reasoning)
        
        # 检查是否需要额外的跳数
        hop = 1
        while (hop < self.max_hops and 
               not reasoning["is_sufficient"] and 
               reasoning["follow_up_queries"]):
            
            if self.verbose:
                print(f"开始跳数 {hop}，有 {len(reasoning['follow_up_queries'])} 个后续查询")
            
            hop_chunks = []
            
            # 处理每个后续查询
            for follow_up_query in reasoning["follow_up_queries"]:
                all_queries.append(follow_up_query)
                debug_info["all_queries"].append(follow_up_query)
                
                # 为后续查询检索
                follow_up_vector = self._vectorize_query(follow_up_query)
                if follow_up_vector.size > 0:
                    follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                    hop_chunks.extend(follow_up_chunks)
                    all_chunks.extend(follow_up_chunks)
                    debug_info["all_chunks"].extend(follow_up_chunks)
            
            # 为此跳数生成推理
            reasoning = self._generate_reasoning(
                query, 
                hop_chunks, 
                previous_queries=all_queries[:-1],
                hop_number=hop
            )
            reasoning_steps.append(reasoning)
            debug_info["reasoning_steps"].append(reasoning)
            
            hop += 1
        
        # 合成最终答案
        answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)
        
        return answer, debug_info

# 基于选定知识库生成索引路径
def get_kb_paths(kb_name: str) -> Dict[str, str]:
    """获取指定知识库的索引文件路径"""
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    return {
        "index_path": os.path.join(kb_dir, "semantic_chunk.index"),
        "metadata_path": os.path.join(kb_dir, "semantic_chunk_metadata.json")
    }

# 基于指定知识库使用多跳推理 RAG 生成答案
def multi_hop_generate_answer(query: str, kb_name: str, use_table_format: bool = False, system_prompt: str = "你是一名专业学术学习助手。") -> Tuple[str, Dict]:
    """使用多跳推理RAG生成答案，基于指定知识库"""
    kb_paths = get_kb_paths(kb_name)
    
    reasoning_rag = ReasoningRAG(
        index_path=kb_paths["index_path"],
        metadata_path=kb_paths["metadata_path"],
        max_hops=3,
        initial_candidates=5,
        refined_candidates=3,
        reasoning_model=Config.llm_model,
        verbose=True
    )
    
    answer, debug_info = reasoning_rag.retrieve_and_answer(query, use_table_format)
    return answer, debug_info

# 使用简单向量检索(朴素的一跳式)生成答案，基于指定知识库
def simple_generate_answer(query: str, kb_name: str, use_table_format: bool = False) -> str:
    """使用简单的向量检索生成答案，不使用多跳推理"""
    try:
        kb_paths = get_kb_paths(kb_name)
        
        # 使用基本向量搜索
        search_results = vector_search(query, kb_paths["index_path"], kb_paths["metadata_path"], limit=5)
        
        if not search_results:
            return "未找到相关信息。"
        
        # 准备背景信息
        background_chunks = "\n\n".join([f"[相关信息 {i+1}]: {result['chunk']}" 
                                       for i, result in enumerate(search_results)])
        
        # 生成答案
        system_prompt = "你是一名专业学术学习助手，请根据以下背景信息和用户提问帮助学习者理解问题。"
        
        # 【方案2：根据学科增强Prompt】
        subject = _current_classification.get('subject', '未分类')
        system_prompt = enhance_prompt_by_subject(system_prompt, subject)

        if use_table_format:
            system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
        
        user_prompt = f"""
        问题：{query}
        
        背景信息：
        {background_chunks}
        
        请基于以上背景信息回答用户的问题。
        """
        
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"生成答案时出错：{str(e)}"

# 修改主要的问题处理函数以支持指定知识库(整套系统里对外的总入口函数)
def ask_question_parallel(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, use_table_format: bool = False, multi_hop: bool = False) -> str:
    """基于指定知识库回答问题"""
    try:
        # 自动分类问题（理科/文科）
        global _current_classification
        classification_info = ""
        if CLASSIFIER_AVAILABLE and _classifier:
            try:
                classification = _classifier.classify(question)
                subject = classification['subject']
                confidence = classification['confidence']
                reason = classification.get('reason', '无')
                
                # 更新全局分类信息（供后续函数使用）
                _current_classification = {"subject": subject, "confidence": confidence}
                
                classification_info = f"[学科分类] {subject} (置信度: {confidence:.2f})"
                print(f"\n{'='*60}")
                print(f"📚 {classification_info}")
                print(f"💡 分类理由: {reason}")
                print(f"{'='*60}\n")
                
                # 【智能路由】根据学科自动选择知识库
                original_kb = kb_name
                if subject == "理科":
                    science_kb_path = os.path.join(KB_BASE_DIR, "science")
                    if os.path.exists(science_kb_path):
                        kb_name = "science"
                        print(f"✅ 智能路由: 理科问题 → 自动切换到知识库 '{kb_name}'")
                    else:
                        print(f"ℹ️  未找到 'science' 知识库，使用 '{kb_name}' (提示：创建science知识库可提升理科问答质量)")
                elif subject == "文科":
                    liberal_kb_path = os.path.join(KB_BASE_DIR, "liberal")
                    if os.path.exists(liberal_kb_path):
                        kb_name = "liberal"
                        print(f"✅ 智能路由: 文科问题 → 自动切换到知识库 '{kb_name}'")
                    else:
                        print(f"ℹ️  未找到 'liberal' 知识库，使用 '{kb_name}' (提示：创建liberal知识库可提升文科问答质量)")
            except Exception as e:
                print(f"⚠️ 分类失败: {e}")
                _current_classification = {"subject": "未分类", "confidence": 0.0}

        sentiment = analyze_sentiment(question)
        tone_instruction = sentiment.get("style_instruction", "")
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        search_background = ""
        local_answer = ""
        debug_info = {}

        # 并行处理
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            if use_search:
                futures[executor.submit(get_search_background, question)] = "search"
                
            if os.path.exists(index_path):
                if multi_hop:
                    # 使用多跳推理
                    futures[executor.submit(multi_hop_generate_answer, question, kb_name, use_table_format)] = "rag"
                else:
                    # 使用简单向量检索
                    futures[executor.submit(simple_generate_answer, question, kb_name, use_table_format)] = "simple"
                
            for future in as_completed(futures):
                result = future.result()
                if futures[future] == "search":
                    search_background = result or ""
                elif futures[future] == "rag":
                    local_answer, debug_info = result
                elif futures[future] == "simple":
                    local_answer = result

        # 如果同时有搜索和本地结果，合并它们
        if search_background and local_answer:
            system_prompt = "你是一名专业的学术学习助手，请整合网络搜索和本地知识库提供全面的解答。"
            
            table_instruction = ""
            if use_table_format:
                table_instruction = """
                请尽可能以Markdown表格的形式呈现你的回答，特别是对于症状、治疗方法、药物等结构化信息。
                
                请确保你的表格遵循正确的Markdown语法：
                | 列标题1 | 列标题2 | 列标题3 |
                | ------- | ------- | ------- |
                | 数据1   | 数据2   | 数据3   |
                """
                
            user_prompt = f"""
            问题：{question}
            
            网络搜索结果：{search_background}
            
            本地知识库分析：{local_answer}
            
            {table_instruction}
            
            请根据以上信息，提供一个综合的回答。
            """
            
            try:
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                combined_answer = response.choices[0].message.content.strip()
                final_answer = combined_answer
            except Exception as e:
                # 如果合并失败，回退到本地答案
                final_answer = local_answer
        elif local_answer: # 没有搜索结果，只有本地答案
            final_answer = local_answer
        elif search_background:
            # 仅从搜索结果生成答案
            system_prompt = "你是一名专业的学术学习助手。"
            if use_table_format:
                system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
            final_answer = generate_answer_from_llm(question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_background}")
        else:
            final_answer = "未找到相关信息。"

        if tone_instruction:
            final_answer = apply_tone_to_answer(final_answer, tone_instruction, sentiment)
        return final_answer

    except Exception as e:
        return f"查询失败：{str(e)}"


def _generate_style_answer(question: str, base_answer: str, style_key: str, use_table_format: bool = False) -> str:
    """根据不同思路风格重写答案，保持事实一致但侧重点不同"""
    style_map = {
        "A": "使用严格的逻辑推理步骤：先分析问题、然后逐步推理、最后得出结论。避免使用类比或例子。",
        "B": "使用直观的类比和例子来解释：先给出生活中的类似场景、然后类比到问题、最后给出答案。避免逻辑推理步骤。"
    }
    style_desc = style_map.get(style_key, "清晰回答")
    table_hint = "如适合，请用 Markdown 表格呈现关键信息。" if use_table_format else ""
    system_prompt = (
        "你是一名学习助手，请在保证事实正确的前提下，用指定思路组织回答。"
        "不要编造超出基础答案的事实，可补充解释方式。"
    )
    user_prompt = (
        f"用户问题：{question}\n"
        f"基础答案（供参考）：{base_answer}\n\n"
        f"请用方案{style_key}的思路撰写：{style_desc} {table_hint}"
    )
    try:
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 回退：返回基础答案以避免中断
        return f"(生成方案{style_key}时出错：{e})\n{base_answer}"


def generate_ab_responses(question: str, base_answer: str, use_table_format: bool = False) -> Dict[str, str]:
    """生成 A/B 两种思路的回答"""
    if not base_answer:
        return {"A": "", "B": ""}
    answer_a = _generate_style_answer(question, base_answer, "A", use_table_format)
    answer_b = _generate_style_answer(question, base_answer, "B", use_table_format)
    return {"A": answer_a, "B": answer_b}


def ask_question_with_ab(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, use_table_format: bool = False, multi_hop: bool = False) -> Tuple[str, Dict[str, str]]:
    """
    基于指定知识库回答问题，并额外生成 A/B 思路的回答
    返回 (基础综合答案, {'A': 方案A, 'B': 方案B})
    """
    base_answer = ask_question_parallel(
        question=question,
        kb_name=kb_name,
        use_search=use_search,
        use_table_format=use_table_format,
        multi_hop=multi_hop
    )
    ab_answers = generate_ab_responses(question, base_answer, use_table_format)
    return base_answer, ab_answers

# 修改以支持多知识库的流式响应函数
def process_question_with_reasoning(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, use_table_format: bool = False, multi_hop: bool = False, chat_history: List = None):
    """增强版process_question，支持流式响应，实时显示检索和推理过程，支持多知识库和对话历史"""
    try:
        # 自动分类问题（理科/文科）
        classification_info = ""
        subject_type = ""
        if CLASSIFIER_AVAILABLE and _classifier:
            try:
                classification = _classifier.classify(question)
                subject_type = classification['subject']
                confidence = classification['confidence']
                classification_info = f"### 📚 学科分类\n**{subject_type}** (置信度: {confidence:.2f})\n\n"
                print(f"[学科分类] 问题: {question}")
                print(f"[学科分类] 结果: {subject_type} (置信度: {confidence:.2f})")
                
                # 可以根据学科自动选择知识库（可选功能）
                # if subject_type == "理科" and os.path.exists(os.path.join(KB_BASE_DIR, "science")):
                #     kb_name = "science"
                # elif subject_type == "文科" and os.path.exists(os.path.join(KB_BASE_DIR, "liberal")):
                #     kb_name = "liberal"
            except Exception as e:
                print(f"⚠️ 分类失败: {e}")

        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        # 构建带对话历史的问题
        if chat_history and len(chat_history) > 0:
            # 构建对话上下文
            context = "之前的对话内容：\n"
            for user_msg, assistant_msg in chat_history[-3:]: # 只取最近3轮对话
                context += f"用户：{user_msg}\n"
                context += f"助手：{assistant_msg}\n"
            context += f"\n当前问题：{question}"
            enhanced_question = f"基于以下对话历史，回答用户的当前问题。\n{context}"
        else:
            enhanced_question = question

        # 初始状态
        search_result = "联网搜索进行中..." if use_search else "未启用联网搜索"
        
        if multi_hop:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行多跳推理检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 推理状态\n{reasoning_status}"
            yield search_display, "正在启动多跳推理流程..."
        else:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行向量检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n{reasoning_status}"
            yield search_display, "正在启动简单检索流程..."

        # 如果启用，并行运行搜索
        search_future = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            if use_search:
                search_future = executor.submit(get_search_background, question)
                
        # 检查索引是否存在
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            # 如果索引不存在，提前返回
            if search_future:
                # 等待搜索结果
                search_result = "等待联网搜索结果..."
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n知识库 '{kb_name}' 中未找到索引"
                yield search_display, "等待联网搜索结果..."
                
                search_result = search_future.result() or "未找到相关网络信息"
                system_prompt = "你是一名专业的学术学习助手。请考虑对话历史并回答用户的问题。"
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                answer = generate_answer_from_llm(enhanced_question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_result}")
                
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n无法在知识库 '{kb_name}' 中进行本地检索（未找到索引）"
                yield search_display, answer
            else:
                yield f"知识库 '{kb_name}' 中未找到索引，且未启用联网搜索", "无法回答您的问题。请先上传文件到该知识库或启用联网搜索。"
            return

        # 开始流式处理
        current_answer = "正在分析您的问题..."
        
        if multi_hop:
            # 使用多跳推理的流式接口
            reasoning_rag = ReasoningRAG(
                index_path=index_path,
                metadata_path=metadata_path,
                max_hops=3,
                initial_candidates=5,
                refined_candidates=3,
                verbose=True
            )
            
            # 使用enhanced_question进行检索
            # 如果后台有联网搜索，尝试获取并将其并入 enhanced_question 作为背景信息
            if search_future:
                try:
                    net_info = search_future.result(timeout=5) or ""
                except Exception:
                    net_info = ""
                if net_info:
                    # 限制长度并加入到 enhanced_question，供多跳推理使用
                    enhanced_question = f"{enhanced_question}\n\n[联网检索摘要]: {net_info[:1500]}"
                    search_result = net_info

            for step_result in reasoning_rag.stream_retrieve_and_answer(enhanced_question, use_table_format):
                # 更新当前状态
                status = step_result["status"]
                reasoning_display = step_result["reasoning_display"]
                
                # 如果有新的答案，更新
                if step_result["answer"]:
                    current_answer = step_result["answer"]
                
                # 如果搜索结果已返回，更新搜索结果
                # 尝试在每一步更新联网搜索结果（若后台完成）
                if search_future and search_future.done():
                    try:
                        search_result = search_future.result() or "未找到相关网络信息"
                    except Exception:
                        search_result = "未找到相关网络信息"
                
                # 构建并返回当前状态
                current_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status}\n\n{reasoning_display}"
                yield current_display, current_answer
        else:
            # 简单向量检索的流式处理
            yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n正在执行向量相似度搜索...", "正在检索相关信息..."
            
            # 执行简单向量搜索，使用enhanced_question
            try:
                # 在简单检索场景中，先等待或获取后台的联网搜索摘要（若存在），并将其并入生成时的背景信息
                net_info = ""
                if search_future:
                    try:
                        net_info = search_future.result(timeout=3) or ""
                    except Exception:
                        net_info = "" 
                    if net_info:
                        search_result = net_info

                search_results = vector_search(enhanced_question, index_path, metadata_path, limit=5)
                
                if not search_results:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n未找到相关信息", f"知识库 '{kb_name}' 中未找到相关信息。"
                    current_answer = f"知识库 '{kb_name}' 中未找到相关信息。"
                else:
                    # 显示检索到的信息
                    chunks_detail = "\n\n".join([f"**相关信息 {i+1}**:\n{result['chunk']}" for i, result in enumerate(search_results[:5])])
                    chunks_preview = "\n".join([f"- {result['chunk'][:100]}..." for i, result in enumerate(search_results[:3])])
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n找到 {len(search_results)} 个相关信息块\n\n### 检索到的信息预览\n{chunks_preview}", "正在生成答案..."
                    
                    # 生成答案
                    # 将联网搜索结果也并入背景块，优先放在最前面以供LLM参考
                    background_chunks_list = []
                    if net_info:
                        background_chunks_list.append(f"[联网检索摘要]: {net_info}")
                    background_chunks_list.extend([f"[相关信息 {i+1}]: {result['chunk']}" for i, result in enumerate(search_results)])
                    background_chunks = "\n\n".join(background_chunks_list)
                    
                    system_prompt = "你是一名专业的学术学习助手。基于提供的背景信息和对话历史回答用户的问题。"
                    
                    # 【方案2：根据学科增强Prompt】
                    subject = _current_classification.get('subject', '未分类')
                    system_prompt = enhance_prompt_by_subject(system_prompt, subject)

                    if use_table_format:
                        system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                    
                    user_prompt = f"""
                    {enhanced_question}
                    
                    背景信息：
                    {background_chunks}
                    
                    请基于以上背景信息和对话历史回答用户的问题。
                    """
                    
                    response = client.chat.completions.create(
                        model=Config.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    
                    current_answer = response.choices[0].message.content.strip()
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n检索完成，已生成答案\n\n### 检索到的内容\n{chunks_detail}", current_answer
                    
            except Exception as e:
                error_msg = f"检索过程中出错: {str(e)}"
                yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{error_msg}", f"检索过程中出错: {str(e)}"
                current_answer = f"检索过程中出错: {str(e)}"
        
        # 检索完成后，如果有搜索结果，可以考虑合并知识
        if search_future and search_future.done():
            search_result = search_future.result() or "未找到相关网络信息"
            
            # 如果同时有搜索结果和本地检索结果，可以考虑合并
            if search_result and current_answer and current_answer not in ["正在分析您的问题...", "本地知识库中未找到相关信息。"]:
                status_text = "正在合并联网搜索和知识库结果..."
                if multi_hop:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status_text}", current_answer
                else:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{status_text}", current_answer
                
                # 合并结果
                system_prompt = "你是一名专业的学术学习助手，请整合网络搜索和本地知识库提供全面的解答。请考虑对话历史。"
                
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                
                user_prompt = f"""
                {enhanced_question}
                
                网络搜索结果：{search_result}
                
                本地知识库分析：{current_answer}
                
                请根据以上信息和对话历史，提供一个综合的回答。确保使用Markdown表格来呈现适合表格形式的信息。
                """
                
                try:
                    response = client.chat.completions.create(
                        model="qwen-plus",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    combined_answer = response.choices[0].message.content.strip()
                    
                    final_status = "已整合联网和知识库结果"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{final_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{final_status}"
                    
                    yield final_display, combined_answer
                except Exception as e:
                    # 如果合并失败，使用现有答案
                    error_status = f"合并结果失败: {str(e)}"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{error_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{error_status}"
                        
                    yield final_display, current_answer
        
    except Exception as e:
        error_msg = f"处理失败：{str(e)}\n{traceback.format_exc()}"
        yield f"### 错误信息\n{error_msg}", f"处理您的问题时遇到错误：{str(e)}"

# ==================== 笔记助手后端 API ====================

# 获取全局笔记文件路径(不再与知识库绑定)
def get_notes_path(kb_name: str) -> str:
    """获取全局笔记文件路径（不再与具体知识库绑定）"""
    return os.path.join(KB_BASE_DIR, "notes.json")

# 加载全局笔记列表
def load_notes(kb_name: str) -> List[Dict[str, Any]]:
    """加载全局笔记列表；兼容从旧路径 knowledge_bases/default/notes.json 迁移"""
    try:
        notes_path = get_notes_path(kb_name)
        # 若全局文件不存在，尝试从旧路径迁移
        if not os.path.exists(notes_path):
            legacy_path = os.path.join(KB_BASE_DIR, DEFAULT_KB, "notes.json")
            if os.path.exists(legacy_path):
                try:
                    legacy_notes = _load_json_file(legacy_path, [])
                    # 将旧笔记迁移到全局
                    with open(notes_path, 'w', encoding='utf-8') as f:
                        json.dump(legacy_notes, f, ensure_ascii=False, indent=2)
                    print(f"已将旧笔记从 {legacy_path} 迁移到 {notes_path}")
                except Exception as mig_e:
                    print(f"迁移旧笔记失败: {mig_e}")
        
        if os.path.exists(notes_path):
            return _load_json_file(notes_path, [])
        return []
    except Exception as e:
        print(f"加载笔记失败: {e}")
        traceback.print_exc()
        return []

# 保存笔记到全局笔记文件
def save_note_to_kb(kb_name: str, note: Dict[str, Any]) -> str:
    """保存笔记到全局笔记文件（与具体知识库无关）"""
    try:
        notes = load_notes(kb_name)
        
        # 如果笔记没有 id，生成一个
        if 'id' not in note or not note['id']:
            import uuid
            note['id'] = f"note_{uuid.uuid4().hex[:12]}"
        
        # 添加/更新时间戳
        if 'created' not in note:
            note['created'] = datetime.now().isoformat()
        note['last_modified'] = datetime.now().isoformat()
        
        # 检查是否是更新现有笔记
        existing_index = None
        for i, existing_note in enumerate(notes):
            if existing_note.get('id') == note['id']:
                existing_index = i
                break
        
        if existing_index is not None:
            notes[existing_index] = note
        else:
            notes.append(note)
        
        # 保存到全局文件
        notes_path = get_notes_path(kb_name)
        with open(notes_path, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        action = "更新" if existing_index is not None else "创建"
        return f"笔记{action}成功！ID: {note['id']}"
    except Exception as e:
        error_msg = f"保存笔记失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

# 删除全局笔记中的指定笔记
def delete_note_from_kb(kb_name: str, note_id: str) -> str:
    """删除全局笔记中的指定笔记（保留签名以兼容UI）"""
    try:
        if not note_id:
            return "错误：笔记ID为空"
        
        notes = load_notes(kb_name)
        original_count = len(notes)
        
        # 过滤掉要删除的笔记
        notes = [n for n in notes if n.get('id') != note_id]
        
        if len(notes) == original_count:
            return f"错误：未找到ID为 '{note_id}' 的笔记"
        
        # 保存更新后的笔记列表到全局
        notes_path = get_notes_path(kb_name)
        with open(notes_path, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        return f"笔记删除成功！ID: {note_id}"
    except Exception as e:
        error_msg = f"删除笔记失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

# 获取单个全局笔记
def get_note_by_id(kb_name: str, note_id: str) -> Optional[Dict[str, Any]]:
    """根据ID获取单个笔记（全局）"""
    try:
        notes = load_notes(kb_name)
        for note in notes:
            if note.get('id') == note_id:
                return note
        return None
    except Exception as e:
        print(f"获取笔记失败: {e}")
        return None

# ==================== 笔记助手后端 API 结束 ====================

# 添加处理函数，批量上传文件到指定知识库
def batch_upload_to_kb(file_objs: List, kb_name: str) -> str:
    """批量上传文件到指定知识库并进行处理"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：未指定知识库"
            
        # 确保知识库目录存在
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)
            
        if not file_objs or len(file_objs) == 0:
            return "错误：未选择任何文件"
            
        return process_and_index_files(file_objs, kb_name)
    except Exception as e:
        return f"上传文件到知识库失败: {str(e)}"

# ==================== 错题本相关 API ====================

# 尝试导入 OCR 相关库（可选依赖）
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# 优先使用 PaddleOCR（推荐），如果不可用再回退到 pytesseract
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

_PADDLE_OCR_CACHE = {}

def _get_paddle_ocr(lang: str = "ch", use_angle_cls: bool = True):
    """获取（缓存的）PaddleOCR 实例，避免每次调用都重复加载模型。"""
    key = (lang or "ch", bool(use_angle_cls))
    if key in _PADDLE_OCR_CACHE:
        return _PADDLE_OCR_CACHE[key]
    if PaddleOCR is None:
        return None
    # PaddleOCR v3+ 初始化时会进行“模型源连通性检查”，在部分环境下会显著拖慢启动；
    # 关闭该检查可减少首次初始化开销（不影响已缓存模型的离线推理）。
    try:
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    except Exception:
        pass
    # 兼容不同版本的 PaddleOCR：部分版本不支持 show_log 参数
    try:
        ocr = PaddleOCR(
            use_angle_cls=bool(use_angle_cls),
            lang=(lang or "ch"),
            show_log=False,
        )
    except Exception as e:
        try:
            ocr = PaddleOCR(
                use_angle_cls=bool(use_angle_cls),
                lang=(lang or "ch"),
            )
        except Exception as e2:
            print(f"PaddleOCR初始化失败: {e2} (原始错误: {e})")
            return None
    _PADDLE_OCR_CACHE[key] = ocr
    return ocr

def _paddleocr_extract_text(
    ocr,
    image_path: str,
    min_score: float = 0.5,
    use_angle_cls: bool = True,
) -> str:
    """对单张图片执行 PaddleOCR，返回按阅读顺序拼接的文本。"""
    if not ocr or not image_path:
        return ""
    # PaddleOCR v3+（paddleocr>=3）通常提供 predict()，且不支持 cls 参数；
    # v2（paddleocr 2.x）常见接口是 ocr(img, cls=True/False)。
    try:
        if hasattr(ocr, "predict"):
            # v3: 用 predict（官方推荐），避免传 cls
            # v3 默认会启用较重的文档预处理（如去畸变/行方向等），CPU 下会很慢；
            # 这里关闭不太必要的步骤以提速（方向分类可按需保留）。
            ocr_result = ocr.predict(
                image_path,
                use_doc_orientation_classify=bool(use_angle_cls),
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            # v2: 尝试带 cls=True（角度分类），不支持则降级
            try:
                ocr_result = ocr.ocr(image_path, cls=True)
            except Exception as e:
                if "cls" in str(e) or "unexpected keyword" in str(e):
                    ocr_result = ocr.ocr(image_path)
                else:
                    raise
    except Exception as e:
        print(f"PaddleOCR错误 ({image_path}): {e}")
        return ""

    # PaddleOCR 的返回：
    # - v2 常见： [ [ [box], (text, score) ], ... ] （外层可能按页/图拆分）
    # - v3 常见： [ OCRResult ]，其中包含 rec_texts/rec_scores/dt_polys 等字段
    lines = []
    if isinstance(ocr_result, list) and ocr_result:
        # ========== v3: OCRResult / dict-like ==========
        # 形如： [{'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...], ...}]
        first = ocr_result[0]
        is_v3_like = False
        try:
            is_v3_like = hasattr(first, "get") and (first.get("rec_texts") is not None)
        except Exception:
            is_v3_like = False

        if is_v3_like:
            for page in ocr_result:
                try:
                    rec_texts = page.get("rec_texts") or []
                    rec_scores = page.get("rec_scores") or []
                    polys = page.get("dt_polys") or page.get("rec_polys") or page.get("rec_boxes") or []
                except Exception:
                    continue

                for i, txt in enumerate(rec_texts):
                    try:
                        score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        if not txt or score < float(min_score):
                            continue
                        poly = polys[i] if i < len(polys) else None
                        x = y = 0.0
                        if isinstance(poly, (list, tuple)) and poly:
                            xs = [pt[0] for pt in poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                            ys = [pt[1] for pt in poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                            if xs and ys:
                                x = float(min(xs))
                                y = float(min(ys))
                        lines.append((y, x, str(txt).strip()))
                    except Exception:
                        continue
        else:
        # 兼容：有的版本返回 [ [..] ]，有的直接返回 [..]
            if len(ocr_result) == 1 and isinstance(ocr_result[0], list) and (not ocr_result[0] or isinstance(ocr_result[0][0], (list, tuple))):
                candidates = ocr_result[0]
            else:
                candidates = ocr_result
            for item in candidates:
                try:
                    if not item or len(item) < 2:
                        continue
                    box = item[0]
                    txt_score = item[1]
                    if not isinstance(txt_score, (list, tuple)) or len(txt_score) < 2:
                        continue
                    txt, score = txt_score[0], float(txt_score[1])
                    if not txt or score < float(min_score):
                        continue
                    # 计算一个用于排序的参考点（box 左上角/中心）
                    x = y = 0.0
                    if isinstance(box, (list, tuple)) and box:
                        xs = [pt[0] for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                        ys = [pt[1] for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                        if xs and ys:
                            x = float(min(xs))
                            y = float(min(ys))
                    lines.append((y, x, str(txt).strip()))
                except Exception:
                    continue

    lines.sort(key=lambda t: (t[0], t[1]))
    return "\n".join([t[2] for t in lines if t[2]])

# 使用 OCR 处理图片文件，返回文本内容
def ocr_images_to_texts(image_paths, lang: str = "auto", min_score: float = 0.5):
    """错题本图片 OCR：优先 PaddleOCR，失败回退 pytesseract。

    返回：[{path, filename, text, engine}]
    - lang: auto/ch/en（PaddleOCR 语言；auto -> ch）
    - min_score: PaddleOCR 置信度阈值（越大越严格）
    """
    use_lang = (lang or "auto").strip().lower()
    if use_lang in ("auto", "zh", "zh-cn", "zh_cn", "chinese"):
        use_lang = "ch"
    elif use_lang in ("en", "eng", "english"):
        use_lang = "en"
    else:
        # 其它 PaddleOCR lang（如 'japan','korean' 等）直接透传
        pass

    results = []
    for p in image_paths or []:
        if not p or not os.path.exists(p):
            continue
        fname = os.path.basename(p)

        text = ""
        engine = ""

        # 1) PaddleOCR（优先）
        ocr = _get_paddle_ocr(lang=use_lang, use_angle_cls=True)
        if ocr is not None:
            text = _paddleocr_extract_text(
                ocr,
                p,
                min_score=float(min_score),
                use_angle_cls=True,
            )
            engine = "paddleocr"

        # 2) 回退到 pytesseract
        if (not text.strip()) and pytesseract and Image:
            try:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                custom_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
                text = pytesseract.image_to_string(
                    img,
                    lang="chi_sim+eng",
                    config=custom_config,
                )
                engine = "pytesseract"
            except Exception as e:
                print(f"OCR错误 ({p}): {e}")
                text = ""

        # 3) 清理（如果文件里定义了 clean_text）
        try:
            clean = clean_text(text)
        except Exception:
            clean = (text or "").strip()

        results.append({"path": p, "filename": fname, "text": clean, "engine": engine})
    return results

# 使用 LLM 分析文本内容，生成练习题
def analyze_text_wrong_problems(text_content: str, level: str = "auto", count: int = 5):
    """Analyze edited text (from OCR or manual) and generate exercises via LLM.
    Returns (display_markdown, json_result).
    """
    try:
        level_desc = f"难度:{level}，数量:{count}"
        system_prompt = "你是一名专业的负责初高中课程的出题与讲解老师。"
        user_prompt = f"请根据以下题目与解析内容，分类知识点并生成{count}道练习题（{level}难度，包含答案与简要解释）：\n\n{text_content}"
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        display = f"### 解析与出题结果\n- {level_desc}\n\n{content}"
        # 简单返回结构化占位
        data = {"level": level, "count": count, "output": content}
        return display, data
    except Exception as e:
        return f"生成失败：{e}", {"error": str(e)}

# 获取错题本存储文件路径
def get_wrong_problems_path():
    """获取错题本存储文件路径"""
    return os.path.join(KB_BASE_DIR, "wrong_problems.json")

# 加载所有错题记录
def load_wrong_problems():
    """加载所有错题记录"""
    try:
        path = get_wrong_problems_path()
        return _load_json_file(path, [])
    except Exception as e:
        print(f"加载错题记录失败: {e}")
        return []

# 保存一条错题记录到错题本
def save_wrong_problem(content: str, level: str = "auto", tags: str = ""):
    """保存一条错题记录到错题本"""
    try:
        problems = load_wrong_problems()
        import uuid
        problem = {
            "id": f"wp_{uuid.uuid4().hex[:12]}",
            "content": content.strip(),
            "level": level,
            "tags": [t.strip() for t in tags.split(',') if t.strip()],
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        problems.append(problem)
        
        path = get_wrong_problems_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        
        return f"✅ 错题已保存！ID: {problem['id']}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

# 删除指定的错题记录
def delete_wrong_problem(problem_id: str):
    """删除指定的错题记录"""
    try:
        problems = load_wrong_problems()
        original_count = len(problems)
        problems = [p for p in problems if p.get('id') != problem_id]
        
        if len(problems) == original_count:
            return f"❌ 未找到ID为 '{problem_id}' 的错题记录"
        
        path = get_wrong_problems_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        
        return f"✅ 错题记录已删除！ID: {problem_id}"
    except Exception as e:
        return f"❌ 删除失败: {str(e)}"

# 以 Markdown 格式显示所有错题记录
def format_wrong_problems_display():
    """格式化错题记录为Markdown显示"""
    try:
        problems = load_wrong_problems()
        if not problems:
            return "### 错题记录\n\n暂无错题记录"
        
        lines = ["### 错题记录\n"]
        for i, p in enumerate(problems, 1):
            created = p.get('created', 'Unknown')[:10]  # 只显示日期
            level = p.get('level', 'auto')
            tags_str = ', '.join(p.get('tags', [])) if p.get('tags') else '无标签'
            content_preview = p.get('content', '')[:80] + "..." if len(p.get('content', '')) > 80 else p.get('content', '')
            lines.append(f"**{i}. [{created}] {level}难度** - {tags_str}")
            lines.append(f"```\n{content_preview}\n```")
            lines.append(f"*ID: {p.get('id')}*\n")
        
        return "\n".join(lines)
    except Exception as e:
        return f"### 错题记录\n\n加载失败: {str(e)}"

# ==================== 错题本 API 结束 ====================

# ==================== 家长视图 API ====================

# 获取学生学习统计数据
def get_learning_statistics():
    """
    收集学生学习统计数据：任务完成情况、学习时长、笔记数量、错题数量等
    返回字典包含各项统计信息
    """
    try:
        from datetime import date, timedelta
        
        # 直接加载学习看板数据（避免循环导入）
        board_path = os.path.join(KB_BASE_DIR, "learning_board.json")
        board = _load_json_file(board_path, {})
        
        tasks = board.get('tasks', [])
        records = board.get('study_records', [])
        
        # 计算任务统计
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.get('completed'))
        completion_rate = int((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0)
        
        # 计算学习时长统计
        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        
        today_minutes = sum([r.get('minutes', 0) for r in records if r.get('date') == today])
        week_minutes = sum([r.get('minutes', 0) for r in records if r.get('date') >= week_ago])
        total_minutes = sum([r.get('minutes', 0) for r in records])
        
        # 计算笔记统计（读取全局notes.json）
        all_notes = []
        global_notes_path = os.path.join(KB_BASE_DIR, "notes.json")
        if os.path.exists(global_notes_path):
            all_notes = _load_json_file(global_notes_path, [])
        note_count = len(all_notes)
        
        # 计算错题统计
        wrong_problems = load_wrong_problems()
        wrong_count = len(wrong_problems)
        wrong_by_level = {}
        for wp in wrong_problems:
            level = wp.get('level', 'auto')
            wrong_by_level[level] = wrong_by_level.get(level, 0) + 1
        
        stats = {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'completion_rate': completion_rate,
            'today_minutes': today_minutes,
            'week_minutes': week_minutes,
            'total_minutes': total_minutes,
            'note_count': note_count,
            'wrong_count': wrong_count,
            'wrong_by_level': wrong_by_level,
            'generated_time': datetime.now().isoformat()
        }
        return stats
    except Exception as e:
        print(f"获取学习统计失败: {e}")
        traceback.print_exc()
        return {}

# 生成家长学习进度报告
def generate_parent_report():
    """
    生成家长学习进度报告（使用LLM总结）
    返回 (report_markdown, stats_dict)
    """
    try:
        stats = get_learning_statistics()
        if not stats:
            return "无法收集学习数据", {}
        
        # 构建统计摘要
        summary = f"""
        学生学习统计摘要：
        - 任务完成情况：{stats['completed_tasks']}/{stats['total_tasks']} （完成度：{stats['completion_rate']}%）
        - 今日学习时长：{stats['today_minutes']} 分钟
        - 本周学习时长：{stats['week_minutes']} 分钟
        - 累计学习时长：{stats['total_minutes']} 分钟
        - 笔记数量：{stats['note_count']} 篇
        - 错题记录：{stats['wrong_count']} 道
        - easy 难度：{stats['wrong_by_level'].get('easy', 0)} 道
        - medium 难度：{stats['wrong_by_level'].get('medium', 0)} 道
        - hard 难度：{stats['wrong_by_level'].get('hard', 0)} 道
        - auto 难度：{stats['wrong_by_level'].get('auto', 0)} 道

        请根据上述学习数据，为家长生成一份简洁的学习进度报告，包括：
        1. 学生的整体学习表现评价
        2. 学习强项与改进空间
        3. 近期学习建议
        4. 家长可以采取的支持措施

        报告应该是鼓励性的、具体的、可操作的。
        """
        
        # 调用LLM生成报告
        system_prompt = "你是一位专业的教育顾问，为家长生成关于学生学习进度的总结报告。报告应该简洁、清晰、具有建设性。"
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary}
            ]
        )
        
        report = response.choices[0].message.content.strip()
        
        # 格式化为Markdown
        markdown_report = f"""**生成时间：** {stats['generated_time'][:10]}

## 学习统计概览

| 指标 | 数值 |
|------|------|
| 任务完成度 | {stats['completion_rate']}% ({stats['completed_tasks']}/{stats['total_tasks']}) |
| 今日学习时长 | {stats['today_minutes']} 分钟 |
| 本周学习时长 | {stats['week_minutes']} 分钟 |
| 累计学习时长 | {stats['total_minutes']} 分钟 |
| 笔记总数 | {stats['note_count']} 篇 |
| 错题记录数 | {stats['wrong_count']} 道 |

## 教育顾问建议

{report}
"""
        
        return markdown_report, stats
    except Exception as e:
        error_msg = f"生成报告失败：{str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, {}

# ==================== 家长视图 API 结束 ====================


# ==================== 知识图谱增强 RAG ====================

# 使用知识图谱增强的RAG查询
def query_with_kg_enhancement(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, 
                               use_kg: bool = True, use_table_format: bool = False) -> Tuple[str, str]:
    """
    使用知识图谱增强的RAG查询
    
    Args:
        question: 用户问题
        kb_name: 知识库名称
        use_search: 是否使用联网搜索
        use_kg: 是否使用知识图谱
        use_table_format: 是否使用表格格式
        
    Returns:
        (回答文本, 调试信息)
    """
    sentiment = analyze_sentiment(question)
    tone_instruction = sentiment.get("style_instruction", "")
    source_info = "### 📚 本次回答的知识来源\n\n"
    
    # 自动分类问题（理科/文科）
    if CLASSIFIER_AVAILABLE and _classifier:
        try:
            classification = _classifier.classify(question)
            subject = classification['subject']
            confidence = classification['confidence']
            source_info += f"### 📚 学科分类\n**{subject}** (置信度: {confidence:.2f})\n\n"
            print(f"[学科分类] 问题: {question}")
            print(f"[学科分类] 结果: {subject} (置信度: {confidence:.2f})")
        except Exception as e:
            print(f"⚠️ 分类失败: {e}")

    # 检查KG是否启用
    if not Config.kg_enabled:
        answer = ask_question_parallel(question, kb_name, use_search, use_table_format, multi_hop=False)
        source_info += "ℹ️ 知识图谱未在配置中启用（Config.kg_enabled = False）\n"
        return answer, source_info
    
    if not use_kg:
        answer = ask_question_parallel(question, kb_name, use_search, use_table_format, multi_hop=False)
        source_info += "ℹ️ 用户未启用知识图谱选项\n"
        return answer, source_info
    
    if not KG_AVAILABLE:
        answer = ask_question_parallel(question, kb_name, use_search, use_table_format, multi_hop=False)
        source_info += "❌ 知识图谱模块加载失败，无法使用KG增强功能。请检查neo4j和kg_construct模块是否正确安装。\n"
        return answer, source_info
    
    try:
        # 1. 从问题中提取实体（提供数据库实体参照）
        print(f"从问题中提取实体...")
        kg_builder = KnowledgeGraphBuilder()
        
        # 先获取数据库中的实体列表作为参照
        db_entities = []
        neo4j_handler_temp = Neo4jHandler()
        try:
            neo4j_handler_temp.connect()
            db_entities = neo4j_handler_temp.get_all_entity_names(limit=500)
            print(f"获取到数据库中 {len(db_entities)} 个实体作为参照")
        except Exception as e:
            print(f"获取数据库实体列表失败: {e}")
        finally:
            neo4j_handler_temp.close()
        
        entities = kg_builder.extract_entities_from_query(question, db_entities=db_entities)
        
        source_info += f"#### 🧠 知识图谱相关实体\n"
        if entities:
            # entities是(实体名, 类型)的元组列表，只提取实体名
            entity_names = [e[0] if isinstance(e, tuple) else e for e in entities]
            source_info += f"从问题中识别到的关键实体: **{', '.join(entity_names)}**\n\n"
        else:
            source_info += "未能识别到关键实体\n\n"
        
        kg_context = ""
        kg_data = {}
        if entities:
            # 2. 在知识图谱中查询相关知识
            print(f"在知识图谱中查询相关知识...")
            neo4j_handler = Neo4jHandler()
            try:
                neo4j_handler.connect()
                kg_data = neo4j_handler.query_related_entities(entities)
                kg_context = neo4j_handler.format_kg_context(kg_data)
                print(f"从知识图谱获取了 {kg_data.get('count', 0)} 条相关知识")
                
                # 展示KG知识
                source_info += f"#### 🕸️ 知识图谱中的相关知识\n"
                matched_entities = kg_data.get('matched_entities', [])
                if matched_entities:
                    source_info += "实体匹配情况：\n"
                    for match in matched_entities:
                        source_info += f"- 查询实体 **{match.get('query', '?')}** 通过 {match.get('method', 'unknown')} 匹配到 **{match.get('matched', '?')}** ({match.get('type', '未知')})\n"
                
                triples = kg_data.get('triples') or []
                if triples:
                    source_info += f"\n为您找到 **{len(triples)}** 条相关知识关系：\n\n"
                    for i, rel in enumerate(triples[:8], 1):
                        subject = rel.get('subject', {}).get('name', '?')
                        predicate = rel.get('predicate', '→')
                        obj = rel.get('object', {}).get('name', '?')
                        source_info += f"{i}. **{subject}** {predicate} **{obj}**\n"
                    if len(triples) > 8:
                        source_info += f"\n...及其他 {len(triples)-8} 条关系\n"
                else:
                    source_info += "未找到相关知识关系\n"
                source_info += "\n"
            except Exception as e:
                print(f"知识图谱查询失败: {e}")
                source_info += f"⚠️ 知识图谱查询失败: {str(e)}\n\n"
            finally:
                neo4j_handler.close()
        
        # 3. 进行常规RAG检索
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]
        
        rag_context = ""
        rag_results = []
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # 使用已有的向量检索函数
            results = vector_search(question, index_path, metadata_path, limit=5)
            rag_results = results
            if results:
                rag_context = "\n\n### RAG检索结果\n"
                for i, result in enumerate(results[:3], 1):
                    # 分块数据中字段名是 'chunk' 而不是 'text'
                    chunk_text = result.get('chunk', result.get('text', ''))
                    rag_context += f"\n**文档 {i}:**\n{chunk_text[:500]}...\n"
                
                # 展示RAG文档
                source_info += f"#### 📄 相关文档片段\n"
                source_info += f"从知识库 **{kb_name}** 中检索到 **{len(results)}** 个相关文档片段：\n\n"
                for i, result in enumerate(results[:5], 1):
                    # 优先使用 'chunk' 字段，备用 'text' 字段
                    chunk_text = result.get('chunk', result.get('text', '')).replace('\n', ' ').strip()
                    if len(chunk_text) > 300:
                        chunk_text = chunk_text[:300] + "..."
                    source_info += f"**片段 {i}:**\n{chunk_text}\n\n"
            else:
                source_info += f"#### 📄 相关文档片段\n未在知识库中找到相关文档\n\n"
        else:
            source_info += f"#### 📄 相关文档片段\n知识库索引不存在\n\n"
        
        # 4. 获取联网搜索结果（如果启用）
        search_context = ""
        if use_search:
            search_result = get_search_background(question)
            if search_result:
                search_context = f"\n\n### 联网搜索结果\n{search_result}\n"
        
        # 5. 组合所有上下文生成回答
        combined_context = ""
        if kg_context:
            combined_context += kg_context + "\n"
        if rag_context:
            combined_context += rag_context + "\n"
        if search_context:
            combined_context += search_context
        
        if not combined_context:
            source_info += "---\n\n⚠️ 未找到任何相关信息\n"
            return "抱歉，没有找到相关信息来回答您的问题。", source_info
        
        source_info += "---\n\n"
        used_sources = []
        if kg_context:
            used_sources.append("知识图谱")
        if rag_context:
            used_sources.append("文档检索")
        if search_context:
            used_sources.append("联网搜索")
        source_info += f"💡 本次回答综合使用了: **{'、'.join(used_sources)}**\n"
        
        # 6. 生成回答
        system_prompt = "你是一名专业的学术助手。请基于提供的背景信息回答用户的问题。"
        if tone_instruction:
            system_prompt += f" 请注意回答语气：{tone_instruction}"
        if use_table_format:
            system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
        
        user_prompt = f"""背景信息：
        {combined_context}

        问题：{question}

        请基于以上信息给出准确、全面的回答。"""

        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        final_answer = response.choices[0].message.content.strip()
        if tone_instruction:
            final_answer = apply_tone_to_answer(final_answer, tone_instruction, sentiment)
        return final_answer, source_info
        
    except Exception as e:
        print(f"KG增强查询失败: {e}")
        traceback.print_exc()
        error_info = f"### 📚 知识来源\n\n❌ 查询过程出错: {str(e)}\n"
        return f"查询失败: {str(e)}", error_info
        # 回退到普通RAG（使用并行管线入口）
        return ask_question_parallel(question, kb_name, use_search, use_table_format, multi_hop=False)

# 为知识库中的文件构建知识图谱
def build_kg_for_kb_file(kb_name: str, filename: str) -> Tuple[bool, str]:
    """
    为知识库中的文件构建知识图谱
    
    Args:
        kb_name: 知识库名称
        filename: 文件名
        
    Returns:
        (成功标志, 消息)
    """
    if not KG_AVAILABLE:
        return False, "知识图谱功能不可用，请安装neo4j驱动：pip install neo4j"
    
    try:
        # 读取文件内容
        file_path = os.path.join(KB_BASE_DIR, kb_name, filename)
        if not os.path.exists(file_path):
            return False, f"文件不存在: {filename}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 提取三元组
        print(f"从文件 {filename} 提取知识图谱...")
        kg_builder = KnowledgeGraphBuilder()
        triples = kg_builder.extract_triples_from_text(text)
        
        if not triples:
            return False, "未能从文件中提取到有效的知识三元组"
        
        # 保存三元组到临时JSON文件
        json_path = os.path.join(OUTPUT_DIR, f"{kb_name}_{filename}_triples.json")
        kg_builder.save_triples_to_json(triples, json_path)
        
        # 加载到Neo4j
        print(f"将三元组加载到Neo4j...")
        neo4j_handler = Neo4jHandler()
        neo4j_handler.connect()
        
        # 不清空数据库，追加三元组
        successful = neo4j_handler.add_triples_batch(triples, batch_size=100)
        
        stats = neo4j_handler.get_statistics()
        neo4j_handler.close()
        
        message = f"成功为文件 {filename} 构建知识图谱！\n"
        message += f"提取了 {len(triples)} 个三元组，成功加载 {successful} 个\n"
        message += f"当前图谱包含 {stats.get('node_count', 0)} 个节点和 {stats.get('relationship_count', 0)} 个关系"
        
        return True, message
        
    except Exception as e:
        error_msg = f"构建知识图谱失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return False, error_msg

# 获取知识图谱统计信息
def get_kg_statistics() -> Dict[str, Any]:
    """
    获取知识图谱统计信息
    
    Returns:
        统计信息字典
    """
    if not KG_AVAILABLE:
        return {"error": "知识图谱功能不可用"}
    
    try:
        neo4j_handler = Neo4jHandler()
        neo4j_handler.connect()
        stats = neo4j_handler.get_statistics()
        neo4j_handler.close()
        return stats
    except Exception as e:
        return {"error": str(e)}

# 为整个知识库构建知识图谱
def build_kg_for_entire_kb(kb_name: str) -> Tuple[bool, str]:
    """
    为整个知识库构建知识图谱（从语义分块中提取）
    
    Args:
        kb_name: 知识库名称
        
    Returns:
        (成功标志, 消息)
    """
    if not KG_AVAILABLE:
        return False, "知识图谱功能不可用，请安装neo4j驱动：pip install neo4j"
    
    try:
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_dir):
            return False, f"知识库不存在: {kb_name}"
        
        # 优先从语义分块元数据中读取内容
        semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")
        text_chunks = []
        
        if os.path.exists(semantic_chunk_metadata):
            # 从语义分块中读取
            try:
                with open(semantic_chunk_metadata, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    if isinstance(chunks_data, list):
                        text_chunks = [chunk.get('chunk', '') for chunk in chunks_data if chunk.get('chunk')]
                        print(f"从 semantic_chunk_metadata.json 读取到 {len(text_chunks)} 个语义分块")
            except Exception as e:
                print(f"读取语义分块失败: {e}")
        
        # 如果没有语义分块，尝试读取独立的笔记JSON文件
        if not text_chunks:
            json_files = [f for f in os.listdir(kb_dir) 
                         if f.endswith('.json') 
                         and f not in ['semantic_chunk_metadata.json', 'semantic_chunk_vector.json']]
            
            for filename in json_files:
                try:
                    file_path = os.path.join(kb_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            text_chunks.append(content)
                except Exception as e:
                    print(f"读取文件 {filename} 失败: {e}")
        
        if not text_chunks:
            return False, f"知识库 {kb_name} 中没有找到可处理的内容\n提示：请先上传文件到知识库"
        
        # 合并所有文本分块
        combined_text = "\n\n".join(text_chunks)
        print(f"合并了 {len(text_chunks)} 个文本分块，总长度: {len(combined_text)} 字符")
        
        # 提取知识图谱三元组
        kg_builder = KnowledgeGraphBuilder()
        neo4j_handler = Neo4jHandler()
        neo4j_handler.connect()
        
        print(f"开始从知识库内容中提取三元组...")
        triples = kg_builder.extract_triples_from_text(combined_text)
        
        if not triples:
            neo4j_handler.close()
            return False, f"未能从知识库 {kb_name} 中提取到有效的知识三元组\n提示：内容可能缺少实体和关系信息"
        
        # 批量加载到Neo4j
        print(f"提取到 {len(triples)} 个三元组，开始加载到Neo4j...")
        successful = neo4j_handler.add_triples_batch(triples, batch_size=100)
        
        stats = neo4j_handler.get_statistics()
        neo4j_handler.close()
        
        # 生成结果消息
        message = f"✅ 知识库 {kb_name} 的知识图谱构建完成！\n\n"
        message += f"📊 处理统计：\n"
        message += f"  - 处理分块数: {len(text_chunks)}\n"
        message += f"  - 提取三元组: {len(triples)} 个\n"
        message += f"  - 成功加载: {successful} 个\n\n"
        message += f"🔍 当前图谱规模：\n"
        message += f"  - 节点数: {stats.get('node_count', 0)}\n"
        message += f"  - 关系数: {stats.get('relationship_count', 0)}\n\n"
        message += f"💡 提示：现在可以在问答助理中勾选 '🧠 启用知识图谱(KG)' 来使用增强检索"
        
        return True, message
        
    except Exception as e:
        return False, f"构建知识图谱时出错: {str(e)}\n{traceback.format_exc()}"


# ==================== 知识图谱功能结束 ====================


if __name__ == "__main__":
    from ui import launch_ui
    launch_ui()
