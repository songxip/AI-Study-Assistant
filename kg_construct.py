import json
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from config import Config

# 定义学术领域的实体类型
ENTITY_TYPES = [
    "概念", "理论", "方法", "算法",
    "模型", "技术", "工具", "框架",
    "定理", "公式", "原则", "策略",
    "人物", "机构", "术语", "领域",
    "数据结构", "系统", "应用", "问题"
]

# 定义学术领域的关系类型
RELATIONSHIP_TYPES = [
    "属于", "包含", "使用", "应用",
    "定义", "实现", "创建", "提出",
    "前置", "后置", "依赖", "关联",
    "派生", "扩展", "优化", "改进",
    "基于", "解决", "证明", "验证",
    "影响", "支持", "对比", "结合"
]


class KnowledgeGraphBuilder:
    """知识图谱构建类 - 使用阿里云LLM API提取三元组"""
    
    def __init__(self):
        """初始化知识图谱构建器"""
        self.client = OpenAI(
            api_key=Config.llm_api_key,
            base_url=Config.llm_base_url
        )
        self.model = Config.kg_extract_model
        
    def extract_triples_from_text(self, text: str, chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """
        从文本中提取知识三元组
        
        Args:
            text: 要提取的文本
            chunk_size: 文本分块大小
            
        Returns:
            三元组列表
        """
        # 如果文本太长，分块处理
        if len(text) > chunk_size:
            chunks = self._split_text(text, chunk_size)
            all_triples = []
            for i, chunk in enumerate(chunks):
                print(f"处理文本块 {i+1}/{len(chunks)}...")
                triples = self._extract_triples_from_chunk(chunk)
                all_triples.extend(triples)
            return all_triples
        else:
            return self._extract_triples_from_chunk(text)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """将长文本分块"""
        chunks = []
        sentences = re.split(r'[。！？\n]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_triples_from_chunk(self, text: str) -> List[Dict[str, Any]]:
        """从单个文本块中提取三元组"""
        system_prompt = """你是一个专业的知识图谱构建助手。你的任务是从学术文本中提取结构化的知识三元组。

        请仔细分析文本，提取其中的实体和关系，并以JSON格式输出。

        实体类型包括：""" + ", ".join(ENTITY_TYPES) + """

        关系类型包括：""" + ", ".join(RELATIONSHIP_TYPES) + """

        输出格式要求：
        1. 必须是有效的JSON数组
        2. 每个三元组包含subject（主语）、predicate（谓语）、object（宾语）
        3. 主语和宾语都需要包含name（名称）和type（类型）字段
        4. 不要添加任何解释文字，只返回JSON

        示例输出：
        [
        {
            "subject": {"name": "深度学习", "type": "领域"},
            "predicate": "包含",
            "object": {"name": "卷积神经网络", "type": "模型"}
        }
        ]"""

        user_prompt = f"""请从以下学术文本中提取知识三元组：

        文本：
        {text}

        请严格按照JSON格式输出三元组，不要添加任何其他文字。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # 使用较低温度以获得更确定的输出
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理响应，移除可能的markdown代码块标记
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = re.sub(r'```\s*|\s*```', '', content)
            
            # 尝试提取JSON数组
            json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if json_match:
                triples = json.loads(json_match.group(0))
                # 验证三元组结构
                validated_triples = []
                for triple in triples:
                    if self._validate_triple(triple):
                        validated_triples.append(triple)
                    else:
                        print(f"跳过无效三元组: {triple}")
                
                print(f"成功提取 {len(validated_triples)} 个有效三元组")
                return validated_triples
            else:
                print(f"无法从响应中提取JSON数组")
                return []
                
        except Exception as e:
            print(f"提取三元组时出错: {e}")
            return []
    
    def _validate_triple(self, triple: Dict) -> bool:
        """验证三元组结构是否有效"""
        if not isinstance(triple, dict):
            return False
        
        # 检查必要字段
        if not all(key in triple for key in ["subject", "predicate", "object"]):
            return False
        
        # 检查subject和object结构
        for entity_key in ["subject", "object"]:
            entity = triple[entity_key]
            if not isinstance(entity, dict):
                return False
            if not all(key in entity for key in ["name", "type"]):
                return False
            if not entity["name"] or not entity["type"]:
                return False
        
        # 检查predicate
        if not triple["predicate"]:
            return False
        
        return True
    
    def extract_triples_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从文件中提取知识三元组
        
        Args:
            file_path: 文件路径
            
        Returns:
            三元组列表
        """
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"从文件 {file_path} 中提取三元组...")
            return self.extract_triples_from_text(text)
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return []
    
    def save_triples_to_json(self, triples: List[Dict[str, Any]], output_path: str):
        """
        保存三元组到JSON文件
        
        Args:
            triples: 三元组列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(triples, f, ensure_ascii=False, indent=2)
            print(f"成功保存 {len(triples)} 个三元组到 {output_path}")
        except Exception as e:
            print(f"保存三元组时出错: {e}")
    
    def extract_entities_from_query(self, query: str, max_entities: int = None, db_entities: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        """
        从用户查询中提取实体
        
        Args:
            query: 用户查询
            max_entities: 最大提取实体数
            db_entities: 数据库中已有的实体列表（提供参照）
            
        Returns:
            实体列表，每个元素为 (实体名, 实体类型)
        """
        if max_entities is None:
            max_entities = Config.kg_max_entities
        
        # 构建数据库实体参照信息
        db_reference = ""
        if db_entities:
            # 按类型分组，便于大模型理解
            entities_by_type = {}
            for name, etype in db_entities:
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append(name)
            
            db_reference = "\n\n**数据库中已有的实体供参考（请尽量从中选择或匹配相近的实体名称）：**\n"
            for etype, names in sorted(entities_by_type.items())[:10]:  # 限制显示类型数
                db_reference += f"\n- {etype}: {', '.join(names[:20])}"  # 每类最多显示20个
                if len(names) > 20:
                    db_reference += f" 等{len(names)}个"
        
        system_prompt = """你是一个专业的实体识别助手。请从用户的问题中提取关键实体。

        实体类型包括：""" + ", ".join(ENTITY_TYPES) + """""" + db_reference + """

        输出格式要求：
        1. 必须是有效的JSON数组
        2. 每个实体包含name（名称）和type（类型）字段
        3. **优先使用数据库中已有的实体名称**，如果问题中的词与数据库实体相近，请直接使用数据库中的实体名
        4. 只提取与问题最相关的核心实体
        5. 不要添加任何解释文字，只返回JSON

        示例输出：
        [
        {"name": "机器学习", "type": "领域"},
        {"name": "决策树", "type": "算法"}
        ]"""

        user_prompt = f"""请从以下问题中提取关键实体（最多{max_entities}个）：

        问题：{query}

        请严格按照JSON格式输出实体列表。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理响应
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = re.sub(r'```\s*|\s*```', '', content)
            
            # 提取JSON数组
            json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group(0))
                result = []
                for entity in entities[:max_entities]:
                    if "name" in entity and "type" in entity:
                        result.append((entity["name"], entity["type"]))
                
                print(f"从查询中提取到 {len(result)} 个实体: {result}")
                return result
            else:
                print("无法从响应中提取实体")
                return []
                
        except Exception as e:
            print(f"提取实体时出错: {e}")
            return []


def build_kg_from_file(file_path: str, output_json_path: str = None) -> List[Dict[str, Any]]:
    """
    从文件构建知识图谱
    
    Args:
        file_path: 输入文件路径
        output_json_path: 输出JSON路径（可选）
        
    Returns:
        三元组列表
    """
    builder = KnowledgeGraphBuilder()
    triples = builder.extract_triples_from_file(file_path)
    
    if output_json_path and triples:
        builder.save_triples_to_json(triples, output_json_path)
    
    return triples


def build_kg_from_text(text: str, output_json_path: str = None) -> List[Dict[str, Any]]:
    """
    从文本构建知识图谱
    
    Args:
        text: 输入文本
        output_json_path: 输出JSON路径（可选）
        
    Returns:
        三元组列表
    """
    builder = KnowledgeGraphBuilder()
    triples = builder.extract_triples_from_text(text)
    
    if output_json_path and triples:
        builder.save_triples_to_json(triples, output_json_path)
    
    return triples


if __name__ == "__main__":
    # 测试代码
    test_text = """
    深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
    卷积神经网络（CNN）是一种特殊的深度学习模型，主要用于图像识别任务。
    循环神经网络（RNN）则擅长处理序列数据，如文本和时间序列。
    """
    
    print("测试从文本提取三元组...")
    triples = build_kg_from_text(test_text, "test_triples.json")
    
    print(f"\n提取到的三元组:")
    for triple in triples:
        print(f"  {triple['subject']['name']} --[{triple['predicate']}]--> {triple['object']['name']}")
