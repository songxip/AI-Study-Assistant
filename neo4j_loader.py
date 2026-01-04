import json
import re
import os
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from config import Config
from kg_construct import ENTITY_TYPES, RELATIONSHIP_TYPES
import numpy as np
from difflib import SequenceMatcher

class Neo4jHandler:
    """Neo4j数据库操作处理器 - 用于学术知识图谱"""
    
    def __init__(self, uri=None, username=None, password=None):
        """初始化Neo4j处理器"""
        self.uri = uri or Config.neo4j_uri
        self.username = username or Config.neo4j_username
        self.password = password or Config.neo4j_password
        self.driver = None
        
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            print("成功连接到Neo4j数据库")
        except Exception as e:
            print(f"连接Neo4j失败: {e}")
            raise
        
    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")
            
    def clear_database(self):
        """从数据库中删除所有节点和关系"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("数据库已清空")
        except Exception as e:
            print(f"清空数据库时出错: {e}")
            
    def create_constraints(self):
        """为实体类型创建约束"""
        try:
            with self.driver.session() as session:
                # 为每个实体类型创建约束
                for entity_type in ENTITY_TYPES:
                    # 净化实体类型以适应Neo4j
                    clean_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', entity_type)
                    if not clean_type:  # 确保清理后的类型不为空
                        clean_type = "未知类型"
                    try:
                        # 创建约束（语法取决于Neo4j版本）
                        try:
                            # Neo4j 4.x+
                            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{clean_type}`) REQUIRE n.name IS UNIQUE")
                        except:
                            try:
                                # Neo4j 3.x
                                session.run(f"CREATE CONSTRAINT ON (n:`{clean_type}`) ASSERT n.name IS UNIQUE")
                            except:
                                print(f"无法为{clean_type}创建约束，跳过")
                    except Exception as e:
                        print(f"为{clean_type}创建约束时出错: {e}")
                print("已为所有实体类型创建约束")
        except Exception as e:
            print(f"设置约束时出错: {e}")
    
    def validate_triples(self, triples):
        """验证三元组数据的有效性，并修复可能导致错误的记录"""
        fixed_triples = []
        invalid_count = 0
        
        for triple in triples:
            try:
                # 确保所有必要的字段都存在
                if not all(key in triple for key in ["subject", "predicate", "object"]):
                    print(f"跳过缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue
                    
                if not all(key in triple["subject"] for key in ["name", "type"]):
                    print(f"跳过主体缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue
                    
                if not all(key in triple["object"] for key in ["name", "type"]):
                    print(f"跳过客体缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue
                
                # 确保类型字段在清理后不为空
                subject_type = triple["subject"]["type"]
                object_type = triple["object"]["type"]
                
                clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)
                
                if not clean_subject_type:
                    print(f"修复主体类型为空的三元组: {triple}")
                    triple["subject"]["type"] = "未知类型"
                
                if not clean_object_type:
                    print(f"修复客体类型为空的三元组: {triple}")
                    triple["object"]["type"] = "未知类型"
                
                fixed_triples.append(triple)
            except Exception as e:
                print(f"验证三元组时出错: {e}, 三元组: {triple}")
                invalid_count += 1
        
        print(f"验证完成: 有效三元组 {len(fixed_triples)}, 无效/已修复三元组 {invalid_count}")
        return fixed_triples
            
    def add_triples_batch(self, triples, batch_size=100):
        """批量向数据库添加三元组"""
        # 首先验证并修复三元组
        validated_triples = self.validate_triples(triples)
        
        successful = 0
        total = len(validated_triples)
        
        for i in range(0, total, batch_size):
            batch = validated_triples[i:i+batch_size]
            try:
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for triple in batch:
                            # 提取三元组信息
                            subject_name = triple["subject"]["name"]
                            subject_type = triple["subject"]["type"]
                            predicate = triple["predicate"]
                            object_name = triple["object"]["name"]
                            object_type = triple["object"]["type"]
                            
                            # 净化类型以适应Neo4j
                            clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                            clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)
                            clean_predicate = re.sub(r'[^\w\u4e00-\u9fa5]', '_', predicate)
                            
                            # 确保类型不为空字符串
                            if not clean_subject_type:
                                clean_subject_type = "未知类型"
                            if not clean_object_type:
                                clean_object_type = "未知类型"
                            if not clean_predicate:
                                clean_predicate = "关联"
                            
                            # 准备来源属性
                            source_props = ""
                            if "sources" in triple:
                                source_list = triple["sources"]
                                source_props = ", r.sources = $sources"
                            elif "source" in triple:
                                source_list = [triple["source"]]
                                source_props = ", r.source = $source"
                            else:
                                source_list = None
                            
                            # 创建三元组
                            query = f"""
                            MERGE (s:`{clean_subject_type}` {{name: $subject_name}})
                            SET s.type = $subject_type
                            MERGE (o:`{clean_object_type}` {{name: $object_name}})
                            SET o.type = $object_type
                            MERGE (s)-[r:`{clean_predicate}`]->(o)
                            SET r.name = $predicate{source_props}
                            """
                            
                            # 准备查询参数
                            params = {
                                "subject_name": subject_name,
                                "subject_type": subject_type,
                                "object_name": object_name,
                                "object_type": object_type,
                                "predicate": predicate
                            }
                            
                            # 添加来源参数
                            if "sources" in triple:
                                params["sources"] = source_list
                            elif "source" in triple and source_list:
                                params["source"] = source_list[0]
                            
                            tx.run(query, **params)
                        
                        # 提交事务
                        tx.commit()
                        successful += len(batch)
                
                print(f"已添加 {i+len(batch)}/{total} 个三元组")
                
            except Exception as e:
                print(f"添加三元组批次 {i//batch_size + 1} 时出错: {e}")
        
        return successful
            
    def get_statistics(self):
        """获取知识图谱的统计信息"""
        try:
            stats = {}
            
            with self.driver.session() as session:
                # 获取节点数
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                stats["node_count"] = node_count
                
                # 获取关系数
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                stats["relationship_count"] = rel_count
                
                # 获取节点类型分布
                node_types_result = session.run(
                    "MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(*) as count ORDER BY count DESC"
                )
                stats["node_types"] = [(record["type"], record["count"]) for record in node_types_result]
                
                # 获取关系类型分布
                rel_types_result = session.run(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(*) as count ORDER BY count DESC"
                )
                stats["relationship_types"] = [(record["type"], record["count"]) for record in rel_types_result]
                
                return stats
        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return {"error": str(e)}
    
    def get_all_entity_names(self, limit: int = 500) -> List[Tuple[str, str]]:
        """获取数据库中所有实体的名称和类型
        
        Args:
            limit: 返回的实体数量上限
            
        Returns:
            实体列表，每个元素为 (实体名, 实体类型)
        """
        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (n)
                    RETURN DISTINCT n.name as name, labels(n)[0] as type
                    LIMIT {limit}
                """)
                entities = [(record["name"], record["type"]) for record in result if record["name"]]
                return entities
        except Exception as e:
            print(f"获取实体列表时出错: {e}")
            return []
    
    def query_related_entities(self, entities: List[Tuple[str, str]], max_depth: int = None) -> Dict[str, Any]:
        """
        根据实体列表查询相关的图谱知识（支持模糊匹配）
        
        Args:
            entities: 实体列表，每个元素为 (实体名, 实体类型)
            max_depth: 查询最大深度
            
        Returns:
            包含相关三元组和实体的字典
        """
        if max_depth is None:
            max_depth = Config.kg_max_depth
        
        try:
            all_triples = []
            all_nodes = set()
            matched_entities = []
            
            with self.driver.session() as session:
                for entity_name, entity_type in entities:
                    # 首先尝试严格匹配
                    matched_entity = self._find_matching_entity(session, entity_name, entity_type)
                    matched_by = "strict"
                    
                    if not matched_entity:
                        # 严格匹配失败，尝试获取最相似的实体
                        print(f"严格匹配失败: {entity_name}，尝试模糊匹配...")
                        matched_entity = self._find_similar_entity(session, entity_name)
                        matched_by = "fuzzy"
                    
                    if matched_entity:
                        print(f"使用匹配的实体: {matched_entity}")
                        triples, nodes = self._query_entity_relations(session, matched_entity[0], matched_entity[1], max_depth)
                        all_triples.extend(triples)
                        all_nodes.update(nodes)
                        matched_entities.append({
                            "query": entity_name,
                            "matched": matched_entity[0],
                            "type": matched_entity[1],
                            "method": matched_by
                        })
                    else:
                        print(f"未找到实体 {entity_name} 或相似实体")
            
            return {
                "triples": all_triples,
                "nodes": [{"name": n[0], "type": n[1]} for n in list(all_nodes)],
                "count": len(all_triples),
                "matched_entities": matched_entities
            }
            
        except Exception as e:
            print(f"查询相关实体时出错: {e}")
            return {"triples": [], "nodes": [], "count": 0}
    
    def _find_matching_entity(self, session, entity_name: str, entity_type: str) -> Tuple[str, str]:
        """
        严格匹配实体
        
        Args:
            session: Neo4j会话
            entity_name: 实体名称
            entity_type: 实体类型
            
        Returns:
            匹配的实体 (名称, 类型) 或 None
        """
        try:
            clean_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', entity_type) or "概念"
            
            query = f"""
            MATCH (n:`{clean_type}` {{name: $entity_name}})
            RETURN n.name as name, labels(n)[0] as type
            LIMIT 1
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            if record:
                return (record["name"], record["type"])
            return None
        except Exception as e:
            print(f"严格匹配出错: {e}")
            return None
    
    def _find_similar_entity(self, session, query_entity: str, top_k: int = 1) -> Tuple[str, str]:
        """
        通过模糊匹配找到最相似的实体
        
        Args:
            session: Neo4j会话
            query_entity: 查询的实体名
            top_k: 返回最相似的前k个
            
        Returns:
            最相似的实体 (名称, 类型) 或 None
        """
        try:
            normalized_query = (query_entity or "").strip()
            if not normalized_query:
                return None

            result = session.run("""
                MATCH (e:Entity)
                RETURN DISTINCT e.name as name, labels(e)[0] as type
                LIMIT 2000
            """)
        
            best_match = None
            highest_score = 0.0
        
            for record in result:
                entity_name = (record["name"] or "").strip()
                entity_type = record["type"]
                if not entity_name:
                    continue
            
                # Normalize for comparison
                q = normalized_query.lower()
                n = entity_name.lower()
            
                # 使用 SequenceMatcher 进行模糊匹配评分
                score = SequenceMatcher(None, q, n).ratio()
            
                # 如果出现包含关系，强力提升分数
                if q in n or n in q:
                    score = max(score, 0.95)
            
                if score > highest_score:
                    highest_score = score
                    best_match = (entity_name, entity_type)
        
            # 设置更宽松的相似度阈值，便于召回近似实体
            if highest_score >= 0.2:
                print(f"找到相似实体: {best_match} (score={highest_score:.2f})")
                return best_match
        
            return None
        except Exception as e:
            print(f"模糊匹配出错: {e}")
            return None
    
    def _query_entity_relations(self, session, entity_name: str, entity_type: str, max_depth: int) -> Tuple[List[Dict], set]:
        """
        查询实体的相关关系
        
        Args:
            session: Neo4j会话
            entity_name: 实体名称
            entity_type: 实体类型
            max_depth: 最大查询深度
            
        Returns:
            (三元组列表, 节点集合)
        """
        all_triples = []
        all_nodes = set()
        
        try:
            # 查询以该实体为起点或终点的关系
            query = f"""
            MATCH path = (start:`{entity_type}` {{name: $entity_name}})-[*1..{max_depth}]-(connected)
            WITH relationships(path) as rels, nodes(path) as nodes
            UNWIND rels as rel
            RETURN 
                startNode(rel).name as subject_name,
                labels(startNode(rel))[0] as subject_type,
                type(rel) as predicate,
                endNode(rel).name as object_name,
                labels(endNode(rel))[0] as object_type
            LIMIT 50
            """
            
            result = session.run(query, entity_name=entity_name)
            for record in result:
                triple = {
                    "subject": {
                        "name": record["subject_name"],
                        "type": record["subject_type"]
                    },
                    "predicate": record["predicate"],
                    "object": {
                        "name": record["object_name"],
                        "type": record["object_type"]
                    }
                }
                all_triples.append(triple)
                all_nodes.add((record["subject_name"], record["subject_type"]))
                all_nodes.add((record["object_name"], record["object_type"]))
                
        except Exception as e:
            print(f"查询实体关系失败: {e}")
        
        return all_triples, all_nodes
    
    def format_kg_context(self, kg_data: Dict[str, Any]) -> str:
        """
        将图谱查询结果格式化为文本上下文
        
        Args:
            kg_data: query_related_entities返回的数据
            
        Returns:
            格式化的文本
        """
        if not kg_data or kg_data.get("count", 0) == 0:
            return ""
        
        lines = ["### 知识图谱相关信息\n"]
        
        triples = kg_data.get("triples", [])
        for triple in triples[:20]:  # 限制数量避免过长
            subject = triple["subject"]["name"]
            predicate = triple["predicate"]
            obj = triple["object"]["name"]
            lines.append(f"- {subject} {predicate} {obj}")
        
        if len(triples) > 20:
            lines.append(f"\n... 还有 {len(triples) - 20} 条相关知识\n")
        
        return "\n".join(lines)


def load_triples_to_neo4j(triples_json_path, neo4j_uri, neo4j_username, neo4j_password, 
                          clear_db=True, batch_size=100):
    """
    将三元组从JSON文件加载到Neo4j
    
    Args:
        triples_json_path: 包含三元组的JSON文件路径
        neo4j_uri: Neo4j数据库的URI
        neo4j_username: Neo4j用户名
        neo4j_password: Neo4j密码
        clear_db: 是否在加载三元组前清空数据库
        batch_size: 批处理大小
    
    Returns:
        包含已加载知识图谱统计信息的字典
    """
    # 从JSON加载三元组
    try:
        with open(triples_json_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)
            
        print(f"已从{triples_json_path}加载{len(triples)}个三元组")
    except Exception as e:
        print(f"从{triples_json_path}加载三元组时出错: {e}")
        return {"error": str(e)}
    
    # 初始化Neo4j处理器
    neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)
    neo4j_handler.connect()
    
    # 如果请求，清空数据库
    if clear_db:
        neo4j_handler.clear_database()
    
    # 创建约束
    neo4j_handler.create_constraints()
    
    # 将三元组添加到数据库
    print("正在将三元组批量添加到Neo4j...")
    successful_triples = neo4j_handler.add_triples_batch(triples, batch_size=batch_size)
    
    # 获取统计信息
    stats = neo4j_handler.get_statistics()
    stats["successful_triples"] = successful_triples
    stats["total_triples"] = len(triples)
    
    # 关闭Neo4j连接
    neo4j_handler.close()
    
    print(f"\n成功添加{successful_triples}个三元组（共{len(triples)}个）到Neo4j")
    print(f"Neo4j现在包含{stats['node_count']}个节点和{stats['relationship_count']}个关系")
    
    return stats


if __name__ == "__main__":
    # 配置参数
    # 优先从环境变量读取配置，便于在不同环境中运行
    TRIPLES_JSON_PATH = os.getenv("TRIPLES_JSON_PATH", "functional_programming_kg_results/functional_programming_triples_unique.json")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # 注意：UI 使用的是环境变量名 NEO4J_USER，因此这里也使用相同名称以保持一致
    NEO4J_USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CLEAR_DB = os.getenv("NEO4J_CLEAR_DB", "True").lower() in ("1", "true", "yes")
    BATCH_SIZE = int(os.getenv("NEO4J_BATCH_SIZE", "100"))  # 每批处理的三元组数量
    
    # 加载三元组到Neo4j
    stats = load_triples_to_neo4j(
        triples_json_path=TRIPLES_JSON_PATH,
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        clear_db=CLEAR_DB,
        batch_size=BATCH_SIZE
    )
    
    # 打印统计信息
    if "error" not in stats:
        print("\n=== 知识图谱统计信息 ===")
        print(f"总节点数: {stats['node_count']}")
        print(f"总关系数: {stats['relationship_count']}")
        
        print("\n节点类型:")
        for node_type, count in stats['node_types']:
            print(f"  {node_type}: {count}")
            
        print("\n关系类型:")
        for rel_type, count in stats['relationship_types']:
            print(f"  {rel_type}: {count}")
    else:
        print(f"错误: {stats['error']}")