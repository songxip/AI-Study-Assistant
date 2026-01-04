import os

class Config():

    #retrievor参数
    topd = 3    #召回文章的数量
    topt = 6    #召回文本片段的数量
    maxlen = 128  #召回文本片段的长度
    topk = 5    #query召回的关键词数量
    bert_path = '/workspace/model/embedding/tao-8k'
    recall_way = 'embed'  #召回方式, keyword, embed

    #generator参数
    max_source_length = 767  #输入的最大长度
    max_target_length = 256  #生成的最大长度
    model_max_length = 1024  #序列最大长度
    
    #embedding API 参数 - 用于 text2vec.py
    use_api = True  # 是否使用API而非本地模型
    api_key = "sk-xxx" # 请修改为实际 API Key
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = "text-embedding-v3"
    dimensions = 1024
    batch_size = 10 
    
    #LLM API 参数 - 用于 rag.py
    llm_api_key = "sk-xxx"  # 请修改为实际 API Key
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 与embedding共用同一个URL
    llm_model = "qwen-plus"  # 默认使用的LLM模型
    
    # 知识库配置
    kb_base_dir = "knowledge_bases"  # 知识库根目录
    default_kb = "default"  # 默认知识库名称
    
    # 输出目录配置 - 现在用作临时文件目录
    output_dir = "output_files"
    
    # Neo4j 图数据库配置
    neo4j_uri = "bolt://localhost:7687"  # Neo4j连接URI
    neo4j_username = "neo4j"  # Neo4j用户名
    neo4j_password = os.getenv("NEO4J_PASSWORD")  # Neo4j密码，请修改为实际密码
    
    # 知识图谱配置
    kg_enabled = True  # 是否启用知识图谱功能
    kg_extract_model = "qwen-plus"  # 用于实体提取的模型
    kg_max_entities = 10  # 每次查询最多提取的实体数
    kg_max_depth = 2  # 图谱检索最大深度