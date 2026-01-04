import os
import torch
from functional import seq
import numpy as np
import torch.nn.functional as F  
from torch import cosine_similarity
from config import Config
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TextVector():
    def __init__(self, cfg):
        self.bert_path = cfg.bert_path
        
        # 从配置文件读取API相关设置
        self.use_api = getattr(cfg, 'use_api', True)
        self.api_key = getattr(cfg, 'api_key', "sk-xxx") # 请修改为实际 API Key
        self.base_url = getattr(cfg, 'base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_name = getattr(cfg, 'model_name', "text-embedding-v3")
        self.dimensions = getattr(cfg, 'dimensions', 1024)
        self.batch_size = getattr(cfg, 'batch_size', 10)
        
        # 只有在不使用API时才加载本地模型
        if not self.use_api:
            self.load_model()

    def load_model(self):
        """载入模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.model = AutoModel.from_pretrained(self.bert_path)

    def mean_pooling(self, model_output, attention_mask):
        """采用序列mean-pooling获得句子的表征向量"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_vec(self, sentences):
        """通过模型获取句子的向量"""
        if self.use_api:
            # 如果使用API，重定向到API方法
            return self.get_vec_api(sentences)
            
        # 否则使用原始BERT方法
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.data.cpu().numpy().tolist()
        return sentence_embeddings
    
    def get_vec_api(self, query, batch_size=None):
        """通过API获取句子的向量"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 空查询检查
        if not query:
            print("Warning: Empty query provided to get_vec_api")
            return []
            
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        if isinstance(query, str):
            query = [query]
        
        # 移除空字符串和None值，确保输入数据有效
        query = [q for q in query if q and isinstance(q, str) and q.strip()]
        if not query:
            print("Warning: No valid text to vectorize after filtering")
            return []
        
        all_vectors = []
        retry_count = 0
        max_retries = 2  # 允许重试几次
        
        while retry_count <= max_retries and not all_vectors:
            try:
                for i in range(0, len(query), batch_size):
                    batch = query[i:i + batch_size]
                    try:
                        completion = client.embeddings.create(
                            model=self.model_name,
                            input=batch,
                            dimensions=self.dimensions,
                            encoding_format="float"
                        )
                        vectors = [embedding.embedding for embedding in completion.data]
                        all_vectors.extend(vectors)
                    except Exception as e:
                        print(f"向量化批次 {i//batch_size + 1} 失败：{str(e)}")
                        # 不立即返回空数组，继续处理其他批次
                        continue
                
                # 检查是否有成功获取的向量
                if all_vectors:
                    break
                else:
                    retry_count += 1
                    print(f"未获取到任何向量，第 {retry_count} 次重试...")
                    
            except Exception as outer_e:
                print(f"向量化过程中发生错误：{str(outer_e)}")
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"第 {retry_count} 次重试...")
                
        # 返回向量数组，如果仍然为空，确保返回一个正确形状的空数组
        if not all_vectors and self.dimensions > 0:
            print("Warning: 返回一个空的向量数组，形状为 [0, dimensions]")
            return np.zeros((0, self.dimensions))
            
        return all_vectors

    def get_vec_batch(self, data, bs=None):
        """batch方式获取，提高效率"""
        if bs is None:
            bs = self.batch_size
            
        if self.use_api:
            # 如果使用API，直接调用API方法
            vectors = self.get_vec_api(data, bs)
            return torch.tensor(np.array(vectors)) if len(vectors) > 0 else torch.tensor(np.array([]))
        
        # 否则使用原始BERT方法
        data = seq(data).grouped(bs)
        all_vectors = []
        for batch in data:
            vecs = self.get_vec(batch)
            all_vectors.extend(vecs)
        all_vectors = torch.tensor(np.array(all_vectors))
        return all_vectors

    def vector_similarity(self, vectors):
        """以[query，text1，text2...]来计算query与text1，text2,...的cosine相似度"""
        # Add dimension checking to prevent errors
        if vectors.size(0) <= 1:
            print("Warning: Not enough vectors for similarity calculation")
            return []
            
        if len(vectors.shape) < 2:
            print("Warning: Vectors must be 2-dimensional")
            return []
        
        vectors = F.normalize(vectors, p=2, dim=1)
        q_vec = vectors[0,:]
        o_vec = vectors[1:,:]
        sim = cosine_similarity(q_vec, o_vec)
        sim = sim.data.cpu().numpy().tolist()
        return sim

cfg = Config()
tv = TextVector(cfg)
get_vector = tv.get_vec_batch  # 修正名称
get_sim = tv.vector_similarity