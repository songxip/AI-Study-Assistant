"""
理科/文科问题分类器
支持两种模式：
1. LLM API模式（推荐）：使用现有大模型API进行分类，无需训练
2. 本地模型模式：使用微调后的BERT模型进行分类（需要先训练）
"""

import json
from typing import Literal, Optional
from openai import OpenAI
from config import Config

class SubjectClassifier:
    """问题学科分类器"""
    
    def __init__(self, mode: Literal["api", "local"] = "api", model_path: Optional[str] = None):
        """
        初始化分类器
        
        Args:
            mode: "api" 使用LLM API分类（推荐），"local" 使用本地微调模型
            model_path: 本地模型路径（仅local模式需要）
        """
        self.mode = mode
        self.client = None
        
        if mode == "api":
            # 使用LLM API模式
            self.client = OpenAI(
                api_key=Config.llm_api_key,
                base_url=Config.llm_base_url
            )
        elif mode == "local":
            # 使用本地模型模式（需要先训练）
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                if not model_path:
                    raise ValueError("local模式需要提供model_path参数")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ 本地分类模型已加载: {model_path}")
            except ImportError:
                raise ImportError("使用local模式需要安装: pip install transformers torch")
            except Exception as e:
                raise Exception(f"加载本地模型失败: {e}")
    
    def classify(self, question: str) -> dict:
        """
        分类问题属于理科还是文科
        
        Args:
            question: 用户问题
            
        Returns:
            {
                "subject": "理科" 或 "文科",
                "confidence": 0.0-1.0 的置信度,
                "reason": "分类理由"
            }
        """
        if self.mode == "api":
            return self._classify_with_api(question)
        else:
            return self._classify_with_local_model(question)
    
    def _classify_with_api(self, question: str) -> dict:
        """使用LLM API进行分类"""
        system_prompt = """你是一个专业的学科分类助手。请判断用户的问题属于理科还是文科。

理科包括：数学、物理、化学、生物、计算机科学、工程、医学、统计学等。
文科包括：语文、历史、政治、地理、文学、哲学、法律、经济、社会学、心理学等。

请以JSON格式返回：
{
    "subject": "理科" 或 "文科",
    "confidence": 0.0-1.0之间的数字,
    "reason": "简要说明分类理由"
}"""

        user_prompt = f"问题：{question}\n\n请判断这个问题属于理科还是文科。"

        try:
            response = self.client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # 降低随机性，提高一致性
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # 验证结果格式
            if result.get("subject") not in ["理科", "文科"]:
                result["subject"] = "理科" if "理科" in result.get("subject", "") else "文科"
            
            if "confidence" not in result:
                result["confidence"] = 0.8
            
            return result
            
        except Exception as e:
            print(f"分类失败: {e}")
            # 默认返回理科
            return {
                "subject": "理科",
                "confidence": 0.5,
                "reason": f"分类过程出错: {str(e)}"
            }
    
    def _classify_with_local_model(self, question: str) -> dict:
        """使用本地微调模型进行分类"""
        import torch
        
        # 编码输入
        inputs = self.tokenizer(
            question,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # 映射到标签（根据训练时的标签顺序）
        # 假设：0=理科, 1=文科（需要根据实际训练时调整）
        subject = "理科" if predicted_class == 0 else "文科"
        
        return {
            "subject": subject,
            "confidence": float(confidence),
            "reason": f"本地模型预测（类别ID: {predicted_class}）"
        }


# 便捷函数：直接调用分类器
def classify_question(question: str, mode: str = "api") -> dict:
    """
    便捷函数：分类问题
    
    Args:
        question: 用户问题
        mode: "api" 或 "local"
        
    Returns:
        分类结果字典
    """
    classifier = SubjectClassifier(mode=mode)
    return classifier.classify(question)


if __name__ == "__main__":
    # 测试代码
    test_questions = [
        "什么是二次函数？",
        "唐朝的建立时间是什么时候？",
        "如何计算矩阵的逆？",
        "《红楼梦》的作者是谁？",
        "什么是深度学习？",
        "文艺复兴时期的主要特点是什么？"
    ]
    
    print("=" * 50)
    print("测试问题分类器（API模式）")
    print("=" * 50)
    
    classifier = SubjectClassifier(mode="api")
    
    for q in test_questions:
        result = classifier.classify(q)
        print(f"\n问题: {q}")
        print(f"分类: {result['subject']} (置信度: {result['confidence']:.2f})")
        print(f"理由: {result['reason']}")

