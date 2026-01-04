# RagHop

简要说明：
RagHop 是一个面向学习场景的检索增强生成（RAG）助手原型。它将本地知识库（文档、PDF、笔记）向量化并建立索引，结合联网检索与可选的知识图谱（Neo4j），为学习者生成多跳推理式的回答、错题解析与练习题，并提供一个基于 Gradio 的交互界面。

主要功能：
- 知识库管理：创建/删除/选择多知识库，支持批量上传文档并持久化到 `knowledge_bases/`。
- 语义分块与向量化：对文档做语义分块（`semantic_chunk`），使用 `text2vec.py` 调用 embedding API 或本地模型生成向量，并保存为 JSON。
- FAISS 索引：基于向量构建 FAISS 索引，用于相似度检索。
- 多跳推理 RAG：`ReasoningRAG` 支持多跳检索-推理循环，逐步补充检索查询并合成答案。
- 知识图谱（可选）：集成 Neo4j（通过 `kg_construct.py` 与 `neo4j_loader.py`）用于实体抽取与图谱增强查询。
- 错题本与 OCR：支持使用 PaddleOCR / pytesseract 对题目图片 OCR、错题解析与练习题生成。
- 笔记与家长报告：保存/管理笔记、统计学习数据并生成家长报告。
- Web 界面：基于 `gradio` 的 UI（入口：`ui.py` / `rag.py`），支持流式显示检索与推理过程。

主要文件说明：
- `rag.py`：核心后端实现，包含语义分块、向量化、索引构建、RAG、KG 增强、错题本、笔记等。
- `ui.py`：Gradio 前端界面，调用 `rag.py` 暴露的接口并提供交互页面。
- `text2vec.py`：文本向量化封装，支持 API 与本地模型两种模式。
- `retrievor.py`：联网检索与候选召回逻辑（Bing + 排序/embedding 召回）。
- `config.py`：全局配置（API keys、目录、Neo4j 连接等），请按需修改或使用环境变量。
- `kg_construct.py` / `neo4j_loader.py`：知识图谱构建与 Neo4j 交互（可选依赖）。

运行环境与依赖：
- 推荐 Python 3.12。
- 安装依赖：

```bash
pip install -r requirements.txt
```

快速启动（本地开发/测试）：
1. 在 `config.py` 中填写/检查 API key 与 base_url（或设置为环境变量），并确保 `kb_base_dir` 与 `output_dir` 存在。
2. 如需启用知识图谱，请安装并启动 Neo4j，并在 `config.py` 或环境变量中设置 `NEO4J_PASSWORD`。
3. 启动 Gradio 界面：

```bash
python rag.py
```

使用说明（概要）：
- 通过界面创建或选择知识库，上传 PDF/文本文件；项目会对文件进行分块、向量化并构建 FAISS 索引。
- 在问答页面输入问题，可选择是否启用联网搜索、多跳推理和知识图谱增强；系统会并行检索并合成答案，支持流式展示。
- 学习看板：记录学习时长和待完成任务
- 错题本：上传图片后可执行 OCR、错题解析并生成练习题；也可将错题保存到错题本。
- 笔记助手：在界面中管理、编辑笔记
- 家长视图：生成学生学习报告。

注意事项：
- `config.py` 中包含示例 API key 与 base_url，请务必替换为自己的密钥与服务地址。
- 知识图谱（Neo4j）为可选组件，未安装或未配置时相关功能会回退到普通 RAG/检索。
- 大规模向量化/索引构建对内存/磁盘有要求，生产部署请在具备相应资源的环境中运行。
- 部分功能依赖 PaddleOCR、pytesseract、faiss、torch 等库，Windows 环境下可能需要额外配置。
