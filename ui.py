import gradio as gr
import os
import json
import tempfile
from datetime import date
from datetime import datetime
import traceback
from typing import Dict, Any

# 从 rag.py 导入所有后端函数
from rag import (
    get_knowledge_bases,
    create_knowledge_base,
    delete_knowledge_base,
    get_kb_files,
    batch_upload_to_kb,
    process_question_with_reasoning,
    build_kg_for_kb_file,
    build_kg_for_entire_kb,
    get_kg_statistics,
    query_with_kg_enhancement,
    KB_BASE_DIR,
    DEFAULT_KB,
    extract_text_from_pdf,  # 添加PDF文本提取函数
    ask_question_parallel,
    ask_question_with_ab,
    generate_ab_responses,
    multi_hop_generate_answer,
    # 笔记助手后端API
    load_notes,
    save_note_to_kb,
    delete_note_from_kb,
    get_note_by_id,
    # 错题本后端API
    ocr_images_to_texts,
    analyze_text_wrong_problems,
    save_wrong_problem,
    delete_wrong_problem,
    load_wrong_problems,
    format_wrong_problems_display,
    # 家长视图后端API
    generate_parent_report,
    get_learning_statistics
)

# Gradio 界面 - 修改为支持多知识库
custom_css = """
/* ========== 全局动态渐变背景 ========== */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 应用到所有可能的容器元素 - 使用更柔和的色调 */
html {
    min-height: 100% !important;
}

body,
.gradio-container,
.gradio-container > .main,
.gradio-container > .wrap,
#component-0,
.contain,
.app,
main {
    background: linear-gradient(-45deg, 
        #f3e7fa,   /* 柔和淡紫 */
        #e8f4fc,   /* 柔和淡蓝 */
        #e6f7ed,   /* 柔和淡绿 */
        #fdf6e9,   /* 柔和淡黄 */
        #fce8e6,   /* 柔和淡粉 */
        #eef3fc,   /* 柔和淡蓝紫 */
        #f3e7fa    /* 回到柔和淡紫 */
    ) !important;
    background-size: 400% 400% !important;
    -webkit-animation: gradientMove 20s ease infinite !important;
    -moz-animation: gradientMove 20s ease infinite !important;
    animation: gradientMove 20s ease infinite !important;
}

body {
    min-height: 100vh !important;
}

.gradio-container {
    min-height: 100vh !important;
    background-attachment: fixed !important;
}

/* 让内部容器透明以显示背景 */
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container > div:not(.gr-group):not(.gr-box),
.tabs > .tabitem,
.tabitem > .gap {
    background: transparent !important;
    background-color: transparent !important;
}

/* 保持卡片和组件有自己的背景 - 使用更柔和的白色 */
.gr-group, .gr-box, .gr-panel, .gr-form, .gr-input, .gr-button {
    background-color: rgba(255, 255, 255, 0.92) !important;
}

.web-search-toggle .form { display: flex !important; align-items: center !important; }
.web-search-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.web-search-toggle .checkbox-wrap { order: 1 !important; background: #e8f0e8 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.web-search-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.web-search-toggle input:checked + .checkbox-wrap { background: #7eb8da !important; }
.web-search-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }

/* ========== 修复复选框对号显示问题 ========== */
/* 确保所有复选框的勾号/对号正常显示 */

/* 针对智能问答助手页面的复选框 */
#qa-assistant-tab input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    accent-color: #1976D2 !important;
    cursor: pointer !important;
}

/* 针对笔记助手页面的复选框（实时预览等） */
#notes-assistant-tab input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    accent-color: #9C27B0 !important;
    cursor: pointer !important;
}

/* 针对错题本页面的复选框（实时预览等） */
#wrong-problem-tab input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    accent-color: #FF9800 !important;
    cursor: pointer !important;
}

/* 通用Gradio复选框样式修复 */
.gr-checkbox input[type="checkbox"],
.web-search-toggle input[type="checkbox"],
.multi-hop-toggle input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    opacity: 1 !important;
    position: relative !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
}

/* 确保复选框可见且有正确的选中样式 */
input[type="checkbox"]:checked {
    accent-color: #1976D2 !important;
    background-color: #1976D2 !important;
}

/* 笔记助手复选框选中样式 */
#notes-assistant-tab input[type="checkbox"]:checked {
    accent-color: #9C27B0 !important;
    background-color: #9C27B0 !important;
}

/* 错题本复选框选中样式 */
#wrong-problem-tab input[type="checkbox"]:checked {
    accent-color: #FF9800 !important;
    background-color: #FF9800 !important;
}

/* 为toggle开关样式的复选框添加对号 */
.web-search-toggle .checkbox-wrap .checkbox-container,
.multi-hop-toggle .checkbox-wrap .checkbox-container {
    position: relative !important;
    background: white !important;
    border-radius: 50% !important;
}

.web-search-toggle input:checked ~ .checkbox-wrap .checkbox-container::after,
.multi-hop-toggle input:checked ~ .checkbox-wrap .checkbox-container::after {
    content: '✓' !important;
    color: #1976D2 !important;
    font-weight: 700 !important;
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    font-size: 14px !important;
    line-height: 1 !important;
}
#search-results { max-height: 400px; overflow-y: auto; border: 1px solid #b8d4e8; border-radius: 8px; padding: 10px; background-color: #f5fafd; }
#question-input { border-color: #b8d4e8 !important; }
#answer-output { background-color: #f5faf5; border-color: #b8d4e8 !important; max-height: 400px; overflow-y: auto; }
.submit-btn { background-color: #7eb8da !important; border: none !important; }
.reasoning-steps { background-color: #f5faf5; border: 1px dashed #a8c8b8; padding: 10px; margin-top: 10px; border-radius: 8px; }
.loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(126, 184, 218, 0.3); border-radius: 50%; border-top-color: #7eb8da; animation: spin 1s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.stream-update { animation: fade 0.5s ease-in-out; }
@keyframes fade { from { background-color: rgba(126, 184, 218, 0.1); } to { background-color: transparent; } }
.status-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; }
.status-processing { background-color: #eef5fb; color: #155074; border-left: 4px solid #2b8fcc; }
.status-success { background-color: #e6f4ea; color: #155724; border-left: 4px solid #155724; }
.status-error { background-color: #fdf2f2; color: #8b2d2d; border-left: 4px solid #b85252; }
.multi-hop-toggle .form { display: flex !important; align-items: center !important; }
.multi-hop-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.multi-hop-toggle .checkbox-wrap { order: 1 !important; background: #e8f0e8 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.multi-hop-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.multi-hop-toggle input:checked + .checkbox-wrap { background: #8cc49a !important; }
    .multi-hop-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }

/* ========== 修复单选按钮（Radio）选中点显示问题 ========== */
/* 针对智能问答助手页面的单选按钮，确保选中小圆点可见 */
#qa-assistant-tab input[type="radio"],
.gr-radio input[type="radio"],
.gradio-radio input[type="radio"] {
    -webkit-appearance: radio !important;
    -moz-appearance: radio !important;
    appearance: radio !important;
    width: 18px !important;
    height: 18px !important;
    accent-color: #6b5dd3 !important;
    cursor: pointer !important;
    position: relative !important;
    background-color: transparent !important;
}

/* 为自定义/伪造的单选样式提供后备的选中点显示（覆盖某些主题引起的隐藏） */
#qa-assistant-tab input[type="radio"]::after,
.gr-radio input[type="radio"]::after,
.gradio-radio input[type="radio"]::after {
    content: '' !important;
    display: block !important;
    width: 10px !important;
    height: 10px !important;
    border-radius: 50% !important;
    background: transparent !important;
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    transition: background 0.12s ease !important;
}

#qa-assistant-tab input[type="radio"]:checked::after,
.gr-radio input[type="radio"]:checked::after,
.gradio-radio input[type="radio"]:checked::after {
    background: #6b5dd3 !important;
}
/* 提高方案 A/B 的可读性：使Markdown渲染文本颜色更深，便于阅读 */
#qa-assistant-tab .gr-markdown, #qa-assistant-tab gradio-markdown, #qa-assistant-tab [data-testid="markdown"] {
    color: #1b1b1b !important;
    font-size: 14px !important;
}
#qa-assistant-tab .gr-markdown h1, #qa-assistant-tab .gr-markdown h2, #qa-assistant-tab .gr-markdown h3 {
    color: #0f1720 !important;
}
.kb-management { border: 1px solid #c8dce8; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #f8fbfd; }
.kb-selector { margin-bottom: 10px; }
/* 缩小文件上传区域高度 */
.compact-upload {
    margin-bottom: 10px;
}

.file-upload.compact {
    padding: 10px;  /* 减小内边距 */
    min-height: 120px; /* 减小最小高度 */
    margin-bottom: 10px;
}

/* 优化知识库内容显示区域 */
.kb-files-list {
    height: 400px;
    overflow-y: auto;
}

/* 确保右侧列有足够空间 */
#kb-files-group {
    height: 100%;
    display: flex;
    flex-direction: column;
}
.kb-files-list { max-height: 250px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-top: 10px; background-color: #fafafa; }
#kb-management-container {
    max-width: 800px !important;
    margin: 0 !important; /* 移除自动边距，靠左对齐 */
    margin-left: 20px !important; /* 添加左边距 */
}
.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.file-upload {
    border: 2px dashed #7eb8da;
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f7ff;
    margin-bottom: 15px;
}
.tabs.tab-selected {
    background-color: #e3f2fd;
    border-bottom: 3px solid #7eb8da;
}
.group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    background-color: #fafafa;
}

/* 退出按钮样式 */
#exit-button {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(238, 90, 111, 0.3) !important;
    transition: all 0.3s ease !important;
    margin-top: 10px !important;
}
#exit-button:hover {
    background: linear-gradient(135deg, #ee5a6f 0%, #ff6b6b 100%) !important;
    box-shadow: 0 4px 12px rgba(238, 90, 111, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* 添加更多针对知识库管理页面的样式 */
#kb-controls, #kb-file-upload, #kb-files-group {
    width: 100% !important;
    max-width: 800px !important;
    margin-right: auto !important;
}

/* 修改Gradio默认的标签页样式以支持左对齐 */
.tabs > .tab-nav > button {
    flex: 0 1 auto !important; /* 修改为不自动扩展，只占用必要空间 */
}
.tabs > .tabitem {
    padding-left: 0 !important; /* 移除左边距，使内容靠左 */
}
/* 对于首页的顶部标题部分 */
#app-container h1, #app-container h2, #app-container h3, 
#app-container > .prose {
    text-align: left !important;
    padding-left: 20px !important;
}

/* 基础主题优化 */
#app-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* ========== 知识库管理 - 柔和蓝色主题 ========== */
#kb-management-tab {
    --kb-primary: #6ba3d6;
    --kb-secondary: #8fc1e8;
    --kb-bg: #f0f7fc;
    --kb-accent: #5a9ac8;
    --kb-light: #d4e8f7;
    --kb-lighter: #f5fafd;
    --kb-dark: #4a8ab8;
    --kb-darker: #3a7aa8;
    --kb-highlight: #7eb8e0;
    --kb-glow: rgba(107, 163, 214, 0.25);
}

#kb-management-tab {
    background: linear-gradient(135deg, var(--kb-lighter) 0%, var(--kb-light) 50%, var(--kb-bg) 100%) !important;
}

#kb-management-tab .gradio-container {
    background: linear-gradient(135deg, rgba(107, 163, 214, 0.06), rgba(143, 193, 232, 0.12), rgba(212, 232, 247, 0.18)) !important;
}

#kb-management-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--kb-highlight), var(--kb-primary), var(--kb-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--kb-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#kb-management-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--kb-primary), var(--kb-dark), var(--kb-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--kb-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#kb-management-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--kb-lighter), var(--kb-light)) !important;
    border: 2px solid var(--kb-secondary) !important;
    color: var(--kb-dark) !important;
}

#kb-management-tab .file-upload.compact {
    border: 2px dashed var(--kb-secondary) !important;
    background: linear-gradient(145deg, var(--kb-lighter), rgba(143, 193, 232, 0.15)) !important;
    transition: all 0.3s ease !important;
}

#kb-management-tab .file-upload.compact:hover {
    border-color: var(--kb-primary) !important;
    background: linear-gradient(145deg, var(--kb-light), rgba(143, 193, 232, 0.25)) !important;
    box-shadow: 0 4px 15px var(--kb-glow) !important;
}

#kb-management-tab .kb-management {
    border: 2px solid var(--kb-primary) !important;
    border-radius: 12px !important;
    background: linear-gradient(145deg, var(--kb-bg), var(--kb-lighter), #ffffff) !important;
    box-shadow: 0 4px 20px var(--kb-glow), inset 0 0 30px rgba(107, 163, 214, 0.04) !important;
}

#kb-management-tab .gr-group {
    background: linear-gradient(145deg, var(--kb-lighter), #ffffff) !important;
    border: 1px solid var(--kb-light) !important;
    box-shadow: 0 2px 10px rgba(107, 163, 214, 0.08) !important;
}

#kb-management-tab input, #kb-management-tab textarea, #kb-management-tab select,
#kb-management-tab .gr-dropdown, #kb-management-tab .gr-dropdown > div {
    border: 2px solid #64B5F6 !important;
    background: linear-gradient(145deg, #ffffff, #f0f8ff) !important;
}

#kb-management-tab input:focus, #kb-management-tab textarea:focus,
#kb-management-tab .gr-dropdown:focus-within, #kb-management-tab .gr-dropdown:focus-within > div {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.25), 0 0 15px rgba(33, 150, 243, 0.3) !important;
}

#kb-management-tab h1, #kb-management-tab h2, #kb-management-tab h3 {
    color: var(--kb-dark) !important;
    border-left: 4px solid var(--kb-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(107, 163, 214, 0.08) !important;
    background: linear-gradient(90deg, var(--kb-lighter), transparent) !important;
}

/* ========== 学习看板 - 柔和绿色主题 ========== */
#learning-board-tab {
    --lb-primary: #7cb88a;
    --lb-secondary: #a3d4af;
    --lb-bg: #f2f9f4;
    --lb-accent: #6aad7a;
    --lb-light: #d4ecd9;
    --lb-lighter: #f5fbf7;
    --lb-dark: #5a9d6a;
    --lb-darker: #4a8d5a;
    --lb-highlight: #8cc49a;
    --lb-mint: #c5e8cc;
    --lb-glow: rgba(124, 184, 138, 0.25);
}

#learning-board-tab {
    background: linear-gradient(135deg, var(--lb-lighter) 0%, var(--lb-light) 50%, var(--lb-mint) 100%) !important;
}

#learning-board-tab .gradio-container {
    background: linear-gradient(135deg, rgba(124, 184, 138, 0.06), rgba(163, 212, 175, 0.12), rgba(212, 236, 217, 0.18)) !important;
}

#learning-board-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--lb-highlight), var(--lb-primary), var(--lb-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--lb-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#learning-board-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--lb-primary), var(--lb-dark), var(--lb-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--lb-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#learning-board-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--lb-lighter), var(--lb-light)) !important;
    border: 2px solid var(--lb-secondary) !important;
    color: var(--lb-dark) !important;
}

#learning-board-tab .timer-display {
    font-family: 'Courier New', monospace !important;
    font-size: 1.8em !important;
    font-weight: bold !important;
    color: var(--lb-darker) !important;
    background: linear-gradient(135deg, var(--lb-mint), var(--lb-light), var(--lb-lighter)) !important;
    padding: 15px 25px !important;
    border-radius: 10px !important;
    border: 2px solid var(--lb-secondary) !important;
    text-align: center !important;
    box-shadow: 0 4px 15px var(--lb-glow), inset 0 2px 10px rgba(255,255,255,0.5) !important;
    text-shadow: 0 1px 2px rgba(124, 184, 138, 0.15) !important;
}

#learning-board-tab .gr-group {
    border: 2px solid var(--lb-secondary) !important;
    border-radius: 12px !important;
    background: linear-gradient(145deg, var(--lb-lighter), var(--lb-light), #ffffff) !important;
    box-shadow: 0 4px 20px var(--lb-glow), inset 0 0 30px rgba(124, 184, 138, 0.04) !important;
}

#learning-board-tab input, #learning-board-tab textarea, #learning-board-tab select,
#learning-board-tab .gr-dropdown, #learning-board-tab .gr-dropdown > div {
    border: 2px solid #81C784 !important;
    background: linear-gradient(145deg, #ffffff, #f0fff0) !important;
}

#learning-board-tab input:focus, #learning-board-tab textarea:focus,
#learning-board-tab .gr-dropdown:focus-within, #learning-board-tab .gr-dropdown:focus-within > div {
    border-color: #4CAF50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.25), 0 0 15px rgba(76, 175, 80, 0.3) !important;
}

#learning-board-tab h1, #learning-board-tab h2, #learning-board-tab h3 {
    color: var(--lb-dark) !important;
    border-left: 4px solid var(--lb-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(124, 184, 138, 0.08) !important;
    background: linear-gradient(90deg, var(--lb-lighter), transparent) !important;
}

/* ========== 笔记助手 - 柔和紫色主题 ========== */
#notes-assistant-tab {
    --na-primary: #b08cc0;
    --na-secondary: #c8a8d4;
    --na-bg: #f9f4fb;
    --na-accent: #a07cb0;
    --na-light: #e6d8ec;
    --na-lighter: #fbf7fd;
    --na-dark: #906ca0;
    --na-darker: #805c90;
    --na-highlight: #c0a0cc;
    --na-lavender: #dcc8e4;
    --na-glow: rgba(176, 140, 192, 0.25);
}

#notes-assistant-tab {
    background: linear-gradient(135deg, var(--na-lighter) 0%, var(--na-light) 50%, var(--na-lavender) 100%) !important;
}

#notes-assistant-tab .gradio-container {
    background: linear-gradient(135deg, rgba(176, 140, 192, 0.06), rgba(200, 168, 212, 0.12), rgba(230, 216, 236, 0.18)) !important;
}

#notes-assistant-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--na-highlight), var(--na-primary), var(--na-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--na-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#notes-assistant-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--na-primary), var(--na-dark), var(--na-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--na-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#notes-assistant-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--na-lighter), var(--na-light)) !important;
    border: 2px solid var(--na-secondary) !important;
    color: var(--na-dark) !important;
}

#notes-assistant-tab textarea, #notes-assistant-tab input[type="text"], #notes-assistant-tab select,
#notes-assistant-tab input, #notes-assistant-tab .gr-dropdown, #notes-assistant-tab .gr-dropdown > div,
#notes-assistant-tab .gr-dropdown input, #notes-assistant-tab .gr-dropdown select,
#notes-assistant-tab [data-testid="dropdown"], #notes-assistant-tab .wrap > input {
    border: 2px solid #BA68C8 !important;
    border-radius: 8px !important;
    background: linear-gradient(145deg, #ffffff, #fdf0ff) !important;
}

#notes-assistant-tab textarea:focus, #notes-assistant-tab input[type="text"]:focus,
#notes-assistant-tab input:focus, #notes-assistant-tab .gr-dropdown:focus-within,
#notes-assistant-tab .gr-dropdown:focus-within > div, #notes-assistant-tab .gr-dropdown:focus-within input {
    border-color: #9C27B0 !important;
    box-shadow: 0 0 0 3px rgba(156, 39, 176, 0.25), 0 0 15px rgba(156, 39, 176, 0.3) !important;
}

#notes-assistant-tab .gr-column {
    border-radius: 12px !important;
    background: linear-gradient(145deg, var(--na-lighter), var(--na-light), #ffffff) !important;
    box-shadow: 0 4px 20px var(--na-glow), inset 0 0 30px rgba(176, 140, 192, 0.04) !important;
    border: 1px solid var(--na-lavender) !important;
}

#notes-assistant-tab .gr-group {
    background: linear-gradient(145deg, var(--na-lighter), #ffffff) !important;
    border: 1px solid var(--na-light) !important;
    box-shadow: 0 2px 10px rgba(176, 140, 192, 0.08) !important;
}

#notes-assistant-tab h1, #notes-assistant-tab h2, #notes-assistant-tab h3 {
    color: var(--na-dark) !important;
    border-left: 4px solid var(--na-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(176, 140, 192, 0.08) !important;
    background: linear-gradient(90deg, var(--na-lighter), transparent) !important;
}

/* ========== 错题本 - 柔和橙色主题 ========== */
#wrong-problem-tab {
    --wp-primary: #e8b87a;
    --wp-secondary: #f0cc9a;
    --wp-bg: #fdf8f2;
    --wp-accent: #d8a86a;
    --wp-light: #f5e4d0;
    --wp-lighter: #fefbf7;
    --wp-dark: #c8986a;
    --wp-darker: #b8885a;
    --wp-highlight: #f0c08a;
    --wp-peach: #f8dcc0;
    --wp-glow: rgba(232, 184, 122, 0.25);
}

#wrong-problem-tab {
    background: linear-gradient(135deg, var(--wp-lighter) 0%, var(--wp-light) 50%, var(--wp-peach) 100%) !important;
}

#wrong-problem-tab .gradio-container {
    background: linear-gradient(135deg, rgba(232, 184, 122, 0.06), rgba(240, 204, 154, 0.12), rgba(245, 228, 208, 0.18)) !important;
}

#wrong-problem-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--wp-highlight), var(--wp-primary), var(--wp-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--wp-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#wrong-problem-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--wp-primary), var(--wp-dark), var(--wp-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--wp-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#wrong-problem-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--wp-lighter), var(--wp-light)) !important;
    border: 2px solid var(--wp-secondary) !important;
    color: var(--wp-dark) !important;
}

#wrong-problem-tab textarea, #wrong-problem-tab input, #wrong-problem-tab select,
#wrong-problem-tab .gr-dropdown, #wrong-problem-tab .gr-dropdown > div {
    border: 2px solid #FFB74D !important;
    border-radius: 8px !important;
    background: linear-gradient(145deg, #ffffff, #fffaf0) !important;
}

#wrong-problem-tab textarea:focus, #wrong-problem-tab input:focus,
#wrong-problem-tab .gr-dropdown:focus-within, #wrong-problem-tab .gr-dropdown:focus-within > div {
    border-color: #FF9800 !important;
    box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.25), 0 0 15px rgba(255, 152, 0, 0.3) !important;
}

#wrong-problem-tab .gr-group {
    background: linear-gradient(145deg, var(--wp-lighter), var(--wp-light), #ffffff) !important;
    border: 1px solid var(--wp-peach) !important;
    box-shadow: 0 4px 20px var(--wp-glow), inset 0 0 30px rgba(232, 184, 122, 0.04) !important;
    border-radius: 12px !important;
}

#wrong-problem-tab .gr-button-stop {
    background: linear-gradient(135deg, #e8a090, #d89090, #c88080) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(216, 144, 144, 0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#wrong-problem-tab .gr-button-stop:hover {
    background: linear-gradient(135deg, #d89090, #c88080, #b87070) !important;
    box-shadow: 0 6px 20px rgba(216, 144, 144, 0.4) !important;
}

#wrong-problem-tab h1, #wrong-problem-tab h2, #wrong-problem-tab h3 {
    color: var(--wp-dark) !important;
    border-left: 4px solid var(--wp-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(232, 184, 122, 0.1) !important;
    background: linear-gradient(90deg, var(--wp-lighter), transparent) !important;
}

/* ========== 智能问答助手 - 柔和深蓝主题 ========== */
#qa-assistant-tab {
    --qa-primary: #6a9cc8;
    --qa-secondary: #8eb8dc;
    --qa-bg: #f0f6fb;
    --qa-accent: #4a7ca8;
    --qa-light: #d0e4f2;
    --qa-lighter: #f5f9fc;
    --qa-dark: #5a8cb8;
    --qa-darker: #4a7ca8;
    --qa-highlight: #7eb0d8;
    --qa-sky: #b8d4ec;
    --qa-glow: rgba(106, 156, 200, 0.25);
}

#qa-assistant-tab {
    background: linear-gradient(135deg, var(--qa-lighter) 0%, var(--qa-light) 50%, var(--qa-sky) 100%) !important;
}

#qa-assistant-tab .gradio-container {
    background: linear-gradient(135deg, rgba(106, 156, 200, 0.06), rgba(142, 184, 220, 0.12), rgba(208, 228, 242, 0.18)) !important;
}

#qa-assistant-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--qa-highlight), var(--qa-primary), var(--qa-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--qa-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#qa-assistant-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--qa-primary), var(--qa-dark), var(--qa-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--qa-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#qa-assistant-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--qa-lighter), var(--qa-light)) !important;
    border: 2px solid var(--qa-secondary) !important;
    color: var(--qa-dark) !important;
}

#qa-assistant-tab .chatbot {
    border: 2px solid var(--qa-secondary) !important;
    border-radius: 12px !important;
    background: linear-gradient(145deg, #ffffff, var(--qa-lighter)) !important;
    box-shadow: 0 4px 25px var(--qa-glow), inset 0 0 30px rgba(106, 156, 200, 0.02) !important;
}

/* 聊天框滚动交给 Gradio 默认处理，避免自定义逻辑干扰 */

#qa-assistant-tab #question-input, #qa-assistant-tab input, #qa-assistant-tab textarea, #qa-assistant-tab select,
#qa-assistant-tab .gr-dropdown, #qa-assistant-tab .gr-dropdown > div {
    border: 2px solid #42A5F5 !important;
    border-radius: 10px !important;
    padding: 15px !important;
    font-size: 1.1em !important;
    background: linear-gradient(145deg, #ffffff, #f0f7ff) !important;
}

#qa-assistant-tab #question-input:focus, #qa-assistant-tab input:focus, #qa-assistant-tab textarea:focus,
#qa-assistant-tab .gr-dropdown:focus-within, #qa-assistant-tab .gr-dropdown:focus-within > div {
    border-color: #1976D2 !important;
    box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.25), 0 0 15px rgba(25, 118, 210, 0.3) !important;
}

#qa-assistant-tab .gr-group {
    background: linear-gradient(145deg, var(--qa-lighter), var(--qa-light), #ffffff) !important;
    border: 1px solid var(--qa-sky) !important;
    box-shadow: 0 4px 20px var(--qa-glow), inset 0 0 30px rgba(106, 156, 200, 0.04) !important;
    border-radius: 12px !important;
}

#qa-assistant-tab h1, #qa-assistant-tab h2, #qa-assistant-tab h3 {
    color: var(--qa-dark) !important;
    border-left: 4px solid var(--qa-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(106, 156, 200, 0.08) !important;
    background: linear-gradient(90deg, var(--qa-lighter), transparent) !important;
}

/* ========== 家长视图 - 柔和青色主题 ========== */
#parent-view-tab {
    --pv-primary: #5ab8c4;
    --pv-secondary: #88d4dc;
    --pv-bg: #f2fbfc;
    --pv-accent: #4aa8b4;
    --pv-light: #c8ecf0;
    --pv-lighter: #f5fdfe;
    --pv-dark: #4a98a4;
    --pv-darker: #3a8894;
    --pv-highlight: #70c8d4;
    --pv-aqua: #a8e0e8;
    --pv-teal: #60c0cc;
    --pv-glow: rgba(90, 184, 196, 0.25);
}

#parent-view-tab {
    background: linear-gradient(135deg, var(--pv-lighter) 0%, var(--pv-light) 50%, var(--pv-aqua) 100%) !important;
}

#parent-view-tab .gradio-container {
    background: linear-gradient(135deg, rgba(90, 184, 196, 0.06), rgba(136, 212, 220, 0.12), rgba(200, 236, 240, 0.18)) !important;
}

#parent-view-tab .gr-button-primary {
    background: linear-gradient(135deg, var(--pv-highlight), var(--pv-primary), var(--pv-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--pv-glow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    color: white !important;
    padding: 15px 30px !important;
    font-size: 1.1em !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

#parent-view-tab .gr-button-primary:hover {
    background: linear-gradient(135deg, var(--pv-teal), var(--pv-dark), var(--pv-darker)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--pv-glow), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

#parent-view-tab .gr-button-secondary {
    background: linear-gradient(135deg, var(--pv-lighter), var(--pv-light)) !important;
    border: 2px solid var(--pv-secondary) !important;
    color: var(--pv-dark) !important;
}

#parent-view-tab .report-display {
    border: 2px solid var(--pv-secondary) !important;
    border-radius: 12px !important;
    background: linear-gradient(145deg, var(--pv-lighter), var(--pv-light), #ffffff) !important;
    box-shadow: 0 4px 25px var(--pv-glow), inset 0 0 30px rgba(90, 184, 196, 0.04) !important;
}

#parent-view-tab .gr-group {
    background: linear-gradient(145deg, var(--pv-lighter), var(--pv-light), #ffffff) !important;
    border: 1px solid var(--pv-aqua) !important;
    box-shadow: 0 4px 20px var(--pv-glow), inset 0 0 30px rgba(90, 184, 196, 0.04) !important;
    border-radius: 12px !important;
}

#parent-view-tab .gr-column {
    background: linear-gradient(145deg, var(--pv-lighter), var(--pv-light), #ffffff) !important;
    border: 1px solid var(--pv-aqua) !important;
    box-shadow: 0 4px 20px var(--pv-glow) !important;
}

#parent-view-tab input, #parent-view-tab textarea, #parent-view-tab select,
#parent-view-tab .gr-dropdown, #parent-view-tab .gr-dropdown > div {
    border: 2px solid #4DD0E1 !important;
    background: linear-gradient(145deg, #ffffff, #f0ffff) !important;
}

#parent-view-tab input:focus, #parent-view-tab textarea:focus,
#parent-view-tab .gr-dropdown:focus-within, #parent-view-tab .gr-dropdown:focus-within > div {
    border-color: #00BCD4 !important;
    box-shadow: 0 0 0 3px rgba(0, 188, 212, 0.25), 0 0 15px rgba(0, 188, 212, 0.3) !important;
}

/* 家长视图艺术标题样式 */
@keyframes titleGlow {
    0% { box-shadow: 0 4px 15px rgba(90, 184, 196, 0.3), 0 0 20px rgba(136, 212, 220, 0.2); }
    50% { box-shadow: 0 4px 25px rgba(90, 184, 196, 0.45), 0 0 35px rgba(136, 212, 220, 0.35); }
    100% { box-shadow: 0 4px 15px rgba(90, 184, 196, 0.3), 0 0 20px rgba(136, 212, 220, 0.2); }
}

@keyframes titleShimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes floatTitle {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-3px); }
}

@keyframes sparkle {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
}

/* Accordion 标题艺术效果 */
#learning-stats-title > .label-wrap, #report-title > .label-wrap {
    background: linear-gradient(135deg, 
        #70c8d4 0%, 
        #5ab8c4 25%, 
        #60c0cc 50%, 
        #5ab8c4 75%, 
        #4a98a4 100%) !important;
    background-size: 200% auto !important;
    color: white !important;
    padding: 15px 25px !important;
    border-radius: 12px !important;
    border: none !important;
    animation: titleGlow 3s ease-in-out infinite, titleShimmer 4s linear infinite !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 255, 255, 0.3) !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    font-size: 1.1em !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}

#learning-stats-title > .label-wrap::before, #report-title > .label-wrap::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent) !important;
    animation: titleShimmer 3s infinite !important;
}

#learning-stats-title > .label-wrap span, #report-title > .label-wrap span {
    color: white !important;
    font-weight: 700 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

#learning-stats-title, #report-title {
    border: 2px solid var(--pv-aqua) !important;
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 25px rgba(90, 184, 196, 0.2) !important;
    margin-bottom: 20px !important;
}

#learning-stats-title > .wrap, #report-title > .wrap {
    background: linear-gradient(145deg, rgba(224, 247, 250, 0.9), rgba(178, 235, 242, 0.8), rgba(255,255,255,0.95)) !important;
    padding: 20px !important;
}

/* 旧的样式保留用于其他可能的元素 */
#parent-view-tab .art-title {
    background: linear-gradient(135deg, 
        var(--pv-highlight) 0%, 
        var(--pv-primary) 25%, 
        var(--pv-teal) 50%, 
        var(--pv-primary) 75%, 
        var(--pv-dark) 100%) !important;
    background-size: 200% auto !important;
    color: white !important;
    padding: 15px 25px !important;
    border-radius: 12px !important;
    border-left: none !important;
    animation: titleGlow 3s ease-in-out infinite, titleShimmer 4s linear infinite, floatTitle 4s ease-in-out infinite !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 255, 255, 0.3) !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    font-size: 1.2em !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
}

#parent-view-tab h1, #parent-view-tab h2, #parent-view-tab h3 {
    color: var(--pv-dark) !important;
    border-left: 4px solid var(--pv-primary) !important;
    padding-left: 10px !important;
    text-shadow: 0 1px 2px rgba(90, 184, 196, 0.08) !important;
    background: linear-gradient(90deg, var(--pv-lighter), transparent) !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

/* ========== 通用优化 ========== */

/* 卡片式设计 */
.gr-group, .gr-column {
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    background-color: white !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
    border: 1px solid #e0e0e0 !important;
}

.gr-group:hover, .gr-column:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-3px) !important;
}

/* 标签页样式优化 */
.tabs .tab-nav {
    background-color: #f8f9fa !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 10px 0 10px !important;
    margin-bottom: 20px !important;
}

.tabs .tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    margin-right: 5px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    border: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    background-color: #e9ecef !important;
}

.tabs .tab-nav button:hover {
    background-color: #dee2e6 !important;
}

.tabs .tab-nav button.selected {
    background-color: white !important;
    border-color: #7eb8da !important;
    border-bottom-color: white !important;
    position: relative !important;
    z-index: 1 !important;
    color: #6ba8ca !important;
}

/* 按钮样式统一 */
.gr-button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
    cursor: pointer !important;
}

.gr-button:hover:not(:disabled) {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

.gr-button-secondary {
    background-color: #f8f9fa !important;
    border-color: #dee2e6 !important;
    color: #495057 !important;
}

/* 输入框和下拉框样式优化 */
input, textarea, select, .gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #90CAF9 !important;
    padding: 12px 15px !important;
    transition: all 0.3s ease !important;
    font-size: 1em !important;
}

input:focus, textarea:focus, select:focus, .gr-dropdown:focus-within {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2), 0 0 12px rgba(33, 150, 243, 0.25) !important;
    outline: none !important;
}

/* 状态提示样式 */
.status-success {
    background-color: #e6f4ea !important;
    color: #155724 !important;
    border-left: 4px solid #155724 !important;
}

.status-error {
    background-color: #fdf2f2 !important;
    color: #8b2d2d !important;
    border-left: 4px solid #b85252 !important;
}

.status-processing {
    background-color: #eef5fb !important;
    color: #155074 !important;
    border-left: 4px solid #2b8fcc !important;
    border-left: 4px solid #7eb8da !important;
}

/* 标题样式优化 */
h1, h2, h3, h4 {
    color: #2C3E50 !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
}

h1 {
    background: linear-gradient(135deg, #7eb8da, #5a8094) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    padding-bottom: 15px !important;
    border-bottom: 3px solid #7eb8da !important;
    font-size: 2.5em !important;
}

h2 {
    color: #6ba8ca !important;
    padding-left: 15px !important;
    border-left: 4px solid #7eb8da !important;
    font-size: 1.8em !important;
}

h3 {
    color: #2C3E50 !important;
    font-size: 1.4em !important;
    margin-top: 1.5rem !important;
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 10px !important;
    height: 10px !important;
}

::-webkit-scrollbar-track {
    background: #f1f1f1 !important;
    border-radius: 5px !important;
}

::-webkit-scrollbar-thumb {
    background: #a8c8da !important;
    border-radius: 5px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #8ab0c4 !important;
}

/* 图标美化 */
.gr-button-primary::before {
    content: "✨ " !important;
}

.gr-button-stop::before {
    content: "⚠️ " !important;
}

/* 加载动画优化 */
.loading-spinner {
    border: 3px solid rgba(126, 184, 218, 0.15) !important;
    border-radius: 50% !important;
    border-top: 3px solid #7eb8da !important;
    animation: spin 1s linear infinite !important;
    width: 30px !important;
    height: 30px !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 响应式优化 */
@media (max-width: 768px) {
    .gr-row {
        flex-direction: column !important;
    }
    
    .gr-column {
        width: 100% !important;
        min-width: 100% !important;
        margin-bottom: 15px !important;
    }
    
    .tabs .tab-nav {
        flex-wrap: wrap !important;
    }
    
    .tabs .tab-nav button {
        flex: 1 1 auto !important;
        margin-bottom: 5px !important;
        padding: 10px 15px !important;
        font-size: 0.9em !important;
    }
    
    h1 {
        font-size: 2em !important;
    }
    
    h2 {
        font-size: 1.5em !important;
    }
    
    h3 {
        font-size: 1.2em !important;
    }
}

/* 确保家长视图标题颜色正确 */
#parent-view-tab .gr-markdown h3:first-child {
    color: #4a98a4 !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

/* 笔记助手按钮大小统一 */
.button-size {
    min-height: 27px !important;
    padding: 4px 8px !important;
    font-size: 6px !important;
}
"""

# 自定义主题
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray",
    text_size="lg",
    spacing_size="md",
    radius_size="md"
)

# Gradio Markdown LaTeX 分隔符配置（支持行内 $...$ 和块级 $$...$$ 公式）
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# 添加简单的JavaScript，通过html组件实现
js_code = """
<script>
// Load MathJax for LaTeX rendering in Markdown previews, including inline $...$
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    processRefs: true,
    packages: {'[+]': ['noerrors', 'noundefined']},
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    multlineWidth: '85%',
    inlineMath_old: [['$', '$'], ['\\(', '\\)']],
    // 确保单$不会被忽略
    processClass: 'tex2jax_process|mathjax',
    ignoreClass: 'tex2jax_ignore'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'annotation', 'annotation-xml'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process|mathjax'
  },
  startup: {
    typeset: false,
    pageReady: function () {
      return MathJax.startup.defaultPageReady().then(function () {
        console.log('MathJax initial typesetting complete');
      });
    }
  },
  svg: { 
    fontCache: 'global',
    scale: 1,
    minScale: 0.5,
    mtextInheritFont: false,
    merrorInheritFont: true,
    mathmlSpacing: false,
    skipAttributes: {},
    exFactor: 0.5,
    displayAlign: 'center',
    displayIndent: '0'
  },
  chtml: {
    scale: 1,
    minScale: 0.5,
    mtextInheritFont: false,
    merrorInheritFont: true,
    mathmlSpacing: false
  }
};

(function() {
  var mj = document.createElement('script');
  mj.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  mj.async = true;
  document.head.appendChild(mj);
})();

let __typesetToken = 0;
let __lastUserScrollTs = 0;

function findScrollEl(root) {
  const host = (root && root.querySelector)
    ? (root.querySelector('#chatbot') || root.querySelector('.qa-chatbot'))
    : null;
  if (!host) return null;
  const nodes = [host, ...host.querySelectorAll('*')];
  return nodes.find(el => {
    const st = getComputedStyle(el);
    const oy = st && st.overflowY;
    return (oy === 'auto' || oy === 'scroll') && (el.scrollHeight > el.clientHeight + 2);
  }) || host;
}

function __hookUserScroll(root) {
  const scrollEl = findScrollEl(root);
  if (!scrollEl || scrollEl.__userScrollHooked) return;
  scrollEl.__userScrollHooked = true;
  
  // 只在用户滚动时更新最后滚动时间戳
  const mark = () => { 
    if (__lastUserScrollTs !== Date.now()) { // 只有在发生变化时更新
      __lastUserScrollTs = Date.now(); 
    }
  };
  
  // 只更新滚动信息，不干扰滚动
  ['scroll','wheel','touchmove','pointerdown','pointerup','mousedown','mouseup'].forEach(ev => {
    scrollEl.addEventListener(ev, mark, { passive: true, capture: true });
  });
}

function typesetAllMath() {
  if (!(window.MathJax && window.MathJax.typesetPromise)) {
    console.log('MathJax not ready yet');
    return;
  }

  __typesetToken += 1;
  __hookUserScroll(document);
  
  console.log('Starting MathJax typesetting...');
  const roots = [document];
  
  // 收集常见的 Shadow DOM 根（Gradio 组件内）
  const withShadow = Array.from(document.querySelectorAll('*')).filter(el => el.shadowRoot);
  withShadow.forEach(el => roots.push(el.shadowRoot));
  
  // 对每个根执行排版（限定在可能包含Markdown的容器内提高效率）
  const targets = [];
  roots.forEach(root => {
    // 扩大选择器范围，确保覆盖所有Markdown渲染区域
    const selectors = [
      '.prose', 'gradio-markdown', '[data-testid="markdown"]', 
      '.chatbot', '.gradio-container', '.markdown', '.gr-markdown',
      '.note-preview', '.wrong-editor-preview', 'body'
    ];
    selectors.forEach(selector => {
      try {
        const elements = root.querySelectorAll(selector);
        targets.push(...elements);
      } catch (e) {
        // 忽略无效选择器
      }
    });
  });
  
  // 确保至少有document.body
  if (!targets.length) targets.push(document.body);
  
  console.log('Typesetting targets:', targets.length);
  window.MathJax.typesetPromise(targets)
    .then(() => {
      console.log('MathJax typesetting complete');
      // 不调整 scrollTop，滚动完全交给用户
    })
    .catch(err => console.warn('MathJax typeset error:', err));
}

/* setupChatScrollLock removed - let Gradio handle scrolling natively */

document.addEventListener('DOMContentLoaded', function() {
    // 自动触发一次排版
    typesetAllMath();
    // setupChatScrollLock removed - Gradio handles scrolling natively

    // 当页面加载完毕后，找到提交按钮，并为其添加点击事件
    const observer = new MutationObserver(function(mutations) {
        // 找到提交按钮
        const submitButton = document.querySelector('button[data-testid="submit"]');
        if (submitButton) {
            submitButton.addEventListener('click', function() {
                // 找到检索标签页按钮并点击它
                setTimeout(function() {
                    const retrievalTab = document.querySelector('[data-testid="tab-button-retrieval-tab"]');
                    if (retrievalTab) retrievalTab.click();
                }, 100);
            });
            observer.disconnect(); // 一旦找到并设置事件，停止观察
        }
    });
    
    // 开始观察文档变化
    observer.observe(document.body, { childList: true, subtree: true });

    // 监听DOM变化，触发MathJax重新排版，使$...$与$$...$$公式正确渲染
    // 使用防抖避免过于频繁的重新排版
    const isMathArea = (node) => {
        if (!node) return false;
        const el = (node.nodeType === 1) ? node : node.parentElement;
        return !!(el && el.closest && el.closest(
            ".gr-markdown, gradio-markdown, [data-testid='markdown'], #chatbot, .qa-chatbot"
        ));
    };

    let typesetTimeout;
    const mjObserver = new MutationObserver(function(mutations) {
        if (!mutations.some(m => isMathArea(m.target))) return;
        clearTimeout(typesetTimeout);
        typesetTimeout = setTimeout(typesetAllMath, 200);
    });
    mjObserver.observe(document.body, { childList: true, subtree: true });
    
    // 页面完全加载以及MathJax脚本加载完成后再排版
    // 多次尝试确保MathJax完全加载
    const tryTypeset = () => {
        if (window.MathJax && window.MathJax.typesetPromise) {
            setTimeout(typesetAllMath, 500);
            setTimeout(typesetAllMath, 1500);
            setTimeout(typesetAllMath, 3000);
        } else {
            setTimeout(tryTypeset, 500);
        }
    };
    
    if (document.readyState === 'complete') {
        tryTypeset();
    } else {
        window.addEventListener('load', tryTypeset);
    }

    // 自动学习计时功能
    // 监听页面可见性变化，离开页面时自动保存
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            console.log('[自动计时] 页面隐藏，触发自动保存');
            // 触发自动保存（通过隐藏的按钮）
            const autoSaveBtn = document.getElementById('auto-save-trigger');
            if (autoSaveBtn) {
                autoSaveBtn.click();
            }
        } else {
            console.log('[自动计时] 页面可见，恢复计时');
        }
    });
    
    // 页面关闭前保存
    window.addEventListener('beforeunload', function(e) {
        console.log('[自动计时] 页面即将关闭，触发自动保存');
        const autoSaveBtn = document.getElementById('auto-save-trigger');
        if (autoSaveBtn) {
            autoSaveBtn.click();
        }
    });
});
</script>
"""

# 用于存储当前活跃的计时会话（内存中）
_active_timer_session = {
    'start_time': None, 
    'elapsed_seconds': 0, 
    'is_running': False,
    'auto_mode': True,  # 自动计时模式
    'last_save_time': None  # 上次保存时间
}

# 获取学习看板文件路径
def get_learning_board_path():
    return os.path.join(KB_BASE_DIR, "learning_board.json")

# 加载学习看板数据
def load_learning_board():
    path = get_learning_board_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"tasks": [], "study_records": []}
    else:
        return {"tasks": [], "study_records": []}

# 保存学习看板数据
def save_learning_board(board: Dict[str, Any]):
    path = get_learning_board_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(board, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存学习看板失败: {e}")
        traceback.print_exc()
        return False

# 构建学习看板 UI 状态
def _build_board_ui_state():
    board = load_learning_board()
    tasks = board.get('tasks', [])
    # 生成任务列表Markdown
    if not tasks:
        tasks_md = "### 待办任务\n\n暂无任务"
        choices = []
    else:
        lines = ["### 待办任务\n"]
        for i, t in enumerate(tasks):
            status = "✅ 已完成" if t.get('completed') else "🔲 未完成"
            lines.append(f"{i+1}. {t.get('title')} — {t.get('est_minutes', 0)} 分钟 — {status}")
        tasks_md = "\n".join(lines)
        choices = [f"{t.get('id')}|{t.get('title')}" for t in tasks]

    # 完成度
    total = len(tasks)
    completed = sum(1 for t in tasks if t.get('completed'))
    percent = int((completed / total) * 100) if total > 0 else 0
    progress_html_str = f"<div style='font-weight:bold'>完成度: {percent}% ({completed}/{total})</div>"

    dropdown_update = gr.update(choices=choices, value=choices[0] if choices else None)
    return tasks_md, dropdown_update, progress_html_str

# 添加学习任务
def add_task_action(title, est_minutes):
    if not title or not str(title).strip():
        return _build_board_ui_state()
    board = load_learning_board()
    tasks = board.get('tasks', [])
    # 生成唯一id
    existing_ids = [int(t['id'][4:]) for t in tasks if isinstance(t.get('id'), str) and t.get('id').startswith('task') and t.get('id')[4:].isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 0
    new_task = {
        'id': f'task{next_id}',
        'title': str(title).strip(),
        'est_minutes': int(est_minutes) if est_minutes else 0,
        'completed': False,
        'created': date.today().isoformat()
    }
    tasks.append(new_task)
    board['tasks'] = tasks
    save_learning_board(board)
    return _build_board_ui_state()

# 解析下拉菜单选项值
def _parse_dropdown_value(val):
    if not val:
        return None
    # format is 'taskX|title'
    if '|' in val:
        return val.split('|', 1)[0]
    return val

# 切换任务完成状态
def toggle_task_action(selection):
    tid = _parse_dropdown_value(selection)
    if not tid:
        return _build_board_ui_state()
    board = load_learning_board()
    tasks = board.get('tasks', [])
    for t in tasks:
        if t.get('id') == tid:
            t['completed'] = not bool(t.get('completed'))
            break
    board['tasks'] = tasks
    save_learning_board(board)
    return _build_board_ui_state()

# 删除学习任务
def delete_task_action(selection):
    tid = _parse_dropdown_value(selection)
    if not tid:
        return _build_board_ui_state()
    board = load_learning_board()
    tasks = board.get('tasks', [])
    tasks = [t for t in tasks if t.get('id') != tid]
    board['tasks'] = tasks
    save_learning_board(board)
    return _build_board_ui_state()

# 格式化时间显示
def _format_time(seconds):
    """格式化秒数为 HH:MM:SS（处理浮点数）"""
    # 使用四舍五入到最近秒，避免因定时器抖动导致跳秒
    total_seconds = int(round(seconds))  # 四舍五入为整数秒
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# 计算当前会话已消耗秒数
def _get_current_session_elapsed():
    """获取当前会话已消耗秒数（保留小数避免跳秒）"""
    if _active_timer_session['is_running'] and _active_timer_session['start_time']:
        elapsed = (datetime.now() - _active_timer_session['start_time']).total_seconds()
        return _active_timer_session['elapsed_seconds'] + elapsed
    return _active_timer_session['elapsed_seconds']

# 自动开始计时（页面加载时）
def auto_start_timer():
    """自动开始计时，在页面加载时调用"""
    if not _active_timer_session['is_running']:
        _active_timer_session['is_running'] = True
        _active_timer_session['start_time'] = datetime.now()
        _active_timer_session['auto_mode'] = True
        print(f"[自动计时] 已启动，开始时间: {_active_timer_session['start_time']}")
    return _get_timer_display()

# 手动开始计时
def start_timer_action():
    """手动开始计时"""
    if not _active_timer_session['is_running']:
        _active_timer_session['is_running'] = True
        _active_timer_session['start_time'] = datetime.now()
    # 返回当前计时显示、开始/暂停按钮状态
    current_elapsed = _get_current_session_elapsed()
    time_str = _format_time(current_elapsed)
    return time_str, gr.update(interactive=False), gr.update(interactive=True)

# 暂停计时
def pause_timer_action():
    """暂停计时"""
    if _active_timer_session['is_running'] and _active_timer_session['start_time']:
        elapsed = (datetime.now() - _active_timer_session['start_time']).total_seconds()
        # 保留浮点秒以保持精度，避免累加整数导致跳秒
        _active_timer_session['elapsed_seconds'] += elapsed
        _active_timer_session['is_running'] = False
        _active_timer_session['start_time'] = None
    current_elapsed = _get_current_session_elapsed()
    time_str = _format_time(current_elapsed)
    return time_str, gr.update(interactive=True), gr.update(interactive=False)

# 获取计时器显示文本（简化版）
def _get_timer_display():
    """获取计时器显示文本"""
    current_elapsed = _get_current_session_elapsed()
    time_str = _format_time(current_elapsed)
    return time_str

# 实时更新计时器（供定时器调用）
def update_timer_display():
    """实时更新计时器显示"""
    return _get_timer_display()

# 退出程序前保存所有数据
def exit_program():
    """退出程序：先保存学习时长，然后退出"""
    import sys
    import os
    
    try:
        # 保存当前学习时长
        current_elapsed = _get_current_session_elapsed()
        minutes = int(current_elapsed / 60) if current_elapsed > 0 else 0
        
        if minutes >= 1:
            board = load_learning_board()
            recs = board.get('study_records', [])
            today_str = date.today().isoformat()
            
            # 检查今天是否已有记录，如果有则累加，没有则新增
            existing_index = next((i for i, r in enumerate(recs) if r.get('date') == today_str), None)
            if existing_index is not None:
                # 今天已有记录，累加分钟数
                recs[existing_index]['minutes'] += minutes
                msg = f"✅ 已累加 {minutes} 分钟（今日总计: {recs[existing_index]['minutes']} 分钟）\n正在退出程序..."
            else:
                # 今天没有记录，新增一条
                recs.append({'date': today_str, 'minutes': minutes})
                msg = f"✅ 已保存 {minutes} 分钟学习时长\n正在退出程序..."
            
            board['study_records'] = recs
            save_learning_board(board)
            print(f"[退出程序] {msg}")
        else:
            msg = "✅ 学习时长不足1分钟\n正在退出程序..."
            print(f"[退出程序] {msg}")
        
        # 延迟一下让用户看到消息
        import time
        time.sleep(0.5)
        
        # 退出程序
        print("[退出程序] 程序即将关闭...")
        os._exit(0)  # 强制退出
        
    except Exception as e:
        error_msg = f"❌ 保存失败: {str(e)}\n仍将退出程序..."
        print(f"[退出程序] {error_msg}")
        import time
        time.sleep(0.5)
        os._exit(1)

# 自动保存学习记录（静默保存）
def auto_save_session():
    """自动保存学习会话，不重置计时器（用于页面离开时）"""
    try:
        current_elapsed = _get_current_session_elapsed()
        minutes = int(current_elapsed / 60) if current_elapsed > 0 else 0
        
        # 如果学习时长小于1分钟，不保存
        if minutes < 1:
            print(f"[自动保存] 学习时长不足1分钟，跳过保存")
            return "学习时长不足1分钟，未保存"
        
        board = load_learning_board()
        recs = board.get('study_records', [])
        today_str = date.today().isoformat()
        
        # 检查今天是否已有记录，如果有则累加，没有则新增
        existing_index = next((i for i, r in enumerate(recs) if r.get('date') == today_str), None)
        if existing_index is not None:
            # 今天已有记录，累加分钟数
            new_total = recs[existing_index]['minutes'] + minutes
            recs[existing_index]['minutes'] = new_total
            print(f"[自动保存] 已累加 {minutes} 分钟，今日累计: {new_total} 分钟")
        else:
            # 今天没有记录，新增一条
            recs.append({'date': today_str, 'minutes': minutes})
            print(f"[自动保存] 新增记录 {minutes} 分钟")
        
        board['study_records'] = recs
        
        # 保存到文件
        if save_learning_board(board):
            _active_timer_session['last_save_time'] = datetime.now()
            _active_timer_session['elapsed_seconds'] = 0  # 重置已保存的时间
            _active_timer_session['start_time'] = datetime.now() if _active_timer_session['is_running'] else None
            print(f"[自动保存] 数据已写入: {get_learning_board_path()}")
            return f"已自动保存 {minutes} 分钟学习时长"
        else:
            return "保存失败"
    except Exception as e:
        print(f"[自动保存] 失败: {e}")
        traceback.print_exc()
        return f"保存失败: {str(e)}"

# 保存本次学习会话（手动保存）
def save_session_action():
    """手动保存本次学习会话"""
    current_elapsed = _get_current_session_elapsed()
    minutes = int(current_elapsed / 60) if current_elapsed > 0 else 0
    if minutes <= 0:
        return _build_board_ui_state()
    
    board = load_learning_board()
    recs = board.get('study_records', [])
    today_str = date.today().isoformat()
    
    # 检查今天是否已有记录，如果有则累加，没有则新增
    existing_index = next((i for i, r in enumerate(recs) if r.get('date') == today_str), None)
    if existing_index is not None:
        # 今天已有记录，累加分钟数
        recs[existing_index]['minutes'] += minutes
    else:
        # 今天没有记录，新增一条
        recs.append({'date': today_str, 'minutes': minutes})
    
    board['study_records'] = recs
    save_learning_board(board)
    
    # 重置计时器
    _active_timer_session['is_running'] = False
    _active_timer_session['start_time'] = None
    _active_timer_session['elapsed_seconds'] = 0
    
    return _build_board_ui_state()

# 清空今日学习记录
def clear_today_action():
    board = load_learning_board()
    today = date.today().isoformat()
    recs = [r for r in board.get('study_records', []) if r.get('date') != today]
    board['study_records'] = recs
    save_learning_board(board)
    return _build_board_ui_state()

# 创建知识库函数
def create_kb_and_refresh(kb_name):
    result = create_knowledge_base(kb_name)
    kbs = get_knowledge_bases()
    # 更新三个下拉菜单（管理界面、对话隐藏同步下拉、笔记助手）
    return (
        result,
        gr.update(choices=kbs, value=kb_name if "创建成功" in result else None),
        gr.update(choices=kbs, value=kb_name if "创建成功" in result else None),
        gr.update(choices=kbs, value=kb_name if "创建成功" in result else None),
    )

# 刷新知识库列表
def refresh_kb_list():
    kbs = get_knowledge_bases()
    # 更新三个下拉菜单（管理界面、对话隐藏同步下拉、笔记助手）
    return (
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
    )

# 删除知识库
def delete_kb_and_refresh(kb_name):
    result = delete_knowledge_base(kb_name)
    kbs = get_knowledge_bases()
    # 更新三个下拉菜单（管理界面、对话隐藏同步下拉、笔记助手）
    return (
        result,
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
        gr.update(choices=kbs, value=kbs[0] if kbs else None),
    )

# 更新知识库文件列表
def update_kb_files_list(kb_name):
    if not kb_name:
        return "未选择知识库"
    
    files = get_kb_files(kb_name)
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))
    
    if not files:
        files_str = "知识库中暂无文件"
    else:
        files_str = "**文件列表:**\n\n" + "\n".join([f"- {file}" for file in files])
    
    index_status = "\n\n**索引状态:** " + ("✅ 已建立索引" if has_index else "❌ 未建立索引")
    
    return f"### 知识库: {kb_name}\n\n{files_str}{index_status}"

# 同步知识库选择 - 管理界面到对话界面
def sync_kb_to_chat(kb_name):
    # 同步到对话（隐藏）和笔记助手下拉
    return gr.update(value=kb_name), gr.update(value=kb_name)

# 同步知识库选择 - 对话界面到管理界面
def sync_chat_to_kb(kb_name):
    # 返回给管理界面、笔记助手，并更新文件列表
    return gr.update(value=kb_name), gr.update(value=kb_name), update_kb_files_list(kb_name)

# 处理文件上传到指定知识库
def process_upload_to_kb(files, kb_name):
    if not kb_name:
        return "错误：未选择知识库"
    
    result = batch_upload_to_kb(files, kb_name)
    # 更新知识库文件列表
    files_list = update_kb_files_list(kb_name)
    return result, files_list

# 知识库选择变化时
def on_kb_change(kb_name):
    if not kb_name:
        return "未选择知识库", "选择知识库查看文件..."
    
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))
    status = f"已选择知识库: {kb_name}" + (" (已建立索引)" if has_index else " (未建立索引)")
    
    # 更新文件列表
    files_list = update_kb_files_list(kb_name)
    
    return status, files_list

# 构建 Gradio UI
def build_ui():
    """构建并返回Gradio UI"""
    with gr.Blocks(title="学术知识问答系统", theme=custom_theme, css=custom_css, elem_id="app-container") as demo:
        with gr.Column(elem_id="header-container"):
            with gr.Row():
                with gr.Column(scale=9):
                    gr.Markdown("""
                    # 📖 AI智能学习助手
                    **智能学术学习助手，支持多知识库管理、多轮对话、普通语义检索和高级多跳推理**  
                    本系统支持创建多个知识库，上传TXT/PDF/MD等常见文本文件，通过语义向量检索或创新的多跳推理机制提供学术信息问答服务。
                    """)
                with gr.Column(scale=1, min_width=120):
                    exit_btn = gr.Button("🚪 退出程序", variant="stop", size="lg", elem_id="exit-button")
                    exit_status = gr.Textbox(label="状态", visible=False, interactive=False)
        
        # 添加JavaScript脚本
        gr.HTML(js_code, visible=False)
        
        # 使用State来存储对话历史
        chat_history_state = gr.State([])
        ab_answer_state = gr.State({})
        ab_choice_state = gr.State("A")
        
        # 创建标签页
        with gr.Tabs() as tabs:
            # 知识库管理标签页
            with gr.TabItem("知识库管理"):
                with gr.Row(elem_id="kb-management-tab", elem_classes="theme-kb"):
                    # 左侧列：控制区conda 
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### 📚 知识库管理与构建")
                        
                        with gr.Row(elem_id="kb-controls"):
                            with gr.Column(scale=1):
                                new_kb_name = gr.Textbox(
                                    label="新知识库名称",
                                    placeholder="输入新知识库名称",
                                    lines=1
                                )
                                create_kb_btn = gr.Button("创建知识库", variant="primary", scale=1)
                        
                            with gr.Column(scale=1):
                                current_kbs = get_knowledge_bases()
                                kb_dropdown = gr.Dropdown(
                                    label="选择知识库",
                                    choices=current_kbs,
                                    value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                                    elem_classes="kb-selector"
                                )
                                
                                with gr.Row():
                                    refresh_kb_btn = gr.Button("刷新列表", size="sm", scale=1)
                                    delete_kb_btn = gr.Button("删除知识库", size="sm", variant="stop", scale=1)
                        
                        kb_status = gr.Textbox(label="知识库状态", interactive=False, placeholder="选择或创建知识库")
                        
                        with gr.Group(elem_id="kb-file-upload", elem_classes="compact-upload"):
                            gr.Markdown("### 📄 上传文件到知识库")
                            file_upload = gr.File(
                                label="选择文件（支持多选：.txt, .md, .json, .csv, .py, .html, .pdf 等文本文件）",
                                type="filepath",
                                file_types=[".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html", ".pdf"],
                                file_count="multiple",
                                elem_classes="file-upload compact"
                            )
                            upload_status = gr.Textbox(label="上传状态", interactive=False, placeholder="上传后显示状态")
                        
                        kb_select_for_chat = gr.Dropdown(
                            label="为对话选择知识库",
                            choices=current_kbs,
                            value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                            visible=False  # 隐藏，仅用于同步
                        )
                            
                    with gr.Column(scale=1, min_width=400):
                        with gr.Group(elem_id="kb-files-group"):
                            gr.Markdown("### 📋 知识库内容")
                            
                            # 知识图谱批量构建按钮
                            with gr.Row():
                                build_kg_all_btn = gr.Button(
                                    "🧠 为当前知识库构建知识图谱", 
                                    variant="primary",
                                    scale=2
                                )
                            kg_build_all_status = gr.Textbox(
                                label="知识图谱构建状态",
                                interactive=False,
                                lines=8,
                                placeholder="点击按钮开始为整个知识库构建知识图谱..."
                            )
                            
                            # 初始化时显示默认知识库的文件列表
                            initial_kb = DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None)
                            initial_files = update_kb_files_list(initial_kb) if initial_kb else "选择知识库查看文件..."
                            kb_files_list = gr.Markdown(
                                value=initial_files,
                                elem_classes="kb-files-list",
                                latex_delimiters=LATEX_DELIMITERS
                            )
                    
                    # 用于对话界面的知识库选择器
                    kb_select_for_chat = gr.Dropdown(
                        label="为对话选择知识库",
                        choices=current_kbs,
                        value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                        visible=False  # 隐藏，仅用于同步
                    )
            
            # 学习看板标签页
            with gr.TabItem("学习看板"):
                with gr.Row(elem_id="learning-board-tab"):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 📝 学习看板")
                        task_title_input = gr.Textbox(label="任务标题", placeholder="例如：复习线性代数", lines=1)
                        task_est_input = gr.Number(label="预计分钟", value=30)
                        with gr.Row():
                            add_task_btn = gr.Button("添加任务", variant="primary")
                            refresh_board_btn = gr.Button("🔄 刷新数据", variant="secondary", size="sm")
                        with gr.Row():
                            task_select_dropdown = gr.Dropdown(label="选择任务（用于操作）", choices=[], value=None)
                            toggle_task_btn = gr.Button("切换完成状态", size="sm")
                            delete_task_btn = gr.Button("删除任务", size="sm", variant="stop")
                        gr.Markdown("#### ⏱️ 自动学习计时器")
                        gr.Markdown("💡 **自动模式:** 打开页面自动开始计时，关闭页面自动保存")
                        with gr.Row():
                            timer_display = gr.Textbox(
                                label="当前学习时长", 
                                value="00:00:00 | 🟢 自动计时中 | 已累计 0 分钟", 
                                interactive=False, 
                                elem_classes="timer-display"
                            )
                        with gr.Row():
                            start_timer_btn = gr.Button("▶️ 开始", size="sm", variant="primary")
                            pause_timer_btn = gr.Button("⏸️ 暂停", size="sm")
                        # 隐藏的自动保存触发按钮
                        auto_save_trigger_btn = gr.Button("自动保存", elem_id="auto-save-trigger", visible=False)
                        clear_today_btn = gr.Button("清空今日记录", size="sm", variant="stop")
                    with gr.Column(scale=2):
                        # 初始化学习看板数据
                        initial_board_state = _build_board_ui_state()
                        tasks_markdown = gr.Markdown(initial_board_state[0], latex_delimiters=LATEX_DELIMITERS)
                        progress_html = gr.HTML(initial_board_state[2])

            # 笔记助手标签页
            with gr.TabItem("笔记助手"):
                with gr.Row(elem_id="notes-assistant-tab"):
                    # 左侧：笔记管理
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### 📝 笔记管理")
                        
                        # 知识库选择器（笔记与知识库关联）
                        kb_dropdown_notes = gr.Dropdown(
                            label="选择知识库",
                            choices=current_kbs,
                            value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                        )
                        
                        # 笔记列表（下拉框）
                        notes_dropdown = gr.Dropdown(
                            label="选择笔记",
                            choices=[],
                            value=None,
                            interactive=True
                        )
                        
                        with gr.Row():
                            new_note_btn = gr.Button("📄 新建笔记", size="sm", variant="primary")
                            refresh_notes_btn = gr.Button("🔄 刷新列表", size="sm")
                            delete_note_btn = gr.Button("🗑️ 删除笔记", size="sm", variant="stop")
                        
                        notes_status = gr.Textbox(label="状态", interactive=False, placeholder="选择或创建笔记")
                    
                    # 中间：笔记编辑器
                    with gr.Column(scale=2):
                        gr.Markdown("### ✍️ 笔记编辑器")
                        
                        note_title_input = gr.Textbox(
                            label="笔记标题",
                            placeholder="输入笔记标题...",
                            lines=1
                        )
                        
                        note_content_input = gr.Textbox(
                            label="笔记内容（支持 Markdown 与 LaTeX）",
                            placeholder="在此输入或粘贴 Markdown 内容，例如：\n# 标题\n段落内容，支持行内公式 $a_1$ 和块级公式 $$E=mc^2$$\n\n提示：未来版本将支持语音转文本功能！",
                            lines=15,
                            elem_classes="note-textarea"
                        )
                        
                        note_tags_input = gr.Textbox(
                            label="标签（用逗号分隔）",
                            placeholder="例如：数学,第一章,重点",
                            lines=1
                        )
                        
                        with gr.Row():
                            save_note_btn = gr.Button("💾 保存笔记", variant="primary", size="sm")
                            clear_note_btn = gr.Button("🔄 清空编辑器", variant="secondary", size="sm")
                            preview_md_btn = gr.Button("🔎 预览 Markdown", size="sm")
                            live_preview_checkbox = gr.Checkbox(label="实时预览", value=False, elem_classes="button-size")
                        
                        # Markdown 渲染预览区域
                        note_preview_md = gr.Markdown(value="", label="预览 (Markdown)", latex_delimiters=LATEX_DELIMITERS)
                        
                        # 隐藏的state用于存储当前笔记ID
                        current_note_id = gr.State(None)
                    
                    # 右侧：知识库文件预览
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 📂 知识库文件预览")
                        
                        kb_files_dropdown = gr.Dropdown(
                            label="选择文件",
                            choices=[],
                            value=None,
                            interactive=True
                        )
                        
                        with gr.Row():
                            refresh_files_btn = gr.Button("🔄 刷新文件", size="sm")
                            open_file_btn = gr.Button("📂 打开文件", size="sm", variant="primary")
                            build_kg_btn = gr.Button("🧠 为文件构建知识图谱", size="sm", variant="primary")
                        
                        file_preview_text = gr.Textbox(
                            label="文件内容预览",
                            placeholder="选择文件查看内容（支持 .txt, .md, .py, .json 等文本文件）",
                            lines=20,
                            interactive=False,
                            max_lines=20
                        )
                        
                        file_info_text = gr.Textbox(
                            label="文件信息",
                            interactive=False,
                            placeholder="文件大小、修改时间等信息"
                        )
                        kg_build_status = gr.Textbox(label="知识图谱构建状态", interactive=False, placeholder="点击上方按钮开始构建...")

            # 错题本标签页（与其它四个页面并列）
            with gr.TabItem("错题本"):
                # 添加状态变量来跟踪错题记录显示状态
                wrong_records_visible = gr.State(False)
                with gr.Row(elem_id="wrong-problem-tab"):
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### 🧩 错题本：上传图片→OCR→编辑→生成\n支持拍照或图片文件（jpg/png）")
                        wrong_images = gr.File(
                            label="上传错题图片（单张或多张）",
                            type="filepath",
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg", ".bmp"]
                        )
                        ocr_btn = gr.Button("识别并填充到编辑框", variant="primary")
                        level_dropdown = gr.Dropdown(
                            label="难度水平",
                            choices=["auto", "easy", "medium", "hard"],
                            value="auto"
                        )
                        count_input = gr.Number(label="生成题目数量", value=5, minimum=1, maximum=20)
                        tags_input = gr.Textbox(label="标签（逗号分隔）", placeholder="例如：数学,代数,难题")
                        with gr.Row():
                            gen_wrong_btn = gr.Button("生成类似题目", variant="primary")
                            save_wrong_btn = gr.Button("💾 加入错题本", variant="secondary")
                        view_records_btn = gr.Button("📖 查看错题记录", variant="secondary")
                        wrong_problem_selector = gr.Dropdown(
                            label="选择错题（用于删除）",
                            choices=[],
                            value=None
                        )
                        with gr.Row():
                            delete_wrong_btn = gr.Button("🗑️ 删除选中错题", variant="stop", size="sm")
                            refresh_wrong_btn = gr.Button("🔄 刷新错题列表", variant="secondary", size="sm")
                        save_status = gr.Textbox(label="操作状态", interactive=False, placeholder="等待操作...")
                    with gr.Column(scale=2):
                        wrong_editor = gr.Textbox(label="识别与编辑内容（Markdown）", lines=12, elem_classes="wrong-textarea")
                        with gr.Row():
                            wrong_preview_btn = gr.Button("🔎 预览编辑内容", size="sm")
                            wrong_live_preview_checkbox = gr.Checkbox(label="实时预览", value=False)
                        wrong_editor_preview = gr.Markdown(label="编辑内容预览 (Markdown)", value="", latex_delimiters=LATEX_DELIMITERS)
                        wrong_result_md = gr.Markdown(label="结果与建议 (Markdown)", value="等待编辑后生成...", latex_delimiters=LATEX_DELIMITERS)
                        wrong_json = gr.JSON(label="结构化结果", value={})
                        wrong_records_display = gr.Markdown(label="错题记录", value="点击'查看错题记录'按钮查看历史", latex_delimiters=LATEX_DELIMITERS)

                def run_ocr_to_editor(images):
                    files = []
                    if images:
                        files = list(images) if isinstance(images, (list, tuple)) else [images]
                    items = ocr_images_to_texts(files)
                    combined = "\n\n".join([it.get("text", "") for it in items if it.get("text")])
                    return combined or "(OCR未识别到文本，请手动输入或检查图片清晰度)"

                def run_generate_from_editor(text_content, level, count):
                    display, data = analyze_text_wrong_problems(text_content or "", level=str(level), count=int(count or 5))
                    return display, data

                # 编辑框 Markdown 渲染与实时预览
                def render_wrong_markdown(content):
                    if not content:
                        return ""
                    return content

                def live_wrong_preview(content, enabled):
                    if enabled:
                        return content or ""
                    return ""

                def run_save_wrong_problem(content, level, tags):
                    if not content or not content.strip():
                        return "❌ 错题内容不能为空", gr.update()
                    result = save_wrong_problem(content, level=str(level), tags=str(tags or ""))
                    # 刷新错题选择器
                    problems = load_wrong_problems()
                    choices = [(f"{p.get('created', 'Unknown')[:10]} - {p.get('content', '')[:30]}...", p.get('id')) for p in problems]
                    return result, gr.update(choices=choices)

                def run_view_records(is_visible):
                    if is_visible:
                        # 隐藏错题记录
                        return "点击'查看错题记录'按钮查看历史", gr.update(choices=[]), gr.update(value="📖 查看错题记录"), False
                    else:
                        # 显示错题记录
                        problems = load_wrong_problems()
                        choices = [(f"{p.get('created', 'Unknown')[:10]} - {p.get('content', '')[:30]}...", p.get('id')) for p in problems]
                        return format_wrong_problems_display(), gr.update(choices=choices), gr.update(value="📖 隐藏错题记录"), True

                def run_refresh_wrong_problems():
                    """刷新错题列表"""
                    problems = load_wrong_problems()
                    choices = [(f"{p.get('created', 'Unknown')[:10]} - {p.get('content', '')[:30]}...", p.get('id')) for p in problems]
                    return gr.update(choices=choices), f"✅ 已刷新，共 {len(problems)} 道错题"

                def run_delete_wrong_problem(problem_id):
                    if not problem_id:
                        return "❌ 请先选择要删除的错题", gr.update(), format_wrong_problems_display()
                    result = delete_wrong_problem(problem_id)
                    # 刷新列表和选择器
                    problems = load_wrong_problems()
                    choices = [(f"{p.get('created', 'Unknown')[:10]} - {p.get('content', '')[:30]}...", p.get('id')) for p in problems]
                    display = format_wrong_problems_display()
                    return result, gr.update(choices=choices, value=None), display

                ocr_btn.click(
                    fn=run_ocr_to_editor,
                    inputs=[wrong_images],
                    outputs=[wrong_editor]
                )
                gen_wrong_btn.click(
                    fn=run_generate_from_editor,
                    inputs=[wrong_editor, level_dropdown, count_input],
                    outputs=[wrong_result_md, wrong_json]
                )
                wrong_preview_btn.click(
                    fn=render_wrong_markdown,
                    inputs=[wrong_editor],
                    outputs=[wrong_editor_preview]
                )
                wrong_editor.change(
                    fn=live_wrong_preview,
                    inputs=[wrong_editor, wrong_live_preview_checkbox],
                    outputs=[wrong_editor_preview]
                )
                save_wrong_btn.click(
                    fn=run_save_wrong_problem,
                    inputs=[wrong_editor, level_dropdown, tags_input],
                    outputs=[save_status, wrong_problem_selector]
                )
                view_records_btn.click(
                    fn=run_view_records,
                    inputs=[wrong_records_visible],
                    outputs=[wrong_records_display, wrong_problem_selector, view_records_btn, wrong_records_visible]
                )
                delete_wrong_btn.click(
                    fn=run_delete_wrong_problem,
                    inputs=[wrong_problem_selector],
                    outputs=[save_status, wrong_problem_selector, wrong_records_display]
                )
                refresh_wrong_btn.click(
                    fn=run_refresh_wrong_problems,
                    inputs=[],
                    outputs=[wrong_problem_selector, save_status]
                )

            # 智能问答助手标签页
            with gr.TabItem("智能问答助手"):
                with gr.Row(elem_id="qa-assistant-tab"):
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 对话设置")
                        
                        kb_dropdown_chat = gr.Dropdown(
                            label="选择知识库进行对话",
                            choices=current_kbs,
                            value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                        )
                        
                        with gr.Row():
                            web_search_toggle = gr.Checkbox(
                                label="🌐 启用联网搜索",
                                value=True,
                                info="获取最新学术动态",
                                elem_classes="web-search-toggle"
                            )
                            table_format_toggle = gr.Checkbox(
                                label="📊 表格格式输出",
                                value=True,
                                info="使用Markdown表格展示结构化回答",
                                elem_classes="web-search-toggle"
                            )
                        
                        multi_hop_toggle = gr.Checkbox(
                            label="🔄 启用多跳推理",
                            value=False,
                            info="使用高级多跳推理机制（较慢但更全面）",
                            elem_classes="multi-hop-toggle"
                        )
                        kg_toggle = gr.Checkbox(
                            label="🧠 启用知识图谱(KG)",
                            value=False,
                            info="结合Neo4j图谱与RAG共同回答"
                        )
                        
                        with gr.Accordion("💡 查看本次回答的知识来源", open=True):
                            search_results_output = gr.Markdown(
                                label="知识来源详情",
                                elem_id="search-results",
                                value="等待提交问题...",
                                elem_classes="source-info",
                                latex_delimiters=LATEX_DELIMITERS
                            )
                        
                    with gr.Column(scale=3):
                        gr.Markdown("### 💬 对话历史")
                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            label="对话历史",
                            height=550,
                            elem_classes="qa-chatbot",
                            autoscroll=False
                        )
                        question_input = gr.Textbox(
                        label="输入学术相关问题",
                        placeholder="例如：证明三角形全等的方法？",
                        lines=2,
                        elem_id="question-input"
                    )
                        ab_toggle = gr.Checkbox(
                            label="生成A/B思路回答",
                            value=True,
                            info="同一问题给出两种不同思路，便于选择",
                        )
                        with gr.Row():
                            answer_a_md = gr.Markdown(label="方案A：链式推理版", value="等待生成...", latex_delimiters=LATEX_DELIMITERS)
                            answer_b_md = gr.Markdown(label="方案B：启发式/案例版", value="等待生成...", latex_delimiters=LATEX_DELIMITERS)
                        ab_choice_radio = gr.Radio(
                            label="选择希望采用的思维方式",
                            choices=[
                                ("A：链式推理+依据", "A"),
                                ("B：启发式/案例说明", "B")
                            ],
                            value="A"
                        )
                        apply_ab_btn = gr.Button("将选中思路写入对话", variant="secondary")
                        ab_status = gr.HTML(value="")
                
                with gr.Row(elem_classes="submit-row"):
                    submit_btn = gr.Button("提交问题", variant="primary", elem_classes="submit-btn")
                    clear_btn = gr.Button("清空输入", variant="secondary")
                    clear_history_btn = gr.Button("清空对话历史", variant="secondary", elem_classes="clear-history-btn")
                
                # 状态显示框
                status_box = gr.HTML(
                    value='<div class="status-box status-processing">准备就绪，等待您的问题</div>',
                    visible=True
                )
                
                gr.Examples(
                    examples=[
                        ["第一次工业革命的意义？"],
                        ["文艺复兴运动的意义？"],
                        ["三角函数和差化积公式？"],
                        ["圆锥表面积计算公式？"],
                        ["鲁迅的代表作？"]
                    ],
                    inputs=question_input,
                    label="示例问题（点击尝试）"
                )

            # 家长视图标签页（现在在最后）
            with gr.TabItem("家长视图", elem_id="parent-view-tab", elem_classes="theme-parent"):
                gr.Markdown("""### 📊 学生学习进度报告
                这是为家长提供的学生学习进展总览。系统会自动收集学生的学习数据，
                并由AI教育顾问生成个性化的进度报告和建议。
                """)
                
                with gr.Row():
                    with gr.Column():
                        gen_report_btn = gr.Button("📈 生成学习进度报告", variant="primary", size="lg")
                        refresh_report_btn = gr.Button("🔄 刷新数据", variant="secondary")
                
                # 学习统计显示
                with gr.Accordion("📌 学习统计数据", open=True, elem_id="learning-stats-title"):
                    stats_display = gr.Markdown(value="点击生成报告后显示统计数据...")
                
                # 报告显示区（使用 Markdown 渲染）
                with gr.Accordion("📋 AI 教育顾问报告", open=True, elem_id="report-title"):
                    report_display = gr.Markdown(value="点击生成报告后显示内容...")
                
                # 报告下载
                report_file = gr.File(
                    label="⬇️ 下载学习进度报告 (Markdown)",
                    file_count="single",
                    interactive=False,
                    visible=False
                )
                
                # 下载提示
                gr.Markdown("💡 **提示**：报告内容可以复制或截图分享给家长。")
                
                def run_generate_report():
                    report_md, stats = generate_parent_report()
                    # 格式化统计数据显示
                    if stats:
                        stats_md = f"""#### 📈 核心指标
                        - **任务完成度**：{stats['completion_rate']}% ({stats['completed_tasks']}/{stats['total_tasks']})
                        - **今日学习时长**：{stats['today_minutes']} 分钟
                        - **本周学习时长**：{stats['week_minutes']} 分钟
                        - **累计学习时长**：{stats['total_minutes']} 分钟  
                        - **笔记总数**：{stats['note_count']} 篇
                        - **错题记录**：{stats['wrong_count']} 道
                        - Easy：{stats['wrong_by_level'].get('easy', 0)} | Medium：{stats['wrong_by_level'].get('medium', 0)} | Hard：{stats['wrong_by_level'].get('hard', 0)} | 其他：{stats['wrong_by_level'].get('auto', 0)}
                        """
                    else:
                        stats_md = "暂无统计数据"
                    
                    # 保存报告为临时文件供下载
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as f:
                            f.write(report_md if report_md else "暂无报告内容")
                            file_path = f.name
                        download_file = gr.update(value=file_path, visible=True)
                    except Exception as e:
                        print(f"生成下载文件失败: {e}")
                        download_file = gr.update(value=None, visible=False)
                    
                    return stats_md, report_md, download_file
                
                gen_report_btn.click(
                    fn=run_generate_report,
                    inputs=[],
                    outputs=[stats_display, report_display, report_file]
                )
                
                refresh_report_btn.click(
                    fn=run_generate_report,
                    inputs=[],
                    outputs=[stats_display, report_display, report_file]
                )
        
        # 学习看板按钮绑定
        add_task_btn.click(
            fn=add_task_action,
            inputs=[task_title_input, task_est_input],
            outputs=[tasks_markdown, task_select_dropdown, progress_html]
        )

        def apply_ab_choice(choice, ab_state, chat_history):
            """将选中的A/B思路写回对话记录"""
            if not ab_state:
                return chat_history or [], '<div class="status-box status-error">当前没有可用的A/B候选，请先提交问题生成</div>', chat_history or []
            selected_key = choice if choice in ("A", "B") else ("A" if str(choice).startswith("A") else "B")
            chosen = ab_state.get(selected_key)
            if not chosen:
                return chat_history or [], '<div class="status-box status-error">未找到所选思路的答案</div>', chat_history or []
            history = chat_history[:] if chat_history else []
            # 替换最后一条助手机器人消息
            replaced = False
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "assistant":
                    history[i]["content"] = chosen
                    replaced = True
                    break
            if not replaced:
                history.append({"role": "assistant", "content": chosen})
            return history, f'<div class="status-box status-success">已采用方案{selected_key}</div>', history

        apply_ab_btn.click(
            fn=apply_ab_choice,
            inputs=[ab_choice_radio, ab_answer_state, chat_history_state],
            outputs=[chatbot, ab_status, chat_history_state]
        )
        ab_choice_radio.change(
            fn=lambda v: v,
            inputs=[ab_choice_radio],
            outputs=[ab_choice_state]
        )
        toggle_task_btn.click(
            fn=toggle_task_action,
            inputs=[task_select_dropdown],
            outputs=[tasks_markdown, task_select_dropdown, progress_html]
        )
        delete_task_btn.click(
            fn=delete_task_action,
            inputs=[task_select_dropdown],
            outputs=[tasks_markdown, task_select_dropdown, progress_html]
        )
        
        # 计时器相关按钮
        start_timer_btn.click(
            fn=start_timer_action,
            inputs=[],
            outputs=[timer_display, start_timer_btn, pause_timer_btn]
        )
        pause_timer_btn.click(
            fn=pause_timer_action,
            inputs=[],
            outputs=[timer_display, start_timer_btn, pause_timer_btn]
        )
        clear_today_btn.click(
            fn=clear_today_action,
            inputs=[],
            outputs=[tasks_markdown, task_select_dropdown, progress_html]
        )
        
        # 退出程序按钮
        exit_btn.click(
            fn=exit_program,
            inputs=[],
            outputs=[]
        )
        
        # 自动保存按钮（隐藏，由JavaScript触发）
        auto_save_trigger_btn.click(
            fn=auto_save_session,
            inputs=[],
            outputs=[]
        )
        
        # 刷新按钮：更新看板数据
        def refresh_all():
            board_state = _build_board_ui_state()
            return board_state[0], board_state[1], board_state[2]
        
        refresh_board_btn.click(
            fn=refresh_all,
            inputs=[],
            outputs=[tasks_markdown, task_select_dropdown, progress_html]
        )

        # ==================== 笔记助手回调函数 ====================
        
        def refresh_notes_list(kb_name):
            """刷新笔记列表"""
            if not kb_name:
                return gr.update(choices=[], value=None), "请先选择知识库", "", ""
            
            notes = load_notes(kb_name)
            if not notes:
                return gr.update(choices=[], value=None), "当前知识库暂无笔记", "", ""
            
            # 构建下拉选项：显示标题，值为ID
            choices = [(f"{note.get('title', '无标题')} ({note.get('last_modified', '未知时间')})", note['id']) 
                      for note in notes]
            return gr.update(choices=choices, value=None), f"已加载 {len(notes)} 条笔记", "", ""
        
        def load_selected_note(kb_name, note_id):
            """加载选中的笔记到编辑器"""
            if not kb_name or not note_id:
                return "", "", "", note_id, "请选择笔记", ""
            
            note = get_note_by_id(kb_name, note_id)
            if not note:
                return "", "", "", None, "❌ 笔记不存在", ""
            
            title = note.get('title', '')
            content = note.get('content', '')
            tags = ', '.join(note.get('tags', []))
            status = f"✅ 已加载笔记：{title}"
            
            # 同时返回渲染后的 Markdown 预览内容（保留原始 $...$）
            rendered = content or ""
            return title, content, tags, note_id, status, rendered
        
        def save_current_note(kb_name, note_id, title, content, tags):
            """保存当前笔记"""
            if not kb_name:
                return "❌ 请先选择知识库", note_id, gr.update()
            
            if not title.strip():
                return "❌ 笔记标题不能为空", note_id, gr.update()
            
            # 构建笔记对象
            note = {
                'title': title.strip(),
                'content': content.strip(),
                'tags': [tag.strip() for tag in tags.split(',') if tag.strip()],
            }
            
            # 如果有ID，说明是更新操作
            if note_id:
                note['id'] = note_id
            
            # 保存笔记
            result = save_note_to_kb(kb_name, note)
            
            # 刷新笔记列表
            notes = load_notes(kb_name)
            choices = [(f"{n.get('title', '无标题')} ({n.get('last_modified', '未知时间')})", n['id']) 
                      for n in notes]
            
            # 如果是新建笔记，获取新生成的ID
            if not note_id and notes:
                latest_note = max(notes, key=lambda x: x.get('last_modified', ''))
                note_id = latest_note['id']
            
            return f"✅ {result}", note_id, gr.update(choices=choices)
        
        def delete_current_note(kb_name, note_id):
            """删除当前笔记"""
            if not kb_name:
                return "❌ 请先选择知识库", None, "", "", "", gr.update()
            
            if not note_id:
                return "❌ 请先选择要删除的笔记", None, "", "", "", gr.update()
            
            result = delete_note_from_kb(kb_name, note_id)
            
            # 刷新笔记列表
            notes = load_notes(kb_name)
            choices = [(f"{n.get('title', '无标题')} ({n.get('last_modified', '未知时间')})", n['id']) 
                      for n in notes]
            
            return f"✅ {result}", None, "", "", "", gr.update(choices=choices, value=None)
        
        def clear_note_editor():
            """清空编辑器"""
            return "", "", "", None, "已清空编辑器", ""
        
        def new_note_action(kb_name=None):
            """新建笔记：仅清空编辑器，等待用户填写并点击保存（不自动保存）。"""
            # 不进行任何文件写入，仅重置编辑器状态
            return "", "", "", None, "✏️ 请输入新笔记内容", ""
        
        def refresh_notes_action(kb_name):
            """刷新笔记列表并重新加载笔记内容"""
            if not kb_name:
                return gr.update(choices=[], value=None), "请先选择知识库"
            
            notes = load_notes(kb_name)
            if not notes:
                return gr.update(choices=[], value=None), "当前知识库暂无笔记"
            
            choices = [(f"{note.get('title', '无标题')} ({note.get('last_modified', '未知时间')})", note['id']) 
                      for note in notes]
            return gr.update(choices=choices, value=None), f"✅ 已刷新，共 {len(notes)} 条笔记"
        
        # 文件预览相关函数
        def load_kb_files(kb_name):
            """加载知识库文件列表"""
            if not kb_name:
                return gr.update(choices=[], value=None)
            
            files = get_kb_files(kb_name)
            return gr.update(choices=files, value=None)
        
        def preview_file(kb_name, filename):
            """预览选中的文件内容"""
            if not kb_name or not filename:
                return "请选择文件", ""
            
            try:
                file_path = os.path.join(KB_BASE_DIR, kb_name, filename)
                
                # 获取文件信息
                file_size = os.path.getsize(file_path)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                
                # 格式化文件大小
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
                
                file_info = f"📄 文件名: {filename}\n📊 大小: {size_str}\n🕒 修改时间: {file_mtime}"
                
                # 检查文件扩展名
                text_extensions = ['.txt', '.md', '.py', '.json', '.xml', '.html', '.css', '.js', '.java', '.cpp', '.c', '.h', '.csv']
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in text_extensions:
                    # 读取文本文件
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # 限制显示长度
                        if len(content) > 50000:
                            content = content[:50000] + "\n\n... (内容过长，已截断)"
                        return content, file_info
                elif file_ext == '.pdf':
                    # 提取PDF文本内容
                    try:
                        pdf_text = extract_text_from_pdf(file_path)
                        if pdf_text and pdf_text.strip():
                            # 限制显示长度
                            if len(pdf_text) > 50000:
                                pdf_text = pdf_text[:50000] + "\n\n... (内容过长，已截断)"
                            return f"📖 PDF文本内容:\n\n{pdf_text}", file_info
                        else:
                            return "❌ PDF文件为空或无法提取文本内容（可能是扫描版PDF）", file_info
                    except Exception as e:
                        return f"❌ PDF文本提取失败: {str(e)}\n\n提示：该PDF可能是扫描版或包含特殊格式", file_info
                else:
                    return f"不支持预览该文件类型 ({file_ext})，仅支持文本文件和PDF格式。", file_info
                    
            except Exception as e:
                return f"❌ 读取文件失败: {str(e)}", ""
        
        def open_file_with_system(kb_name, filename):
            """用系统默认程序打开文件（特别适用于PDF）"""
            if not kb_name or not filename:
                return "请先选择文件"
            
            try:
                import subprocess
                import platform
                
                file_path = os.path.join(KB_BASE_DIR, kb_name, filename)
                
                if not os.path.exists(file_path):
                    return f"❌ 文件不存在: {filename}"
                
                # 根据操作系统使用不同的打开命令
                system = platform.system()
                
                if system == 'Windows':
                    # Windows: 使用 os.startfile
                    os.startfile(file_path)
                elif system == 'Darwin':  # macOS
                    subprocess.run(['open', file_path], check=True)
                else:  # Linux
                    subprocess.run(['xdg-open', file_path], check=True)
                
                return f"✅ 已用系统默认程序打开: {filename}"
                
            except Exception as e:
                return f"❌ 打开文件失败: {str(e)}"
        
        
        # 这部分事件绑定已在上面定义，此处为重复代码，可删除或注释

        # 笔记助手事件绑定
        kb_dropdown_notes.change(
            fn=refresh_notes_list,
            inputs=[kb_dropdown_notes],
            outputs=[notes_dropdown, notes_status, note_content_input, note_preview_md]
        )
        
        notes_dropdown.change(
            fn=load_selected_note,
            inputs=[kb_dropdown_notes, notes_dropdown],
            outputs=[note_title_input, note_content_input, note_tags_input, current_note_id, notes_status, note_preview_md]
        )
        
        save_note_btn.click(
            fn=save_current_note,
            inputs=[kb_dropdown_notes, current_note_id, note_title_input, note_content_input, note_tags_input],
            outputs=[notes_status, current_note_id, notes_dropdown]
        )
        
        delete_note_btn.click(
            fn=delete_current_note,
            inputs=[kb_dropdown_notes, current_note_id],
            outputs=[notes_status, current_note_id, note_title_input, note_content_input, note_tags_input, notes_dropdown]
        )
        
        new_note_btn.click(
            fn=new_note_action,
            inputs=[kb_dropdown_notes],
            outputs=[note_title_input, note_content_input, note_tags_input, current_note_id, notes_status, note_preview_md]
        )
        
        refresh_notes_btn.click(
            fn=refresh_notes_action,
            inputs=[kb_dropdown_notes],
            outputs=[notes_dropdown, notes_status]
        )
        
        clear_note_btn.click(
            fn=clear_note_editor,
            inputs=[],
            outputs=[note_title_input, note_content_input, note_tags_input, current_note_id, notes_status, note_preview_md]
        )
        
        # 文件预览事件绑定
        kb_dropdown_notes.change(
            fn=load_kb_files,
            inputs=[kb_dropdown_notes],
            outputs=[kb_files_dropdown]
        )
        
        kb_files_dropdown.change(
            fn=preview_file,
            inputs=[kb_dropdown_notes, kb_files_dropdown],
            outputs=[file_preview_text, file_info_text]
        )

        # Markdown 渲染与实时预览支持
        def render_markdown(content):
            if not content:
                return ""
            return content

        def live_preview(content, enabled):
            if enabled:
                return content or ""
            return ""

        preview_md_btn.click(
            fn=render_markdown,
            inputs=[note_content_input],
            outputs=[note_preview_md]
        )

        note_content_input.change(
            fn=live_preview,
            inputs=[note_content_input, live_preview_checkbox],
            outputs=[note_preview_md]
        )
        
        refresh_files_btn.click(
            fn=load_kb_files,
            inputs=[kb_dropdown_notes],
            outputs=[kb_files_dropdown]
        )
        
        open_file_btn.click(
            fn=open_file_with_system,
            inputs=[kb_dropdown_notes, kb_files_dropdown],
            outputs=[file_info_text]
        )

        # 创建知识库按钮功能
        create_kb_btn.click(
            fn=create_kb_and_refresh,
            inputs=[new_kb_name],
            outputs=[kb_status, kb_dropdown, kb_dropdown_chat, kb_dropdown_notes]
        ).then(
            fn=lambda: "",  # 清空输入框
            inputs=[],
            outputs=[new_kb_name]
        )
        
        # 刷新知识库列表按钮功能
        refresh_kb_btn.click(
            fn=refresh_kb_list,
            inputs=[],
            outputs=[kb_dropdown, kb_dropdown_chat, kb_dropdown_notes]
        )
        
        # 删除知识库按钮功能
        delete_kb_btn.click(
            fn=delete_kb_and_refresh,
            inputs=[kb_dropdown],
            outputs=[kb_status, kb_dropdown, kb_dropdown_chat, kb_dropdown_notes]
        ).then(
            fn=update_kb_files_list,
            inputs=[kb_dropdown],
            outputs=[kb_files_list]
        )
        
        # 知识库选择变化时 - 管理界面
        kb_dropdown.change(
            fn=on_kb_change,
            inputs=[kb_dropdown],
            outputs=[kb_status, kb_files_list]
        ).then(
            fn=sync_kb_to_chat,
            inputs=[kb_dropdown],
            outputs=[kb_dropdown_chat, kb_dropdown_notes]
        )
        
        # 知识库选择变化时 - 对话界面
        kb_dropdown_chat.change(
            fn=sync_chat_to_kb,
            inputs=[kb_dropdown_chat],
            outputs=[kb_dropdown, kb_dropdown_notes, kb_files_list]
        )
        
        # 处理文件上传
        file_upload.upload(
            fn=process_upload_to_kb,
            inputs=[file_upload, kb_dropdown],
            outputs=[upload_status, kb_files_list]
        )
        
        # 为整个知识库构建知识图谱
        def run_build_kg_all(kb_name):
            if not kb_name:
                return "请先选择一个知识库"
            success, message = build_kg_for_entire_kb(kb_name)
            return message
        
        build_kg_all_btn.click(
            fn=run_build_kg_all,
            inputs=[kb_dropdown],
            outputs=[kg_build_all_status]
        )
        
        # 清空输入按钮功能
        clear_btn.click(
            fn=lambda: "",
            inputs=[],
            outputs=[question_input]
        )
        
        # 清空对话历史按钮功能
        def clear_history():
            return (
                [],
                [],
                "等待生成...",
                "等待生成...",
                gr.update(value="A"),
                "",
                {},
                "A"
            )

        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, chat_history_state, answer_a_md, answer_b_md, ab_choice_radio, ab_status, ab_answer_state, ab_choice_state]
        )
        
        # 提交按钮 - 开始流式处理
        def update_status(is_processing=True, is_error=False):
            if is_processing:
                return '<div class="status-box status-processing">正在处理您的问题...</div>'
            elif is_error:
                return '<div class="status-box status-error">处理过程中出现错误</div>'
            else:
                return '<div class="status-box status-success">回答已生成完毕</div>'
        
        # 处理问题并更新对话历史(连接前端与 RAG 流水线)
        def process_and_update_chat(question, kb_name, use_search, use_table_format, multi_hop, chat_history):
            if not question.strip():
                return chat_history, update_status(False, True), "等待提交问题..."
            
            try:
                # 首先更新聊天界面，显示用户问题
                chat_history.append([question, "正在思考..."])
                yield chat_history, update_status(True), f"开始处理您的问题，使用知识库: {kb_name}..."
                
                # 用于累积检索状态和答案
                last_search_display = ""
                last_answer = ""
                
                # 使用生成器进行流式处理(调用流式处理函数)
                for search_display, answer in process_question_with_reasoning(question, kb_name, use_search, use_table_format, multi_hop, chat_history[:-1]):
                    # 更新检索状态和答案
                    last_search_display = search_display
                    last_answer = answer
                    
                    # 更新聊天历史中的最后一条（当前的回答）
                    if chat_history:
                        chat_history[-1][1] = answer
                        yield chat_history, update_status(True), search_display
                
                # 处理完成，更新状态
                yield chat_history, update_status(False), last_search_display
                
            except Exception as e:
                # 发生错误时更新状态和聊天历史
                error_msg = f"处理问题时出错: {str(e)}"
                if chat_history:
                    chat_history[-1][1] = error_msg
                yield chat_history, update_status(False, True), f"### 错误\n{error_msg}"
        
        # 连接提交按钮
        def _append_message(history, role, content):
            history = history[:] if history else []
            history.append({"role": role, "content": content})
            return history

        def _replace_last_assistant(history, content):
            history = history[:] if history else []
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "assistant":
                    history[i]["content"] = content
                    return history
            history.append({"role": "assistant", "content": content})
            return history

        def submit_with_optional_kg(question, kb_name, use_search, use_table, use_multi_hop, use_kg, enable_ab, chat_history, ab_choice_selected=None):
            """
            智能问答主函数，支持：
            - 联网搜索 (use_search)
            - 多跳推理 (use_multi_hop)
            - 知识图谱增强 (use_kg)
            - A/B方案生成 (enable_ab)
            
            无论启用哪些功能，都确保返回详细的知识来源信息 (source_info)
            """
            try:
                if not question.strip():
                    return (
                        chat_history or [],
                        '<div class="status-box status-error">请输入问题</div>',
                        "",
                        "未生成（请输入问题）",
                        "未生成（请输入问题）",
                        "",
                        chat_history or [],
                        {}
                    )

                chat_history_safe = chat_history[:] if chat_history else []
                chat_history_safe = _append_message(chat_history_safe, "user", question)

                base_answer = ""
                source_info = ""
                ab_answers = {}

                if use_kg:
                    # ====== 知识图谱增强分支 ======
                    # query_with_kg_enhancement 返回 (answer, source_info)
                    # source_info 包含：学科分类、KG实体、KG关系、文档片段、联网搜索等
                    base_answer, source_info = query_with_kg_enhancement(
                        question,
                        kb_name=kb_name,
                        use_search=use_search,
                        use_kg=True,
                        use_table_format=use_table
                    )
                else:
                    # ====== 非KG分支：使用流式处理 ======
                    # process_question_with_reasoning 是生成器，yield (display, answer)
                    # display 包含联网搜索结果、检索状态、检索到的文档分块等
                    try:
                        gen = process_question_with_reasoning(question, kb_name, use_search, use_table, use_multi_hop, chat_history)
                        last_display = ""
                        last_answer = ""
                        for disp, ans in gen:
                            # 迭代到结束，保留最后一次返回的展示与答案
                            if disp:
                                last_display = disp
                            if ans:
                                last_answer = ans

                        base_answer = last_answer or ""
                        source_info = last_display or ""

                        # 如果流式接口返回的 source_info 为空或过于简短，尝试补充
                        if not source_info or len(source_info) < 50:
                            # 对于多跳模式，尝试使用非流式接口获取详细信息
                            if use_multi_hop:
                                try:
                                    _ans, debug = multi_hop_generate_answer(question, kb_name, use_table_format=use_table)
                                    chunks = debug.get('all_chunks', []) if isinstance(debug, dict) else []
                                    if chunks:
                                        chunks_preview = "\n\n".join([f"**检索块 {i+1}**:\n{c.get('chunk','')[:600]}" for i, c in enumerate(chunks[:5])])
                                        if len(chunks) > 5:
                                            chunks_preview += f"\n\n...以及另外 {len(chunks)-5} 个块（总计 {len(chunks)} 个）"
                                        source_info = f"### 📚 检索到的文档分块\n**知识库**: {kb_name}\n**模式**: 多跳推理\n\n{chunks_preview}"
                                    else:
                                        source_info = f"### 📚 检索信息\n**知识库**: {kb_name}\n**模式**: 多跳推理\n\n未检索到相关文档分块"
                                except Exception as e:
                                    source_info = f"### 📚 检索信息\n**知识库**: {kb_name}\n**模式**: 多跳推理\n\n获取详细信息失败: {str(e)}"
                            else:
                                source_info = f"### 📚 检索信息\n**知识库**: {kb_name}\n**模式**: 简单向量检索\n\n流式处理未返回详细检索信息"

                    except Exception as e:
                        # 回退到原有同步接口以保证健壮性
                        print(f"流式处理失败，回退到同步接口: {e}")
                        try:
                            if enable_ab:
                                base_answer, ab_answers = ask_question_with_ab(
                                    question, kb_name, use_search, use_table, multi_hop=use_multi_hop
                                )
                            else:
                                base_answer = ask_question_parallel(
                                    question, kb_name, use_search, use_table, multi_hop=use_multi_hop
                                )
                        except Exception as inner_e:
                            base_answer = f"查询失败：{str(inner_e)}"
                        source_info = f"### 📚 检索信息\n**知识库**: {kb_name}\n**联网搜索**: {'启用' if use_search else '关闭'}\n**多跳推理**: {'启用' if use_multi_hop else '关闭'}\n\n（同步接口未返回详细检索信息）"

                # ====== 生成 A/B 方案 ======
                if enable_ab and not ab_answers:
                    try:
                        ab_answers = generate_ab_responses(question, base_answer, use_table_format=use_table)
                    except Exception as e:
                        ab_answers = {"A": base_answer, "B": f"(生成B方案失败：{e})\n{base_answer}"}

                answer_a = ab_answers.get("A", base_answer if enable_ab else "未启用A/B回答")
                answer_b = ab_answers.get("B", base_answer if enable_ab else "未启用A/B回答")

                # ====== 构建最终消息 ======
                if enable_ab and ab_answers:
                    if ab_choice_selected in ("A", "B"):
                        selected_key = ab_choice_selected
                    else:
                        selected_key = "A" if (not ab_choice_selected or str(ab_choice_selected).startswith("A")) else "B"
                    chosen_answer = ab_answers.get(selected_key, base_answer)
                    combined_msg = f"【已选方案{selected_key}】\n{chosen_answer}"
                    ab_status_html = f'<div class="status-box status-success">已生成 A/B 两种思路，当前采用方案{selected_key}</div>'
                elif enable_ab:
                    combined_msg = base_answer
                    ab_status_html = '<div class="status-box status-error">A/B 生成失败，已回退为单答案</div>'
                else:
                    combined_msg = base_answer
                    ab_status_html = '<div class="status-box status-processing">A/B 功能已关闭</div>'

                chat_history_safe = _append_message(chat_history_safe, "assistant", combined_msg)

                # 构建状态提示
                mode_info = []
                if use_kg:
                    mode_info.append("KG增强")
                if use_multi_hop:
                    mode_info.append("多跳推理")
                if use_search:
                    mode_info.append("联网搜索")
                mode_str = "、".join(mode_info) if mode_info else "基础检索"
                status = f'<div class="status-box status-success">回答已生成（{mode_str}）</div>'

                return (
                    chat_history_safe,
                    status,
                    source_info,
                    answer_a if enable_ab else "（已关闭A/B生成）",
                    answer_b if enable_ab else "（已关闭A/B生成）",
                    ab_status_html,
                    chat_history_safe,
                    ab_answers
                )
            except Exception as e:
                err = f"查询失败：{str(e)}"
                chat_history_safe = _append_message(chat_history or [], "user", question)
                chat_history_safe = _append_message(chat_history_safe, "assistant", err)
                return (
                    chat_history_safe,
                    '<div class="status-box status-error">查询失败</div>',
                    f"### 错误\n{err}",
                    "生成失败",
                    "生成失败",
                    '<div class="status-box status-error">A/B 生成失败</div>',
                    chat_history_safe,
                    {}
                )

        submit_btn.click(
            fn=submit_with_optional_kg,
            inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle, kg_toggle, ab_toggle, chat_history_state, ab_choice_state],
            outputs=[chatbot, status_box, search_results_output, answer_a_md, answer_b_md, ab_status, chat_history_state, ab_answer_state],
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[question_input]
        )
        
        # 支持Enter键提交
        question_input.submit(
            fn=submit_with_optional_kg,
            inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle, kg_toggle, ab_toggle, chat_history_state, ab_choice_state],
            outputs=[chatbot, status_box, search_results_output, answer_a_md, answer_b_md, ab_status, chat_history_state, ab_answer_state]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[question_input]
        )
        
        # 页面加载时自动启动学习计时器
        demo.load(
            fn=auto_start_timer,
            inputs=[],
            outputs=[timer_display]
        )
        
        # 添加定时器：降低刷新频率到1秒，减少无关DOM更新对滚动的影响
        timer_updater = gr.Timer(value=1.0, active=True)
        timer_updater.tick(
            fn=update_timer_display,
            inputs=[],
            outputs=[timer_display]
        )

    return demo

def launch_ui():
    """启动Gradio UI"""
    import atexit
    
    # 注册程序退出时的保存钩子
    def save_on_exit():
        """程序退出时保存学习时长"""
        try:
            current_elapsed = _get_current_session_elapsed()
            minutes = int(current_elapsed / 60) if current_elapsed > 0 else 0
            
            if minutes >= 1:
                board = load_learning_board()
                recs = board.get('study_records', [])
                today_str = date.today().isoformat()
                
                # 检查今天是否已有记录，如果有则累加，没有则新增
                existing_index = next((i for i, r in enumerate(recs) if r.get('date') == today_str), None)
                if existing_index is not None:
                    # 今天已有记录，累加分钟数
                    recs[existing_index]['minutes'] += minutes
                    print(f"[退出保存] 已累加 {minutes} 分钟（今日总计: {recs[existing_index]['minutes']} 分钟）")
                else:
                    # 今天没有记录，新增一条
                    recs.append({'date': today_str, 'minutes': minutes})
                    print(f"[退出保存] 已保存 {minutes} 分钟学习时长")
                
                board['study_records'] = recs
                save_learning_board(board)
            else:
                print(f"[退出保存] 学习时长不足1分钟，未保存")
        except Exception as e:
            print(f"[退出保存] 失败: {e}")
    
    atexit.register(save_on_exit)
    
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    launch_ui()
