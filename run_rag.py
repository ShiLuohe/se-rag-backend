import argparse
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Tuple
import uvicorn
import faiss
import numpy as np
import os
import json

from openai import OpenAI

from embedding import get_embedding

# 创建 FastAPI 实例
app = FastAPI()

# 添加 CORS 中间件以允许处理 OPTIONS 请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可根据需要替换为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 定义输入数据模型
class RagRequest(BaseModel):
    userQuestion: str
    catagory: int

# 定义返回数据结构（可根据需要调整）
class RagResponse(BaseModel):
    status: str
    data: Dict[str, Any]

# 全局变量：向量索引与元数据
faiss_index = None
metadata_list = []

# 参数：由主程序设置
llm_api_base = None
llm_api_key = None
llm_model = None

# 加载资料库
def load_rag_database(db_path: str):
    global faiss_index, metadata_list
    index_path = os.path.join(db_path, "faiss.index")
    meta_path = os.path.join(db_path, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("资料库文件不存在")

    faiss_index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

# 搜索函数：根据问题和分类过滤返回 top-k 结果
def RagSearch(user_question: str, catagory: int, top_k: int = 30) -> List[Dict[str, str]]:
    if faiss_index is None:
        raise RuntimeError("向量索引未加载")

    query_vec = get_embedding(user_question).reshape(1, -1)
    distances, indices = faiss_index.search(query_vec, top_k * 5)  # 先多取，后过滤

    results = []
    for idx in indices[0]:
        if idx < len(metadata_list):
            meta = metadata_list[idx]
            if catagory == 0 or meta.get("catagory") == catagory:
                results.append(meta)
                if len(results) >= top_k:
                    break
    return results


SEP_TOKEN = "<|Result|>"
SYSTEM_PROMPT = f'''
你是一个课程选择助手。
在用户的输入部分，你会得到一个json格式的字符串，叫做课程列表，以及一段查询。
json格式的字符串是一个列表，列表中的每个元素是一个字符串，描述了一个课程。
查询就是一个朴素的字符串，是用户的选课需求。
你的任务是：从课程列表中选择一到三门课程作为用户的选课推荐，要求尽可能满足用户的选课需求。
请注意，课程列表可能混入一些随机数据，也可能是一个空列表，没有可用或满足要求课程的情况下你可以不推荐任何课程。
你可以输出任何思考过程，但是最终需要形式化的给出结果。具体地说，你可以先输出任何东西，比如解析用户的需求，\
分析提供的课程列表等。然后你需要输出一个特别标志 {SEP_TOKEN}，在该标志后面是一个json格式的列表。列表中的\
每个元素是一个字典，包含"课程名称"和"理由"两个项目。
具体地说，你的输出应该保持如下格式：

这里是你的分析过程。{SEP_TOKEN}'''.strip() + '''
[{"course": "你推荐课程的名称1", "reason": "你推荐课程的理由1"}, \
{"course": "你推荐课程的名称2", "reason": "你推荐课程的理由2"}, \
{"course": "你推荐课程的名称3", "reason": "你推荐课程的理由3"}]

以下是用户的输入

'''.strip()

def parsePrompt(results: List[Dict[str, str]], user_question: str) -> str:
    stringed_courses = f"[\"{"\", \"".join([result["text"] for result in results])}\"]"
    return f"课程列表: {stringed_courses}. \n用户提问: {user_question}"

def LLMCall(results: List[Dict[str, str]], user_question: str) -> str:
    global llm_model, oai_client
    prompt = parsePrompt(results, user_question)

    try:
        response = oai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR]: {str(e)}"

def LLMParse(llm_response: str) -> Tuple[int, str, List[Dict[str, str]]]:

    splits = [split.strip() for split in llm_response.split(SEP_TOKEN)]
    if len(splits)!= 2: return 0, None, None

    try:
        res_list = json.loads(splits[1])
    except:
        return 1, splits[0], None

    return 3, splits[0], res_list

# POST 接口：/rag
@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(request_data: RagRequest):
    global rag_entry
    try:
        results = RagSearch(request_data.userQuestion, int(request_data.catagory), rag_entry)
        llm_response = LLMCall(results, request_data.userQuestion)
        llm_parse_status, llm_response_text, llm_response_list = LLMParse(llm_response)
        return RagResponse(status="success", data={
            "rag_results": results,
            "llm_output": llm_response,
            "llm_parse_status": llm_parse_status,
            "llm_response_text": llm_response_text,
            "llm_response_list": llm_response_list,
        })
    except Exception as e:
        return RagResponse(status="error", data={"message": str(e)})

@app.options("/rag")
async def options_handler():
    return JSONResponse(content={}, status_code=204)

# 主函数：支持从命令行加载资料库路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG search server.")
    parser.add_argument("--db", type=str, default="./db", help="Path to RAG database")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--rag_entry", type=int, default=30, help="No. of entries RAG module return.")
    parser.add_argument("--llm_api_base", type=str, required=True, help="LLM API base URL")
    parser.add_argument("--llm_api_key", type=str, required=True, help="LLM API Key")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model name")
    args = parser.parse_args()

    rag_entry = args.rag_entry

    llm_model = args.llm_model
    oai_client = OpenAI(api_key=args.llm_api_key, base_url=args.llm_api_base)

    load_rag_database(args.db)
    uvicorn.run(app, host=args.host, port=args.port)

