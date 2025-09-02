# turning query into embedding and searching in vector DB (HF-only)

import os
import argparse

from dotenv import load_dotenv

# LangChain (新包路径，避免弃用警告)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# 本地 HuggingFace LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# 读取 .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Please check your .env file")

CHROMA_PATH = "chroma"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 与建库一致
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
""".strip()

def build_args():
    p = argparse.ArgumentParser(description="RAG query (HF-only)")
    p.add_argument("query_text", type=str, help="The query text.")
    p.add_argument("--persist-dir", default=CHROMA_PATH, help="Chroma persist directory.")
    p.add_argument("--embed-model", default=EMBED_MODEL, help="HF embedding model (must match your index).")
    p.add_argument("--llm-model", default=DEFAULT_LLM, help="HF CausalLM model to use for generation.")
    p.add_argument("--k", type=int, default=3, help="Top-K retrieved documents.")
    p.add_argument("--threshold", type=float, default=0.5, help="Relevance score threshold (0~1).")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate.")
    return p.parse_args()

def load_hf_llm(model_id: str, temperature: float, max_new_tokens: int):
    """加载 HF 模型；把 HF_TOKEN 传入 from_pretrained。"""
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto",
        dtype="auto"
    )
    gen_pipe = pipeline(
        task="text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0)
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def main():
    args = build_args()
    query_text = args.query_text

    # Embedding（务必与建库一致）
    embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)

    # 连接已持久化的 Chroma
    db = Chroma(persist_directory=args.persist_dir, embedding_function=embeddings)

    # 诊断：库里文档数
    try:
        print("Docs in DB:", db._collection.count())
    except Exception:
        print("Docs in DB: (unknown)")

    # 检索
    results = db.similarity_search_with_relevance_scores(query_text, k=args.k)
    if not results:
        print("Unable to find matching results (empty result).")
        return

    for i, (doc, score) in enumerate(results):
        preview = (doc.page_content or "")[:200].replace("\n", " ")
        print(f"{i} score={score:.3f} source={doc.metadata.get('source')} | {preview}\n")

    top_score = results[0][1]
    if top_score < args.threshold:
        print(f"Unable to find matching results. Top score={top_score:.3f} < {args.threshold}")
        return

    # 组织 Prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )
    print("\n[Prompt to HF LLM]\n", prompt, "\n")

    # 生成（完全本地/HF，无需 OpenAI Key）
    llm = load_hf_llm(args.llm_model, args.temperature, args.max_new_tokens)
    resp = llm.invoke(prompt)           # HuggingFacePipeline 返回 str
    response_text = resp if isinstance(resp, str) else str(resp)

    sources = [doc.metadata.get("source") for doc, _ in results]
    print(f"Response:\n{response_text}\n\nSources: {sources}")

if __name__ == "__main__":
    main()
