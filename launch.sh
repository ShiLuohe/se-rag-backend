python run_rag.py \
    --db ./db \
    --host 0.0.0.0 \
    --port 8000 \
    --rag_entry 30 \
    --llm_api_base "https://api.deepseek.com/v1" \
    --llm_api_key "sk-762cb9c74e154a71b083185e5f0e766e" \
    --llm_model "deepseek-chat" \
    # --llm_model deepseek-reasoner \
