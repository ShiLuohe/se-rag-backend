curl \
    -X POST http://localhost:8000/rag \
    -H "Content-Type: application/json" \
    -d '{"userQuestion": "给分好的公共课", "catagory": 3}'