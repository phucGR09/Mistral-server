from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import threading
# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Thread-safe lock
llm_lock = threading.Lock()

# Khởi tạo mô hình Mistral GGUF
llm = Llama(
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
    n_threads=4, 
    n_gpu_layers=20
)

# Pydantic model để nhận dữ liệu JSON
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

# Endpoint xử lý sinh văn bản
@app.post("/generate")
async def generate(data: GenerateRequest):
    try:
        with llm_lock:
            result = llm(
                data.prompt,
                max_tokens=data.max_tokens,
                stop=["</s>"]
            )
        return {"text": result["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý mô hình: {str(e)}")

# Endpoint kiểm tra API hoạt động
@app.get("/")
def root():
    return {"message": "Mistral LLM API is up and running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
