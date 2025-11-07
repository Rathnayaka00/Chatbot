from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route import router

app = FastAPI(
    title="Common Chatbot API",
    description="A unified chatbot API that uses function calling to route queries to either a RAG system or a SQL database.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Common Chatbot API. Use the /chat endpoint to ask questions."}