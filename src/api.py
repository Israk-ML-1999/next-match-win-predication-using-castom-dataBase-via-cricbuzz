from fastapi import FastAPI
from pydantic import BaseModel
from vector_search import search_vector_database
from groq_llm import ask_llm

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

@app.post("/predict", response_model=QueryResponse)
def predict_winner(request: QueryRequest):
    query = request.query
    results = search_vector_database(query)

    if results:
        # If results are found in the FAISS database
        return QueryResponse(result=f"Based on the database, the team with higher probability of winning is mentioned above: {results}")
    else:
        # If no results are found, query the LLM
        answer = ask_llm(query)
        return QueryResponse(result=f"Based on the database and LLM model, the next win probability team: {answer.strip()}")

@app.get("/")
def root():
    return {"message": "Welcome to the Cricket Match Prediction API!"}