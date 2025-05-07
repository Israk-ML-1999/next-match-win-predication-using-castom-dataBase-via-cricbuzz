import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")


def ask_llm(query):
    if not openai.api_key:
        raise ValueError("GROQ_API_KEY is missing in the environment variables.")
    
    try:
        response = openai.ChatCompletion.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "You are a cricket expert AI that predicts match winners. You analyze team performance, player statistics, pitch conditions, and recent match history to provide accurate predictions."},
                {"role": "user", "content": query},
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error querying LLM: {e}"