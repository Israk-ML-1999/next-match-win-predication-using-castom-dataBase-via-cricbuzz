from vector_search import search_vector_database
from groq_llm import ask_llm

def decide_cricket_winner(query):
    results = search_vector_database(query)
    if results:
        print("\n🔍 Found in local database:")
        for res in results:
            print(res)
        print("\nBased on the database, the team with higher probability of winning is mentioned above.")
    else:
        print("\n🤖 No match found in the database. Asking LLM...")
        answer = ask_llm(query)
        print(f"\nBased on the database and LLM model, the next win probability team: {answer.strip()}")

if __name__ == "__main__":
    q = input("Enter your query (e.g., 'Who will win next India vs Australia match?'): ").strip()
    if q:
        decide_cricket_winner(q)
    else:
        print("Query cannot be empty. Please try again.")
