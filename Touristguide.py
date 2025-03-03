from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import re

def generate_ai_response(prompt_chain, llm_engine):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    response = processing_pipeline.invoke({})
    # Remove <think> tags and their content
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response

def build_prompt_chain(user_query, message_log):
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a seasoned and charismatic tourist guide with a knack for storytelling, bringing history, culture, and local secrets to life. "
        "Your responses should be engaging, entertaining, and packed with fascinating facts, hidden gems, and insider tips. "
        "Add humor, enthusiasm, and a touch of drama to make the experience immersive—like a guide who knows all the best spots, the funniest legends, and the smartest travel hacks. "
        "Only discuss the city and the history of the place asked about. "
        "Avoid generic or robotic responses. Your tone should be warm, enthusiastic, and filled with personality—like a real guide who knows every alley, every secret, and every local legend. "
        "Make travelers feel the pulse of the city, giving them reasons to explore beyond the usual tourist spots! "
        "Avoid giving what you think in the output; only provide information about what is asked."
    )
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def main():
    llm_engine = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.7)
    message_log = [{"role": "ai", "content": "Hello, traveler! 🌍 Where are we exploring today?"}]
    
    while True:
        user_query = input("Enter a city name or ask about a place (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye, traveler! Safe journeys! 🛫")
            break
        
        message_log.append({"role": "user", "content": user_query})
        prompt_chain = build_prompt_chain(user_query, message_log)
        ai_response = generate_ai_response(prompt_chain, llm_engine)
        
        message_log.append({"role": "ai", "content": ai_response})
        print("\nTour Guide: ", ai_response, "\n")

if __name__ == "__main__":
    main()