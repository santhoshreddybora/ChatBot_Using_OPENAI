import streamlit as st
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from langchain_pinecone import Pinecone
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone as PC
load_dotenv()


client=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


model=SentenceTransformer("all-MiniLM-L6-v2")
index = pinecone.Index(index_name='langchain',
                       api_key=os.getenv('PINECONE_API_KEY'),
                       host="https://langchain-0lqasoa.svc.aped-4627-b74a.pinecone.io")

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string



def query_refiner(conversation, query):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
    ],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    # matches = result['matches']
    # contexts = [match['metadata']['text'] for match in matches if 'metadata' in match and 'text' in match['metadata']]
    # return "\n\n".join(contexts)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']