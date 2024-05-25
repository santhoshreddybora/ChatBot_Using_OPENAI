from dotenv import load_dotenv
import os
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ( SystemMessagePromptTemplate,
                               HumanMessagePromptTemplate,
                               ChatPromptTemplate,
                               MessagesPlaceholder)
from streamlit_chat import message
from utils import *


# Load environment variables
load_dotenv()
st.subheader("Chat bot with OPENAI,Pinecone,Langchain and StreamLit")

if "responses" not in st.session_state:
    st.session_state['responses'] =['How can i assist you!!']

if "requests" not in st.session_state:
    st.session_state['requests'] =[]

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,
        # memory_key="history",
        # max_input_size=3,
        # max_output_size=10,
        return_messages=True
    )


system_msg_template=SystemMessagePromptTemplate.from_template(template="""Answer the question as truthful as 
                                                              possible i need only correct and concise 
                                                              answers if you dont know the answer please 
                                                              say "I don't know" """)

human_message_template=HumanMessagePromptTemplate.from_template(template="{input}")

chat_prompt_template=ChatPromptTemplate.from_messages([system_msg_template,
                                         MessagesPlaceholder(variable_name="history"),
                                         human_message_template])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'))

conversation=ConversationChain(llm=llm,
                               memory=st.session_state.buffer_memory,
                               prompt=chat_prompt_template,
                               verbose=True)


##Setting title and containers for response and text messages
st.title('Langchain chat bot for your data.')
response_container=st.container()
text_container=st.container()

##Will take your query and refine it and match on refined query and gives you response
with text_container:
    query=st.text_input("Query: ",key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string=get_conversation_string()
            refined_query=query_refiner(conversation_string,query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context=find_match(refined_query)
            # st.write("Context:")
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
 

## will check for session state response and return in chat message  
with response_container:
    if st.session_state["responses"]:
        for i in range(0,len(st.session_state["responses"])):
            message(st.session_state["responses"][i],key=str(i))
            if i <len(st.session_state["requests"]):
                message(st.session_state["requests"][i],is_user=True,key=str(i)+'_user')









