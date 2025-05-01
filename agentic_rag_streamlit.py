# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_nomic import NomicEmbeddings
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = NomicEmbeddings(model="nomic-embed-text-v1")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
# initiating llm
llm = ChatTogether(temperature=0, model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', max_tokens=200)

# pulling prompt from hub
# prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Custom System Message
system_message = SystemMessage(content="""
You are TCE's official chatbot, providing reliable and verified information only about Thiagarajar College of Engineering.
If a question is outside your knowledge, say 'I don’t know' instead of guessing.
If relevant, use the 'retrieve' function to fetch factual data from the college's database.
""")

# Creating a ChatPromptTemplate with agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),  # Maintains conversation history
    ("user", "{input}"),  # User input
    MessagesPlaceholder(variable_name="agent_scratchpad")  # Required for tool use
])



# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="TCE Chatbot", page_icon="🤖")


# College logo file path (local) or URL
logo_path = "tcelogo.png"  # If it's a local file

# Create layout for logo and title
col1, col2 = st.columns([0.2, 0.8])  # Adjust proportions if needed

with col1:
    st.image(logo_path, use_container_width=True)  # New recommended parameter

with col2:
    st.markdown("<h1 style='margin-top: 10px;'>TCE Chatbot</h1>", unsafe_allow_html=True)  # Proper alignment




# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("What do you want to know about TCE?")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

