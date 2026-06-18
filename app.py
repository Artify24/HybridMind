import streamlit as st 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever , create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
# from langchain.agents import initialize_agent,AgentType

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

st.title("Welcome to HybrideMind!")

upload_files = st.file_uploader("Choose a pdf file", accept_multiple_files=True)
user_input = st.text_input("Your question:")



defaul_session = "Chat_1"

if 'store' not in st.session_state:
     st.session_state.store = {}
        
def run_rag_chain(user_input,retriever):
    ### Giving better question to llm base on chat history
    contextualize_q_system_prompt=(
            """
                Given a chat history and the latest user question which might reference context in the chat history, 
                formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
                just reformulate it if needed and otherwise return it as is.
            """
        )
    
    contextualize_q__prompt = ChatPromptTemplate.from_messages(
         [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
         ]
    )

    history_aware_retriver = create_history_aware_retriever(llm,retriever,contextualize_q__prompt)

    # Question answer chain 
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriver,question_answer_chain)

    def get_session_history(session_id: str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return  st.session_state.store[session_id] 
    
    convenstional_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    if user_input:
        session_history=get_session_history(defaul_session)
        response = convenstional_rag_chain.invoke(
            {"input": user_input},
                config={
                    "configurable": {"session_id":defaul_session}
                },  
            )
        # st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        # st.write("Chat History:", session_history.messages)

def hybrid_chain(user_input, retriever):

    docs = retriever.invoke(user_input)

    pdf_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    try:
        web_context = search.run(user_input)
    except:
        web_context = ""

    try:
        wiki_context = wiki.run(user_input)
    except:
        wiki_context = ""

    try:
        arxiv_context = arxiv.run(user_input)
    except:
        arxiv_context = ""

    full_context = f"""
    PDF Context:
    {pdf_context}

    Web Search:
    {web_context}

    Wikipedia:
    {wiki_context}

    Arxiv:
    {arxiv_context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are HybridMind.

                Use PDF context as the primary source.

                Use Web Search, Wikipedia and Arxiv
                to supplement missing information.

                If information differs, explain both.

                Context:
                {context}
                """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    chain = prompt | llm

    def get_session_history(
        session_id: str
    ) -> BaseChatMessageHistory:

        if session_id not in st.session_state.store:
            st.session_state.store[
                session_id
            ] = ChatMessageHistory()

        return st.session_state.store[
            session_id
        ]

    conversational_chain = (
        RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    )

    response = conversational_chain.invoke(
        {
            "input": user_input,
            "context": full_context
        },
        config={
            "configurable": {
                "session_id": defaul_session
            }
        }
    )

    st.write("Assistant:", response.content)


## Wrapper and agents
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")

tools=[search,arxiv,wiki]


# Agent Function no agent 
def agents(question):

    search_context = search.run(question)

    try:
        wiki_context = wiki.run(question)
    except:
        wiki_context = ""

    prompt = f"""
    Question:
    {question}

    DuckDuckGo Results:
    {search_context}

    Wikipedia Results:
    {wiki_context}

    Answer the user's question using the information above.
    """

    response = llm.invoke(prompt)

    st.write(response.content)

# intiallizing session state 
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 

# Handling Uploads 
documents=[]
if upload_files and st.session_state.vector_store == None:
    for files in upload_files:
        os.makedirs("temp", exist_ok=True)
        tempFile = f"temp/{files.name}"
        with open(tempFile, 'wb') as f:
            f.write(files.getvalue())
            f_name = files.name

        loader = PyPDFLoader(tempFile)
        docs = loader.load()
        documents.extend(docs)

    
    output_parser = StrOutputParser()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splites = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents=splites,embedding=embedding)
    st.session_state.vector_store = vector_store


# Handling User input 
if user_input and upload_files:
    similarity_score = st.session_state.vector_store.similarity_search_with_score(
        user_input,
        k=3
    ) [0][1]

    retriever = st.session_state.vector_store.as_retriever()


    st.write(
    f"Distance Score: {similarity_score}")

    if similarity_score < 0.4:

        st.success("Mode: RAG")

        run_rag_chain(
            user_input=user_input,
            retriever=retriever
        )

    elif similarity_score < 1.0:

        st.info("Mode: HYBRID")

        hybrid_chain(
            user_input=user_input,
            retriever=retriever
        )

    else:

        st.warning("Mode: SEARCH")

        agents(user_input)
       
elif user_input and not upload_files:
    agents(user_input)