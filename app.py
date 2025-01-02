import streamlit as st
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

# Set page config
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Add title and description
st.title("I2E Chatbot")
st.markdown("Type your question about i2e Consulting...")

# Initialize session state for storing the RAG chain
if 'rag_chain' not in st.session_state:
    # Initialize the RAG system
    @st.cache_resource
    def initialize_rag_system():
        # Initialize embeddings and load existing vector store
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        db = FAISS.load_local("bge_small_faiss", embeddings, allow_dangerous_deserialization=True)
        
        # Initialize LLM
        llm = ChatGroq(
            temperature=0,
            #groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_api_key= "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",
            model_name="mixtral-8x7b-32768"
        )
        
        # Setup retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Setup prompt template
        prompt_template = """
        Using the information contained in the context, give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.
                
        Question: {question}
        Context: {context}
        
        Answer: """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )
        
        # Create LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Create RAG chain
        rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
        
        return rag_chain

    # Initialize the RAG system
    with st.spinner("Initializing the system..."):
        st.session_state.rag_chain = initialize_rag_system()

# Create the Q&A interface
st.subheader("Ask a Question")
question = st.text_input("Enter your question here:", key="question_input")

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Get the answer
                response = st.session_state.rag_chain.invoke(question)
                
                # Display the answer
                st.subheader("Answer")
                st.markdown(response['text'])
                
                # Add an expandable section for debugging info
                with st.expander("Show Debug Info"):
                    st.write("Question:", question)
                    st.write("Raw Response:", response)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")

# Add sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses RAG (Retrieval Augmented Generation) to answer questions about your documents.
    
    Features:
    - Uses pre-computed FAISS embeddings for efficient retrieval
    - Powered by Mixtral-8x7b-32768 via Groq
    - Uses BGE embeddings for document retrieval
    """)