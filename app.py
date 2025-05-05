import streamlit as st
import weaviate
from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from schema import GRANT_CLASS_NAME
from weaviate.auth import AuthApiKey
import weaviate.classes as wvc

# --- Use st.secrets for environment variables ---
WEAVIATE_URL = st.secrets["WEAVIATE_REST_URL"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Environment variable check ---
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    st.error("Missing one or more required secrets: WEAVIATE_REST_URL, WEAVIATE_API_KEY, or OPENAI_API_KEY.")
    st.stop()
# --- (End env var checks) ---


# --- Weaviate Connection and Retriever Setup ---
@st.cache_resource # Cache the connection and chain setup
def setup_rag_chain():
    """Initializes the Weaviate client, LangChain components, and the RAG chain."""
    try:
        # === CORRECTED CONNECTION ===
        # Connect using the helper function, removing explicit gRPC args
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY} # Pass OpenAI key
        )
        # === END CORRECTED CONNECTION ===

        # Check connection status using client.is_live() or client.is_ready()
        if not client.is_ready(): # is_ready() checks connectivity and schema readiness
            st.error("Failed to connect to Weaviate Cloud. Please check connection details and API keys.")
            client.close() # Close the potentially partially open client
            return None

        # Setup LangChain components
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=GRANT_CLASS_NAME,
            text_key="combinedText",
            embedding=embeddings,
            attributes=["purpose", "division", "regionServed", "amountCommitted", "grantId"]
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        st.success("RAG chain setup successful!")
        # Note: Don't close the client here; LangChain needs it. Streamlit cache handles resource cleanup.
        return qa_chain

    except Exception as e:
        st.error(f"Error setting up RAG chain: {e}")
        st.error("Check Weaviate connection, API keys, and collection name/schema.")
        # Ensure client is closed if setup fails partially
        if 'client' in locals() and client and client.is_connected():
            client.close()
        return None

# --- (Rest of the app.py Streamlit UI code remains the same) ---
# Initialize the RAG chain
qa_chain = setup_rag_chain()

# --- Streamlit App UI ---
st.title("Grant Information Q&A 💰")
st.markdown("Ask questions about grant purposes, funding divisions, regions served, or committed amounts.")

query = st.text_input("Enter your question:", placeholder="e.g., What grants support projects in Africa?")

if st.button("Get Answer"):
    if qa_chain and query:
        with st.spinner("Searching for answers..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.subheader("Answer:")
                st.write(result['result'])

                st.subheader("Sources Used:")
                if 'source_documents' in result and result['source_documents']:
                    for i, doc in enumerate(result['source_documents']):
                        metadata = doc.metadata
                        purpose = metadata.get('purpose', 'N/A')
                        division = metadata.get('division', 'N/A')
                        region = metadata.get('regionServed', 'N/A')
                        amount = metadata.get('amountCommitted', 'N/A')
                        grant_id = metadata.get('grantId', 'N/A')

                        st.markdown(f"**Source {i+1} (Grant ID: {grant_id})**")
                        st.markdown(f"- **Purpose:** {purpose}")
                        st.markdown(f"- **Division:** {division}")
                        st.markdown(f"- **Region Served:** {region}")
                        try:
                            amount_formatted = f"${float(amount):,.2f}"
                        except (ValueError, TypeError):
                            amount_formatted = str(amount)
                        st.markdown(f"- **Amount Committed:** {amount_formatted}")
                        st.divider()
                else:
                    st.warning("No source documents were found for this answer.")

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
                st.error("Please check the logs for more details.")
    elif not query:
        st.warning("Please enter a query.")


