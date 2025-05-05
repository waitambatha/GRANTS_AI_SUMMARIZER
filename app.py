import weaviate.classes as wvc
from weaviate.classes.query import Filter # Import Filter
from weaviate.classes.query import Sort
import streamlit as st
import weaviate
import re # Import regular expressions for query parsing
from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from schema import GRANT_CLASS_NAME
from weaviate.auth import AuthApiKey


# --- Use st.secrets for environment variables ---
WEAVIATE_URL = st.secrets["WEAVIATE_REST_URL"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Helper Function to Parse Query ---
def parse_structured_query(query: str):
    """
    Parses the query to find requests for top/lowest N grants by amount in a region[cite: 2].
    Returns a dictionary with parsed parameters or None if not a match[cite: 3].
    """
    # Regex to find patterns like "top 5 grants in Asia by amount", "lowest 3 grants in Africa", etc.
    # Making "grants", "by amount", "committed amount" optional and flexible
    pattern = re.compile(
        r"(top|lowest|highest|bottom)\s+(\d+)\s+(?:grants?\s+)?(?:in|for)\s+([\w\s]+?)\s*(?:by\s+amount|by\s+committed\s+amount)?$",
        re.IGNORECASE
    )
    match = pattern.search(query)

    if match:
        direction = match.group(1).lower()
        limit = int(match.group(2))
        region = match.group(3).strip() [cite: 5]

        # *** CORRECTED LINE BELOW ***
        sort_order = wvc.config.SortOrder.DESC if direction in ["top", "highest"] else wvc.config.SortOrder.ASC

        return {
            "limit": limit,
            "region": region,
            "sort_by": "amountCommitted",
            "sort_order": sort_order,
            "type": "structured_filter_sort"
        } [cite: 5] # Note: The original source [cite: 6] ended here, but logically the dict belongs with source [cite: 5]

    # Regex for simpler "top N grants" or "lowest N grants" (without region)
    pattern_simple = re.compile(
        r"(top|lowest|highest|bottom)\s+(\d+)\s+grants?.*(?:by\s+amount|by\s+committed\s+amount)?$",
        re.IGNORECASE
    )
    match_simple = pattern_simple.search(query)
    if match_simple:
        direction = match_simple.group(1).lower()
        limit = int(match_simple.group(2))
        # *** CORRECTED LINE BELOW ***
        sort_order = wvc.config.SortOrder.DESC if direction in ["top", "highest"] else wvc.config.SortOrder.ASC
        return {
            "limit": limit,
            "region": None, # No region specified
            "sort_by": "amountCommitted",
            "sort_order": sort_order,
            "type": "structured_sort_only"
        } [cite: 7] # Note: The original source [cite: 6] ended before this return block

    # Regex for just specifying number, e.g., "show me 15 grants" (uses semantic search but adjusts k)
    pattern_k = re.compile(r"(?:show|list|find|get|display)\s+(\d+)\s+grants?", re.IGNORECASE) [cite: 8]
    match_k = pattern_k.search(query)
    if match_k:
        limit = int(match_k.group(1))
        # Clamp limit to a reasonable max if needed, e.g., 50
        limit = min(limit, 50)
        return {
            "limit": limit,
            "type": "semantic_search_k"
        }

    return None

# --- Weaviate Connection and Setup ---
@st.cache_resource # Cache the connection, client, and chain setup
def setup_weaviate_and_rag():
    """Initializes the Weaviate client, LangChain components, and the RAG chain."""
    try:
        # Connect using the helper function
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )

        if not client.is_ready():
            st.error("Failed to connect to Weaviate Cloud. Check connection details and API keys.")
            client.close()
            return None, None # Return None for both client and chain

        # Setup LangChain components
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=GRANT_CLASS_NAME,
            text_key="combinedText", # Ensure this matches your schema's vectorizable property
            embedding=embeddings,
            attributes=["purpose", "division", "regionServed", "amountCommitted", "grantId"] # Attributes to retrieve
        )
        # Default retriever, 'k' might be adjusted later if needed
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        st.success("Weaviate client and RAG chain setup successful!")
        # Return both the client (for direct queries) and the QA chain
        return client, qa_chain

    except Exception as e:
        st.error(f"Error setting up Weaviate/RAG chain: {e}")
        st.error("Check Weaviate connection, API keys, collection name/schema, and OpenAI setup.")
        if 'client' in locals() and client and client.is_connected():
            client.close()
        return None, None # Return None for both
# --- Execute Custom Weaviate Query ---
# --- Execute Custom Weaviate Query ---
def execute_custom_query(client: weaviate.WeaviateClient, params: dict):
    """ Executes a direct Weaviate query based on parsed parameters """
    try:
        grants = client.collections.get(GRANT_CLASS_NAME)
        filters = None
        if params.get("region"):
            # Use Weaviate's filtering capabilities
            filters = Filter.by_property("regionServed").equal(params["region"]) # Make sure Filter is imported

        response = grants.query.fetch_objects(
            limit=params["limit"],
            filters=filters,  # Removed the [cite: 16] marker
            sort=Sort.by_property(name=params["sort_by"], order=params["sort_order"]) # Make sure Sort is imported and order uses the SortOrder enum correctly
        )
        return response.objects # Return the list of Weaviate objects

    except Exception as e:
        st.error(f"Error executing custom Weaviate query: {e}")
        return []

