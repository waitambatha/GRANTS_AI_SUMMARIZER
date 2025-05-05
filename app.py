import streamlit as st
import weaviate
import re # Import regular expressions for query parsing
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

# --- Helper Function to Parse Query ---
def parse_structured_query(query: str):
    """
    Parses the query to find requests for top/lowest N grants by amount in a region.
    Returns a dictionary with parsed parameters or None if not a match.
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
        region = match.group(3).strip()

        sort_order = weaviate.classes.query.SortOrder.DESC if direction in ["top", "highest"] else weaviate.classes.query.SortOrder.ASC

        return {
            "limit": limit,
            "region": region,
            "sort_by": "amountCommitted",
            "sort_order": sort_order,
            "type": "structured_filter_sort"
        }

    # Regex for simpler "top N grants" or "lowest N grants" (without region)
    pattern_simple = re.compile(
        r"(top|lowest|highest|bottom)\s+(\d+)\s+grants?.*(?:by\s+amount|by\s+committed\s+amount)?$",
        re.IGNORECASE
    )
    match_simple = pattern_simple.search(query)
    if match_simple:
        direction = match_simple.group(1).lower()
        limit = int(match_simple.group(2))
        sort_order = weaviate.classes.query.SortOrder.DESC if direction in ["top", "highest"] else weaviate.classes.query.SortOrder.ASC
        return {
            "limit": limit,
            "region": None, # No region specified
            "sort_by": "amountCommitted",
            "sort_order": sort_order,
            "type": "structured_sort_only"
        }

    # Regex for just specifying number, e.g., "show me 15 grants" (uses semantic search but adjusts k)
    pattern_k = re.compile(r"(?:show|list|find|get|display)\s+(\d+)\s+grants?", re.IGNORECASE)
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
def execute_custom_query(client: weaviate.WeaviateClient, params: dict):
    """ Executes a direct Weaviate query based on parsed parameters """
    try:
        grants = client.collections.get(GRANT_CLASS_NAME)
        filters = None
        if params.get("region"):
            # Use Weaviate's filtering capabilities
            filters = Filter.by_property("regionServed").equal(params["region"])

        response = grants.query.fetch_objects(
            limit=params["limit"],
            filters=filters,
            sort=Sort.by_property(params["sort_by"], params["sort_order"])
        )
        return response.objects # Return the list of Weaviate objects

    except Exception as e:
        st.error(f"Error executing custom Weaviate query: {e}")
        return []

# --- Initialize Client and RAG Chain ---
client, qa_chain = setup_weaviate_and_rag()

# --- Streamlit App UI ---
st.title("Grant Information Q&A 💰")
st.markdown("""
Ask questions about grant purposes, funding divisions, regions served, or committed amounts.
You can also ask for specific rankings like:
- `top 10 grants in Europe by amount`
- `lowest 5 grants for Asia`
- `show me 15 grants` (adjusts number of semantic search results)
""")

query = st.text_input("Enter your question:", placeholder="e.g., What grants support projects in Africa?")

if st.button("Get Answer"):
    if not client or not qa_chain:
        st.error("System not initialized properly. Please check connection details and restart.")
    elif not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                # 1. Try parsing for structured query
                parsed_params = parse_structured_query(query)

                if parsed_params and parsed_params["type"] in ["structured_filter_sort", "structured_sort_only"]:
                    # --- Handle Structured Query (Filter/Sort) ---
                    st.subheader(f"Direct Query Results for: '{query}'")
                    results = execute_custom_query(client, parsed_params)

                    if results:
                        st.markdown(f"Found {len(results)} grants matching your criteria:")
                        for i, obj in enumerate(results):
                            # Access properties directly from the Weaviate object
                            props = obj.properties
                            purpose = props.get('purpose', 'N/A')
                            division = props.get('division', 'N/A')
                            region = props.get('regionServed', 'N/A')
                            amount = props.get('amountCommitted', 'N/A')
                            grant_id = props.get('grantId', 'N/A') # Or use obj.uuid if grantId isn't a property

                            st.markdown(f"**Result {i+1} (Grant ID: {grant_id})**")
                            st.markdown(f"- **Purpose:** {purpose}")
                            st.markdown(f"- **Division:** {division}")
                            st.markdown(f"- **Region Served:** {region}")
                            try:
                                amount_formatted = f"${float(amount):,.2f}"
                            except (ValueError, TypeError):
                                amount_formatted = str(amount) # Keep as string if conversion fails
                            st.markdown(f"- **Amount Committed:** {amount_formatted}")
                            st.divider()
                    else:
                        st.warning("No grants found matching your specific criteria.")

                elif parsed_params and parsed_params["type"] == "semantic_search_k":
                     # --- Handle Semantic Search with adjusted K ---
                     st.subheader(f"Semantic Search Results for: '{query}' (showing up to {parsed_params['limit']} results)")
                     # Temporarily adjust retriever's k value if possible (or re-create retriever - simpler here)
                     temp_retriever = qa_chain.retriever.vectorstore.as_retriever(search_kwargs={'k': parsed_params['limit']})
                     temp_qa_chain = RetrievalQA.from_chain_type(
                         llm=qa_chain.combine_documents_chain.llm_chain.llm, # Reuse LLM
                         chain_type="stuff",
                         retriever=temp_retriever,
                         return_source_documents=True
                     )
                     result = temp_qa_chain.invoke({"query": query}) # Use the original general query text
                     st.subheader("Answer:")
                     st.write(result['result'])
                     st.subheader("Sources Used:")
                     # Display source documents (similar to the original code)
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


                else:
                    # --- Handle General RAG Query ---
                    st.subheader(f"Answer based on semantic search for: '{query}'")
                    # Use the default QA chain (k=5 or original default)
                    result = qa_chain.invoke({"query": query})
                    st.write(result['result'])

                    st.subheader("Sources Used:")
                    # Display source documents (using the same logic as before)
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
                st.error(f"An error occurred during query processing: {e}")
                st.error("Please check the logs or query syntax for more details.")

# Ensure the client is closed gracefully when Streamlit exits (though @st.cache_resource helps)
# This part might be tricky with Streamlit's lifecycle and caching.
# A more robust solution might involve explicit session state management if needed.
