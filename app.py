#
import streamlit as st
import weaviate
import re
import os
import sys
import traceback
from dotenv import load_dotenv

# Import for Filter and SortOrder (v4.x)
import weaviate.classes.query as wvc_query

# Try importing SortOrder for v4.x, fallback to None for robustness
try:
    from weaviate.classes.query import SortOrder
    print("Debug: Successfully imported weaviate.classes.query.SortOrder") # Added debug
except ImportError:
    SortOrder = None  # Fallback will use string representation if needed
    print("Debug: Failed to import weaviate.classes.query.SortOrder. SortOrder set to None.") # Added debug

# --- Add explicit check after try-except ---
if SortOrder:
     print(f"Debug: SortOrder type after import attempt: {type(SortOrder)}")
     # Let's also check its attributes to be sure it's the real enum
     try:
         # Accessing the enum members to confirm it loaded correctly
         print(f"Debug: SortOrder members check: ASC={SortOrder.ASC}, DESC={SortOrder.DESC}")
     except AttributeError:
         print("Debug: WARNING - Imported SortOrder object does not have expected ASC/DESC members. Treating as failed import.")
         SortOrder = None # Force to None if members are missing
else:
     print("Debug: SortOrder is None after import attempt.")


# Using langchain_core.documents.Document is fine, but we primarily work with dicts from Weaviate results now
# from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
# Ensure schema.py exists and defines GRANT_CLASS_NAME
try:
    #
    from schema import GRANT_CLASS_NAME
except ImportError:
    st.error("Could not import GRANT_CLASS_NAME from schema.py. Please ensure schema.py exists in the same directory and defines GRANT_CLASS_NAME.")
    st.stop()

from weaviate.auth import AuthApiKey

import pandas as pd  # Import pandas for data manipulation
import plotly.express as px  # Import plotly for charting


# --- Use st.secrets for environment variables ---
WEAVIATE_URL = st.secrets["WEAVIATE_REST_URL"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Helper Function to Parse Query ---

# --- Environment variable check ---
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    #
    st.error("Missing one or more required environment variables: WEAVIATE_REST_URL, WEAVIATE_API_KEY, or OPENAI_API_KEY. Please check your .env file.")
    st.stop()
# --- (End env var checks) ---

# --- Weaviate Client Version Check ---
st.sidebar.subheader("Weaviate Client Check")
try:
    client_version = weaviate.__version__
    st.sidebar.write(f"Detected Weaviate Client Version: {client_version}")
    # Add a check for V4 specifically
    if not client_version.startswith("4."):
         st.sidebar.warning(f"Warning: Expected Weaviate client v4.x, but found {client_version}. Compatibility issues may arise.")
except AttributeError:
    st.sidebar.error("Could not retrieve Weaviate client version.")
    st.error("Configuration Error: Could not verify Weaviate client setup. Please check installation.")
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred during version check: {e}")
    st.error(f"Configuration Error: An unexpected error occurred during setup check: {e}")
    print(f"Debug: Error during version check: {traceback.format_exc()}")


# --- Helper Function to Parse Query ---
#
# (This function remains the same as the previous version, passing SortOrder or "asc"/"desc")
def parse_structured_query(query: str):
    """
    Parses the query to find requests for top/lowest N grants by amount in a region,
    or simple top/lowest N requests, or requests for a specific number of results.
    Returns a dictionary with parsed parameters or None if not a match.
    """
    pattern_region_sort = re.compile(
        r"(top|lowest|highest|bottom)\s+(\d+)\s+(?:grants?\s+)?(?:in|for)\s+([\w\s]+?)\s*(?:by\s+amount|by\s+committed\s+amount)?$",
        re.IGNORECASE
    )
    match_region_sort = pattern_region_sort.search(query)
    if match_region_sort:
        direction = match_region_sort.group(1).lower()
        limit = int(match_region_sort.group(2))
        region = match_region_sort.group(3).strip()
        sort_order_value = SortOrder.DESC if SortOrder and direction in ["top", "highest"] else SortOrder.ASC if SortOrder else "desc" if direction in ["top", "highest"] else "asc"
        return {"limit": limit, "region": region, "sort_by": "amountCommitted", "sort_order": sort_order_value, "type": "structured_filter_sort"}

    pattern_sort_only = re.compile(
        r"(top|lowest|highest|bottom)\s+(\d+)\s+grants?.*(?:by\s+amount|by\s+committed\s+amount)?$",
        re.IGNORECASE
    )
    match_sort_only = pattern_sort_only.search(query)
    if match_sort_only:
        direction = match_sort_only.group(1).lower()
        limit = int(match_sort_only.group(2))
        sort_order_value = SortOrder.DESC if SortOrder and direction in ["top", "highest"] else SortOrder.ASC if SortOrder else "desc" if direction in ["top", "highest"] else "asc"
        return {"limit": limit, "region": None, "sort_by": "amountCommitted", "sort_order": sort_order_value, "type": "structured_sort_only"}

    pattern_k = re.compile(r"(?:show|list|find|get|display)\s+(\d+)\s+grants?", re.IGNORECASE)
    match_k = pattern_k.search(query)
    if match_k:
        limit = int(match_k.group(1))
        limit = min(limit, 100)
        return {"limit": limit, "type": "semantic_search_k"}

    return {"type": "semantic_search", "limit": 5}


# --- Weaviate Connection and Setup ---
#
# (This function remains the same as the previous version)
@st.cache_resource(ttl="1h")
def setup_weaviate_and_rag():
    """Initializes the Weaviate client, LangChain components, and the RAG chain."""
    client = None
    try:
        st.info("Attempting to connect to Weaviate and setup RAG chain...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        if not client.is_ready():
            st.error("Failed to connect to Weaviate Cloud. Check connection details and API keys.")
            if client and client.is_connected(): client.close()
            return None, None
        st.info("Weaviate connection successful. Setting up LangChain components...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = WeaviateVectorStore(
            client=client, index_name=GRANT_CLASS_NAME, text_key="combinedText",
            embedding=embeddings, attributes=["purpose", "division", "regionServed", "amountCommitted", "grantId"]
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )
        st.success("Weaviate client and RAG chain setup successful!")
        return client, qa_chain
    except Exception as e:
        st.error(f"Error setting up Weaviate/RAG chain: {e}")
        st.error("Check Weaviate connection, API keys, schema, and OpenAI setup.")
        print(f"Debug: Error during setup: {traceback.format_exc()}")
        if client and client.is_connected():
            try: client.close()
            except Exception: pass
        return None, None

# --- Execute Custom Weaviate Query ---
#
def execute_custom_query(client: weaviate.WeaviateClient, params: dict):
    """Executes a direct Weaviate query based on parsed parameters (v4.x compatible)"""
    results = []
    collection_names = []
    # Phase 1: List and Validate Collections
    try:
        if not client or not hasattr(client, 'is_connected') or not client.is_connected():
             st.error("Weaviate client is not connected/invalid before executing custom query.")
             print("Debug: execute_custom_query called with invalid/disconnected client.")
             return []

        all_collections_response = None
        if hasattr(client, 'collections') and client.collections:
             all_collections_response = client.collections.list_all()
             response_type = type(all_collections_response)
             print(f"Debug: Type of client.collections.list_all() response: {response_type}")

             if response_type is dict:
                  print("Debug: Handling dict response from list_all().")
                  collection_names = list(all_collections_response.keys())
                  print(f"Debug: Extracted collection names from dict keys: {collection_names}")
             elif hasattr(all_collections_response, 'collections'):
                  print("Debug: Handling expected object response from list_all().")
                  collection_objects = all_collections_response.collections
                  if isinstance(collection_objects, list) and all(hasattr(c, 'name') for c in collection_objects):
                       collection_names = [c.name for c in collection_objects]
                       print(f"Debug: Extracted collection names from object attribute: {collection_names}")
                  else:
                       st.error("List_all() response has '.collections', but it's not a list of expected objects.")
                       print(f"Debug: Unexpected structure within list_all().collections: {collection_objects}")
             else:
                  st.error("Unexpected response from Weaviate when listing collections (neither dict nor object with '.collections').")
                  print(f"Debug: Unexpected response object from list_all(): {all_collections_response} (Type: {response_type})")
        else:
             st.error("Weaviate client object does not have the expected 'collections' attribute.")
             print("Debug: client object missing 'collections' attribute.")
             return [] # Critical failure

        if not collection_names:
             st.warning("Could not retrieve collection names from Weaviate. Cannot proceed.")
             print(f"Debug: collection_names list is empty after attempting retrieval.")
             return []

        if GRANT_CLASS_NAME not in collection_names:
            st.error(f"Weaviate collection '{GRANT_CLASS_NAME}' not found among available collections: {collection_names}.")
            return []

    except Exception as e:
        st.error(f"Error during collection listing/validation phase: {e}")
        print(f"Debug: Error in execute_custom_query (collection listing/validation): {traceback.format_exc()}")
        return []

    # Phase 2: Build and Execute Query
    try:
        grants = client.collections.get(GRANT_CLASS_NAME)
        filters = None
        if params.get("region"):
            try:
                filters = wvc_query.Filter.by_property("regionServed").equal(params["region"])
            except Exception as e:
                 st.error(f"Error creating filter for regionServed: {e}")
                 print(f"Debug: Error creating filter: {traceback.format_exc()}")
                 return []

        # --- Start Update: Correct Keyword to 'name' ---
        sort_object = None
        if params.get("sort_by") and params.get("sort_order"):
            sort_prop_name = params["sort_by"] # The name of the property to sort by
            raw_sort_order_value = params["sort_order"] # Enum or "asc"/"desc"

            print(f"Debug: Building sort for property '{sort_prop_name}', order value: {raw_sort_order_value} (Type: {type(raw_sort_order_value)})")

            # Preferred: Use SortOrder enum if available and value is the enum type
            if SortOrder and isinstance(raw_sort_order_value, SortOrder):
                print(f"Debug: Using SortOrder enum path.")
                try:
                    # Correct keyword is 'name', value is the property name
                    # Correct keyword is 'order', value is the SortOrder enum member
                    sort_object = wvc_query.Sort.by_property(name=sort_prop_name, order=raw_sort_order_value)
                    print(f"Debug: Created sort object using enum: {sort_object}")
                except Exception as e:
                    st.error(f"Error creating sort object using SortOrder enum: {e}")
                    print(f"Debug: Error during enum sort object creation: {traceback.format_exc()}")
                    return []
            else:
                # Fallback: Convert value ("asc", "desc", etc.) to boolean for 'ascending'
                print(f"Debug: Using boolean 'ascending' path (SortOrder enum not available/used).")
                try:
                     is_ascending = str(raw_sort_order_value).lower() == "asc"
                     print(f"Debug: Determined ascending={is_ascending}")
                     # Correct keyword is 'name', value is the property name
                     # Correct keyword is 'ascending', value is the boolean
                     sort_object = wvc_query.Sort.by_property(name=sort_prop_name, ascending=is_ascending)
                     print(f"Debug: Created sort object using ascending={is_ascending}: {sort_object}")
                except TypeError as te:
                     # This catches if 'ascending' itself is an unexpected keyword
                     st.error(f"TypeError creating sort object using boolean 'ascending': {te}. Client version mismatch likely.")
                     print(f"Debug: TypeError during boolean sort object creation: {traceback.format_exc()}")
                     return []
                except Exception as e:
                     st.error(f"Error creating sort object using boolean 'ascending': {e}")
                     print(f"Debug: Non-TypeError during boolean sort object creation: {traceback.format_exc()}")
                     return []
        # --- End Update ---

        # Execute the query
        print(f"Debug: Executing fetch_objects with limit={params['limit']}, filters={filters}, sort={sort_object}")
        response = grants.query.fetch_objects(
            limit=params["limit"],
            filters=filters,
            sort=sort_object
        )
        print(f"Debug: fetch_objects response received. Number of objects: {len(response.objects)}")
        results = [obj.properties for obj in response.objects]

    except Exception as e:
        st.error(f"Error executing custom Weaviate query phase: {e}")
        st.error("Check collection access, query parameters (sort?), connection, or query structure.")
        print(f"Debug: Error in execute_custom_query (query execution phase): {traceback.format_exc()}")
        if 'sort_object' in locals(): print(f"Debug: Sort object at time of error: {sort_object}")
        return []
    return results


# --- Display Results and Visualizations ---
#
# (This function remains the same as the previous version)
def display_results_and_charts(results, is_source_documents=False):
    """Displays the query results as a list and table, and generates charts if possible"""
    if not results:
        if not is_source_documents: st.warning("No results found for the specific query.")
        return

    if is_source_documents:
        processed_results = [doc.metadata for doc in results if hasattr(doc, 'metadata')]
        if not processed_results:
            st.warning("Source documents found, but could not extract metadata.")
            return
        st.subheader("Sources Used (Metadata):")
    else:
        processed_results = results
        st.subheader("Detailed Results List:")

    for i, props in enumerate(processed_results):
        purpose = props.get('purpose', 'N/A')
        division = props.get('division', 'N/A')
        region = props.get('regionServed', 'N/A')
        amount = props.get('amountCommitted', 'N/A')
        grant_id = props.get('grantId', 'N/A')
        st.markdown(f"**Result {i+1} (Grant ID: {grant_id})**")
        st.markdown(f"- **Purpose:** {purpose}")
        st.markdown(f"- **Division:** {division}")
        st.markdown(f"- **Region Served:** {region}")
        try:
            amount_numeric = pd.to_numeric(amount, errors='coerce')
            amount_formatted = f"${amount_numeric:,.2f}" if pd.notna(amount_numeric) else str(amount)
        except (ValueError, TypeError): amount_formatted = str(amount)
        st.markdown(f"- **Amount Committed:** {amount_formatted}")
        st.divider()

    st.subheader("Results Table:")
    try:
        results_df = pd.DataFrame(processed_results)
        display_cols = [col for col in ['grantId', 'purpose', 'division', 'regionServed', 'amountCommitted'] if col in results_df.columns]
        if display_cols:
            if 'amountCommitted' in results_df.columns:
                 results_df_display = results_df.copy()
                 results_df_display['displayAmount'] = results_df_display['amountCommitted'].apply(lambda x: f"${pd.to_numeric(x, errors='coerce'):,.2f}" if pd.notna(pd.to_numeric(x, errors='coerce')) else str(x))
                 if 'displayAmount' in results_df_display.columns and 'amountCommitted' in display_cols:
                      display_cols[display_cols.index('amountCommitted')] = 'displayAmount'
                 st.dataframe(results_df_display[display_cols])
            else: st.dataframe(results_df[display_cols])
        elif not results_df.empty: st.dataframe(results_df)
        else: st.warning("Result table is empty.")
    except Exception as e:
         st.error(f"Could not display results in table format: {e}")
         print(f"Debug: Error creating DataFrame for table: {traceback.format_exc()}")

    if not is_source_documents:
        st.subheader("Visualizations:")
        try:
            if not processed_results:
                 st.warning("No data available for visualization.")
                 return
            results_df = pd.DataFrame(processed_results)
            if 'amountCommitted' in results_df.columns and not results_df['amountCommitted'].isnull().all():
                results_df['amountCommitted_numeric'] = pd.to_numeric(results_df['amountCommitted'], errors='coerce')
                results_df_clean = results_df.dropna(subset=['amountCommitted_numeric']).copy()
                if not results_df_clean.empty:
                    if 'regionServed' in results_df_clean.columns and results_df_clean['regionServed'].nunique() > 0:
                        region_amount = results_df_clean[results_df_clean['regionServed'].astype(str).str.strip() != ''].groupby('regionServed')['amountCommitted_numeric'].sum().reset_index().sort_values('amountCommitted_numeric', ascending=False).head(15)
                        if not region_amount.empty:
                            fig_region = px.bar(region_amount, x='regionServed', y='amountCommitted_numeric', title='Total Amount Committed by Region (Top 15)', labels={'regionServed': 'Region Served', 'amountCommitted_numeric': 'Total Amount Committed ($)'}, hover_data={'amountCommitted_numeric': ':.2f'}, color='regionServed')
                            st.plotly_chart(fig_region, use_container_width=True)
                        else: st.info("No data for 'Amount by Region' chart.")
                    if 'division' in results_df_clean.columns and results_df_clean['division'].nunique() > 0:
                        division_amount = results_df_clean[results_df_clean['division'].astype(str).str.strip() != ''].groupby('division')['amountCommitted_numeric'].sum().reset_index().sort_values('amountCommitted_numeric', ascending=False).head(15)
                        if not division_amount.empty:
                            fig_division = px.bar(division_amount, x='division', y='amountCommitted_numeric', title='Total Amount Committed by Division (Top 15)', labels={'division': 'Division', 'amountCommitted_numeric': 'Total Amount Committed ($)'}, hover_data={'amountCommitted_numeric': ':.2f'}, color='division')
                            st.plotly_chart(fig_division, use_container_width=True)
                        else: st.info("No data for 'Amount by Division' chart.")
                    if 'regionServed' in results_df_clean.columns and results_df_clean['regionServed'].nunique() > 1:
                        region_count = results_df_clean[results_df_clean['regionServed'].astype(str).str.strip() != '']['regionServed'].value_counts().reset_index()
                        if 'count' not in region_count.columns and len(region_count.columns) == 2: region_count.columns = ['regionServed', 'count']
                        if not region_count.empty and 'count' in region_count.columns:
                            fig_pie_region = px.pie(region_count.head(10), values='count', names='regionServed', title='Distribution of Grants by Region (Top 10)', hover_data={'count': True})
                            fig_pie_region.update_traces(textposition='inside', textinfo='percent+label', hole=.3)
                            st.plotly_chart(fig_pie_region, use_container_width=True)
                        else: st.info("No data for 'Distribution by Region' pie chart.")
                else: st.warning("No valid numeric 'amountCommitted' data found after cleaning.")
            else: st.warning("'amountCommitted' column missing, empty, or non-numeric.")
        except Exception as e:
            st.error(f"An error occurred while generating charts: {e}")
            print(f"Debug: Error in chart generation: {traceback.format_exc()}")


# --- Initialize Client and RAG Chain ---
client, qa_chain = setup_weaviate_and_rag()

# --- Streamlit App UI ---
#
st.title("Grant Information Q&A 💰")
st.markdown("""
Ask questions about grant purposes, funding divisions, regions served, or committed amounts.
You can also ask for specific rankings or numbers of grants like:
- `top 10 grants in Europe by amount`
- `lowest 5 grants for Asia`
- `show me 15 grants` (adjusts number of semantic search results)
- `What grants support projects in Africa?` (general semantic search)
""")

query = st.text_input("Enter your question:", placeholder="e.g., What grants support projects in Africa?")

if st.button("Get Answer"):
    if not client or not qa_chain:
        st.error("System not initialized properly. Cannot process query.")
        st.info("Check environment variables, Weaviate connection, API keys, schema, OpenAI setup, and logs.")
    elif not query:
        #
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                parsed_params = parse_structured_query(query)

                # --- Structured Query Execution ---
                #
                if parsed_params and parsed_params["type"] in ["structured_filter_sort", "structured_sort_only"]:
                    st.subheader(f"Direct Query Results for: '{query}'")
                    results = execute_custom_query(client, parsed_params)
                    display_results_and_charts(results, is_source_documents=False)

                # --- Semantic Search with Specific K ---
                #
                elif parsed_params and parsed_params["type"] == "semantic_search_k":
                    st.subheader(f"Semantic Search Results for: '{query}' (showing up to {parsed_params['limit']} results)")
                    try:
                        current_embeddings = None
                        if hasattr(qa_chain, 'retriever') and hasattr(qa_chain.retriever, 'vectorstore') and hasattr(qa_chain.retriever.vectorstore, 'embedding'):
                            current_embeddings = qa_chain.retriever.vectorstore.embedding
                        else: raise ValueError("Embedding model not found in QA chain")

                        current_llm = None
                        if hasattr(qa_chain, 'combine_documents_chain') and hasattr(qa_chain.combine_documents_chain, 'llm_chain') and hasattr(qa_chain.combine_documents_chain.llm_chain, 'llm'):
                             current_llm = qa_chain.combine_documents_chain.llm_chain.llm
                        else: raise ValueError("LLM not found in QA chain")

                        temp_vectorstore = WeaviateVectorStore(
                            client=client, index_name=GRANT_CLASS_NAME, text_key="combinedText",
                            embedding=current_embeddings, attributes=["purpose", "division", "regionServed", "amountCommitted", "grantId"]
                        )
                        temp_retriever = temp_vectorstore.as_retriever(search_kwargs={'k': parsed_params['limit']})
                        temp_qa_chain = RetrievalQA.from_chain_type(
                            llm=current_llm, chain_type="stuff", retriever=temp_retriever, return_source_documents=True
                        )
                        result = temp_qa_chain.invoke({"query": query})

                        st.subheader("Answer:")
                        st.write(result.get('result', 'No answer generated.'))

                        #
                        if 'source_documents' in result and result['source_documents']:
                             display_results_and_charts(result['source_documents'], is_source_documents=True)
                        else: st.warning("No source documents retrieved/used for this answer.")

                    except Exception as e:
                         st.error(f"Failed to execute semantic search with k={parsed_params['limit']}: {e}")
                         print(f"Debug: Error in semantic_search_k block: {traceback.format_exc()}")

                # --- Default Semantic Search (RAG) ---
                #
                else:
                    st.subheader(f"Answer based on semantic search for: '{query}'")
                    result = qa_chain.invoke({"query": query})
                    st.write(result.get('result', 'No answer generated.'))
                    if 'source_documents' in result and result['source_documents']:
                        display_results_and_charts(result['source_documents'], is_source_documents=True)
                    else: st.warning("No source documents retrieved/used for this answer.")

            except Exception as e:
                #
                st.error(f"An unexpected error occurred during query processing: {e}")
                st.error("Please check the application logs (terminal output) for the full traceback.")
                print(f"Debug: An unexpected error occurred during query processing: {traceback.format_exc()}")
