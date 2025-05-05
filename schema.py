# schema.py (Corrected Section)
import weaviate
import os
from dotenv import load_dotenv
# Import AuthApiKey directly from weaviate.auth as in your working script
from weaviate.auth import AuthApiKey
# Import necessary components from weaviate.classes.config
import weaviate.classes.config as wvc_config
# Import the main classes module
import weaviate.classes as wvc

# Load environment variables
load_dotenv()
WEAVIATE_REST_URL = os.getenv("WEAVIATE_REST_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Needed for text2vec-openai module

# Define the class name for the grants
GRANT_CLASS_NAME = "Grant"

def create_weaviate_schema():
    """
    Defines and creates the schema for the Grant class in Weaviate.
    Connects to Weaviate Cloud using connect_to_weaviate_cloud and weaviate.auth.AuthApiKey (v4 client).
    """
    client = None # Initialize client to None for finally block
    if not all([WEAVIATE_REST_URL, WEAVIATE_API_KEY, OPENAI_API_KEY]): # [cite: 2]
        print("Error: Missing one or more environment variables (WEAVIATE_REST_URL, WEAVIATE_API_KEY, OPENAI_API_KEY).") # [cite: 2]
        print("Please check your .env file.") # [cite: 2]
        return # [cite: 2]

    # Connect to Weaviate Cloud using v4 client's connect_to_weaviate_cloud
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_REST_URL, # [cite: 2]
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY), # Use AuthApiKey from weaviate.auth [cite: 2]
            # Add OpenAI API key to additional headers for vectorization
            headers={ # [cite: 3]
                "X-OpenAI-Api-Key": OPENAI_API_KEY # [cite: 3]
            }
        )
        print("Weaviate client initialized successfully using v4 connect_to_weaviate_cloud.") # [cite: 3]

    except Exception as e:
        print(f"Failed to initialize Weaviate client: {e}") # [cite: 3]
        print("Please check your Weaviate and OpenAI credentials in the .env file.") # [cite: 4]
        return # [cite: 4]

    # Check if the collection (class) already exists
    try:
        if client.collections.exists(GRANT_CLASS_NAME): # Use client.collections.exists for v4 [cite: 4]
            print(f"Collection '{GRANT_CLASS_NAME}' already exists. Deleting and recreating.") # [cite: 4, 5]
            client.collections.delete(GRANT_CLASS_NAME) # Use client.collections.delete for v4 [cite: 5]

        print(f"Creating schema for collection '{GRANT_CLASS_NAME}'...") # [cite: 5]
        # Define the properties for the collection
        properties = [
            wvc_config.Property( # Use wvc_config for Property [cite: 5]
                name="grantId", # [cite: 5]
                data_type=wvc_config.DataType.TEXT, # Use wvc_config for DataType [cite: 6]
                description="Unique identifier for the grant", # [cite: 6]
                index_filterable=True, # [cite: 6]
                index_searchable=False, # [cite: 6]
            ),
            wvc_config.Property(
                name="purpose", # [cite: 7]
                data_type=wvc_config.DataType.TEXT, # [cite: 7]
                description="The main purpose or objective of the grant", # [cite: 7]
                index_filterable=False, # [cite: 7]
                index_searchable=True, # [cite: 7]
            ),
             wvc_config.Property(
                name="division", # [cite: 8]
                data_type=wvc_config.DataType.TEXT, # [cite: 8]
                description="The division or department funding the grant", # [cite: 8]
                index_filterable=True, # [cite: 8]
                index_searchable=True, # [cite: 8]
            ),
            wvc_config.Property(
                name="regionServed", # [cite: 9]
                data_type=wvc_config.DataType.TEXT, # [cite: 9]
                description="The geographical region served by the grant", # [cite: 9]
                index_filterable=True, # [cite: 9]
                index_searchable=True, # [cite: 9]
            ), # [cite: 10]
             wvc_config.Property(
                name="amountCommitted", # [cite: 10]
                data_type=wvc_config.DataType.NUMBER, # Use NUMBER for amounts [cite: 10]
                description="The amount of money committed to the grant", # [cite: 10]
                index_filterable=True, # [cite: 10]
                index_searchable=False, # [cite: 11]
            ),
            # Add a property to store the combined text for RAG context
            wvc_config.Property(
                name="combinedText", # [cite: 11]
                data_type=wvc_config.DataType.TEXT, # [cite: 11]
                description="Combined text fields for RAG context", # [cite: 12]
                index_filterable=False, # [cite: 12]
                index_searchable=True, # [cite: 12]
            )
        ]

        # Define the vectorizer configuration
        vectorizer_config = wvc_config.Configure.Vectorizer.text2vec_openai( # Use wvc_config for Configure [cite: 12]
            model="ada", # Or text-embedding-3-small/large [cite: 13]
            vectorize_collection_name=True # Use the correct parameter name [cite: 13]
        )

        # Define generative configuration (Optional)
        # generative_config = wvc_config.Configure.Generative.openai() # [cite: 14]

        # === CORRECTED PART ===
        # Create the collection by passing arguments directly
        client.collections.create( # [cite: 14]
            name=GRANT_CLASS_NAME, # [cite: 14]
            description="Data about grants including purpose, division, region, and amount.", # [cite: 14]
            properties=properties, # [cite: 14]
            vectorizer_config=vectorizer_config, # Pass vectorizer config here
            # generative_config=generative_config # Optional: Uncomment if you plan to use generative module [cite: 14]
        )
        # === END CORRECTED PART ===

        print(f"Schema (collection) '{GRANT_CLASS_NAME}' created successfully with text2vec-openai vectorizer.") # [cite: 14]

    except Exception as e:
        print(f"An unexpected error occurred while creating schema: {e}") # [cite: 15]

    finally:
        # Close the client connection
        if client and client.is_connected(): # [cite: 15]
            client.close() # [cite: 15]
            print("Weaviate client connection closed.") # [cite: 15]


if __name__ == "__main__":
    create_weaviate_schema() # [cite: 15]