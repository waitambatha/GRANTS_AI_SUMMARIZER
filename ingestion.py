# ingestion.py (Updated)

import pandas as pd
import weaviate
import os
import logging
from dotenv import load_dotenv
# Make sure you have a schema.py file with GRANT_CLASS_NAME defined
# e.g., schema.py could contain: GRANT_CLASS_NAME = "Grants"
try:
    from schema import GRANT_CLASS_NAME
except ImportError:
    print("Error: Could not import GRANT_CLASS_NAME from schema.py.")
    print("Please ensure schema.py exists and defines GRANT_CLASS_NAME.")
    # Define a default or raise an error if critical
    GRANT_CLASS_NAME = "Grants" # Example default, adjust if needed
    print(f"Warning: Using default GRANT_CLASS_NAME='{GRANT_CLASS_NAME}'")

from weaviate.auth import AuthApiKey
import weaviate.classes as wvc
import weaviate.util
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
# Use UTC time for consistency in logs
log_file = os.path.join(log_dir, f"ingestion_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
# Set higher level for noisy libraries if needed
# logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_REST_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(csv_file_path="grants.csv"):
    """
    Loads and preprocesses grant data from a CSV file.
    Handles potential errors during loading and cleaning.
    Returns a list of processed dictionaries or None if critical errors occur.
    """
    logger.info(f"Starting data loading and preprocessing from {csv_file_path}")
    try:
        # Assuming the actual header is on the second row (index 1)
        df = pd.read_csv(csv_file_path, header=1)
        logger.info(f"Successfully read CSV file with {len(df)} rows")
    except FileNotFoundError:
        logger.error(f"CRITICAL: CSV file '{csv_file_path}' not found.")
        return None
    except Exception as e:
        logger.error(f"CRITICAL: Error reading CSV file '{csv_file_path}': {str(e)}", exc_info=True)
        return None

    required_columns = ['GRANT ID', 'PURPOSE', 'DIVISION', 'REGION SERVED', 'AMOUNT COMMITTED']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logger.error(
            f"CRITICAL: Missing required columns in {csv_file_path}. "
            f"Required: {', '.join(required_columns)}, Missing: {', '.join(missing)}"
        )
        return None

    # Handle potential NaN values before processing
    df.fillna({'PURPOSE': '', 'DIVISION': '', 'REGION SERVED': '', 'GRANT ID': '', 'AMOUNT COMMITTED': 0}, inplace=True)


    # Check for duplicate GRANT ID values
    duplicates = df[df['GRANT ID'].astype(str).duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_ids = duplicates['GRANT ID'].unique().tolist()
        logger.warning(f"Found {len(duplicate_ids)} unique GRANT IDs with duplicate entries in CSV: {duplicate_ids[:10]}{'...' if len(duplicate_ids) > 10 else ''}")
        logger.warning("Using generate_uuid5 based on GRANT ID means only the last occurrence of a duplicate ID might be ingested/updated.")

    logger.info("All required columns found. Preprocessing data...")
    processed_data = []
    for index, row in df.iterrows():
        item = {}
        try:
            # Convert Grant ID to string, handle potential errors
            item['grantId'] = str(row['GRANT ID']).strip()
            if not item['grantId']:
                 logger.warning(f"Row {index+2}: Missing Grant ID, skipping row.") # +2 accounts for 0-index and header row
                 continue

            # Convert amount, handle potential errors
            try:
                amount = float(row['AMOUNT COMMITTED'])
                item['amountCommitted'] = amount if not pd.isna(amount) else 0.0
            except (ValueError, TypeError):
                logger.warning(f"Row {index+2} (Grant ID: {item['grantId']}): Invalid amount '{row['AMOUNT COMMITTED']}', setting to 0.0")
                item['amountCommitted'] = 0.0

            # Convert other fields to string, ensure they are not NaN
            item['purpose'] = str(row['PURPOSE'])
            item['division'] = str(row['DIVISION'])
            item['regionServed'] = str(row['REGION SERVED'])

            # Create combinedText
            item['combinedText'] = (
                f"Purpose: {item['purpose']} \n"
                f"Division: {item['division']} \n"
                f"Region Served: {item['regionServed']}"
            )

            processed_data.append(item)

        except Exception as e:
             logger.error(f"Row {index+2}: Error processing row data: {str(e)}. Skipping row.", exc_info=True)
             continue # Skip rows with unexpected processing errors

    logger.info(f"Successfully preprocessed {len(processed_data)} records")
    return processed_data

# --- Data Ingestion ---
def ingest_data(data):
    """
    Ingests preprocessed data into Weaviate using dynamic batching.
    Logs progress, captures batch errors, and verifies ingestion success.
    """
    logger.info("Starting data ingestion process")
    if not data:
        logger.error("No data provided for ingestion. Aborting.")
        return False

    # Check environment variables
    if not all([WEAVIATE_URL, WEAVIATE_API_KEY, OPENAI_API_KEY]):
        missing_vars = [v for v in ['WEAVIATE_URL', 'WEAVIATE_API_KEY', 'OPENAI_API_KEY'] if not os.getenv(v)]
        logger.error(f"CRITICAL: Missing environment variables: {', '.join(missing_vars)}. Check .env file. Aborting.")
        return False

    client = None
    success = False
    objects_prepared = 0
    prep_errors = [] # Errors during preparation loop
    initial_size = -1
    final_size = -1
    added_count = -1

    try:
        logger.info(f"Attempting to connect to Weaviate at {WEAVIATE_URL}")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
            # Add timeouts (optional but recommended)
            # startup_period=10 # seconds to wait for Weaviate to start/respond initially
        )

        # Check connection status
        if not client.is_ready():
             logger.error("CRITICAL: Failed to connect to Weaviate Cloud or instance is not ready. Please check connection details, API keys, and instance status.")
             if client: client.close()
             return False
        logger.info("Weaviate client connected successfully")

        # Check if collection exists
        if not client.collections.exists(GRANT_CLASS_NAME):
            logger.error(f"CRITICAL: Collection '{GRANT_CLASS_NAME}' does not exist in Weaviate. Please create it first (e.g., using schema.py). Aborting.")
            client.close()
            return False
        logger.info(f"Using existing collection '{GRANT_CLASS_NAME}'")
        grants_collection = client.collections.get(GRANT_CLASS_NAME)

        # Get initial size for comparison (using aggregate for potentially better accuracy)
        try:
             aggregation = grants_collection.aggregate.over_all(total_count=True)
             initial_size = aggregation.total_count if aggregation else 0
             logger.info(f"Initial collection size: {initial_size} objects")
        except Exception as e:
            logger.warning(f"Could not determine initial collection size: {str(e)}. Verification will be less precise.")
            initial_size = -1 # Indicate unknown size

        # --- Start Batching ---
        logger.info(f"Starting dynamic batch ingestion for {len(data)} processed records")
        # Weaviate recommends keeping batch size reasonable, 'dynamic' handles some of this.
        # You can also set client.batch.configure(batch_size=...) if needed before the 'with' block.

        with grants_collection.batch.dynamic() as batch:
            for item in data:
                try:
                    properties = {
                        "grantId": item['grantId'],
                        "purpose": item['purpose'],
                        "division": item['division'],
                        "regionServed": item['regionServed'],
                        "amountCommitted": item['amountCommitted'],
                        "combinedText": item['combinedText']
                    }
                    # Generate UUID based on Grant ID for idempotency
                    # Note: This means importing duplicates will overwrite based on the LAST occurrence in the CSV
                    uuid = weaviate.util.generate_uuid5(item['grantId'], GRANT_CLASS_NAME) # Add class name as namespace

                    batch.add_object(
                        properties=properties,
                        uuid=uuid
                        # vector=... # Add vector here if generating embeddings client-side
                    )
                    objects_prepared += 1
                    if objects_prepared % 2000 == 0: # Adjust logging frequency as needed
                        logger.info(f"Added {objects_prepared} objects to batch queue...")

                except Exception as e:
                    grant_id = item.get('grantId', 'N/A')
                    logger.error(f"Error preparing object grantId {grant_id} for batch: {str(e)}")
                    prep_errors.append({"grantId": grant_id, "error": str(e)})

        # --- Batch Execution and Verification ---
        # The `with` block automatically handles batch execution on exit.
        # We need to check for errors reported by the batch manager itself, if available,
        # and compare expected vs actual size change.

        logger.info(f"Batch preparation completed: {objects_prepared} objects queued, {len(prep_errors)} preparation errors encountered.")

        # Check for errors reported by the batch context manager (if any are exposed directly - varies by client version/method)
        # In v4 with `dynamic`, explicit error objects might not be returned directly from the context manager exit.
        # We rely more on logs and the size check. If using manual `.execute()`, you'd check `batch_results.has_errors`.

        # Check final size
        try:
             # Give Weaviate a moment to update counts (optional, adjust as needed)
             # import time
             # time.sleep(2)
             aggregation = grants_collection.aggregate.over_all(total_count=True)
             final_size = aggregation.total_count if aggregation else 0
             logger.info(f"Final collection size: {final_size} objects")
             if initial_size != -1:
                 added_count = final_size - initial_size
                 logger.info(f"Object count change reported by Weaviate: {added_count}")
             else:
                 added_count = -1 # Cannot calculate change accurately
        except Exception as e:
            logger.warning(f"Could not determine final collection size: {str(e)}. Verification incomplete.")
            final_size = -1
            added_count = -1

        # --- Determine Success ---
        expected_added = objects_prepared - len(prep_errors)
        logger.info(f"Expected objects to be added (Prepared - Prep Errors): {expected_added}")

        if len(prep_errors) > 0:
             logger.error(f"Ingestion failed: Encountered {len(prep_errors)} errors during object preparation phase.")
             for error in prep_errors:
                 logger.error(f"  Prep Error for grantId {error['grantId']}: {error['error']}")
             success = False
        elif expected_added > 0 and added_count == 0 and initial_size != -1:
             logger.error(f"CRITICAL FAILURE: Expected ~{expected_added} objects to be added, but size check indicates 0 were added.")
             logger.error("This suggests a potential silent batch execution failure, connection issue during commit, or that all prepared objects already existed (check for UUID conflicts/duplicates).")
             success = False
        elif added_count != -1 and added_count < expected_added:
             logger.warning(f"Partial Success/Discrepancy: Expected {expected_added} additions, but size increased by only {added_count}.")
             logger.warning("Some objects may have failed during batch execution on the server side. Check Weaviate logs if possible.")
             # Consider this a failure for strictness
             success = False
        elif added_count != -1 and added_count >= expected_added:
             logger.info(f"Batch ingestion successful: {added_count} objects reflected in count change (expected {expected_added}).")
             success = True
        elif expected_added == 0:
             logger.info("Ingestion finished: No new objects were prepared or expected to be added (due to prep errors or empty input).")
             success = True # Script ran, but nothing new was added as expected.
        else: # Case where size checks failed
             logger.warning("Ingestion status uncertain: Could not reliably verify object count changes.")
             logger.warning(f"Prepared {objects_prepared} objects with {len(prep_errors)} prep errors.")
             logger.warning("Assuming failure due to lack of verification. Check collection manually.")
             success = False


        # Log a sample object if successful and possible
        if success and final_size > 0:
            try:
                sample_query = grants_collection.query.fetch_objects(limit=1)
                if sample_query.objects:
                    logger.info(f"Sample object from collection post-ingestion: {sample_query.objects[0].properties}")
                else:
                    # This case should ideally not happen if final_size > 0 and success is True
                    logger.warning("Collection size > 0, but failed to fetch a sample object.")
            except Exception as e:
                logger.warning(f"Could not query sample object after ingestion: {str(e)}")
        elif not success and final_size == 0 and initial_size == 0 :
             logger.warning("Ingestion failed, and collection appears to be empty.")
        elif not success:
            logger.warning("Ingestion failed or was incomplete. Sample object query skipped.")


    except weaviate.exceptions.AuthenticationFailedError as e:
         logger.error(f"CRITICAL: Weaviate Authentication Failed: {str(e)}. Check WEAVIATE_API_KEY.")
         success = False
    except Exception as e:
        logger.error(f"CRITICAL: Ingestion process failed with an unhandled exception: {str(e)}", exc_info=True)
        success = False
    finally:
        if client and client.is_connected():
            client.close()
            logger.info("Weaviate client connection closed.")

    logger.info(f"===== Ingestion process finished. Success: {success} =====")
    return success

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("========== Starting Grant Ingestion Script ==========")
    # Define CSV path (can be changed)
    csv_to_ingest = "grants.csv"

    grant_data = load_and_preprocess_data(csv_to_ingest)

    if grant_data is not None: # Check if preprocessing returned data (not None)
        if not grant_data:
             logger.warning("Preprocessing returned an empty list. No data to ingest.")
             logger.info("Script finished: No data to process.")
        else:
             ingestion_success = ingest_data(grant_data)
             if ingestion_success:
                 logger.info("========== Script completed successfully ==========")
             else:
                 logger.error("========== Script completed with errors or discrepancies. Please review logs above. ==========")
    else:
        logger.error("CRITICAL: Data loading and preprocessing failed. Ingestion aborted.")
        logger.error("========== Script completed with critical errors during data load phase. ==========")