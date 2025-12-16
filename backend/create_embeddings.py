import os
import sys
import time
import uuid
import pandas as pd
import traceback
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hib-insurance-index")
DOC_PATH = BACKEND_DIR / "data"
PINECONE_BATCH_SIZE = 96

print("\n" + "=" * 70)
print("📤 PINECONE EMBEDDING UPSERTER")
print("=" * 70)

# Validate API key
if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_DUMMY_API_KEY":
    print("❌ ERROR: PINECONE_API_KEY not found or is dummy value.")
    print("   Please check your .env file")
    sys.exit(1)

print(f"✅ PINECONE_API_KEY loaded: {PINECONE_API_KEY[:30]}...")
print(f"📍 Index name: {PINECONE_INDEX_NAME}")
print(f"📁 Data directory: {DOC_PATH}")

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ Pinecone client initialized")
except Exception as e:
    print(f"❌ Pinecone initialization error: {e}")
    sys.exit(1)

# Check if index exists, create if needed
try:
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"\n⚠️ Index '{PINECONE_INDEX_NAME}' not found. Creating...")
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "text"}
            }
        )
        print(f"✅ Index '{PINECONE_INDEX_NAME}' created")
    else:
        print(f"✅ Index '{PINECONE_INDEX_NAME}' exists")
except Exception as e:
    print(f"❌ Index check/creation error: {e}")
    sys.exit(1)

# Connect to index
try:
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Connection error: {e}")
    sys.exit(1)

# Load data from Excel files
print("\n" + "=" * 70)
print("📂 LOADING DATA FROM EXCEL FILES")
print("=" * 70)

if not DOC_PATH.exists():
    print(f"❌ Data directory '{DOC_PATH}' not found.")
    print("   Please create 'data/' folder and add Excel files")
    sys.exit(1)

excel_files = list(DOC_PATH.glob("*.xlsx")) + list(DOC_PATH.glob("*.xls"))

if not excel_files:
    print(f"❌ No Excel files found in {DOC_PATH}")
    sys.exit(1)

print(f"📂 Found {len(excel_files)} Excel files")

records = []


def combine_row_data(row, column_names):
    """Combines all column values in a row into a single string."""
    parts = []
    for col in column_names:
        value = str(row[col]).strip()
        if value and value != 'nan':
            parts.append(f"{col}: {value}")
    return " | ".join(parts) if parts else "Empty row"


# Process each Excel file
for file_path in excel_files:
    print(f"\n📄 Processing: {file_path.name}")
    try:
        df = pd.read_excel(file_path)
        
        if df.empty:
            print(f"   ⚠️ File is empty. Skipping.")
            continue

        df = df.astype(str).fillna('')
        columns = df.columns.tolist()

        # Create the "text" field for embeddings
        df['text'] = df.apply(
            lambda row: combine_row_data(row, columns), 
            axis=1
        )

        # Create records
        file_record_count = 0
        for i, row in df.iterrows():
            pinecone_record = {
                'id': str(uuid.uuid4()),
                'text': row['text'],  # CRITICAL: Must be named 'text'
                'source_file': file_path.name
            }
            
            # Add original columns as metadata
            for col in columns:
                if col not in ['text', 'id']: 
                    pinecone_record[col] = str(row[col])
                    
            records.append(pinecone_record)
            file_record_count += 1
        
        print(f"   ✅ Loaded {file_record_count} records")

    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        traceback.print_exc()

# Summary before upserting
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"Total records to upsert: {len(records)}")
print(f"Batch size: {PINECONE_BATCH_SIZE}")
print(f"Total batches: {(len(records) + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE}")

if not records:
    print("\n❌ No records to upsert. Exiting.")
    sys.exit(1)

# Ask for confirmation
print("\n⚠️ WARNING: This will upsert all records to Pinecone.")
response = input("Continue? (yes/no): ").strip().lower()

if response != "yes":
    print("❌ Cancelled by user")
    sys.exit(0)

# Upsert records in batches
print("\n" + "=" * 70)
print("📤 UPSERTING RECORDS")
print("=" * 70)

total_upserted = 0

for batch_idx in range(0, len(records), PINECONE_BATCH_SIZE):
    batch = records[batch_idx:batch_idx + PINECONE_BATCH_SIZE]
    batch_num = (batch_idx // PINECONE_BATCH_SIZE) + 1
    total_batches = (len(records) + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE
    
    try:
        upsert_response = pinecone_index.upsert_records(
            "namespace1",
            batch
        )
        
        total_upserted += len(batch)
        print(f"✅ Batch {batch_num}/{total_batches}: Upserted {len(batch)} records")
        
        # Small delay between batches
        if batch_idx + PINECONE_BATCH_SIZE < len(records):
            time.sleep(0.1)
    
    except Exception as e:
        print(f"❌ Batch {batch_num} error: {e}")
        traceback.print_exc()

# Final summary
print("\n" + "=" * 70)
print("✅ UPSERTING COMPLETE")
print("=" * 70)
print(f"Total upserted: {total_upserted} records")

if records:
    print(f"\nExample record:")
    example = records[0]
    print(f"  ID: {example['id']}")
    print(f"  Source: {example.get('source_file', 'unknown')}")
    print(f"  Text: {example['text'][:100]}...")

print("\n✅ Embeddings successfully uploaded to Pinecone!")
print(f"   Index: {PINECONE_INDEX_NAME}")
print(f"   Namespace: namespace1")
print(f"   Total vectors: {total_upserted}")