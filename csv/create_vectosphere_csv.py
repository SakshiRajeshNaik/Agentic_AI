from sentence_transformers import SentenceTransformer
import pandas as pd
import json

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Your data (you can expand later)
sentences = [
    "I love pizza",
    "Pizza is my favourite food",
    "Stock market is up today",
    "TATA AIG offers comprehensive car insurance",
    "Health insurance covers hospitalization expenses"
]

print("Generating embeddings...")
embeddings = model.encode(sentences)

# 🔥 VERY IMPORTANT: convert to JSON string
emb_list = [json.dumps(emb.tolist()) for emb in embeddings]

# Create dataframe
df = pd.DataFrame({
    "id": range(1, len(sentences) + 1),
    "text": sentences,
    "embeddings": emb_list
})

# Save CSV
output_file = "vectosphere_ready.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ CSV saved as {output_file}")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)