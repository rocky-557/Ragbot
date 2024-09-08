import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

data1 = load_json('outputs/vol3.pdf.json')
data2 = load_json('outputs/vol3.pdf.json')
data3 = load_json('outputs/vol3.pdf.json')

def extract_tables(json_data):
    tables = []
    for item in json_data:
        if item['type'] == 'Table':
            table_text = item['metadata']
            table = table_text['text_as_html']
            tables.append(table)
    return tables

vol1 = extract_tables(data1)
vol2 = extract_tables(data2)
vol3 = extract_tables(data3)

full = vol1 + vol2 + vol3

table_embeddings = model.encode(full)

# Create a FAISS index
dimension = table_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(table_embeddings))

def find_relevant_tables(query, model, index, tables, top_k=1):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    relevant_tables = [tables[i] for i in indices[0]]
    return relevant_tables

def give_rel_tabs(query, index = index, tables = full ,model =SentenceTransformer('all-MiniLM-L6-v2') ):
    relev = find_relevant_tables(query, model, index, tables)
    dfs = pd.read_html(relev[0])
    df = dfs[0]
    return df
