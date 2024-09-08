#?///////////                  VEC_DB.PY                    ///////////?#.

import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.schema import Document # Import the Document class

text = ''
with open('full.txt') as f:
    text = f.read()

class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"):
        self.model= sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

# Define the chunk size
chunk_size = 256  # Example chunk size, adjust as needed

def split_text_with_tokenizer(text, chunk_size=chunk_size, tokenizer=tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=False)

    # Retrieve input IDs and chunk them
    input_ids = inputs['input_ids'][0]
    chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

    # Convert token IDs back to text
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    # Create Document objects from text chunks
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    return docs


encoder = Encoder()

class FaissDb:
    def __init__(self):
        self.db = None

    def createDB(self,docs,embedding_function):
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)
        vs=self.db
        vs.add_documents(docs)
        vs.save_local('faiss')
        print('DB Saved Succesfully!')

    def retrieveDB(self,encoder):
        self.db = FAISS.load_local('faiss', encoder.model,allow_dangerous_deserialization=True)
        print('Loaded Sucessfully !')


    def similarity_search(self, question,encoder=encoder, k: int = 3):
      # Ensure self.db has the similarity_search method
        # Call embed_query to get the embedding of the question using the correct model
        embedding = encoder.embedding_function.embed_query(question)
        # Use the embedding for similarity search
        retrieved_docs = self.db.similarity_search_by_vector(embedding, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
        
vecdb = FaissDb()
vecdb.retrieveDB(encoder=encoder)