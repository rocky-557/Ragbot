import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer




class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"):
        self.model= sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device})

def load_and_split_pdfs(chunk_size= 256):
    pdf_path='vol_3.pdf'
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    pages = data


    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        strip_whitespace=True)

    docs = text_splitter.split_documents(pages)
    return docs

encoder = Encoder()

class FaissDb:
    def __init__(self):
        self.db = None

    def createDB(self,docs,embedding_function):
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)
        vs=self.db
        vs.add_documents(docs)
        vs.save_local('faiss_index')
        print('DB Saved Succesfully!')

    def retrieveDB(self,encoder):
        self.db = FAISS.load_local('faiss_index', encoder.model,allow_dangerous_deserialization=True)
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
vecdb.retrieveDB(encoder)
vecdb.similarity_search(question='heart')