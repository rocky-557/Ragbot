from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer


class FaissDb:
    def __init__(self, docs, embedding_function):
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context

class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"):
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
dat = load_and_split_pdfs()
vecdb = FaissDb(dat,embedding_function=encoder.embedding_function)