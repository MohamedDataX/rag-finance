import os
import shutil
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

class FinancialRAG:
    def __init__(self, persist_directory="./data/chroma_db", model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.persist_directory = persist_directory
        
        print(f"Chargement du modèle d'embedding : {model_name}...")
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        
        # init de la Vector DB 
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        print(f"✅ ChromaDB initialisé dans {self.persist_directory}")

    def ingest_text(self, text, metadata):

        if not text:
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.create_documents([text], metadatas=[metadata])
        
        print(f"Texte découpé en {len(chunks)} chunks. Indexation en cours...")
        
        # stockage dans ChromaDB
        # Chroma gère auto la conversion Texte -> Vecteur via l'embedding_function définie
        self.db.add_documents(chunks)
        print(" Indexation terminée")

    def search(self, query, k=3, filter_dict=None):
        print(f" Recherche: '{query}'")
        
        # Recherche par similarité cosinus
        results = self.db.similarity_search(query, k=k, filter=filter_dict)
        
        return results

    def reset_db(self):
 
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(" Base de données vectorielle supprimée.")
            # Réinitialisation de l'objet DB
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )