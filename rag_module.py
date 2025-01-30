from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

# Enable verbose debugging
set_debug(True)
set_verbose(True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG with verbose debugging."""

    def __init__(self, 
                 llm_model: str = "deepseek-r1:8b", 
                 embedding_model: str = "nomic-embed-text", 
                 mongo_connection_str: str = "mongodb://localhost:27017/?directConnection=true", 
                 database_name: str = "knowledge_base", 
                 collection_name: str = "documents"):
        """
        Initialize the ChatPDF instance with an LLM, embedding model, and MongoDB Atlas configuration.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document and the ongoing conversation.
            
            Conversation History:
            {conversation_history}
            
            Context from Documents:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        
        # Setup MongoDB connection
        self.client = MongoClient(mongo_connection_str)
        self.collection = self.client[database_name][collection_name]

        # Verbose connection check
        doc_count = self.collection.count_documents({})
        logger.info(f"MongoDB Connection Established - Document Count: {doc_count}")

        # Initialize the vector store with MongoDB Atlas
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine"
        )

        # Create vector search index on the collection
        # Adjust dimensions based on your embedding model
        self.vector_store.create_vector_search_index(dimensions=768)

        logger.info("Vector Store Initialized")

        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, chunk its contents, and store the embeddings in MongoDB Atlas.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()

        logger.info(f"Loaded {len(docs)} pages from {pdf_file_path}")

        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} document chunks")

        # Optional: Log some sample chunks for verification
        for i, chunk in enumerate(chunks[:3]):
            logger.debug(f"Chunk {i+1} Content: {chunk.page_content[:200]}...")

        chunks = filter_complex_metadata(chunks)

        # Add documents to vector store and check embeddings
        self.vector_store.add_documents(documents=chunks)
        logger.info("Document embeddings stored successfully in MongoDB Atlas.")

    def ask(self, query: str, conversation_history: list = None, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline with verbose debugging and conversation history.

        Parameters:
        - query (str): The user's question.
        - conversation_history (list): List of previous messages in the conversation.
        - k (int): Number of retrieved documents.
        - score_threshold (float): Similarity score threshold for retrieval.

        Returns:
        - str: The assistant's response.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        # Generate and log query embeddings
        query_embedding = self.embeddings.embed_query(query)
        logger.info(f"User Query: {query}")
        logger.debug(f"Query Embedding (sample values): {query_embedding[:10]}... [Total Length: {len(query_embedding)}]")

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            logger.warning("No relevant documents retrieved.")
            return "No relevant context found in the document to answer your question."

        logger.info(f"Retrieved {len(retrieved_docs)} document(s)")
        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"Document {i+1}: {doc.page_content[:200]}...")

        # Format the input for the LLM, including conversation history
        formatted_input = {
            "conversation_history": "\n".join(conversation_history) if conversation_history else "",
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        response = chain.invoke(formatted_input)
        logger.debug(f"LLM Response: {response}")
        return response

    def clear(self):
        """
        Reset the retriever and optionally clear the vector store or other states.
        """
        logger.info("Clearing retriever and resetting state.")
        self.retriever = None

