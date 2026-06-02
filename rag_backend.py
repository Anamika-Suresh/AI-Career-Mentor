import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Document Loaders & Splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Try importing modern LangChain Chroma DB, fallback to community version
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Embedding Models & LLMs
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# File parsers
import pypdf
import docx2txt

ROLE_PROMPTS = {
    "Data Analyst": {
        "focus": "data cleaning, SQL query optimization (CTEs, joins, window functions), descriptive statistics, dashboard design (Tableau/PowerBI), business KPIs, and basic A/B testing support.",
        "tone": "detail-oriented, business-savvy, explaining metric shifts clearly, and emphasizing data extraction and dashboarding best practices."
    },
    "Data Scientist": {
        "focus": "predictive modeling, machine learning algorithms (Random Forests, Gradient Boosting, regression), statistical testing (hypothesis tests, p-values, ANOVA), experimental design (A/B test sample size, power analysis), and storytelling with data.",
        "tone": "statistically rigorous, research-driven, discussing model trade-offs (e.g., bias-variance, ROC-AUC vs Accuracy), and statistical validity."
    },
    "ML Engineer": {
        "focus": "machine learning system design (candidate generation vs ranking), production training pipelines (Airflow/Kubeflow), containerization (Docker), model serving APIs (FastAPI/gRPC), latency optimization (quantization, ONNX), and MLOps monitoring.",
        "tone": "production-focused, systems-oriented, stressing code quality, testing (pytest), latency, scalability, and automated deployments."
    },
    "AI Engineer": {
        "focus": "generative AI solutions, Large Language Models (LLMs), RAG architectures (chunking strategies, hybrid search, rerankers), prompt engineering (few-shot, CoT), vector databases (Chroma, Pinecone), agentic workflows (LangGraph), and LLM evaluation (faithfulness, relevance).",
        "tone": "cutting-edge, prompt-fluent, focusing on text-retrieval techniques, embedding models, orchestrating multi-agent flows, and LLM safety/hallucination checks."
    }
}

class CareerMentorRAG:
    def __init__(self, provider: str, api_key: str):
        """
        Initializes the Career Mentor RAG backend with selected LLM and Embeddings provider.
        
        Args:
            provider (str): "Google Gemini" or "OpenAI"
            api_key (str): Corresponding API Key
        """
        self.provider = provider
        self.api_key = api_key.strip() if api_key else ""
        
        # Define directories for Chroma databases based on provider
        # This keeps embeddings distinct to avoid distance metric crashes
        if provider == "Google Gemini":
            self.db_dir = os.path.join("chroma_db", "gemini")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=self.api_key
            )
        elif provider == "OpenAI":
            self.db_dir = os.path.join("chroma_db", "openai")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key
            )
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize Vector Store
        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # Verify the API key is valid by running a test query embedding
        try:
            self.embeddings.embed_query("test_connection")
        except Exception as e:
            # Clear vectorstore reference to prevent locks
            self.vectorstore = None
            raise ValueError(f"API Key validation failed. Please check that your key is correct and active. Details: {str(e)}")

    def parse_file(self, file_bytes: bytes, filename: str, file_type: str) -> str:
        """
        Parses text out of TXT, PDF, and DOCX files.
        """
        if file_type == "text/plain" or filename.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")
            
        elif file_type == "application/pdf" or filename.endswith(".pdf"):
            # Write to a temporary file, read it, and delete it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
                
            try:
                reader = pypdf.PdfReader(tmp_path)
                text = ""
                for page in reader.pages:
                    text_page = page.extract_text()
                    if text_page:
                        text += text_page + "\n"
                return text
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        elif filename.endswith(".docx") or file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Write to a temporary file, parse, and delete
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
                
            try:
                text = docx2txt.process(tmp_path)
                return text
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            raise ValueError("Unsupported file format. Please upload .txt, .pdf, or .docx files.")

    def add_document(self, file_content: bytes, filename: str, file_type: str, category: str) -> bool:
        """
        Parses, chunks, embeds, and saves a career document into the vector store.
        """
        try:
            # Parse text
            raw_text = self.parse_file(file_content, filename, file_type)
            if not raw_text.strip():
                return False
                
            # Chunking document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(raw_text)
            
            # Prepare LangChain Document objects with rich metadata
            documents = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "category": category,
                        "chunk_id": i,
                        "uploaded_at": timestamp
                    }
                )
                documents.append(doc)
                
            # Add to Chroma Vector Store
            self.vectorstore.add_documents(documents)
            return True
            
        except Exception as e:
            print(f"Error adding document {filename}: {e}")
            raise e

    def query(self, question: str, target_role: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Queries the RAG system: retrieves relevant documents and uses LLM to synthesize a tailored response.
        
        Args:
            question (str): User's career query
            target_role (str): "Data Analyst", "Data Scientist", "ML Engineer", or "AI Engineer"
            category_filter (str): Optional category filter like "Job Description"
            
        Returns:
            Dict: Containing "answer" and list of retrieved "sources"
        """
        if not self.vectorstore:
            return {"answer": "Vector database not initialized.", "sources": []}
            
        # 1. Setup Retrieval Filter
        search_kwargs = {"k": 4}
        if category_filter and category_filter != "All Categories":
            search_kwargs["filter"] = {"category": category_filter}
            
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # 2. Retrieve Relevant Chunks
        try:
            retrieved_docs = retriever.invoke(question)
        except Exception as e:
            return {
                "answer": f"⚠️ **RAG Error**: Failed to retrieve relevant documents. This is typically caused by an invalid or expired API Key.\n\n**Details**: {str(e)}\n\nPlease double check your API key in the sidebar settings and verify it is correct.",
                "sources": []
            }
        
        # Format the context
        context_str = ""
        sources = []
        for i, doc in enumerate(retrieved_docs):
            context_str += f"--- Document: {doc.metadata.get('source', 'Unknown')} (Category: {doc.metadata.get('category', 'General')}) ---\n"
            context_str += f"{doc.page_content}\n\n"
            
            # Save source info to present in the UI
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "General"),
                "content": doc.page_content,
                "index": i + 1
            })
            
        # 3. Build Dynamic System Prompt
        role_info = ROLE_PROMPTS.get(target_role, {
            "focus": "general data and AI careers.",
            "tone": "encouraging, informative, and professional."
        })
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""You are a professional AI Career Mentor specializing in helping graduates and job seekers prepare for the **{target_role}** role.
Your tone should be: {role_info['tone']}.
Your advice and analysis should align with the core expectations of a {target_role}, which are: {role_info['focus']}.

Use the provided retrieved context documents (Job Descriptions, Resume Tips, and Interview Experiences) to answer the user's question. 
- You must tailor your advice to support the **{target_role}** path.
- Provide actionable, specific feedback, citing the sources you use. E.g. "[resume_tips_general.txt] recommended..."
- If the retrieved context contains conflicting information or is missing details, address it honestly, but give high-quality industry advice relevant to the {target_role} career track.
- If no documents are currently loaded, or if they don't contain any relevant facts, politely inform the user that you are providing general **{target_role}** advice because there are no specific matching documents in the database.

Retrieved Context:
{{context}}"""),
            ("human", "{question}")
        ])
        
        # 4. Invoke LLM
        messages = prompt_template.format_messages(context=context_str, question=question)
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}\n\nPlease check your API Key and internet connection."
            
        return {
            "answer": answer,
            "sources": sources
        }

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieves metadata of all unique files currently stored in Chroma.
        """
        try:
            data = self.vectorstore.get()
            metadatas = data.get("metadatas", [])
            
            unique_docs = {}
            for meta in metadatas:
                if not meta:
                    continue
                filename = meta.get("source", "Unknown")
                category = meta.get("category", "General")
                uploaded_at = meta.get("uploaded_at", "N/A")
                
                if filename not in unique_docs:
                    unique_docs[filename] = {
                        "filename": filename,
                        "category": category,
                        "chunks": 1,
                        "uploaded_at": uploaded_at
                    }
                else:
                    unique_docs[filename]["chunks"] += 1
                    
            return list(unique_docs.values())
        except Exception as e:
            print(f"Error listing database documents: {e}")
            return []

    def clear_database(self) -> bool:
        """
        Deletes the Chroma DB collection and clears local folder contents.
        """
        try:
            # Recreate vectorstore to close current database locks
            self.vectorstore = None
            
            # Wait briefly and remove the directory
            if os.path.exists(self.db_dir):
                shutil.rmtree(self.db_dir)
                
            # Reinitialize the Chroma DB cleanly
            os.makedirs(self.db_dir, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
