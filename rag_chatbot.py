import os
from dotenv import load_dotenv

load_dotenv()

class RAGChatbot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("WARNING: No API key found. Running in DEMO mode.\n")
            self.demo_mode = True
            self.documents = []
        else:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_openai import ChatOpenAI
            from langchain.chains import RetrievalQA
            
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
            self.vectorstore = None
            self.qa_chain = None
            self.demo_mode = False
    
    def load_documents(self, pdf_path):
        if self.demo_mode:
            with open(pdf_path, 'rb') as f:
                self.documents.append({"path": pdf_path, "size": len(f.read())})
            return 5
        
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        from langchain.chains import RetrievalQA
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        return len(chunks)
    
    def ask(self, question):
        if not self.documents and not self.qa_chain:
            return "Please upload a document first."
        
        if self.demo_mode:
            return f"[DEMO] Based on the uploaded document, here's a sample answer. Add your OpenAI API key to .env for real RAG-based answers."
        
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

def main():
    chatbot = RAGChatbot()
    
    print("=== RAG Document Chatbot ===\n")
    
    while True:
        print("\n1. Upload PDF")
        print("2. Ask Question")
        print("3. Exit\n")
        
        choice = input("Choose option (1-3): ")
        
        if choice == "3":
            print("Goodbye!")
            break
        
        if choice == "1":
            pdf_path = input("\nEnter PDF path: ")
            if os.path.exists(pdf_path):
                print("\nLoading document...")
                try:
                    chunks = chatbot.load_documents(pdf_path)
                    print(f"Document loaded! Created {chunks} chunks.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("File not found.")
        
        elif choice == "2":
            question = input("\nYour question: ")
            print("\nThinking...\n")
            answer = chatbot.ask(question)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
