import chromadb
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

def setup_chromadb(collection_name="youtube_videos"):
    """Initialize ChromaDB client and collection"""
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        # print(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        # print(f"Created new collection: {collection_name}")
    
    return collection

def add_chunks_to_chromadb(chunks_file: str, collection_name="youtube_videos"):
    """Load chunks and add them to ChromaDB - now uses separate collections"""
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Setup ChromaDB with the specific collection name
    collection = setup_chromadb(collection_name)
    
    # Rest stays the same...
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk['unique_id'])
        embeddings.append(chunk['embedding'])
        documents.append(chunk['text'])
        
        metadata = {
            'chunk_id': chunk['chunk_id'],
            'video_title': chunk['video_title'],
            'video_url': chunk['video_url'],
            'uploader': chunk['uploader'],
            'word_count': chunk['word_count'],
            
            # ADD THESE LINES FOR CITATION TRACKING:
            'start_time': chunk.get('start_time', 'Unknown'),
            'end_time': chunk.get('end_time', 'Unknown'),
            'start_seconds': chunk.get('start_seconds', 0),
            'end_seconds': chunk.get('end_seconds', 0),
            'duration': chunk.get('duration', 0)
        }
        metadatas.append(metadata)
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Added {len(chunks)} chunks to collection: {collection_name}")
    return collection

def search_videos(query: str, collection_name="youtube_videos", n_results=3):
    """Search for relevant video chunks"""
    
    # Setup ChromaDB
    collection = setup_chromadb(collection_name)
    
    # Search
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results

def answer_question(query: str, collection_name="youtube_videos"):
    """Enhanced Q&A with citation tracking"""
    
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    # Search for relevant chunks
    results = search_videos(query, collection_name, n_results=3)
    
    if not results['documents'][0]: #type: ignore
        return "No relevant information found."
    
    # Prepare context with citation info
    context_chunks = []
    citations = []
    
    for i, doc in enumerate(results['documents'][0]): #type: ignore
        metadata = results['metadatas'][0][i] #type: ignore
        
        # Create citation
        citation = f"Source {i+1}: '{metadata['video_title']}' at {metadata.get('start_time', 'Unknown')}-{metadata.get('end_time', 'Unknown')}"
        if metadata.get('video_url') and metadata.get('start_seconds'):
            timestamped_url = f"{metadata['video_url']}&t={int(metadata.get('start_seconds', 0))}s" #type: ignore
            citation += f" ({timestamped_url})"
        
        citations.append(citation)
        context_chunks.append(f"[Source {i+1}] {doc}")
    
    context = "\n\n".join(context_chunks)
    
    # Generate answer with citations
    llm = ChatOpenAI(temperature=0)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following video transcript context, answer the question and include source references.

Context:
{context}

Question: {question}

Answer with citations (use [Source X] format):"""
    )
    
    response = llm.invoke(prompt.format(context=context, question=query))
    answer = response.content.strip() # type: ignore
    
    # Append full citations
    full_response = f"{answer}\n\n" + "="*50 + "\nSources:\n" + "\n".join(citations)
    
    return full_response

# # Example usage
# if __name__ == "__main__":
    
#     # Step 1: Add chunks to ChromaDB
#     chunks_file = "./vectordb_data/video_chunks.json"
    
#     if os.path.exists(chunks_file):
#         collection = add_chunks_to_chromadb(chunks_file)
        
#         # Step 2: Test search
#         query = "What are the main points?"
#         results = search_videos(query)
        
#         print(f"\nSearch results for: '{query}'")
#         for i, doc in enumerate(results['documents'][0]):
#             metadata = results['metadatas'][0][i]
#             print(f"\n{i+1}. From: {metadata['video_title']}")
#             print(f"Text: {doc[:200]}...")
        
#         # Step 3: Test Q&A
#         answer = answer_question(query)
#         print(f"\nAnswer:\n{answer}")
    
#     else:
#         print(f"Chunks file not found: {chunks_file}")
#         print("Please run the chunking step first.")