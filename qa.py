import chromadb
import json
from typing import List, Dict
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi # type: ignore
import numpy as np

load_dotenv()

_bm25_cache = {}

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


def build_bm25_index(collection_name="youtube_videos"):
    """Build BM25 index for the specified collection"""
    
    global _bm25_cache
    if collection_name in _bm25_cache:
        print(f"BM25 index already exists for collection: {collection_name}")
        return _bm25_cache[collection_name]
    

    
    # Setup ChromaDB
    collection = setup_chromadb(collection_name)
    all_results = collection.get()

    if not all_results['documents']:
        print(f"No documents found in collection: {collection_name}")
        return None
    tokenized_docs = [doc.lower().split() for doc in all_results['documents']]
    bm25 = BM25Okapi(tokenized_docs)
    _bm25_cache[collection_name] = {
        'bm25': bm25,
        'documents': all_results['documents'],
        'metadatas': all_results['metadatas'],
        'ids': all_results['ids']
    }
    return _bm25_cache[collection_name]


def add_chunks_to_chromadb(chunks_file: str, collection_name="youtube_videos"):
    """Load chunks and add them to ChromaDB - now uses separate collections"""
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Setup ChromaDB with the specific collection name
    collection = setup_chromadb(collection_name)
    
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
    """Enhanced search with vector + BM25 hybrid approach"""
    
    collection = setup_chromadb(collection_name)
    
    # 1. Vector search (semantic)
    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results * 2  # Get more candidates
    )
    
    # 2. BM25 search (keyword)
    bm25_data = build_bm25_index(collection_name)
    # print(bm25_data)

    bm25_results = {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}
    
    if bm25_data:
        tokenized_query = query.lower().split()
        bm25_scores = bm25_data['bm25'].get_scores(tokenized_query)
        
        # Get top BM25 results
        top_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
        
        bm25_ids = [bm25_data['ids'][i] for i in top_indices if bm25_scores[i] > 0]
        bm25_docs = [bm25_data['documents'][i] for i in top_indices if bm25_scores[i] > 0]
        bm25_metas = [bm25_data['metadatas'][i] for i in top_indices if bm25_scores[i] > 0]
       
        
        bm25_results = {
            'ids': [bm25_ids],
            'documents': [bm25_docs], 
            'metadatas': [bm25_metas]
        }
    
    # 3. Combine results (simple fusion)
    combined_docs = {}
    
    # Add vector results
    if vector_results['documents'][0]: # type: ignore
        for i, doc_id in enumerate(vector_results['ids'][0]):
            combined_docs[doc_id] = {
                'text': vector_results['documents'][0][i], # type: ignore
                'metadata': vector_results['metadatas'][0][i], # type: ignore
                'score': 1.0 - vector_results['distances'][0][i]  # type: ignore
            }
    
    # Boost scores for BM25 matches
    if bm25_results['documents'][0]:
        for doc_id in bm25_results['ids'][0]:
            if doc_id in combined_docs:
                combined_docs[doc_id]['score'] += 0.5  # Boost hybrid matches if document appears in both results
            # Note: We don't add BM25-only results to keep it simple
    
    # 4. Sort and return top results
    sorted_results = sorted(combined_docs.items(), 
                          key=lambda x: x[1]['score'], 
                          reverse=True)[:n_results]
    
    # Format back to original structure
    final_ids = [item[0] for item in sorted_results]
    final_docs = [item[1]['text'] for item in sorted_results]
    final_metas = [item[1]['metadata'] for item in sorted_results]
    final_distances = [1.0 - item[1]['score'] for item in sorted_results]
    
    return {
        'ids': [final_ids],
        'documents': [final_docs],
        'metadatas': [final_metas],
        'distances': [final_distances]
    }

def answer_question(query: str, collection_name="youtube_videos"):
    """Enhanced Q&A with citation tracking"""
    
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    import re
    
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
    # Remove source references from the answer
    answer = re.sub(r'\[Source \d+\]', '', answer).strip()
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