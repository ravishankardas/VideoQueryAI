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
        print(f"collection_name: {collection_name}")
        collection = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
    
    return collection

def add_chunks_to_chromadb(chunks_file: str, collection_name="youtube_videos"):
    """Load chunks and add them to ChromaDB - with proper error handling"""
    
    try:
        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from file: {chunks_file}")
        
        # Setup ChromaDB with the specific collection name
        collection = setup_chromadb(collection_name)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        print("Processing chunks...")
        for i, chunk in enumerate(chunks):
            # Validate chunk data
            if 'unique_id' not in chunk:
                print(f"Warning: Chunk {i} missing unique_id")
                continue
            if 'embedding' not in chunk:
                print(f"Warning: Chunk {i} missing embedding")
                continue
            if 'text' not in chunk:
                print(f"Warning: Chunk {i} missing text")
                continue
                
            ids.append(str(chunk['unique_id']))  # Ensure string
            
            # Validate and fix embedding
            embedding = chunk['embedding']
            if isinstance(embedding, list):
                # Ensure all values are floats
                embedding = [float(x) for x in embedding]
            else:
                print(f"Warning: Chunk {i} has invalid embedding format")
                continue
                
            embeddings.append(embedding)
            documents.append(str(chunk['text']))  # Ensure string
            
            # Clean metadata - ChromaDB is picky about data types
            metadata = {
                'chunk_id': int(chunk.get('chunk_id', i)),
                'video_title': str(chunk.get('video_title', 'Unknown')),
                'video_url': str(chunk.get('video_url', '')),
                'uploader': str(chunk.get('uploader', 'Unknown')),
                'word_count': int(chunk.get('word_count', 0)),
                'start_time': str(chunk.get('start_time', 'Unknown')),
                'end_time': str(chunk.get('end_time', 'Unknown')),
                'start_seconds': float(chunk.get('start_seconds', 0)),
                'end_seconds': float(chunk.get('end_seconds', 0)),
                'duration': float(chunk.get('duration', 0))
            }
            metadatas.append(metadata)
            
            # Debug first chunk
            if i == 0:
                print(f"First chunk validation:")
                print(f"  ID: {ids[0]} (type: {type(ids[0])})")
                print(f"  Embedding length: {len(embeddings[0])}")
                print(f"  Document length: {len(documents[0])}")
                print(f"  Metadata keys: {list(metadata.keys())}")
        
        print(f"Processed {len(ids)} valid chunks")
        
        # Validate lengths
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            print("Error: Lengths don't match!")
            print(f"ids: {len(ids)}, embeddings: {len(embeddings)}, documents: {len(documents)}, metadatas: {len(metadatas)}")
            return None
        
        if len(ids) == 0:
            print("Error: No valid chunks to add!")
            return None
        
        # Add in smaller batches to avoid memory issues
        batch_size = 50
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(ids))
            
            print(f"Adding batch {batch_num + 1}/{total_batches} (items {start_idx}-{end_idx-1})")
            
            try:
                collection.add(
                    ids=ids[start_idx:end_idx],
                    embeddings=embeddings[start_idx:end_idx],
                    documents=documents[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx]
                )
                print(f"  ✅ Batch {batch_num + 1} added successfully")
                
            except Exception as batch_error:
                print(f"  ❌ Error in batch {batch_num + 1}: {batch_error}")
                print(f"  Error type: {type(batch_error).__name__}")
                
                # Try to debug the specific error
                if "dimension" in str(batch_error).lower():
                    print(f"  Embedding dimension issue detected")
                    print(f"  First embedding shape: {len(embeddings[start_idx])}")
                elif "metadata" in str(batch_error).lower():
                    print(f"  Metadata issue detected")
                    print(f"  First metadata: {metadatas[start_idx]}")
                
                import traceback
                traceback.print_exc()
                return None
        
        print(f"✅ Successfully added {len(ids)} chunks to collection: {collection_name}")
        return collection
        
    except Exception as e:
        print(f"❌ Error in add_chunks_to_chromadb: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def build_bm25_index(collection_name="youtube_videos"):
    """Build BM25 index for the specified collection"""
    
    global _bm25_cache
    if collection_name in _bm25_cache:
        print(f"BM25 index already exists for collection: {collection_name}")
        return _bm25_cache[collection_name]
    
    try:
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
    except Exception as e:
        print(f"❌ Error building BM25 index: {e}")
        return None

def search_videos(query: str, collection_name="youtube_videos", n_results=3):
    """Enhanced search with vector + BM25 hybrid approach"""
    
    try:
        collection = setup_chromadb(collection_name)
        
        # 1. Vector search (semantic)
        vector_results = collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get more candidates
        )
        
        # 2. BM25 search (keyword)
        bm25_data = build_bm25_index(collection_name)
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
                    combined_docs[doc_id]['score'] += 0.5
        
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
    except Exception as e:
        print(f"❌ Error in search_videos: {e}")
        return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

def answer_question(query: str, collection_name="youtube_videos"):
    """Enhanced Q&A with citation tracking"""
    
    try:
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
            citation = f"Source {i+1}: '{metadata.get('video_title', 'Unknown')}'"
            if metadata.get('start_time'):
                citation += f" at {metadata['start_time']}"
            if metadata.get('video_url') and metadata.get('start_seconds'):
                timestamped_url = f"{metadata['video_url']}&t={int(metadata.get('start_seconds', 0))}s"
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
    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating answer: {e}"