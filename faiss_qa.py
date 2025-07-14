import json
import os
import numpy as np
import faiss #type: ignore
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class FAISSVectorStore:
    """FAISS-based vector storage to replace ChromaDB"""
    
    def __init__(self, storage_dir="./faiss_vectordb"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
    
    def add_chunks(self, chunks_file: str, collection_name: str):
        """Store chunks using FAISS index"""
        try:
            print(f"📁 Loading chunks from: {chunks_file}")
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            print(f"✅ Loaded {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print("❌ No chunks to process")
                return False
            
            # Extract embeddings and metadata
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Get embedding
                    embedding = chunk.get('embedding', [])
                    if not embedding or len(embedding) != self.dimension:
                        print(f"Warning: Chunk {i} has invalid embedding, skipping")
                        continue
                    
                    embeddings.append(embedding)
                    documents.append(chunk.get('text', ''))
                    ids.append(chunk.get('unique_id', f'chunk_{i}'))
                    
                    # Clean metadata
                    metadata = {
                        'video_title': chunk.get('video_title', 'Unknown'),
                        'video_url': chunk.get('video_url', ''),
                        'start_time': chunk.get('start_time', 'Unknown'),
                        'end_time': chunk.get('end_time', 'Unknown'),
                        'start_seconds': float(chunk.get('start_seconds', 0)),
                        'uploader': chunk.get('uploader', 'Unknown'),
                        'word_count': int(chunk.get('word_count', 0)),
                        'chunk_id': int(chunk.get('chunk_id', i))
                    }
                    metadatas.append(metadata)
                    
                except Exception as e:
                    print(f"Warning: Error processing chunk {i}: {e}")
                    continue
            
            if len(embeddings) == 0:
                print("❌ No valid embeddings found")
                return False
            
            print(f"📊 Processing {len(embeddings)} valid chunks")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            print(f"✅ Created embeddings array: {embeddings_array.shape}")
            
            # Create FAISS index
            print("🔧 Creating FAISS index...")
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            index.add(embeddings_array)
            print(f"✅ Added {index.ntotal} vectors to FAISS index")
            
            # Save everything
            collection_dir = os.path.join(self.storage_dir, collection_name)
            os.makedirs(collection_dir, exist_ok=True)
            
            # Save FAISS index
            index_file = os.path.join(collection_dir, "index.faiss")
            faiss.write_index(index, index_file)
            
            # Save metadata
            metadata_file = os.path.join(collection_dir, "metadata.pkl")
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': documents,
                    'metadatas': metadatas,
                    'ids': ids,
                    'count': len(embeddings)
                }, f)
            
            # Also save as JSON for debugging
            json_file = os.path.join(collection_dir, "metadata.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': documents,
                    'metadatas': metadatas,
                    'ids': ids,
                    'count': len(embeddings)
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved FAISS index and metadata for collection: {collection_name}")
            print(f"   Index file: {index_file}")
            print(f"   Metadata: {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in FAISS storage: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, query: str, collection_name: str, n_results=3) -> Dict:
        """Search using FAISS index"""
        try:
            collection_dir = os.path.join(self.storage_dir, collection_name)
            index_file = os.path.join(collection_dir, "index.faiss")
            metadata_file = os.path.join(collection_dir, "metadata.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                print(f"❌ Collection not found: {collection_name}")
                return {'documents': [[]], 'metadatas': [[]], 'ids': [[]], 'distances': [[]]}
            
            # Load FAISS index
            index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
            
            documents = data['documents']
            metadatas = data['metadatas']
            ids = data['ids']
            
            print(f"🔍 Searching {len(documents)} chunks for: '{query}'")
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            query_array = np.array(query_embedding, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = index.search(query_array, min(n_results, len(documents)))
            
            # Format results
            top_indices = indices[0]
            top_scores = scores[0]
            
            # Convert to distances (ChromaDB style)
            distances = [1.0 - score for score in top_scores]
            
            results = {
                'documents': [[documents[i] for i in top_indices if i < len(documents)]],
                'metadatas': [[metadatas[i] for i in top_indices if i < len(metadatas)]],
                'ids': [[ids[i] for i in top_indices if i < len(ids)]],
                'distances': [distances]
            }
            
            print(f"✅ Found {len(top_indices)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error searching FAISS: {e}")
            import traceback
            traceback.print_exc()
            return {'documents': [[]], 'metadatas': [[]], 'ids': [[]], 'distances': [[]]}

# Global instance
vector_store = FAISSVectorStore()

# Drop-in replacement functions for your existing code
def add_chunks_to_chromadb(chunks_file: str, collection_name="youtube_videos"):
    """Replacement function that uses FAISS"""
    print(f"🚀 Using FAISS instead of ChromaDB")
    return vector_store.add_chunks(chunks_file, collection_name)

def search_videos(query: str, collection_name="youtube_videos", n_results=3):
    """FAISS-based search function"""
    return vector_store.search(query, collection_name, n_results)

def answer_question(query: str, collection_name="youtube_videos"):
    """Q&A using FAISS search"""
    try:
        # Search for relevant chunks
        results = search_videos(query, collection_name, n_results=3)
        
        if not results['documents'][0]:
            return "No relevant information found."
        
        # Prepare context with citation info
        context_chunks = []
        citations = []
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            
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
        
        # Try to use OpenAI for better answers
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            import re
            
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
            
        except Exception as llm_error:
            print(f"Warning: LLM failed ({llm_error}), using simple response")
            
            # Fallback: simple concatenation
            simple_answer = f"Based on the video content:\n\n{' '.join(results['documents'][0][:2])}"
            full_response = f"{simple_answer}\n\n" + "="*50 + "\nSources:\n" + "\n".join(citations)
            return full_response
        
    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        return f"Error generating answer: {e}"

# # Test function
# def test_faiss_setup():
#     """Test if FAISS is working"""
#     try:
#         import faiss
#         print(f"✅ FAISS version: {faiss.__version__}")
        
#         # Create a simple test
#         d = 384
#         index = faiss.IndexFlatIP(d)
#         print(f"✅ FAISS index created successfully")
        
#         return True
#     except ImportError:
#         print("❌ FAISS not installed. Install with: pip install faiss-cpu")
#         return False
#     except Exception as e:
#         print(f"❌ FAISS error: {e}")
#         return False

# if __name__ == "__main__":
#     test_faiss_setup()