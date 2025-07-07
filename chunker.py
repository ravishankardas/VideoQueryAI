import json
import hashlib
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def find_chunk_timestamps(chunk_text: str, segments: List[Dict]) -> Dict:
    """
    Find the timestamp range for a chunk by matching it to transcript segments
    """
    
    if not segments:
        return {
            'start_time': '0:00:00',
            'end_time': '0:00:00', 
            'start_seconds': 0,
            'end_seconds': 0
        }
    
    # Clean chunk text
    chunk_words = chunk_text.lower().strip().split()
    if len(chunk_words) < 3:
        # Use first segment for very short chunks
        first_segment = segments[0]
        return {
            'start_time': first_segment.get('start_time', '0:00:00'),
            'end_time': first_segment.get('end_time', '0:00:00'),
            'start_seconds': first_segment.get('start_seconds', 0),
            'end_seconds': first_segment.get('end_seconds', 0)
        }
    
    # Try to match chunk to segments by finding best overlap
    # Look for first few words and last few words of chunk
    first_phrase = ' '.join(chunk_words[:5]).lower()
    last_phrase = ' '.join(chunk_words[-5:]).lower()
    
    start_segment = None
    end_segment = None
    best_start_score = 0
    best_end_score = 0
    
    # Find segment that best matches the start of our chunk
    for segment in segments:
        segment_text = segment.get('text', '').lower().strip()
        
        # Check how well this segment matches our chunk start
        if first_phrase in segment_text:
            score = len(first_phrase)  # Simple scoring
            if score > best_start_score:
                best_start_score = score
                start_segment = segment
        
        # Also check for partial matches
        words_match = sum(1 for word in chunk_words[:10] if word in segment_text.split())
        if words_match >= 3 and words_match > best_start_score:
            best_start_score = words_match
            start_segment = segment
    
    # Find segment that best matches the end of our chunk  
    for segment in segments:
        segment_text = segment.get('text', '').lower().strip()
        
        # Check how well this segment matches our chunk end
        if last_phrase in segment_text:
            score = len(last_phrase)
            if score > best_end_score:
                best_end_score = score
                end_segment = segment
        
        # Also check for partial matches
        words_match = sum(1 for word in chunk_words[-10:] if word in segment_text.split())
        if words_match >= 3 and words_match > best_end_score:
            best_end_score = words_match
            end_segment = segment
    
    # If we found good matches, use them
    if start_segment and end_segment:
        return {
            'start_time': start_segment.get('start_time', '0:00:00'),
            'end_time': end_segment.get('end_time', '0:00:00'),
            'start_seconds': start_segment.get('start_seconds', 0),
            'end_seconds': end_segment.get('end_seconds', 0)
        }
    
    # Fallback: try to find any segment that contains chunk words
    for segment in segments:
        segment_text = segment.get('text', '').lower().strip()
        words_in_segment = sum(1 for word in chunk_words if word in segment_text.split())
        
        # If segment contains good portion of chunk words, use it
        if words_in_segment >= min(5, len(chunk_words) // 2):
            return {
                'start_time': segment.get('start_time', '0:00:00'),
                'end_time': segment.get('end_time', '0:00:00'),
                'start_seconds': segment.get('start_seconds', 0),
                'end_seconds': segment.get('end_seconds', 0)
            }
    
    # Final fallback: use first segment
    first_segment = segments[0]
    return {
        'start_time': first_segment.get('start_time', '0:00:00'),
        'end_time': first_segment.get('end_time', '0:00:00'),
        'start_seconds': first_segment.get('start_seconds', 0),
        'end_seconds': first_segment.get('end_seconds', 0)
    }

def process_transcription_for_rag_langchain(transcription_file: str, video_metadata: Dict) -> List[Dict]:
    """
    Enhanced pipeline with citation tracking - handles both fast and slow transcript formats
    
    Args:
        transcription_file: Path to transcription JSON file
        video_metadata: Video information from download
    
    Returns:
        List of processed chunks with citation tracking info
    """
    
    # Load transcription data
    with open(transcription_file, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    
    # Debug: Print the structure
    print(f"📋 Transcript keys: {list(transcription_data.keys())}")
    print(f"📝 Source type: {transcription_data.get('source_type', 'unknown')}")
    
    # Get full text and segments - handle both formats
    full_text = transcription_data.get('full_text', '')
    segments = transcription_data.get('segments', [])
    
    # Debug segments
    if segments:
        print(f"🎬 Total segments: {len(segments)}")
        print(f"🔍 First segment keys: {list(segments[0].keys())}")
        print(f"📍 First segment example: {segments[0]}")
    else:
        print("❌ No segments found!")
    
    # Initialize LangChain text splitter for semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Characters per chunk
        chunk_overlap=200,      # Overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs, sentences, etc.
    )
    
    # Split text into chunks
    text_chunks = text_splitter.split_text(full_text)
    
    print(f"Created {len(text_chunks)} chunks using LangChain")
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    
    # Process chunks with timestamp mapping
    print("Mapping chunks to timestamps...")
    chunks = []
    for i, (text_chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        
        # Find timestamps for this chunk
        timestamp_info = find_chunk_timestamps(text_chunk, segments)
        
        # Debug timestamp mapping for first few chunks
        if i < 3:
            print(f"\n🔍 Chunk {i+1} timestamp mapping:")
            print(f"   Text preview: {text_chunk[:50]}...")
            print(f"   Raw timestamp_info: {timestamp_info}")
            print(f"   Mapped to: {timestamp_info['start_time']} - {timestamp_info['end_time']}")
            print(f"   Seconds: {timestamp_info['start_seconds']} - {timestamp_info['end_seconds']}")
        
        chunk = {
            'chunk_id': i,
            'text': text_chunk,
            'embedding': embedding.tolist(),
            'word_count': len(text_chunk.split()),
            
            # Citation tracking info - DEBUG these values
            'start_time': timestamp_info['start_time'],
            'end_time': timestamp_info['end_time'], 
            'start_seconds': timestamp_info['start_seconds'],
            'end_seconds': timestamp_info['end_seconds'],
            'duration': timestamp_info['end_seconds'] - timestamp_info['start_seconds'],
            
            # Video metadata
            'video_title': video_metadata.get('title', 'Unknown'),
            'video_url': video_metadata.get('url', ''),
            'uploader': video_metadata.get('uploader', 'Unknown'),
            'video_duration': video_metadata.get('duration', 0),
            'upload_date': video_metadata.get('upload_date', ''),
            
            # Unique ID
            'unique_id': hashlib.md5(f"{video_metadata.get('url', '')}_{i}_{text_chunk[:50]}".encode()).hexdigest()
        }
        
        # Debug the actual chunk for first few
        if i < 2:
            print(f"\n📦 Chunk {i+1} keys: {list(chunk.keys())}")
            print(f"📦 Chunk {i+1} timestamps in chunk: {chunk['start_time']} - {chunk['end_time']}")
        
        chunks.append(chunk)
    
    # Print summary with citation info
    avg_words = sum(chunk['word_count'] for chunk in chunks) / len(chunks)
    avg_duration = sum(chunk['duration'] for chunk in chunks) / len(chunks)
    
    print(f"\nChunking Summary:")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Average duration per chunk: {avg_duration:.1f} seconds")
    print(f"Citation tracking: ✅ Enabled")
    
    # Show example chunk with citation
    if chunks:
        example = chunks[0]
        print(f"\nExample chunk with citation:")
        print(f"Text: {example['text'][:100]}...")
        print(f"Timestamp: {example['start_time']} - {example['end_time']}")
        print(f"Video: {example['video_title']}")
    
    return chunks

def save_chunks_for_vectordb(chunks: List[Dict], output_file: str):
    """Save processed chunks to file for vector database ingestion"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Chunks with citation tracking saved to: {output_file}")

def create_citation_from_chunk(chunk: Dict) -> str:
    """
    Create a properly formatted citation from a chunk
    
    Args:
        chunk: Chunk dictionary with citation info
        
    Returns:
        Formatted citation string
    """
    video_title = chunk.get('video_title', 'Unknown Video')
    start_time = chunk.get('start_time', '0:00:00')
    end_time = chunk.get('end_time', '0:00:00')
    video_url = chunk.get('video_url', '')
    
    # Create YouTube URL with timestamp
    if video_url and 'youtube.com' in video_url:
        start_seconds = chunk.get('start_seconds', 0)
        timestamped_url = f"{video_url}&t={int(start_seconds)}s"
        citation = f"Source: '{video_title}' at {start_time}-{end_time} ({timestamped_url})"
    else:
        citation = f"Source: '{video_title}' at {start_time}-{end_time}"
    
    return citation

# Example usage and testing
# if __name__ == "__main__":
#     # Test the citation creation
#     test_chunk = {
#         'video_title': 'How to Build RAG Systems',
#         'start_time': '0:02:30',
#         'end_time': '0:02:45', 
#         'video_url': 'https://www.youtube.com/watch?v=example123',
#         'start_seconds': 150,
#         'text': 'This is an example chunk about RAG systems...'
#     }
    
#     citation = create_citation_from_chunk(test_chunk)
#     print(f"Example citation: {citation}")