import os
import uuid
import json
from audio_downloader import download_youtube_audio
from audio_transcriber import (
    transcribe_webm_directly, 
    format_and_save_transcription,
    process_youtube_fast,
    get_video_metadata_fast
)
from chunker import process_transcription_for_rag_langchain, save_chunks_for_vectordb
from qa import add_chunks_to_chromadb, answer_question

class YouTubeRAGPipeline:
    def __init__(self, base_dir="./rag_data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def process_video(self, youtube_url):
        """Complete pipeline with caption fallback"""
        
        # Check if this URL was already processed
        existing_video = self._find_existing_video(youtube_url)
        if existing_video:
            print(f"✅ Video already processed: {existing_video['title']}")
            print(f"UUID: {existing_video['uuid']}")
            return existing_video
        
        # Generate unique ID for this video
        video_uuid = str(uuid.uuid4())[:8]
        print(f"Processing video with UUID: {video_uuid}")
        
        # Create directories for this video
        video_dir = os.path.join(self.base_dir, video_uuid)
        downloads_dir = os.path.join(video_dir, "downloads")
        metadata_dir = os.path.join(video_dir, "metadata")
        transcripts_dir = os.path.join(video_dir, "transcripts")
        chunks_dir = os.path.join(video_dir, "chunks")
        
        try:
            # Step 1: Try captions first
            print("\n=== Trying captions first ===")
            transcript_data, video_metadata = process_youtube_fast(youtube_url)
            
            if transcript_data and video_metadata:
                # SUCCESS: Use captions
                print("✅ Found captions!")
                
                os.makedirs(transcripts_dir, exist_ok=True)
                os.makedirs(metadata_dir, exist_ok=True)
                
                # Save transcript
                transcription_file = os.path.join(transcripts_dir, "transcript.json")
                formatted_transcript = {
                    'language': transcript_data['language'],
                    'full_text': transcript_data['full_text'],
                    'segments': transcript_data['segments'],
                    'total_duration': transcript_data['total_duration'],
                    'source_type': 'captions'
                }
                
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    json.dump(formatted_transcript, f, indent=2, ensure_ascii=False)
                
                # Save metadata
                video_metadata['uuid'] = video_uuid
                metadata_file = os.path.join(metadata_dir, "metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(video_metadata, f, indent=4, ensure_ascii=False)
                
                audio_file = None
                
            else:
                # FALLBACK: Download and transcribe
                print("❌ No captions. Downloading audio...")
                
                audio_file = download_youtube_audio(
                    youtube_url, 
                    output_path=downloads_dir, 
                    metadata_path=metadata_dir
                )
                
                if not audio_file:
                    print("Failed to download audio")
                    return None
                
                print("Transcribing audio...")
                result = transcribe_webm_directly(audio_file)
                
                if not result:
                    print("Failed to transcribe audio")
                    return None
                
                # Save transcription
                os.makedirs(transcripts_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                transcription_file = os.path.join(transcripts_dir, f"{base_name}_transcription.json")
                
                formatted_result = format_and_save_transcription(result, audio_file)
                
                # Move transcription to correct location
                original_transcript = f"./transcripts/{base_name}_transcription.json"
                if os.path.exists(original_transcript):
                    os.rename(original_transcript, transcription_file)
                
                # Load metadata
                metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
                metadata_file = os.path.join(metadata_dir, metadata_files[0])
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    video_metadata = json.load(f)
                
                video_metadata['uuid'] = video_uuid
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(video_metadata, f, indent=4, ensure_ascii=False)
            
            # Create chunks
            print("Creating chunks...")
            os.makedirs(chunks_dir, exist_ok=True)
            
            chunks = process_transcription_for_rag_langchain(transcription_file, video_metadata)
            chunks_file = os.path.join(chunks_dir, "video_chunks.json")
            save_chunks_for_vectordb(chunks, chunks_file)
            
            # Add to ChromaDB with separate collection for this video
            print("Adding to database...")
            collection_name = f"video_{video_uuid}"
            add_chunks_to_chromadb(chunks_file, collection_name=collection_name)
            
            print(f"\n✅ Done! Video: {video_metadata['title']}")
            
            return {
                'uuid': video_uuid,
                'metadata': video_metadata,
                'audio_file': audio_file,
                'transcription_file': transcription_file,
                'chunks_file': chunks_file
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    
    def ask_question(self, question, video_uuid=None):
        """Ask a question - either to a specific video or all videos"""
        if video_uuid:
            collection_name = f"video_{video_uuid}"
            return answer_question(question, collection_name=collection_name)
        else:
            # Search across all videos
            videos = self.list_videos()
            if not videos:
                return "No videos processed yet."
            
            # Try each video and combine results
            all_results = []
            for video in videos:
                collection_name = f"video_{video['uuid']}"
                try:
                    result = answer_question(question, collection_name=collection_name)
                    if result and result != "No relevant information found.":
                        all_results.append(f"From '{video['title']}':\n{result}")
                except:
                    continue
            
            if all_results:
                return "\n\n" + "="*50 + "\n\n".join(all_results)
            else:
                return "No relevant information found in any video."
    
    def list_videos(self):
        """List all processed videos"""
        videos = []
        if not os.path.exists(self.base_dir):
            return videos
        
        for video_uuid in os.listdir(self.base_dir):
            video_dir = os.path.join(self.base_dir, video_uuid)
            metadata_dir = os.path.join(video_dir, "metadata")
            
            if os.path.exists(metadata_dir):
                metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
                if metadata_files:
                    metadata_file = os.path.join(metadata_dir, metadata_files[0])
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    videos.append({
                        'uuid': video_uuid,
                        'title': metadata.get('title', 'Unknown'),
                        'uploader': metadata.get('uploader', 'Unknown'),
                        'url': metadata.get('url', '')
                    })
        
        return videos
    
    def _find_existing_video(self, youtube_url):
        """Check if a video with this URL was already processed"""
        videos = self.list_videos()
        for video in videos:
            if video['url'] == youtube_url:
                return video
        return None