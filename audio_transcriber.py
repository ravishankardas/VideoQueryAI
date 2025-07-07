import re
from youtube_transcript_api._api import YouTubeTranscriptApi
import json
import os
from datetime import timedelta
import whisper # type: ignore

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript_fast(youtube_url):
    """
    Fast transcript extraction using YouTube's existing captions
    Falls back to None if no captions available
    """
    
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return None
    
    try:
        print(f"🔍 Fetching transcript for video ID: {video_id}")
        
        # Try to get transcript (auto-generated or manual)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Prefer manual captions over auto-generated
        transcript = None
        try:
            # Try manual captions first
            transcript = transcript_list.find_manually_created_transcript(['en'])
            print("✅ Found manual captions")
        except:
            try:
                # Fall back to auto-generated
                transcript = transcript_list.find_generated_transcript(['en'])
                print("✅ Found auto-generated captions")
            except:
                # Try any available language
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    transcript = available_transcripts[0]
                    print(f"✅ Found transcript in {transcript.language}")
        
        if not transcript:
            print("❌ No transcripts available")
            return None
        
        # Fetch the transcript
        transcript_data = transcript.fetch()
        
        # Format the data
        formatted_transcript = {
            'language': transcript.language,
            'is_generated': transcript.is_generated,
            'segments': [],
            'full_text': '',
            'total_duration': 0
        }
        
        full_text_parts = []
        
        for item in transcript_data:
            segment = {
                'start_time': str(timedelta(seconds=int(item.start))),
                'end_time': str(timedelta(seconds=int(item.start + item.duration))),
                'start_seconds': item.start,
                'end_seconds': item.start + item.duration,
                'text': item.text.strip(),
                'duration': item.duration
            }
            
            formatted_transcript['segments'].append(segment)
            full_text_parts.append(item.text)
        
        formatted_transcript['full_text'] = ' '.join(full_text_parts)
        if formatted_transcript['segments']:
            formatted_transcript['total_duration'] = formatted_transcript['segments'][-1]['end_seconds']
        
        print(f"✅ Transcript extracted successfully!")
        print(f"Language: {formatted_transcript['language']}")
        print(f"Generated: {formatted_transcript['is_generated']}")
        print(f"Duration: {formatted_transcript['total_duration']:.1f} seconds")
        print(f"Segments: {len(formatted_transcript['segments'])}")
        
        return formatted_transcript
        
    except Exception as e:
        print(f"❌ Error fetching transcript: {str(e)}")
        return None

def transcribe_webm_directly(webm_file_path):
    """
    Transcribe WebM file directly with Whisper
    """
    
    # Check if file exists
    if not os.path.exists(webm_file_path):
        print(f"❌ File not found: {webm_file_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(webm_file_path) / (1024*1024)
    print(f"✅ Found WebM file: {os.path.basename(webm_file_path)}")
    print(f"File size: {file_size:.2f} MB")
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("small")
        print("✅ Model loaded")
        
        # Transcribe directly
        print("Transcribing WebM file (this may take a few minutes)...")
        result = model.transcribe(
            webm_file_path,
            verbose=False,
            word_timestamps=False,
            language=None,
            fp16=False
        )
        
        print("✅ Transcription completed!")
        return result
        
    except Exception as e:
        print(f"❌ Error from transcription: {e}")
        return None

def format_and_save_transcription(result, webm_file_path):
    """
    Format transcription and save to files
    """
    
    # Create output directory
    output_dir = "./transcripts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Format segments with timestamps
    formatted_segments = []
    for segment in result['segments']:
        start_time = str(timedelta(seconds=int(segment['start'])))
        end_time = str(timedelta(seconds=int(segment['end'])))
        
        formatted_segment = {
            'start_time': start_time,
            'end_time': end_time,
            'start_seconds': segment['start'],
            'end_seconds': segment['end'],
            'text': segment['text'].strip(),
            'confidence': segment.get('avg_logprob', 0)
        }
        formatted_segments.append(formatted_segment)
    
    # Create formatted result
    formatted_result = {
        'language': result['language'],
        'full_text': result['text'],
        'segments': formatted_segments,
        'total_duration': formatted_segments[-1]['end_seconds'] if formatted_segments else 0,
        'source_file': webm_file_path
    }
    
    # Save JSON transcription
    base_name = os.path.splitext(os.path.basename(webm_file_path))[0]
    json_file = os.path.join(output_dir, f"{base_name}_transcription.json")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    
    # Save readable transcript
    txt_file = os.path.join(output_dir, f"{base_name}_transcript.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcript: {base_name}\n")
        f.write(f"Language: {formatted_result['language']}\n")
        f.write(f"Duration: {formatted_result['total_duration']:.2f} seconds\n")
        f.write("=" * 50 + "\n\n")
        
        for segment in formatted_segments:
            f.write(f"[{segment['start_time']} - {segment['end_time']}]\n")
            f.write(f"{segment['text']}\n\n")
    
    print(f"✅ Transcription saved to:")
    print(f"   JSON: {json_file}")
    print(f"   Text: {txt_file}")
    
    # Print summary
    print(f"\nTranscription Summary:")
    print(f"Language: {formatted_result['language']}")
    print(f"Duration: {formatted_result['total_duration']:.2f} seconds")
    print(f"Segments: {len(formatted_segments)}")
    print(f"Words: ~{len(formatted_result['full_text'].split())}")
    
    # Show first few segments
    print(f"\nFirst few segments:")
    for i, segment in enumerate(formatted_segments[:3]):
        print(f"{i+1}. [{segment['start_time']}] {segment['text'][:80]}...")
    
    return formatted_result

def save_fast_transcript(transcript_data, youtube_url, output_dir="./transcripts"):
    """Save the fast transcript to JSON file"""
    
    if not transcript_data:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from video ID
    video_id = extract_video_id(youtube_url)
    filename = f"fast_transcript_{video_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Add source info
    transcript_data['source'] = 'youtube_captions'
    transcript_data['video_url'] = youtube_url
    transcript_data['video_id'] = video_id
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Transcript saved to: {filepath}")
    return filepath

def get_video_metadata_fast(youtube_url):
    """
    Get basic video metadata without downloading
    Uses yt-dlp extract_info with download=False
    """
    import yt_dlp # type: ignore
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            metadata = {
                'title': info.get('title', 'Unknown'),
                'url': youtube_url,
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
                'description': info.get('description', '')[:500] + '...' if info.get('description') else ''
            }
            
            return metadata
            
    except Exception as e:
        print(f"❌ Error getting metadata: {str(e)}")
        return None

# Combined fast processing function
def process_youtube_fast(youtube_url):
    """
    Fast processing: Get transcript + metadata without downloading audio
    """
    
    print("🚀 Fast Processing Mode")
    print("="*50)
    
    # Step 1: Get transcript (fast)
    print("\n📝 Step 1: Getting transcript...")
    transcript_data = get_youtube_transcript_fast(youtube_url)
    
    if not transcript_data:
        print("❌ No captions available. Would need to use Whisper (slow mode).")
        return None, None
    
    # Step 2: Get metadata (fast)
    print("\n📊 Step 2: Getting metadata...")
    metadata = get_video_metadata_fast(youtube_url)
    
    if not metadata:
        print("❌ Could not get video metadata")
        return None, None
    
    # Step 3: Save transcript
    print("\n💾 Step 3: Saving transcript...")
    transcript_file = save_fast_transcript(transcript_data, youtube_url)
    
    print("\n✅ Fast processing complete!")
    print(f"Time saved: ~5-10 minutes vs Whisper approach")
    
    return transcript_data, metadata

# Example usage
# if __name__ == "__main__":
#     youtube_url = "https://www.youtube.com/watch?v=w5unVTO7mLQ"
    
#     transcript_data, metadata = process_youtube_fast(youtube_url)
    
#     if transcript_data and metadata:
#         print(f"\n📋 Summary:")
#         print(f"Title: {metadata['title']}")
#         print(f"Duration: {transcript_data['total_duration']:.1f} seconds")
#         print(f"Language: {transcript_data['language']}")
#         print(f"Word count: ~{len(transcript_data['full_text'].split())}")
#         print(f"First 100 chars: {transcript_data['full_text'][:100]}...")
#     else:
#         print("❌ Fast processing failed - would need Whisper fallback")