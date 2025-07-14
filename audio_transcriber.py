import re
from youtube_transcript_api._api import YouTubeTranscriptApi
import json
from datetime import timedelta
import whisper # type: ignore

import os
import requests
import xml.etree.ElementTree as ET
import tempfile
import yt_dlp # type: ignore


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

def parse_youtube_captions_xml(xml_content):
    """Parse YouTube captions XML format"""
    try:
        root = ET.fromstring(xml_content)
        
        formatted_transcript = {
            'language': 'en',
            'is_generated': True,
            'segments': [],
            'full_text': '',
            'total_duration': 0
        }
        
        full_text_parts = []
        
        for text_elem in root.findall('.//text'):
            start = float(text_elem.get('start', 0))
            duration = float(text_elem.get('dur', 0))
            text_content = text_elem.text or ''
            
            # Clean text
            text_content = text_content.strip()
            if not text_content:
                continue
            
            segment = {
                'start_time': str(timedelta(seconds=int(start))),
                'end_time': str(timedelta(seconds=int(start + duration))),
                'start_seconds': start,
                'end_seconds': start + duration,
                'text': text_content,
                'duration': duration
            }
            
            formatted_transcript['segments'].append(segment)
            full_text_parts.append(text_content)
        
        formatted_transcript['full_text'] = ' '.join(full_text_parts)
        if formatted_transcript['segments']:
            formatted_transcript['total_duration'] = formatted_transcript['segments'][-1]['end_seconds']
        
        return formatted_transcript
        
    except Exception as e:
        print(f"Error parsing captions XML: {e}")
        return None

def get_youtube_transcript_api_fallback(youtube_url):
    """Fallback using YouTube Data API for cloud platforms"""
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("❌ No YouTube API key found in environment")
        return None
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("❌ Could not extract video ID")
        return None
    
    try:
        print(f"🔑 Using YouTube Data API for video: {video_id}")
        
        # Step 1: Get captions list
        captions_url = f"https://www.googleapis.com/youtube/v3/captions"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': api_key
        }
        
        response = requests.get(captions_url, params=params)
        
        if response.status_code == 403:
            error_data = response.json()
            if 'error' in error_data:
                error_msg = error_data['error'].get('message', 'Unknown error')
                print(f"❌ API 403 Error: {error_msg}")
                if 'quota' in error_msg.lower():
                    print("💡 Quota exceeded - try again tomorrow or upgrade quota")
                elif 'disabled' in error_msg.lower():
                    print("💡 API not enabled - enable YouTube Data API v3 in Google Cloud")
                elif 'key' in error_msg.lower():
                    print("💡 Invalid API key - check your key in Hugging Face secrets")
            return None
        elif response.status_code != 200:
            print(f"❌ API request failed: {response.status_code} - {response.text}")
            return None
        
        captions_data = response.json()
        
        if 'items' not in captions_data or not captions_data['items']:
            print("❌ No captions found via API")
            return None
        
        # Find English captions (prefer manual over auto-generated)
        caption_track = None
        for item in captions_data['items']:
            lang = item['snippet']['language']
            track_kind = item['snippet']['trackKind']
            
            if lang == 'en':
                if track_kind == 'standard':  # Manual captions
                    caption_track = item
                    break
                elif track_kind == 'ASR' and not caption_track:  # Auto-generated
                    caption_track = item
        
        if not caption_track:
            # Try any available language
            caption_track = captions_data['items'][0]
        
        caption_id = caption_track['id']
        print(f"✅ Found captions: {caption_track['snippet']['language']} ({caption_track['snippet']['trackKind']})")
        
        # Step 2: Download caption content
        download_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}"
        download_params = {
            'key': api_key,
            'tfmt': 'ttml'  # Get XML format for easier parsing
        }
        
        caption_response = requests.get(download_url, params=download_params)
        
        if caption_response.status_code != 200:
            print(f"❌ Caption download failed: {caption_response.status_code}")
            return None
        
        # Step 3: Parse the captions
        formatted_transcript = parse_youtube_captions_xml(caption_response.text)
        
        if formatted_transcript:
            print(f"✅ API transcript extracted successfully!")
            print(f"Language: {formatted_transcript['language']}")
            print(f"Duration: {formatted_transcript['total_duration']:.1f} seconds")
            print(f"Segments: {len(formatted_transcript['segments'])}")
        
        return formatted_transcript
        
    except Exception as e:
        print(f"❌ YouTube API fallback error: {str(e)}")
        return None

def parse_timestamp_to_seconds(timestamp):
    """Convert timestamp string to seconds"""
    try:
        # Handle both VTT (00:01:30.500) and SRT (00:01:30,500) formats
        timestamp = timestamp.replace(',', '.')
        
        # Remove any extra formatting
        timestamp = re.sub(r'<[^>]+>', '', timestamp).strip()
        
        # Parse time components
        if '.' in timestamp:
            time_part, ms_part = timestamp.split('.')
            ms = float('0.' + ms_part)
        else:
            time_part = timestamp
            ms = 0
        
        time_components = time_part.split(':')
        
        if len(time_components) == 3:
            hours, minutes, seconds = map(int, time_components)
            total_seconds = hours * 3600 + minutes * 60 + seconds + ms
        elif len(time_components) == 2:
            minutes, seconds = map(int, time_components)
            total_seconds = minutes * 60 + seconds + ms
        else:
            total_seconds = float(timestamp)
        
        return total_seconds
        
    except Exception as e:
        print(f"Error parsing timestamp '{timestamp}': {e}")
        return 0

def format_seconds_to_time(seconds):
    """Convert seconds back to time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"

def parse_subtitle_file(content, video_info):
    """Parse subtitle file (SRT/VTT) into our transcript format"""
    
    segments = []
    full_text_parts = []
    
    # Remove VTT headers if present
    content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.MULTILINE | re.DOTALL)
    content = re.sub(r'^NOTE.*?\n', '', content, flags=re.MULTILINE)
    
    # Split by double newlines to get blocks
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            # Find timestamp line
            timestamp_line = None
            text_lines = []
            
            for i, line in enumerate(lines):
                if '-->' in line:
                    timestamp_line = line
                    text_lines = lines[i + 1:]
                    break
                elif line.strip().isdigit():
                    # Skip sequence numbers in SRT
                    continue
                elif not timestamp_line:
                    # Look for timestamp in next lines
                    continue
            
            if timestamp_line and text_lines:
                try:
                    # Parse timestamp
                    times = timestamp_line.split(' --> ')
                    start_time = times[0].strip()
                    end_time = times[1].strip()
                    
                    # Convert timestamp to seconds
                    start_seconds = parse_timestamp_to_seconds(start_time)
                    end_seconds = parse_timestamp_to_seconds(end_time)
                    
                    # Get text (clean HTML tags and extra formatting)
                    text = ' '.join(text_lines).strip()
                    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
                    
                    if text and text != ' ':
                        segments.append({
                            'start_time': format_seconds_to_time(start_seconds),
                            'end_time': format_seconds_to_time(end_seconds),
                            'start_seconds': start_seconds,
                            'end_seconds': end_seconds,
                            'text': text,
                            'duration': end_seconds - start_seconds
                        })
                        full_text_parts.append(text)
                        
                except Exception as e:
                    print(f"Error parsing timestamp: {e}")
                    continue
    
    if segments:
        return {
            'language': 'en',
            'is_generated': True,
            'segments': segments,
            'full_text': ' '.join(full_text_parts),
            'total_duration': segments[-1]['end_seconds'] if segments else 0
        }
    
    return None

def get_youtube_transcript_ytdlp_fallback(youtube_url):
    """Ultra-reliable fallback using yt-dlp for captions"""
    
    try:
        print("🛠️ Trying yt-dlp caption extraction...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                # These options help with cloud platforms
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],
                        'skip': ['hls', 'dash']
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info and download subtitles
                info = ydl.extract_info(youtube_url, download=False)
                
                # Actually download the subtitles
                ydl.download([youtube_url])
                
                # Look for subtitle files
                subtitle_files = []
                for file in os.listdir(temp_dir):
                    if file.endswith(('.vtt', '.srt')):
                        subtitle_files.append(os.path.join(temp_dir, file))
                
                if subtitle_files:
                    print(f"✅ Found subtitle file: {os.path.basename(subtitle_files[0])}")
                    # Parse the first subtitle file
                    with open(subtitle_files[0], 'r', encoding='utf-8') as f:
                        subtitle_content = f.read()
                    
                    # Convert to our format
                    return parse_subtitle_file(subtitle_content, info)
                else:
                    print("❌ No subtitle files found")
                
        return None
        
    except Exception as e:
        print(f"❌ yt-dlp fallback failed: {e}")
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
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            metadata = {
                'title': info.get('title', 'Unknown'), # type: ignore
                'url': youtube_url,
                'uploader': info.get('uploader', 'Unknown'), # type: ignore
                'duration': info.get('duration', 0), # type: ignore
                'upload_date': info.get('upload_date', ''), # type: ignore
                'view_count': info.get('view_count', 0), # type: ignore
                'description': info.get('description', '')[:500] + '...' if info.get('description') else '' # type: ignore
            }
            
            return metadata
            
    except Exception as e:
        print(f"❌ Error getting metadata: {str(e)}")
        return None

def process_youtube_fast(youtube_url):
    """
    Fast processing with triple fallback: direct → API → yt-dlp
    """
    
    print("🚀 Fast Processing Mode")
    print("="*50)
    
    # Step 1: Try original transcript method
    print("\n📝 Step 1: Trying direct transcript extraction...")
    transcript_data = get_youtube_transcript_fast(youtube_url)
    
    # Step 2: If failed, try API fallback
    if not transcript_data:
        print("\n🌩️ Step 2: Direct method failed, trying YouTube Data API...")
        transcript_data = get_youtube_transcript_api_fallback(youtube_url)
    
    # Step 3: If API failed, try yt-dlp fallback
    if not transcript_data:
        print("\n🛠️ Step 3: API failed, trying yt-dlp fallback...")
        transcript_data = get_youtube_transcript_ytdlp_fallback(youtube_url)
    
    if not transcript_data:
        print("❌ No captions available via any method. Would need Whisper transcription.")
        return None, None
    
    # Get metadata
    print("\n📊 Getting metadata...")
    metadata = get_video_metadata_fast(youtube_url)
    
    if not metadata:
        print("❌ Could not get video metadata")
        return None, None
    
    # Save transcript
    print("\n💾 Saving transcript...")
    transcript_file = save_fast_transcript(transcript_data, youtube_url)
    
    print("\n✅ Fast processing complete!")
    
    return transcript_data, metadata