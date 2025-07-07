import yt_dlp # type: ignore
import os
import json

def download_youtube_audio(url, output_path="./downloads", metadata_path="./metadata"):
    """
    Download audio from YouTube video
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the audio file
    
    Returns:
        str: Path to the downloaded audio file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best audio quality
        'extractaudio': True,        # Extract audio from video
        'audioformat': 'wav',        # Convert to wav (better for Whisper)
        'audioquality': '192K',      # Set audio quality
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # Output filename template
        'quiet': False,              # Set to True to suppress output
        'postprocessors': [{         # Post-processing to ensure conversion
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            print(f"[Downloader] Title: {video_title}")
            print(f"[Downloader] Duration: {duration // 60}:{duration % 60:02d}")
            
            metadata = {
                'title': info.get('title', 'Unknown'),
                'url': url,
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', '')
            }
            
            # Save metadata to JSON file (use safe filename)
            safe_filename = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length and replace spaces
            metadata_file = os.path.join(metadata_path, f"{safe_filename}.json")
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"[Downloader] Metadata saved to: {metadata_file}")
            print(f"[Downloader] metadata: {metadata}")
            
            # Download the audio
            ydl.download([url])
            
            # Construct the expected filename
            safe_title = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.wav'
            
            return safe_title
            
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return None

# # Example usage
# if __name__ == "__main__":
#     youtube_url = "https://www.youtube.com/watch?v=8kMaTybvDUw"  # Replace with actual URL
    
#     audio_file = download_youtube_audio(youtube_url)
    
#     if audio_file:
#         print(f"Audio downloaded successfully: {audio_file}")
#     else:
#         print("Failed to download audio")