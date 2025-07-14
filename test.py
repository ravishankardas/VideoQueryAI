from pipeline import YouTubeRAGPipeline

# Initialize pipeline
pipeline = YouTubeRAGPipeline()
youtube_url = "https://www.youtube.com/watch?v=DfibPOHnRxc"
video_uuid = "7efe2c26"
collection_name = f"video_{video_uuid}"
result = pipeline.process_video(youtube_url)
print(result)