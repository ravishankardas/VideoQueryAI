# Simple usage
from pipeline import YouTubeRAGPipeline


pipeline = YouTubeRAGPipeline()

# Process a video
youtube_url = input("URL: ").strip()

result = pipeline.process_video(youtube_url)
# print(result)
uuid =  result.get('uuid', 'unknown')
title = result.get('title', 'Unknown Title')
print(f"Processed video: {title} (UUID: {uuid})")
print(f"Processed video UUID: {uuid}")

if result:
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    # Interactive Q&A
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = pipeline.ask_question(question, uuid)
        print(f"\nAnswer: {answer}")

# Show all videos
print("\n=== All Processed Videos ===")
videos = pipeline.list_videos()
for i, video in enumerate(videos, 1):
    print(f"{i}. {video['title']} (UUID: {video['uuid']})")