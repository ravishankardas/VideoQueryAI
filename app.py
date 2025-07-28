import os

# Railway port configuration - get port from environment or default to 7860
PORT = int(os.environ.get("PORT", 7860))

# For local development, use localhost
# For Railway deployment, use 0.0.0.0
if os.environ.get("RAILWAY_ENVIRONMENT_NAME"):
    # Running on Railway
    HOST = "0.0.0.0"
else:
    # Running locally
    HOST = "127.0.0.1"

os.environ["GRADIO_SERVER_NAME"] = HOST
os.environ["GRADIO_SERVER_PORT"] = str(PORT)

import gradio as gr # type: ignore
import re
import requests
from PIL import Image
import io

# Your existing code here...
from pipeline import YouTubeRAGPipeline

# Initialize pipeline
pipeline = YouTubeRAGPipeline()

# Store processed videos globally
processed_videos = {}

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

def get_video_thumbnail(youtube_url):
    """Get video thumbnail from YouTube URL"""
    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return None
        
        # Try different thumbnail qualities (maxresdefault, hqdefault, mqdefault, sddefault)
        thumbnail_urls = [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/sddefault.jpg"
        ]
        
        for url in thumbnail_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Check if it's a valid image (not a default "no thumbnail" image)
                    img = Image.open(io.BytesIO(response.content))
                    # YouTube's default thumbnail is 120x90, we want higher quality
                    if img.size[0] > 120:
                        return img
            except:
                continue
        
        return None
    except Exception as e:
        print(f"Error getting thumbnail: {e}")
        return None

def validate_youtube_url(url):
    """Validate if URL is from YouTube"""
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
    return bool(url and re.match(youtube_regex, url))

def process_video(youtube_url):
    """Process a YouTube video"""
    if not youtube_url:
        return "❌ Please enter a YouTube URL", "", "", None
    
    if not validate_youtube_url(youtube_url):
        return "❌ Please enter a valid YouTube URL", "", "", None
    
    try:
        # Get thumbnail first
        thumbnail = get_video_thumbnail(youtube_url)
        
        # Check if already processed
        for uuid, info in processed_videos.items():
            if info['url'] == youtube_url:
                return f"✅ Video already processed: {info['title']}", info['title'], uuid, info.get('thumbnail', thumbnail)
        
        result = pipeline.process_video(youtube_url)
        
        if result:
            uuid = result.get('uuid', 'unknown')
            
            # Try different ways to get the title
            title = 'Unknown Title'
            
            # Method 1: From metadata dict
            if 'metadata' in result and result['metadata']:
                title = result['metadata'].get('title', title)
            
            # Method 2: Direct from result
            elif 'title' in result:
                title = result['title']
            
            # Method 3: From nested structure
            elif hasattr(result, 'get') and result.get('metadata'):
                metadata = result.get('metadata', {})
                title = metadata.get('title', title)
            
            print(f"Debug - Extracted title: {title}")
            print(f"Debug - Result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
            
            # Store processed video with thumbnail
            processed_videos[uuid] = {
                'title': title,
                'url': youtube_url,
                'metadata': result.get('metadata', {}),
                'thumbnail': thumbnail
            }
            
            return f"✅ Successfully processed: {title}", title, uuid, thumbnail
        else:
            return "❌ Failed to process video. Please try again.", "", "", thumbnail
            
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", None

def ask_question(question, video_uuid, chat_history):
    """Ask a question about the processed video and update chat"""
    if not video_uuid:
        return chat_history, ""
    
    if not question:
        return chat_history, ""
    
    try:
        answer = pipeline.ask_question(question, video_uuid)
        
        # Add to chat history
        chat_history.append([question, answer])
        
        return chat_history, ""  # Clear input
    except Exception as e:
        error_msg = f"❌ Error getting answer: {str(e)}"
        chat_history.append([question, error_msg])
        return chat_history, ""

def reset_everything():
    """Reset all data"""
    global processed_videos
    processed_videos.clear()
    return "", "", "", [], "", "", None  # Clear all components including thumbnail

def update_thumbnail_display(video_title, video_uuid):
    """Update thumbnail when video changes"""
    if video_uuid and video_uuid in processed_videos:
        return processed_videos[video_uuid].get('thumbnail', None)
    return None

# Create Gradio interface
with gr.Blocks(title="VideoQuery AI", theme=gr.themes.Soft()) as app:
    
    # Header
    gr.Markdown("""
    # 🎥 VideoQuery AI
    ### Intelligent YouTube Video Analysis & Question Answering
    Process YouTube videos and ask questions about their content using advanced RAG technology.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📹 Process Video")
            
            # Video processing section
            youtube_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1
            )
            
            with gr.Row():
                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                # reset_btn = gr.Button("🔄 Reset All", variant="secondary")
            
            process_status = gr.Textbox(label="Status", interactive=False, lines=2)
            
            # Video thumbnail display
            video_thumbnail = gr.Image(
                label="Video Thumbnail",
                show_label=True,
                height=200,
                interactive=False
            )
            
            # Video info display
            # with gr.Group():
            #     gr.Markdown("### 📊 Video Info")
            #     selected_video_display = gr.Textbox(
            #         label="Current Video",
            #         interactive=False,
            #         lines=2
            #     )
            
            # Hidden components to store video info
            current_video_title = gr.Textbox(visible=False)
            current_video_uuid = gr.Textbox(visible=False)
            
        with gr.Column(scale=2):
            gr.Markdown("## ❓ Ask Questions")
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False
            )
            
            # Question input at bottom
            with gr.Row():
                question_input = gr.Textbox(
                    label="",
                    placeholder="Ask a question about the video...",
                    scale=4,
                    show_label=False,
                    lines=1
                )
                ask_btn = gr.Button("Send", variant="primary", scale=1, size="lg")
    
    # Examples section
    with gr.Row():
        gr.Markdown("## 📝 Example Questions")
    
    with gr.Row():
        gr.Examples(
            examples=[
                ["What is the main topic of this video?"],
                ["Can you summarize the key points?"],
                ["What technologies were mentioned?"],
                ["Who are the speakers in this video?"],
                ["What are the main takeaways?"]
            ],
            inputs=[question_input],
            label="Click on any example to try it:"
        )
    
    # Footer
   
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[youtube_url],
        outputs=[process_status, current_video_title, current_video_uuid, video_thumbnail]
    )
    
    # reset_btn.click(
    #     fn=reset_everything,
    #     outputs=[youtube_url, process_status, current_video_title, chatbot, current_video_uuid, video_thumbnail]
    # )
    
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, current_video_uuid, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, current_video_uuid, chatbot],
        outputs=[chatbot, question_input]
    )
    
    
    
    # Update thumbnail when video changes
    current_video_uuid.change(
        fn=update_thumbnail_display,
        inputs=[current_video_title, current_video_uuid],
        outputs=[video_thumbnail]
    )


if __name__ == "__main__":
    print("🚀 Starting Gradio app...")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Environment: {'Railway' if os.environ.get('RAILWAY_ENVIRONMENT_NAME') else 'Local'}")
    
    app.launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        show_error=True,
        debug=True,
        favicon_path=None,
        app_kwargs={
            "docs_url": None,
            "redoc_url": None,
        }
    )
    
    # Print the correct URL to visit
    if HOST == "127.0.0.1":
        print(f"\n🌐 Visit: http://localhost:{PORT}")
        print(f"🌐 Or: http://127.0.0.1:{PORT}")
    else:
        print(f"\n🌐 App running on all interfaces: {HOST}:{PORT}")
