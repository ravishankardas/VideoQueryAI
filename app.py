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

# Your existing code here...
from pipeline import YouTubeRAGPipeline

# Initialize pipeline
pipeline = YouTubeRAGPipeline()

# Store processed videos globally
processed_videos = {}

def validate_youtube_url(url):
    """Validate if URL is from YouTube"""
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
    return bool(url and re.match(youtube_regex, url))

def process_video(youtube_url):
    """Process a YouTube video"""
    if not youtube_url:
        return "❌ Please enter a YouTube URL", "", ""
    
    if not validate_youtube_url(youtube_url):
        return "❌ Please enter a valid YouTube URL", "", ""
    
    try:
        # Check if already processed
        for uuid, info in processed_videos.items():
            if info['url'] == youtube_url:
                return f"✅ Video already processed: {info['title']}", info['title'], uuid
        
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
            
            # Store processed video
            processed_videos[uuid] = {
                'title': title,
                'url': youtube_url,
                'metadata': result.get('metadata', {})
            }
            
            return f"✅ Successfully processed: {title}", title, uuid
        else:
            return "❌ Failed to process video. Please try again.", "", ""
            
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

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
    return "", "", "", [], "", ""  # Clear all components

# Create Gradio interface
with gr.Blocks(title="VideoQuery AI") as app:
    
    # Header
    gr.Markdown("""
    # 🎥 VideoQuery AI
    ### Intelligent YouTube Video Analysis & Question Answering
    Process YouTube videos and ask questions about their content using advanced RAG technology.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📹 Process Video")
            
            # Video processing section
            youtube_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            process_btn = gr.Button("🚀 Process Video", variant="primary")
            reset_btn = gr.Button("🔄 Reset All", variant="secondary")
            process_status = gr.Textbox(label="Status", interactive=False)
            
            # Hidden components to store video info
            current_video_title = gr.Textbox(visible=False)
            current_video_uuid = gr.Textbox(visible=False)
            
        with gr.Column():
            gr.Markdown("## ❓ Ask Questions")
            
            selected_video_display = gr.Textbox(
                label="Current Video",
                interactive=False
            )
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=True
            )
            
            # Question input at bottom
            with gr.Row():
                question_input = gr.Textbox(
                    label="",
                    placeholder="Ask a question about the video...",
                    scale=4,
                    show_label=False
                )
                ask_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Examples section
    gr.Markdown("## 📝 Example Questions")
    gr.Examples(
        examples=[
            ["What is the main topic of this video?"],
            ["Can you summarize the key points?"],
            ["What technologies were mentioned?"],
            ["Who are the speakers in this video?"],
            ["What are the main takeaways?"]
        ],
        inputs=[question_input]
    )
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[youtube_url],
        outputs=[process_status, current_video_title, current_video_uuid]
    )
    
    reset_btn.click(
        fn=reset_everything,
        outputs=[youtube_url, process_status, current_video_title, chatbot, selected_video_display, current_video_uuid]
    )
    
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
    
    current_video_title.change(
        fn=lambda title: title,
        inputs=[current_video_title],
        outputs=[selected_video_display]
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
        debug=True
    )
    
    # Print the correct URL to visit
    if HOST == "127.0.0.1":
        print(f"\n🌐 Visit: http://localhost:{PORT}")
        print(f"🌐 Or: http://127.0.0.1:{PORT}")
    else:
        print(f"\n🌐 App running on all interfaces: {HOST}:{PORT}")