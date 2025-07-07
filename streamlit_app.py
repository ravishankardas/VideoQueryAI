import streamlit as st
import os
import tempfile

# Fix PyTorch/Streamlit compatibility issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import torch first to avoid issues
try:
    import torch
    torch.set_num_threads(1)
except:
    pass

from pipeline import YouTubeRAGPipeline

# Configure page
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = YouTubeRAGPipeline()
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("🎥 YouTube RAG Assistant")
st.markdown("Process YouTube videos and ask questions about their content!")

# Sidebar
with st.sidebar:
    st.header("📹 Processed Videos")
    
    # Refresh videos list
    if st.button("🔄 Refresh List"):
        st.session_state.processed_videos = st.session_state.pipeline.list_videos()
    
    # Display processed videos
    if st.session_state.processed_videos:
        for video in st.session_state.processed_videos:
            with st.expander(f"🎬 {video['title'][:30]}..."):
                st.write(f"**Uploader:** {video['uploader']}")
                st.write(f"**UUID:** {video['uuid']}")
                st.code(video['url'])
    else:
        st.info("No videos processed yet")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Process Video")
    
    # URL input
    youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    # Process button
    if st.button("🚀 Process Video", disabled=not youtube_url):
        with st.spinner("Processing video..."):
            try:
                # Check if already processed
                videos = st.session_state.pipeline.list_videos()
                existing_video = None
                for video in videos:
                    if video['url'] == youtube_url:
                        existing_video = video
                        break
                
                if existing_video:
                    st.success(f"✅ Video already processed!")
                    st.info(f"**Title:** {existing_video['title']}")
                    st.info(f"**UUID:** {existing_video['uuid']}")
                    
                    # Load existing data
                    chunks_file = f"./rag_data/{existing_video['uuid']}/chunks/video_chunks.json"
                    if os.path.exists(chunks_file):
                        from qa import add_chunks_to_chromadb
                        add_chunks_to_chromadb(chunks_file)
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Chunks file not found")
                else:
                    # Process new video
                    result = st.session_state.pipeline.process_video(youtube_url)
                    
                    if result:
                        st.success("✅ Video processed successfully!")
                        st.info(f"**Title:** {result['metadata']['title']}")
                        st.info(f"**UUID:** {result['uuid']}")
                    else:
                        st.error("❌ Failed to process video")
                
                # Refresh videos list
                st.session_state.processed_videos = st.session_state.pipeline.list_videos()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.header("💬 Chat with Videos")
    
    # Chat history
    chat_container = st.container()
    
    # Question input
    question = st.text_input("Ask a question:", placeholder="What is this video about?")
    
    # Ask button
    if st.button("❓ Ask Question", disabled=not question):
        with st.spinner("Thinking..."):
            try:
                # Add question to history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Get answer
                answer = st.session_state.pipeline.ask_question(question)
                
                # Add answer to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display chat history
    with chat_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("---")
        else:
            st.info("No conversation yet. Process a video and start asking questions!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()