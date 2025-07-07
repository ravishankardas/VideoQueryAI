import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
from pipeline import YouTubeRAGPipeline

class YouTubeRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube RAG Assistant")
        self.root.geometry("800x600")
        
        # Initialize pipeline
        self.pipeline = YouTubeRAGPipeline()
        self.current_video = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # type: ignore
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YouTube RAG Assistant", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # URL input section
        ttk.Label(main_frame, text="YouTube URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.url_entry = ttk.Entry(main_frame, width=50)
        self.url_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0)) # type: ignore
        
        self.process_btn = ttk.Button(main_frame, text="Process Video", command=self.process_video)
        self.process_btn.grid(row=1, column=2, pady=5, padx=(5, 0))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Chat section
        chat_frame = ttk.LabelFrame(main_frame, text="Chat", padding="5")
        chat_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0)) # type: ignore
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, state=tk.DISABLED)
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)) # type: ignore
        
        # Question input
        self.question_entry = ttk.Entry(chat_frame, width=60)
        self.question_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5) # type: ignore
        self.question_entry.bind('<Return>', lambda e: self.ask_question())
        
        self.ask_btn = ttk.Button(chat_frame, text="Ask", command=self.ask_question)
        self.ask_btn.grid(row=1, column=1, pady=5, padx=(5, 0))
        
        # Initially disable chat
        self.question_entry.config(state=tk.DISABLED)
        self.ask_btn.config(state=tk.DISABLED)
        
        # Video list section
        list_frame = ttk.LabelFrame(main_frame, text="Processed Videos", padding="5")
        list_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)) # type: ignore
        list_frame.columnconfigure(0, weight=1)
        
        # Videos listbox
        self.videos_listbox = tk.Listbox(list_frame, height=5)
        self.videos_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5) # type: ignore
        
        refresh_btn = ttk.Button(list_frame, text="Refresh List", command=self.refresh_videos)
        refresh_btn.grid(row=0, column=1, pady=5, padx=(5, 0))
        
        # Load existing videos
        self.refresh_videos()
    
    def add_to_chat(self, message, sender="System"):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def process_video(self):
        """Process YouTube video in a separate thread"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        # Disable button during processing
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...", foreground="orange")
        
        # Run in separate thread to prevent UI freezing
        thread = threading.Thread(target=self._process_video_thread, args=(url,))
        thread.daemon = True
        thread.start()
    
    def _process_video_thread(self, url):
        """Background thread for video processing"""
        try:
            # Check if video already processed
            videos = self.pipeline.list_videos()
            existing_video = None
            for video in videos:
                if video['url'] == url:
                    existing_video = video
                    break
            
            if existing_video:
                # Load existing video
                self.root.after(0, lambda: self.status_label.config(text="Loading existing video...", foreground="blue"))
                
                chunks_file = f"./rag_data/{existing_video['uuid']}/chunks/video_chunks.json"
                if os.path.exists(chunks_file):
                    from qa import add_chunks_to_chromadb
                    add_chunks_to_chromadb(chunks_file)
                    
                    self.current_video = existing_video
                    self.root.after(0, lambda: self.add_to_chat(f"Loaded existing video: {existing_video['title']}"))
                    self.root.after(0, lambda: self.status_label.config(text=f"Ready - {existing_video['title']}", foreground="green"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="Error: Chunks file not found", foreground="red"))
                    return
            else:
                # Process new video
                self.root.after(0, lambda: self.add_to_chat(f"Processing new video: {url}"))
                
                result = self.pipeline.process_video(url)
                if result:
                    self.current_video = result
                    self.root.after(0, lambda: self.add_to_chat(f"Successfully processed: {result['metadata']['title']}"))
                    self.root.after(0, lambda: self.status_label.config(text=f"Ready - {result['metadata']['title']}", foreground="green"))
                else:
                    self.root.after(0, lambda: self.add_to_chat("Failed to process video"))
                    self.root.after(0, lambda: self.status_label.config(text="Error: Failed to process video", foreground="red"))
                    return
            
            # Enable chat
            self.root.after(0, lambda: self.question_entry.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            self.root.after(0, self.refresh_videos)
            
        except Exception as e:
            self.root.after(0, lambda: self.add_to_chat(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Error occurred", foreground="red"))
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def ask_question(self):
        """Ask a question"""
        question = self.question_entry.get().strip()
        if not question:
            return
        
        # Add question to chat
        self.add_to_chat(question, "You")
        self.question_entry.delete(0, tk.END)
        
        # Disable input during processing
        self.question_entry.config(state=tk.DISABLED)
        self.ask_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Thinking...", foreground="blue")
        
        # Process in background thread
        thread = threading.Thread(target=self._ask_question_thread, args=(question,))
        thread.daemon = True
        thread.start()
    
    def _ask_question_thread(self, question):
        """Background thread for question processing"""
        try:
            answer = self.pipeline.ask_question(question)
            self.root.after(0, lambda: self.add_to_chat(answer, "Assistant"))
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
        except Exception as e:
            self.root.after(0, lambda: self.add_to_chat(f"Error: {str(e)}", "System"))
            self.root.after(0, lambda: self.status_label.config(text="Error occurred", foreground="red"))
        finally:
            # Re-enable input
            self.root.after(0, lambda: self.question_entry.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.question_entry.focus())
    
    def refresh_videos(self):
        """Refresh the videos list"""
        self.videos_listbox.delete(0, tk.END)
        videos = self.pipeline.list_videos()
        for video in videos:
            self.videos_listbox.insert(tk.END, f"{video['title']} ({video['uuid']})")

def main():
    root = tk.Tk()
    app = YouTubeRAGApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()