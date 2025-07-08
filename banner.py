from colorama import Fore, Back, Style, init
import sys

# Initialize colorama for Windows compatibility
init(autoreset=True)

def print_videoquery_banner():
    """Print colored ASCII art banner for VideoQuery AI"""
    
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}
██╗   ██╗██╗██████╗ ███████╗ ██████╗  ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗
██║   ██║██║██╔══██╗██╔════╝██╔═══██╗██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝
██║   ██║██║██║  ██║█████╗  ██║   ██║██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝ 
╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║██║▄▄ ██║██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝  
 ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████╗██║  ██║   ██║   
  ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   
{Style.RESET_ALL}
{Fore.YELLOW}{Style.BRIGHT}                              █████╗ ██╗
                             ██╔══██╗██║
                             ███████║██║
                             ██╔══██║██║
                             ██║  ██║██║
                             ╚═╝  ╚═╝╚═╝
{Style.RESET_ALL}
{Fore.MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Fore.GREEN}{Style.BRIGHT}            🎥 Intelligent YouTube Video Analysis & Question Answering 🎥
{Fore.MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Style.RESET_ALL}
{Fore.WHITE}{Style.BRIGHT}    Features:{Style.RESET_ALL}
    {Fore.CYAN}✓{Style.RESET_ALL} Hybrid Search (Vector + BM25)
    {Fore.CYAN}✓{Style.RESET_ALL} Semantic Chunking
    {Fore.CYAN}✓{Style.RESET_ALL} Citation Tracking
{Style.RESET_ALL}
"""
    
    print(banner)

def print_simple_banner():
    """Simple alternative banner"""
    
    banner = f"""
{Fore.RED}{Style.BRIGHT}
╔╗ ╔╗┬┌┬┐┌─┐┌─┐╔═╗ ┬ ┬┌─┐┬─┐┬ ┬  ╔═╗╦
╚╗╔╝│ ││├┤ │ │║═╬╗│ │├┤ ├┬┘└┬┘  ╠═╣║
 ╚╝ ┴─┴┘└─┘└─┘╚═╝╚└─┘└─┘┴└─ ┴   ╩ ╩╩
{Style.RESET_ALL}
{Fore.YELLOW}    🚀 YouTube Video Intelligence System 🚀{Style.RESET_ALL}
{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}
"""
    
    print(banner)

def print_minimal_banner():
    """Minimal clean banner"""
    
    print(f"""
{Fore.CYAN}{Style.BRIGHT}
    ╦  ╦┬┌┬┐┌─┐┌─┐  ╔═╗ ┬ ┬┌─┐┬─┐┬ ┬  ╔═╗╦
    ╚╗╔╝│ ││├┤ │ │  ║═╬╗│ │├┤ ├┬┘└┬┘  ╠═╣║
     ╚╝ ┴─┴┘└─┘└─┘  ╚═╝╚└─┘└─┘┴└─ ┴   ╩ ╩╩
{Style.RESET_ALL}
{Fore.GREEN}           🎯 Smart Video Analysis & Q&A{Style.RESET_ALL}
""")

def print_loading_animation():
    """Print animated loading"""
    import time
    
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    for i in range(20):
        frame = frames[i % len(frames)]
        print(f"\r{Fore.YELLOW}{frame} Loading VideoQuery AI...{Style.RESET_ALL}", end="", flush=True)
        time.sleep(0.1)
    
    print(f"\r{Fore.GREEN}✓ VideoQuery AI Ready!{Style.RESET_ALL}                    ")

def print_startup_sequence():
    """Complete startup sequence with banner"""
    
    # Clear screen (optional)
    # import os
    # os.system('cls' if os.name == 'nt' else 'clear')
    
    # Main banner
    print_videoquery_banner()
    
    # Loading animation
    print_loading_animation()
    
    # Status message
    print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 System initialized successfully!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Ready to process YouTube videos...{Style.RESET_ALL}\n")

# # Usage examples:
# if __name__ == "__main__":
#     print("1. Full Banner:")
#     print_videoquery_banner()
    
#     print("\n" + "="*50 + "\n")
    
#     print("2. Simple Banner:")
#     print_simple_banner()
    
#     print("\n" + "="*50 + "\n")
    
#     print("3. Minimal Banner:")
#     print_minimal_banner()
    
#     print("\n" + "="*50 + "\n")
    
#     print("4. Startup Sequence:")
#     print_startup_sequence()