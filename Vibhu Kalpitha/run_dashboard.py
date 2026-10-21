import os
import sys
import webbrowser
import time

# Ensure Vibhu Kalpitha is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if __name__ == "__main__":
    import uvicorn
    
    url = "http://127.0.0.1:8000"
    print("=" * 70)
    print("🚀 Launching Agentic AI Market Intelligence Platform...")
    print(f"📡 Dashboard & API URL: {url}")
    print("💡 To open React dev server with Hot Reload: cd frontend && npm run dev")
    print("=" * 70)
    
    # Auto open browser after slight delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
