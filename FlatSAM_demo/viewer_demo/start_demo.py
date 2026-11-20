"""Serve this folder on localhost and open the viewer. Screenshot and MP4 need an http page (a file:// page cannot read its own images).

Usage: python3 start_demo.py [port]      (Python 3 standard library only; stop with Ctrl+C)
"""
import functools
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parent
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass


Handler = functools.partial(Handler, directory=str(root))
socketserver.TCPServer.allow_reuse_address = True
for p in range(port, port + 20):
    try:
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", p), Handler)
        break
    except OSError:
        continue
else:
    sys.exit("no free port found")
url = f"http://127.0.0.1:{p}/index.html"
print(f"FlatSAM viewer demo: {url}   (Ctrl+C to stop)")
threading.Timer(0.5, lambda: webbrowser.open(url)).start()
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
