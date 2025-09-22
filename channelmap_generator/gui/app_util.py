# For general panel serving, both in Docker and in Ploomber

from pathlib import Path

import psutil
import time
import socket
import threading

import panel as pn

# sys.path.insert(0, str(Path(__file__).parent))
from channelmap_generator.gui.gui import create_app

pn.extension(notifications=True)

def find_free_port(start_port=5007):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports found")

def memory_monitorer():
    process = psutil.Process()
    while True:
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
        time.sleep(10)

def monitor_memory():
    threading.Thread(target=memory_monitorer, daemon=True).start()


def main(show=True, local=True):
    # Create app
    app = create_app()

    # Monitor potential memory leak
    monitor_memory()

    # Serve the app
    print("Starting app...")
    if local:
        port = find_free_port(5003)
        pn.serve(app, port=port, show=show, title="Neuropixels Channelmap Generator", verbose=True)
    else:
        app.servable(title="Neuropixels Channelmap Generator")


if __name__ == "__main__":
    main(show=True, local=True)
