# For general panel serving, both in Docker and in Ploomber

import os
import sys
from pathlib import Path

import panel as pn

from channelmap_generator.gui.gui import create_app

pn.extension(notifications=True)

sys.path.insert(0, str(Path(__file__).parent))

app = create_app()

print("Starting app...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# port = int(os.environ.get("PORT", 80))
# pn.serve(app, port=port, address="0.0.0.0", allow_websocket_origin="*", show=False)
app.servable(title="Neuropixels Channelmap Generator")
