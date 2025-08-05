# For Ploomber serving

import os
import sys
from pathlib import Path
import panel as pn

pn.extension()

sys.path.insert(0, str(Path(__file__).parent))
from channelmap_generator.gui.gui import create_app

app = create_app()

print("Starting app...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# port = int(os.environ.get("PORT", 80))
# pn.serve(app, port=port, address="0.0.0.0", allow_websocket_origin="*", show=False)
app.servable()