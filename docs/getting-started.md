# Getting Started

## Requirements

- Python 3.10 or later

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) handles virtual environments and dependencies automatically:

```bash
git clone https://github.com/m-beau/channelmap_generator.git
cd channelmap_generator
uv run cmap_gui  # installs dependencies and launches the GUI
```

To install dependencies without launching the GUI:

```bash
uv sync
```

### Using pip

Create a virtual environment first (e.g. with conda):

```bash
conda create -n pixelmap python=3.12
conda activate pixelmap
pip install .
```

### Using Docker

Run without any local Python installation:

```bash
docker run --rm \
  --name channelmap-app \
  -p 5008:5008 \
  --pull=always \
  ghcr.io/m-beau/channelmap_generator:latest
```

The application will be available at `http://localhost:5008`.

:::{note}
On Apple Silicon (M1/M2) or ARM Linux, add `--platform linux/amd64` to the Docker command.
:::

For production deployments, use **Docker Compose** — see the included `docker-compose.yml` in the repository.

## Quick Start

### Option 1: Browser GUI

Launch the GUI locally:

```bash
uv run cmap_gui   # or just: cmap_gui (if installed with pip)
```

Or use the [online version](https://neuropixels-channelmap-generator.pni.princeton.edu) directly — no installation required.

1. **Select your probe type** from the dropdown.
2. **Choose a preset** or select electrodes manually.
3. **Download the `.imro` file** once you've reached the target electrode count.
4. **Load the `.imro` file** in SpikeGLX via the IM-Setup tab.

### Option 2: Python API

```python
import channelmap_generator as cmg

# Generate a channel map using a preset
imro_list = cmg.generate_imro_channelmap(
    probe_type="2.0-4shanks",
    layout_preset="tips_all",
    wiring_file="wiring_maps/2.0-4shanks_wiring.csv"
)

# Save to file
cmg.save_to_imro_file(imro_list, "my_channelmap.imro")
```

See the [Python API reference](api.md) for full details and `generate_channel_maps.ipynb` in the repository for more examples.
