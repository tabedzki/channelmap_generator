<img src="https://raw.githubusercontent.com/m-beau/channelmap_generator/main/channelmap_generator/gui/assets/npix_map_logo.png" width="150" align="right" vspace="0">

# PixelMap: a browser-based GUI to generate Neuropixels channelmaps

[![Tests](https://github.com/m-beau/channelmap_generator/actions/workflows/tests.yml/badge.svg)](https://github.com/m-beau/channelmap_generator/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

<div align="center">

**Generate IMRO channelmaps for Neuropixels probes that respect electrode-ADC wiring constraints.**

[Online GUI](https://neuropixels-channelmap-generator.pni.princeton.edu) | [Documentation](#usage) | [Paper](#citation)

</div>

## What is PixelMap?

- Browser-based **generation of `.imro` files** for [SpikeGLX](https://billkarsh.github.io/SpikeGLX/help/imroTables/) — the tables that tell SpikeGLX which channels to record from, what reference to use, and recording amplification gain.
- **Arbitrary selection of electrode geometries**, within the boundaries of Neuropixels electrode-ADC hardware wiring constraints.
- Common **presets** available out of the box (reach out to suggest other common geometries!).
- Option to **load a pre-existing `.imro` file** as a starting point (or simply to visualize a probe geometry).

> [!NOTE]
> This tool is in **beta**. The IMRO editor tools in SpikeGLX and Open Ephys work correctly — what we provide here are tools to specify IMRO tables more easily. If you import one of our tables into SpikeGLX or Open Ephys, please double-check that all site selections, referencing, gains, and filter settings are what you intended.

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/m-beau/channelmap_generator.git
cd channelmap_generator
uv run cmap_gui  # installs dependencies automatically and launches the GUI
```

<details>
<summary>Using pip</summary>

Create a virtual environment first, e.g. with conda:

```bash
conda create -n pixelmap python=3.12
conda activate pixelmap
pip install .
```

</details>

<details>
<summary>Using Docker (no installation required)</summary>

```bash
docker run --rm \
  --name channelmap-app \
  -p 5008:5008 \
  --pull=always \
  ghcr.io/m-beau/channelmap_generator:latest
  # add --platform linux/amd64 on Apple Silicon or ARM Linux
```

The application will be available at http://localhost:5008.

For a more robust deployment, use **Docker Compose** — see the included `docker-compose.yml` for configuration details.

</details>

## Usage

### Step 1. Create and download your channelmap (`.imro` file)

#### Option A — Browser-based GUI

Use the [online GUI](https://neuropixels-channelmap-generator.pni.princeton.edu), or launch it locally:

```bash
uv run cmap_gui        # if using uv
cmap_gui               # if installed with pip
```

Neuropixels electrodes are [hardwired](https://www.neuropixels.org/support) to specific ADCs in the probe's head. When you select an electrode, others become unavailable because they share the same recording lines. This GUI lets you build a channelmap around those constraints: selected channels turn **red**, and unavailable channels turn **black**.

You can mix and match four selection methods:

- **Presets** — pre-configured channelmaps that respect wiring constraints
- **Textual selection** — type electrode ranges (e.g., `1-10,20-25`) to add to the current selection
- **Interactive** — click electrodes directly or drag boxes (selection, deselection, or zigzag selection)
- **Load from IMRO file** — pre-load an existing `.imro` file as a starting point

Once you reach the **target number of electrodes** for the selected probe type (384 or 1536), you can **download your channelmap** as an IMRO file alongside a PDF rendering.

![GUI screenshot](channelmap_generator/gui/assets/GUI_screenshot.png)

#### Option B — Python API / Jupyter Notebook

See `generate_channel_maps.ipynb` for examples covering all supported probe types and presets.

Minimal working example:

```python
import channelmap_generator as cmg

# Generate a channel map
imro_list = cmg.generate_imro_channelmap(
    probe_type="2.0-4shanks",
    layout_preset="tips_all",
    wiring_file="wiring_maps/2.0-4shanks_wiring.csv"
)

# Save to file
cmg.save_to_imro_file(imro_list, "my_channelmap.imro")
```

### Step 2. Load the IMRO file in SpikeGLX

Load the `.imro` file as you normally would (see [SpikeGLX documentation](https://billkarsh.github.io/SpikeGLX/help/imroTables/)):

- **Before recording:** through the IM-Setup tab
- **During recording:** through the live probe view (the probe-shaped heatmap)

## Troubleshooting

<details>
<summary><b>The online GUI is unresponsive</b></summary>

Reloading the page should fix most issues. If you have a slow network connection, consider installing PixelMap locally (see [Installation](#installation)).

</details>

<details>
<summary><b>SpikeGLX ignores my <code>.imro</code> file upon upload</b></summary>

Make sure the **probe subtype** is correct. You can find the subtype as the **first number of the first tuple of the probe's IMRO table**, either in:
- The default `.imro` file made by SpikeGLX (save it to file through the IM Setup tab), or
- The `.meta` file saved alongside any recording from that probe (`~imroTbl` field, typically the last field of the file).

> **Note:** SpikeGLX's probe visualizer may not display the correct probe subtype (e.g. it may show NP2014 for a 2013 probe). Always check the manually-saved `.imro` or `.meta` file to be certain.

</details>

<details>
<summary><b>After loading the <code>.imro</code> file, SpikeGLX IMRO editing options are greyed out</b></summary>

This is expected behavior for non-canonical IMRO tables. SpikeGLX disables editing for imported tables that don't match its canonical format (whole/half-shank width boxes enclosing all AP channels with attributes for all channels), allowing it to use external tables "as is" without modifying them. If you need to make adjustments, use the [online GUI](https://neuropixels-channelmap-generator.pni.princeton.edu) again: upload your `.imro` file as a starting point, modify it, then re-download and re-import into SpikeGLX.

</details>

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=channelmap_generator --cov-report=term
```

Tests run automatically via GitHub Actions on every push and pull request.

## Roadmap

Supported Neuropixels probe types:
- [x] 1.0
- [x] 2.0, 1-shank
- [x] 2.0, 4-shanks
- [ ] Quadbase
- [ ] NXT

UHD and Opto are not currently on our roadmap — reach out if you need support for these.

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/m-beau/channelmap_generator/issues). Pull requests are also welcome — please open an issue first to discuss significant changes.

## Citation

If you use PixelMap in your research, please cite:

> Beau M., Tabedzki C., Brody C.D. (2024). *PixelMap: An Application for Flexible Electrode Selection on Neuropixels Probes.* Journal of Open Source Software. <!-- TODO: update with DOI once published -->

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE.txt).
