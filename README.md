# Channelmap Generator for Neuropixels Probes </h1> <img src="https://raw.githubusercontent.com/m-beau/channelmap_generator/main/channelmap_generator/gui/assets/npix_map_logo.png" width="150" align="right" vspace = "0">

<div align="center"> ⚡ Generate IMRO channelmaps for Neuropixels probes ⚡<br>⚡ that respect electrode-ADC wiring constraints! ⚡</div>

<div align="center"> <a href="https://neuropixels-channelmap-generator.pni.princeton.edu">>>> Online GUI <<<</a> </div><br>

Purpose of this tool:
- Convenient browser-based <span style="color: magenta;">generation of `.imro` files for SpikeGLX</span>.
    - [IMRO tables](https://billkarsh.github.io/SpikeGLX/help/imroTables/) are used to tell SpikeGLX what channels to record from (Neuropixels feature more physical electrodes than can be recorded from simultaneously); what reference to use; and sometimes recording amplification gain (1.0 only).
- Enables **arbitrary selection of electrode geometries**, within the boundaries of Neuropixels electrode-ADC hardware wiring constraints
- Common **presets** available out of the box (reach out to suggest other common geometries!)
- Option to **load pre-existing `.imro` file** as starting point (or simply to remember the probe geometry of a specific file)


**Disclaimer**: this tool is in <span style="color: magenta;">beta</span> release and should be considered a work in progress, deployed early to gather feedback. The imro editor tools in SpikeGLX and Open Ephys work correctly. What we are providing here are tools to specify imro tables more easily and conveniently. However, it will take some time to debug all the issues in this beta. If you import one of our tables into SpikeGLX or Open Ephys, please be sure to double check that all the site selections, referencing, gains, and filter settings that get imported are what you intended.

# Local installation

Clone the repository and navigate to it:

```bash
git clone http://github.com/m-beau/channelmap_generator.git
cd channelmap_generator
```

With `uv`, you can either install dependencies first and then run the application, or run it directly in one command:

```bash
uv sync # install dependencies alone
uv run cmap_gui # install dependencies automatically and run
```

This approach is particularly convenient as `uv` will automatically create a virtual environment, install all dependencies from `pyproject.toml`, and run the GUI in a single command.

<details>
  <summary>Using pip or docker</summary>

## Install the package using pip

In this case, you must create a virtual environment yourself, e.g. a new conda environment:

```bash
conda create -n my_environment python=3.12
conda activate my_environment
uv pip install . # fast! run pip install uv first.
# or traditionally with pip only:
pip install .
```

## Run using Docker (Installation-free)

Run the latest stable Docker image directly without any local installation. The application will be available at http://localhost:5008.

```bash
docker run --rm --name channelmap-app -p 5008:5008 --pull=always ghcr.io/m-beau/channelmap_generator:latest # add --platform linux/amd64 on M1 macs or linux machines
```

For a more robust deployment, use **Docker Compose**. See the included `docker-compose.yml` for configuration details.

</details>
<br>

# Usage

## Step 1. Use the app to make and download your channelmap (`.imro` file)

### Option 1 - Browser-based GUI

Either use the [online gui](https://neuropixels-channelmap-generator.pni.princeton.edu), or launch it **locally** from the cloned repository using one of these methods:

```bash
# If you're using uv (recommended):
uv run cmap_gui  # Automatically manages dependencies and virtual environment
# If you installed with pip:
cmap_gui  # Alias for: python ./channelmap_generator/gui/gui.py
```

Neuropixels electrodes are [hardwired](https://www.neuropixels.org/support) to specific ADCs in the probe's head. When you select an electrode, others become unavailable because they share the same recording lines. This GUI allows you to build a channelmap around those constraints: when you select channels, they turn **red**, and those that become unavailable because they share the same lines turn **black**.

> [!WARNING]
> If SpikeGLX seems to ignore the `.imro` file when you try to load it, make sure that the <span style="color: magenta;">probe subtype</span> is correct.
> You can find the subtype as the **first number of the first tuple of the probe's imro table**, either in the default `.imro` file made by SpikeGLX (save it to file though the IM setup tab), or in the `.meta` file saved alongside any recording from that probe (~imroTbl field, the last field of the file).

You can mix and match four selection methods:\
• **Presets:** Pre-configured channelmaps that respect wiring constraints\
• **Textual selection:** Type electrode ranges (e.g., "1-10,20-25") to add to the current selection\
• **Interactive:** Click electrodes directly or drag boxes (selection, deselection, or "zigzag selection") to manually select multiple sites\
• **Selection from pre-existing IMRO file:** you can pre-load an IMRO file as a starting point before doing any of the above.

Once you reach the **target number of electrodes** for the selected probe type (384 or 1536), you can **download your channelmap** as an IMRO file alongside a PDF rendering to easily remember what your channelmap looks like.

> [!TIP]
> **Online version**: This GUI is available online at https://neuropixels-channelmap-generator.pni.princeton.edu - you can use it directly in your browser without installing anything.

![](channelmap_generator/gui/assets/GUI_screenshot.png)

### Option 2 - Python API / Jupyter Notebook

Check out the code in `generate_channel_maps.ipynb` to reproducibly create custom channel maps. The notebook provides examples for all supported probe types and presets.

Here's a MWE:

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

## Step 2. Load the IMRO file in SpikeGLX

Like you would with any `.imro` file (see [SpikeGLX documentation](https://billkarsh.github.io/SpikeGLX/help/imroTables/)).

- Before initiating the recording: through the IM-Setup tab
- Once the recording has started: through the live probe view (the probe-shaped heatmap that represents ongoing activity across the whole shank)

# Troubleshooting

<span style="color: orange;">The online GUI is unresponsive!</span><br>
Reloading the page should fix most issues. If you have crappy network on your machine, consider installing it locally (see installation section above).

<span style="color: orange;">SpikeGLX ignores my `.imro` file upon upload!</span><br>
If SpikeGLX seems to ignore the `.imro` file when you try to load it, make sure that the <span style="color: magenta;">probe subtype</span> is correct. You can find the subtype as the **first number of the first tuple of the probe's imro table**, either in the default `.imro` file made by SpikeGLX (save it to file though the IM setup tab), or in the `.meta` file saved alongside any recording from that probe (~imroTbl field, the last field of the file).

<span style="color: orange;">After loading the `.imro` file, SpikeGLX IMRO editing options become greyed out!</span><br>
This is expected behavior for non-canonical IMRO tables. SpikeGLX greys out editing features for imported tables that don't match its canonical format (being whole/half-shank width boxes that enclose all AP channels with attributes for all channels). This allows SpikeGLX to use external tables "as is" without knowing how to modify them safely. If you need to make adjustments, use the [online gui](https://neuropixels-channelmap-generator.pni.princeton.edu) again: upload your `.imro`file as a starting point, then modify your `.imro`table before re-downlading it and re-importing into SpikeGLX.


# Roadmap

Supported Neuropixels versions:
- [x] 1.0
- [x] 2.0, 1-shank
- [x] 2.0, 4-shanks
- [ ] Quadbase
- [ ] NXT

UHD and opto currently not on our roadmap, reach out if you need it implemented.