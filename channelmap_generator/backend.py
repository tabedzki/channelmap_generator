#############
## Imports ##
#############

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


######################
## Global variables ##
######################

# Probe type to SpikeGLX type mapping
PROBE_TYPE_MAP = {
    "1.0": [0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300],
    "2.0-1shank": [21, 2003, 2004],
    "2.0-4shanks": [24, 2013, 2014],
    "NXT": [2020, 2021]
}

PROBE_N = { # N: number of physical electrodes; n: number of ACDs or channels
    "1.0": {'N': 960, 'n':384, 'n_per_shank':384},
    "2.0-1shank": {'N': 1280, 'n':384, 'n_per_shank':384},
    "2.0-4shanks": {'N': 5120, 'n':384, 'n_per_shank':384},
    "NXT": {'N': 5120, 'n':1536, 'n_per_shank':908}
}

# Reference electrode definitions
REF_ELECTRODES = {
    21: {'ext':0, 'tip':1},
    2003: {'ext':0, 'gnd':1, 'tip':2},
    2004: {'ext':0, 'gnd':1, 'tip':2},
    24: {'ext':0, 'tip':[1,2,3,4]},
    2013: {'ext':0, 'gnd':1, 'tip':[2, 3, 4, 5]},
    2014: {'ext':0, 'gnd':1, 'tip':[2, 3, 4, 5]},
    2020: {'ext':0, 'gnd':1, 'tip':[2, 3, 4, 5]},
    2021: {'ext':0, 'gnd':1, 'tip':[2, 3, 4, 5]},
}
REF_ELECTRODES = {tp: {'ext':0, 'tip':1} for tp in PROBE_TYPE_MAP['1.0']} | REF_ELECTRODES

REF_BANKS = {
    "1.0": {0:0, 1:1, 2:2},
    "2.0-1shank": {0:0, 1:2, 2:4, 3:8}, # wtf
    "2.0-4shanks": {0:0, 1:1, 2:2, 3:3},
    "NXT": {0:0, 1:1, 2:2, 3:3}
}

SUPPORTED_1shank_PRESETS = [
    # 1 shank presets
    "tip",
    "tip_b0_top_b1",
    "top_b0_tip_b1", 
    "zigzag",
]

SUPPORTED_4shanks_PRESETS = [
    # 4 shanks presets
    "tips_all",
    "tip_s0",
    "tip_s1", 
    "tip_s2",
    "tip_s3",
    "tips_0_3",
    "tips_1_2",
    "tip_b0_top_b1_s0",
    "tip_b0_top_b1_s1",
    "tip_b0_top_b1_s2", 
    "tip_b0_top_b1_s3",
    "top_b0_tip_b1_s0",
    "top_b0_tip_b1_s1",
    "top_b0_tip_b1_s2",
    "top_b0_tip_b1_s3",
    "tip_s0b0_top_s2b0",
    "tip_s2b0_top_s0b0",
    "tip_s1b0_top_s3b0",
    "tip_s3b0_top_s1b0",
    "gliding_0-3",
    "gliding_3-0",
    "zigzag_0",
    "zigzag_1",
    "zigzag_2",
    "zigzag_3"
]

############################
## Channel map generation ##
############################

def generate_imro_channelmap(
    probe_type,
    layout_preset = None,
    reference_id = 'ext',
    probe_subtype = None,
    custom_electrodes = None,
    wiring_file = None,
    ap_gain = 500,
    lf_gain = 250,
    hp_filter = 1):
    """
    Generate IMRO-formatted channelmap for Neuropixels probes.

    Args:
        probe_type: Type of probe ("1.0", "2.0-1shank", "2.0-4shanks", "NXT")
        layout_preset: Preset layout configuration
        reference_id: Reference electrode selection ('ext', 'tip', 'gnd')
        probe_subtype: Specific SpikeGLX type number (optional)
        custom_electrodes: list of custom (shank_id, electrode_id) pairs (overrides preset)
        positions_file: Path to positions CSV file
        wiring_file: Path to wiring CSV file
        ap_gain: AP band gain (for 1.0 probes)
        lf_gain: LF band gain (for 1.0 probes)
        hp_filter: High-pass filter setting (for 1.0 probes)

    Returns:
        IMRO-formatted string for channelmap
    """

    # 1) Process probe type and load CSVs
    if probe_subtype is None:
        probe_subtype = PROBE_TYPE_MAP[probe_type][0]

    wiring_df = pd.read_csv(wiring_file)

    # 2) Select electrodes from presets or custom
    selected_electrodes = get_electrodes(
            probe_type, wiring_df, layout_preset, custom_electrodes
        )

    # 3) Generate IMRO table with appropriate format
    imro_list = format_imro_string(
        selected_electrodes, wiring_df, probe_type,
        probe_subtype, reference_id, ap_gain, lf_gain, hp_filter
    )

    n_selected = len(imro_list) - 1
    n_possible = PROBE_N[probe_type]['n']

    if n_selected != n_possible:
        print(f"\n!! WARNING !!\nYou selected {n_selected} electrodes, but {probe_type} probes must record from {n_possible} simultaneously!\n")

    return imro_list


def get_electrodes(probe_type,
                   wiring_df,
                   preset = None,
                   custom_electrodes = None):
    """
    Get electrode selection based on preset, avoiding channel conflicts.
    - probe_type: Type of probe (e.g., "1.0", "2.0-1shank", "2.0-4shanks", "NXT")
    - wiring_df: Wiring DataFrame containing wiring information
    - preset: Preset layout configuration
    - custom_electrodes: Optional list of custom (shank_id, electrode_id) pairs
    """

    if custom_electrodes is None:
        assert preset is not None, "You must provide a preset if you do not provide custom electrodes!"
        selected_electrodes = get_preset_candidates(preset, probe_type, wiring_df)
    else:
        if custom_electrodes.ndim == 1: # assume single shank
            custom_electrodes = np.vstack([custom_electrodes * 0, custom_electrodes]).T
        selected_electrodes = np.array(custom_electrodes).astype(int)

    verify_hardware_violations(probe_type,
                               selected_electrodes,
                               wiring_df)

    return selected_electrodes


def verify_hardware_violations(probe_type, selected_electrodes, wiring_df):

    # Check for probe type illegal numbers
    max_n_per_shank = PROBE_N[probe_type]['n_per_shank']
    for shank in np.unique(selected_electrodes[:, 0]):
        n_electrodes_shank = np.sum(shank == selected_electrodes[:, 0])
        assert n_electrodes_shank <= max_n_per_shank,\
            f"Violation - too many electrodes ({n_electrodes_shank}) on shank {shank}"

    # Check for wiring conflicts
    forbidden_electrodes = find_forbidden_electrodes(selected_electrodes, wiring_df)
    selected_e_str = [f"{se[0]}_{se[1]}" for se in selected_electrodes]
    forbidden_e_str = [f"{se[0]}_{se[1]}" for se in forbidden_electrodes]
    conflicts = np.isin(selected_e_str, forbidden_e_str)
    assert not np.any(conflicts), \
        f"Selected electrodes conflict with wiring diagram ({' and '.join(list(np.array(selected_e_str)[conflicts]))}). Please choose a different preset or custom electrodes."


def find_forbidden_electrodes(selected_electrodes, wiring_df):
    "Return list of forbidden electrodes [[shank_id, electrode_id], ...] given selected electrodes and wiring diagram."

    df = format_wiring_df(wiring_df)

    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    forbidden_electrodes = []
    for shank_id, electrode_id in selected_electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        assert np.any(table_bool), f"electrode {electrode_id} on shank {shank_id} not found in wiring map!"
        table_loc = np.nonzero(table_bool)
        table_row = table_loc[0][0]
        # grab non-nan and non-self electrodes on row of wiring table
        forbidden_electrodes += [se for se in table_ids[table_row] if (~np.any(np.isnan(se))) & ~((shank_id==se[0])&(electrode_id==se[1]))]

    return np.array(forbidden_electrodes).astype(int)


def format_wiring_df(wiring_df):
    """
    Fetch shank id from column header of wiring_df
    and replace electrode_id with (shank_id, electrode_id) tuple in each cell
    """
    # add shank id to electrode id, in dataframe's cells
    df = wiring_df.copy()
    for column in df.columns[1:]:
        shank_id = int(column[5])
        electrode_ids = df.loc[:, column].values
        shank_electrode_ids = np.vstack([np.zeros(len(electrode_ids)) + shank_id, electrode_ids]).T
        shank_electrode_ids = [tuple(se) for se in shank_electrode_ids]
        df[column] = df[column].astype(object)
        df[column] = shank_electrode_ids

    return df


def format_imro_string(electrodes,
                        wiring_df,
                        probe_type,
                        probe_subtype,
                        reference_id,
                        ap_gain,
                        lf_gain,
                        hp_filter):
    """
    Format IMRO string based on probe type.
    See see https://billkarsh.github.io/SpikeGLX/help/imroTables/ for details.
    """

    assert probe_subtype is not None

    df_coordinates = find_electrode_coordinates(electrodes, wiring_df)
    electrodes = [[int(se[0]), int(se[1])] for se in electrodes]

    # Generate entries based on probe type
    entries = []
    if probe_type == "1.0":
        # Format: (channel_id bank ref ap_gain lf_gain hp_filter)
        ref_value = REF_ELECTRODES[probe_subtype][reference_id]

        for ((shank_id, electrode_id), (row, col)) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, 'channel'])
            bank = int(wiring_df.columns[col][-1])
            entry = (channel, bank, ref_value, ap_gain, lf_gain, hp_filter)
            entries.append(entry)

    elif probe_type == "2.0-1shank":
        # Format: (channel_id bank_mask ref electrode_id)
        ref_value = REF_ELECTRODES[probe_subtype][reference_id]

        for ((shank_id, electrode_id), (row, col)) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, 'channel'])
            bank = int(wiring_df.columns[col][-1])
            bank_mask = REF_BANKS["2.0-1shank"][bank] # {1=bnk-0, 2=bnk-1, 4=bnk-2, 8=bnk-3}
            entry = (channel, bank_mask, ref_value, electrode_id)
            entries.append(entry)

    elif probe_type in ["2.0-4shanks", "NXT"]:
        if reference_id == 'tip' and isinstance(REF_ELECTRODES[probe_subtype]['tip'], list):
            ref_values = REF_ELECTRODES[probe_subtype]['tip']
        else: # same for 4 shanks if not tip
            ref_values = [REF_ELECTRODES[probe_subtype][reference_id]] * 4

        for ((shank_id, electrode_id), (row, col)) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, 'channel'])
            bank = int(wiring_df.columns[col][-1])
            entry = (channel, shank_id, bank, ref_values[shank_id], electrode_id)
            entries.append(entry)

    # Sort by channel and format
    entries.sort(key=lambda x: x[0])
    header = (probe_subtype, len(entries))
    imro_list = [header] + entries

    return imro_list

def find_electrode_coordinates(electrodes, wiring_df):

    df = format_wiring_df(wiring_df)

    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    coordinates = []
    for shank_id, electrode_id in electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        row, col = np.nonzero(table_bool)
        coordinates.append([int(row[0]), int(col[0]) + 1])

    return coordinates

################################
## IMRO file I/O utilities ##
################################

def save_to_imro_file(imro_list, filename = "channelmap.imro"):
    """
    Save IMRO list to a text file in SpikeGLX format.

    Args:
        imro_list: List of tuples from generate_imro_channelmap
        filename: Output filename (default: "channelmap.imro")
    """
    if '.imro' not in filename:
        filename = filename + '.imro'
    filename = "_".join(filename.replace("\n", " ").split(" "))

    with open(filename, 'w') as f:
        # Write header
        header = imro_list[0]
        f.write(f"({header[0]},{header[1]})")

        # Write channel entries
        for entry in imro_list[1:]:
            # Format tuple as space-separated values in parentheses
            entry_str = " ".join(str(x) for x in entry)
            f.write(f"({entry_str})")

        # Add newline at end of file
        f.write("\n")

    print(f"IMRO file saved: {filename}")

def read_imro_file(filepath):
    """
    Read IMRO file and return imro_list format.

    Args:
        filepath: Path to .imro file

    Returns:
        imro_list: List of tuples matching generate_imro_channelmap output
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()

    return parse_imro_file(content)

def parse_imro_file(content):
    """
    Parse IMRO file content and return imro_list format.

    Args:
        content: str, contents of .imro file

    Returns:
        imro_list: List of tuples matching generate_imro_channelmap output
    """
    # Split by ')(' to get individual entries
    entries = content.split(')(')

    # Clean up parentheses from first and last entries
    entries[0] = entries[0].lstrip('(')
    entries[-1] = entries[-1].rstrip(')')

    # Parse header (first entry: "24,384")
    header_parts = entries[0].split(',')
    header = (int(header_parts[0]), int(header_parts[1]))

    # Parse channel entries (format: "0 1 0 2 288")
    channel_entries = []
    for entry_str in entries[1:]:
        values = [int(x) for x in entry_str.split()]
        channel_entries.append(tuple(values))

    return [header] + channel_entries

def parse_imro_list(imro_list):
   """
   Parse imro_list to extract electrode selection and parameters.

   Args:
       imro_list: List from read_imro_file or generate_imro_channelmap

   Returns:
       tuple: (selected_electrodes, probe_type, probe_subtype, reference_id, ap_gain, lf_gain, hp_filter)
              selected_electrodes: numpy array of (shank_id, electrode_id) pairs
              probe_type: "1.0", "2.0-1shank", "2.0-4shanks", or "NXT"
              Other parameters: as used in original generation
   """
   header = imro_list[0]
   probe_subtype = header[0]
   entries = imro_list[1:]

   # Determine probe type from subtype
   if probe_subtype in PROBE_TYPE_MAP["1.0"]:
       probe_type = "1.0"
   elif probe_subtype in PROBE_TYPE_MAP["2.0-1shank"]:
       probe_type = "2.0-1shank"
   elif probe_subtype in PROBE_TYPE_MAP["2.0-4shanks"]:
       probe_type = "2.0-4shanks"
   elif probe_subtype in PROBE_TYPE_MAP["NXT"]:
       probe_type = "NXT"

   selected_electrodes = []

   if probe_type == "1.0":
       # Format: (channel, bank, ref, ap_gain, lf_gain, hp_filter)
       reference_id = entries[0][2]  # Same for all entries
       ap_gain = entries[0][3]
       lf_gain = entries[0][4]
       hp_filter = entries[0][5]

       # Convert reference value back to string
       ref_map = {v: k for k, v in REF_ELECTRODES[probe_subtype].items()}
       reference_id = ref_map[reference_id]

       # Extract electrodes: channel + 384*bank gives electrode_id, shank is always 0
       for entry in entries:
           channel, bank = entry[0], entry[1]
           electrode_id = channel + 384 * bank
           selected_electrodes.append([0, electrode_id])

   elif probe_type == "2.0-1shank":
       # Format: (channel, bank_mask, ref, electrode_id)
       reference_id = entries[0][2]
       ap_gain = lf_gain = hp_filter = None  # Not used in 2.0

       # Convert reference value back to string
       ref_map = {v: k for k, v in REF_ELECTRODES[probe_subtype].items()}
       reference_id = ref_map[reference_id]

       # Extract electrodes: electrode_id is directly stored, shank is always 0
       for entry in entries:
           electrode_id = entry[3]
           selected_electrodes.append([0, electrode_id])

   else:  # 2.0-4shanks or NXT
       # Format: (channel, shank_id, bank, ref, electrode_id)
       reference_id = entries[0][3]
       ap_gain = lf_gain = hp_filter = None  # Not used in 2.0

       # Convert reference value back to string
       ref_electrodes = REF_ELECTRODES[probe_subtype]
       if 'tip' in ref_electrodes and isinstance(ref_electrodes['tip'], list):
           if reference_id in ref_electrodes['tip']:
               reference_id = 'tip'
           else:
               ref_map = {v: k for k, v in ref_electrodes.items() if k != 'tip'}
               reference_id = ref_map[reference_id]
       else:
           ref_map = {v: k for k, v in ref_electrodes.items()}
           reference_id = ref_map[reference_id]

       # Extract electrodes: shank_id and electrode_id are directly stored
       for entry in entries:
           shank_id = entry[1]
           electrode_id = entry[4]
           selected_electrodes.append([shank_id, electrode_id])

   selected_electrodes = np.array(selected_electrodes, dtype=int)

   return selected_electrodes, probe_type, probe_subtype, reference_id, ap_gain, lf_gain, hp_filter

###########################
## preset configurations ##
###########################

def get_preset_candidates(preset, probe_type, wiring_df):
    """Get candidate (shank_id, electrode_id) pairs for a preset based on probe type and wiring."""

    if probe_type in ["1.0", "2.0-1shank"]:
        assert preset in SUPPORTED_1shank_PRESETS, \
            f"Preset {preset} is not supported for probe type {probe_type}. Supported presets: {SUPPORTED_1shank_PRESETS}"
    elif probe_type in ["2.0-4shanks", "NXT"]:
        assert preset in SUPPORTED_4shanks_PRESETS, \
            f"Preset {preset} is not supported for probe type {probe_type}. Supported presets: {SUPPORTED_4shanks_PRESETS}"

    preset_electrodes = []

    if probe_type == "1.0":
        # Single shank configurations for 1.0
        if preset == "tip":
            # 0-383 of bank 0
            for row in range(384):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "tip_b0_top_b1":
            # 0-191 of bank 0, 192-383 of bank 1
            for row in range(192):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, 'shank0-bank1']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "top_b0_tip_b1":
            # 192-383 of bank 0, 0-191 of bank 1
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192):
                electrode_id = wiring_df.loc[row, 'shank0-bank1']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "zigzag":
            # channels 0, 2, 4, 6, 8, 10... of bank 0 (even channels)
            for row in range(0, 384, 2):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            # channels 1, 3, 5, 7... of bank 1 (odd channels)
            for row in range(1, 384, 2):
                electrode_id = wiring_df.loc[row, 'shank0-bank1']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

    elif probe_type == "2.0-1shank":
        # Single shank configurations for 2.0-1shank
        if preset == "tip":
            # 0-383 of bank 0
            for row in range(384):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "tip_b0_top_b1":
            # 0-191 of bank 0, 192-383 of bank 1
            for row in range(192):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, 'shank0-bank1']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "top_b0_tip_b1":
            # 192-383 of bank 0, 0-191 of bank 1
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, 'shank0-bank0']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192):
                electrode_id = wiring_df.loc[row, 'shank0-bank1']
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "zigzag":
            # For 2.0 bank 0: channels 0, 3, 4, 7, 8, 11, 12... on bank 0
            # Pattern: 0,3 then +4 repeatedly: 4,7,8,11,12,15,16,19...
            channels_bank0 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank0.append(i)
                channels_bank0.append(i + 3)
                i += 4

            # For 2.0 bank 1: channels 1, 2, 5, 6, 8, 10, 13... on bank 0
            channels_bank1 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank1.append(i + 1)
                channels_bank1.append(i + 2)
                i += 4

            electrodes_bank0 = wiring_df.loc[channels_bank0, "shank0-bank0"]
            electrodes_bank1 = wiring_df.loc[channels_bank1, "shank0-bank1"]
            electrodes_bank0 = np.vstack([electrodes_bank0 * 0, electrodes_bank0]).T
            electrodes_bank1 = np.vstack([electrodes_bank1 * 0, electrodes_bank1]).T
            preset_electrodes = np.vstack([electrodes_bank0, electrodes_bank1])

    elif probe_type in ["2.0-4shanks", "NXT"]:
        # Multi-shank configurations
        if preset == "tips_all":
            # 0-95 (384/4) of each shank's bank 0
            preset_electrodes = np.arange(96)
            preset_electrodes = np.vstack([preset_electrodes * 0, preset_electrodes]).T
            for shank in range(3):
                preset_electrodes_ = preset_electrodes.copy()
                preset_electrodes_[:, 0] = shank + 1
                preset_electrodes = np.vstack([preset_electrodes, preset_electrodes_])

        elif preset.startswith("tip_s") and len(preset) == 6 and preset[5].isdigit():
            # "tip_sX" with X in [0-3] - 0-383 on shank X
            shank = int(preset[5])
            for row in range(384):
                col = f'shank{shank}-bank0'
                electrode_id = wiring_df.loc[row, col]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([shank, int(electrode_id)])

        elif preset == "tips_0_3":
            # 0-191 on shank 0 and 0-191 on shank 3
            for e in range(192):
                preset_electrodes.append([0, e])
            for e in range(192):
                preset_electrodes.append([3, e])

        elif preset == "tips_1_2":
            # 0-191 on shank 1 and 0-191 on shank 2
            for e in range(192):
                preset_electrodes.append([1, e])
            for e in range(192):
                preset_electrodes.append([2, e])

        elif preset.startswith("tip_b0_top_b1_s"):
            # "tip_sXb0_top_sXb1" with X in [0-3]
            shank = int(preset[-1])
            # 0-191 of bank 0, 192-383 of bank 1
            for e in range(192):
                preset_electrodes.append([shank, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([shank, e])

        elif preset.startswith("top_b0_tip_b1_s"):
            # "top_sXb0_tip_sXb1" with X in [0-3]
            shank = int(preset[-1])
            # 0-191 of bank 1, 192-383 of bank 0
            for e in range(192, 384):
                preset_electrodes.append([shank, e])
            for e in range(384, 192 + 384):
                preset_electrodes.append([shank, e])

        elif preset == "tip_s0b0_top_s2b0":
            # 0-191 of bank 0 shank 0, 192-383 of bank 0 shank 2
            for e in range(192):
                preset_electrodes.append([0, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([2, e])

        elif preset == "tip_s2b0_top_s0b0":
            # 0-191 of bank 0 shank 2, 192-383 of bank 0 shank 0
            for e in range(192):
                preset_electrodes.append([2, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([0, e])

        elif preset == "tip_s1b0_top_s3b0":
            # 0-191 of bank 0 shank 1, 192-383 of bank 0 shank 3
            for e in range(192):
                preset_electrodes.append([1, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([3, e])

        elif preset == "tip_s3b0_top_s1b0":
            # 0-191 of bank 0 shank 3, 192-383 of bank 0 shank 1
            for e in range(192):
                preset_electrodes.append([3, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([1, e])

        elif preset == "gliding_0-3":
            # 0-95 of shank 0, 96-191 of shank 1, 192-287 of shank 2, 288-383 of shank 3
            for e in range(96):
               preset_electrodes.append([0, e])
            for e in range(96, 192):
                preset_electrodes.append([1, e])
            for e in range(192, 288):
                preset_electrodes.append([2, e])
            for e in range(288, 384):
                preset_electrodes.append([3, e])

        elif preset == "gliding_3-0":
            # 0-95 of shank 3, 96-191 of shank 2, 192-287 of shank 1, 288-383 of shank 0
            for e in range(96):
               preset_electrodes.append([3, e])
            for e in range(96, 192):
                preset_electrodes.append([2, e])
            for e in range(192, 288):
                preset_electrodes.append([1, e])
            for e in range(288, 384):
                preset_electrodes.append([0, e])

        elif preset.startswith("zigzag_") and len(preset) == 8 and preset[7].isdigit():
            # For 2.0 bank 0: channels 0, 3, 4, 7, 8, 11, 12... on bank 0
            # Pattern: 0,3 then +4 repeatedly: 4,7,8,11,12,15,16,19...
            shank = int(preset[7])
            channels_bank0 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank0.append(i)
                channels_bank0.append(i + 3)
                i += 4

            # For 2.0 bank 1: channels 1, 2, 5, 6, 8, 10, 13... on bank 0
            channels_bank1 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank1.append(i + 1)
                channels_bank1.append(i + 2)
                i += 4

            electrodes_bank0 = wiring_df.loc[channels_bank0, "shank0-bank0"]
            electrodes_bank1 = wiring_df.loc[channels_bank1, "shank0-bank1"]
            electrodes_bank0 = np.vstack([electrodes_bank0 * 0 + shank, electrodes_bank0]).T
            electrodes_bank1 = np.vstack([electrodes_bank1 * 0 + shank, electrodes_bank1]).T
            preset_electrodes = np.vstack([electrodes_bank0, electrodes_bank1])

    return np.array(preset_electrodes, dtype=int)  # (n_electrodes, 2) array - [[shank_id, electrode_id], ...]

##########################
## Channel map plotting ##
##########################

def find_selected_electrodes(imro_list):
    "imro_list: list of tuples starting with (version, )"
    probe_type, n_channels = imro_list[0]
    if probe_type in PROBE_TYPE_MAP["1.0"]:
        probe_type = "1.0"
    elif probe_type in PROBE_TYPE_MAP["2.0-1shank"]:
        probe_type = "2.0-1shank"
    elif probe_type in PROBE_TYPE_MAP["2.0-4shanks"]:
        probe_type = "2.0-4shanks"
    elif probe_type in PROBE_TYPE_MAP["NXT"]:
        probe_type = "NXT"

    imro_table = np.array(imro_list[1:])
    if probe_type in ["1.0"]:
        selected_electrodes = imro_table[:, 0] + 384 * imro_table[:, 1]
        selected_shanks = selected_electrodes * 0
    elif probe_type in ["2.0-1shank"]:
        selected_electrodes = imro_table[:, -1]
        selected_shanks = selected_electrodes * 0
    elif probe_type in ["2.0-4shanks", "NXT"]:
        selected_electrodes = imro_table[:, -1]
        selected_shanks = imro_table[:, 1]
    selected_electrodes = np.vstack([selected_shanks, selected_electrodes]).T

    return selected_electrodes

def plot_probe_layout(probe_type,
                      imro_list,
                      positions_file,
                      wiring_file,
                      title,
                      figsize=(2, 30),
                      save_plot=False,
                      saveDir=None):
    """
    Create visualization of probe layout with selected electrodes

    figsize: (width, height) per shank in inch

    """

    # Format parameters
    selected_color = "red"
    unselected_color = "lightgrey"
    forbidden_color = "black"

    if probe_type in ["1.0", "2.0-1shank"]:
        n_shanks = 1
    elif probe_type in ["2.0-4shanks", "NXT"]:
        n_shanks = 4
    electrode_vpitch = {"1.0": 20, "2.0-1shank": 15, "2.0-4shanks": 15, "NXT": 15}

    # Get physical electrode ids
    selected_electrodes = find_selected_electrodes(imro_list)

    # Define forbidden electrodes from wiring map
    wiring_df = pd.read_csv(wiring_file)
    forbidden_electrodes = find_forbidden_electrodes(selected_electrodes, wiring_df)

    # Load positions
    positions_df = pd.read_csv(positions_file)
    positions = positions_df.values

    # Single shank
    if n_shanks == 1:

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Scale factor to make electrodes visible
        tick_width = 1
        shank_width = 20      # Shank width parameter
        electrode_width_ratio = 0.2   # Electrode width as fraction of shank width
        electrode_height = 10   # Reasonable height
        electrode_width = shank_width * electrode_width_ratio

        # Draw shank outline with pointy tip
        max_y = np.max(positions[:, -1])
        min_y = np.min(positions[:, -1])
        shank_height = max_y - min_y
        tip_height = shank_height * 0.05  # 5% of total shank height

        # Create polygon for shank shape (rectangle + triangle tip)
        shank_points = [
            (-shank_width//2, max_y + 50),  # top left
            (shank_width//2, max_y + 50),   # top right
            (shank_width//2, min_y),        # bottom right at electrode level
            (0, min_y - tip_height),        # tip point
            (-shank_width//2, min_y),       # bottom left at electrode level
        ]
        shank_poly_back = patches.Polygon(shank_points, linewidth=0, edgecolor='black',
                                   facecolor=(0.9, 0.9, 0.9, 0.2), zorder=-100)
        ax.add_patch(shank_poly_back)
        shank_poly_front = patches.Polygon(shank_points, linewidth=2, edgecolor='grey',
                                   facecolor=(0.9, 0.9, 0.9, 0), zorder=100)
        ax.add_patch(shank_poly_front)

        # Draw bank borders
        x_borders = [-shank_width//2, shank_width//2]
        for bank_i in np.arange(0, len(positions), 384):
            bank_y = positions[bank_i, -1] - electrode_vpitch[probe_type]//2
            text_y = bank_y
            if bank_i == 0: text_y -= 50
            ax.plot(x_borders, [bank_y, bank_y], ls='-', lw=1.5, c='grey', zorder=-10)
            ax.text(x_borders[1] + tick_width * 2, text_y,
                    f'Bank {bank_i//384}\nonset', ha='left', va='center',
                    fontsize=8, color='grey', zorder=-10)

        # Draw electrodes
        for (electrode, orig_x, y) in positions[:, 1:]:
            if electrode in selected_electrodes[:, 1]:
                color = selected_color
                alpha = 1.0
            elif electrode in forbidden_electrodes[:, 1]:
                color = forbidden_color
                alpha = 1
            else:
                color = unselected_color
                alpha = 0.7

            # Map original x positions to shank width (normalized then scaled)
            if probe_type in ["1.0"]:
                # Original positions: 11,27,43,59 μm (range: 48 μm, center at 35)
                # Normalize to [-1, 1] range: (x - 35) / 24
                x_norm = (orig_x - 35) / 24
                x = x_norm * (shank_width * 0.8) / 2  # Use 80% of shank width
            else:  # linear
                # Original positions: 0,32 μm (range: 32 μm, center at 16)
                # Normalize to [-1, 1] range: (x - 16) / 16
                x_norm = (orig_x - 16) / 16
                x = x_norm * (shank_width * 0.6) / 2  # Use 60% of shank width

            rect = patches.Rectangle((x - electrode_width//2, y - electrode_height//2),
                                   electrode_width, electrode_height,
                                   linewidth=0, edgecolor='black',
                                   facecolor=color, alpha=alpha)
            ax.add_patch(rect)

        ax.set_xlim(-shank_width//2 - 20, shank_width//2 + 20)
        ax.set_ylim(min_y - tip_height - 50, max_y + 100)
        ax.set_title(title)

        # Style the plot - remove frame, grid, x-ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])

        # Remove all default ticks and labels
        ax.set_yticks([])

        # Add electrode index ticks on the left (every 50 electrodes)
        electrode_ticks = range(0, len(positions), 50)
        for electrode_idx in electrode_ticks:
            if electrode_idx < len(positions):
                y_pos = positions[electrode_idx, -1]
                # Tick mark on left edge of rightmost active shank
                ax.plot([-shank_width//2, -shank_width//2 - tick_width],
                       [y_pos, y_pos], 'k-', linewidth=1)
                # Label on left side
                ax.text(-shank_width//2 - tick_width * 2,
                        y_pos,
                        str(electrode_idx),
                        ha='right', va='center', fontsize=8)

        # Add distance ticks on the right (every 500 μm)
        distance_ticks = range(0, int(max_y), 500)
        for distance in distance_ticks:
            # Tick mark on right edge of shank
            ax.plot([shank_width//2, shank_width//2 + tick_width],
                    [distance, distance],
                    'k-', linewidth=1)
            # Label on right side
            ax.text(shank_width//2 + tick_width * 2,
                    distance,
                    f'{distance}',
                    ha='left', va='center', fontsize=8)

    # multi-shank
    else:

        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] * n_shanks * 0.8, figsize[1]))

        shank_width = 80
        shank_spacing = 150
        electrode_width_ratio = 0.18
        electrode_height = 10
        electrode_width = shank_width * electrode_width_ratio
        tick_width = 5

        for shank_id in range(n_shanks):
            shank_m = positions[:,0] == shank_id
            shank_m_selected = selected_electrodes[:, 0] == shank_id
            x_center = shank_id * shank_spacing

            # Draw shank outline with pointy tip
            max_y = np.max(positions[shank_m, -1])
            min_y = np.min(positions[shank_m, -1])
            shank_height = max_y - min_y
            tip_height = shank_height * 0.05  # 5% of total shank height

            # Create polygon for shank shape (rectangle + triangle tip)
            shank_points = [
                (x_center - shank_width//2, max_y + 50),  # top left
                (x_center + shank_width//2, max_y + 50),  # top right
                (x_center + shank_width//2, min_y),       # bottom right at electrode level
                (x_center, min_y - tip_height),           # tip point
                (x_center - shank_width//2, min_y),       # bottom left at electrode level
            ]
            shank_poly = patches.Polygon(shank_points, linewidth=2, edgecolor='black',
                                       facecolor='lightgray', alpha=0.2)
            ax.add_patch(shank_poly)
            shank_poly_front = patches.Polygon(shank_points, linewidth=2, edgecolor='grey',
                                    facecolor=(0.9, 0.9, 0.9, 0), zorder=100)
            ax.add_patch(shank_poly_front)

            # Draw bank borders
            x_borders = [x_center-shank_width//2, x_center+shank_width//2]
            for bank_i in np.arange(0, len(positions[shank_m]), 384):
                bank_y = positions[shank_m][bank_i, -1] - electrode_vpitch[probe_type]//2
                text_y = bank_y
                if bank_i == 0: text_y -= 50
                ax.plot(x_borders, [bank_y, bank_y], ls='-', lw=1.5, c='grey', zorder=-10)
                if shank_id == 3:
                    ax.text(x_borders[1] + tick_width * 2, text_y,
                            f'Bank {bank_i//384}\nonset', ha='left', va='center',
                            fontsize=8, color='grey', zorder=-10)

            # Draw electrodes
            for (shank_id_pos, electrode, orig_x, y) in positions[shank_m]:
                if shank_id_pos != shank_id:
                    continue
                # Check if this electrode is selected
                if np.isin(shank_id_pos, selected_electrodes[shank_m_selected,0]) \
                 & np.isin(electrode, selected_electrodes[shank_m_selected,1]):
                    electrode_color = selected_color
                    alpha = 1.0
                elif np.isin(shank_id_pos, forbidden_electrodes[:,0]) \
                 & np.isin(electrode, forbidden_electrodes[:,1]):
                    electrode_color = forbidden_color
                    alpha = 1
                else:
                    electrode_color = unselected_color
                    alpha = 0.7

                # Map electrode positions to shank width (normalized then scaled)
                # Original positions: 0,32 μm (center at 16) - always 2.0
                x_norm = (orig_x - 16) / 16
                x = x_center + x_norm * (shank_width * 0.5) / 2

                rect = patches.Rectangle((x - electrode_width//2, y - electrode_height//2),
                                       electrode_width, electrode_height,
                                       linewidth=0, edgecolor='black',
                                       facecolor=electrode_color, alpha=alpha)
                ax.add_patch(rect)

            # Add shank label
            ax.text(x_center, min_y - tip_height - tip_height * 0.2, f'Shank {shank_id}',
                   ha='center', va='center', fontsize=12, fontweight='bold')

        # Get overall bounds for multi-shank layout

        overall_max_y = np.max(positions[:, -1])
        overall_min_y = np.min(positions[:, -1])
        overall_tip_height = (overall_max_y - overall_min_y) * 0.05

        ax.set_xlim(-shank_width, n_shanks * shank_spacing)
        ax.set_ylim(overall_min_y - overall_tip_height - 50, overall_max_y + 100)
        ax.set_title(title)

        # Style the plot - remove frame, grid, x-ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])

        # Remove all default ticks and labels
        ax.set_yticks([])

        # For multi-shank, add ticks to the leftmost and rightmost shank
        leftmost_x = 0 * shank_spacing - shank_width//2
        rightmost_x = 3 * shank_spacing + shank_width//2
        leftmost_positions = positions_df.loc[positions_df.shank == 0, "x":"y"].values
        rightmost_positions = positions_df.loc[positions_df.shank == 3, "x":"y"].values

        # Add electrode index ticks on the left of leftmost shank (every 50 electrodes)
        electrode_ticks = range(0, len(leftmost_positions), 50)
        for electrode_idx in electrode_ticks:
            if electrode_idx < len(rightmost_positions):
                y_pos = rightmost_positions[electrode_idx][1]
                # Tick mark on left edge of rightmost active shank
                ax.plot([leftmost_x, leftmost_x - tick_width],
                       [y_pos, y_pos], 'k-', linewidth=1)
                # Label on left side
                ax.text(leftmost_x - 2 * tick_width, y_pos, str(electrode_idx),
                       ha='right', va='center', fontsize=8)

        # Add distance ticks on the right of rightmost active shank (every 500 μm)
        distance_ticks = range(0, int(overall_max_y), 500)
        for distance in distance_ticks:
            # Tick mark on right edge of rightmost active shank
            ax.plot([rightmost_x, rightmost_x + tick_width],
                   [distance, distance], 'k-', linewidth=1)
            # Label on right side
            ax.text(rightmost_x + 2 * tick_width, distance, f'{distance}',
                   ha='left', va='center', fontsize=8)

    ax.set_ylabel("Vertical position (channel/μm)")
    # Remove grid
    ax.grid(False)

    # Add legend positioned away from probe
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=selected_color,
               markersize=10, label='Selected'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=unselected_color,
               markersize=10, alpha=0.7, label='Unselected'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=forbidden_color,
               markersize=10, alpha=1, label='Forbidden'),
    ]
    ax.legend(title="Electrode state:",
              handles=legend_elements,
              bbox_to_anchor=(0.9, 0))

    plt.tight_layout()

    if save_plot:
        if saveDir is None:
            saveDir = Path.cwd()
        pdf_filename = "_".join(title.replace("\n", " ").split(" ")) + ".pdf"
        plt.savefig(Path(saveDir) / pdf_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {pdf_filename}")