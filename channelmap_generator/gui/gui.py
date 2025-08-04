#!/usr/bin/env python3
"""
Interactive GUI for Neuropixels Channelmap Generation
Using Bokeh for better interactivity with hover, click, and rectangular selection
"""

import sys
import socket
from pathlib import Path
import numpy as np
import pandas as pd
import panel as pn
import param
import re
from io import BytesIO, StringIO
import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, TapTool, BoxSelectTool, ResetTool, PanTool, WheelZoomTool,
    CustomJS, Range1d,
)
from bokeh import events

# Handle imports that work both as script and as package module
try:
    # Try relative import (works when run as module: python -m channelmap_generator.gui.gui)
    from .. import backend
except ImportError:
    # Fall back to absolute import (works when run as script: python gui.py)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import backend

# Paths to assets
WIRING_MAPS_DIR = Path(__file__).resolve().parent.parent / "wiring_maps"
GUI_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Enable Panel extensions
pn.extension('tabulator')

class ChannelmapGUIBokeh(param.Parameterized):
    """Main GUI class for interactive channelmap generation using Bokeh"""

    # Parameters - will take their value as attributes after class initialization
    default_type = "2.0-4shanks"

    probe_type = param.Selector(
        default=default_type,
        objects=list(backend.PROBE_TYPE_MAP.keys()),
        doc="Neuropixels probe type"
    )

    probe_subtype = param.Selector(
        default=backend.PROBE_TYPE_MAP[default_type][0],
        objects=backend.PROBE_TYPE_MAP[default_type],
        doc="Specific probe subtype\n(don't worry too much about it - does not affect probe geometry or imro file structure)"
    )

    reference_id = param.Selector(
        default='tip',
        objects=['tip', 'ext', 'gnd'],
        doc=("Reference to use for recording (probe tip, external pad, or circuit ground (possible for some versions)."
             " Specific channels not implemented."),
    )

    hardware_hp_filter_on = param.Selector(
        default=1,
        objects=[0, 1, None],
        doc=("Whether to turn on Neuropixels 1.0 on-board hardware high-pass filter"
            " (analog equivalent to 1st-order 300Hz hp butterworth)")
    )

    preset = param.Selector(
        default=backend.SUPPORTED_4shanks_PRESETS[0],
        objects=backend.SUPPORTED_4shanks_PRESETS,
        doc="Channel map common presets"
    )
    
    shank_selector = param.Selector(
        default=0,
        objects=[0, 1, 2, 3],
        doc="Shank of electrodes selected in textbox below"
    )
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialize data
        self.wiring_maps_dir = WIRING_MAPS_DIR
        self.selected_electrodes = set()  # Set of (shank_id, electrode_id) tuples
        self.forbidden_electrodes = set()  # Set of (shank_id, electrode_id) tuples
        self.probe_plot_height = 2500
        self.probe_plot_width = 1000
        self.ap_gain_default = 500
        self.lf_gain_default = 250
        
        # Track which box select tool is being used
        self.selectbox_id = None
        
        # Load initial data
        self.load_probe_data()
        
        # Create Bokeh plot
        self.setup_bokeh_plot()
        
        # Create widgets
        self.create_widgets()


    #####################################
    ##### Probe geometry and wiring #####
    #####################################

    def load_probe_data(self):
        """Load wiring and position data for current probe type"""
        # File mapping
        file_map = {
            "1.0": ("1.0_positions.csv", "1.0_wiring.csv"),
            "2.0-1shank": ("2.0-1shank_positions.csv", "2.0-1shank_wiring.csv"),
            "2.0-4shanks": ("2.0-4shanks_positions.csv", "2.0-4shanks_wiring.csv"),
            "NXT": ("2.0-4shanks_positions.csv", "2.0-4shanks_wiring.csv")
        }
        
        pos_file, wire_file = file_map[self.probe_type]
        self.positions_file = self.wiring_maps_dir / pos_file
        self.wiring_file = self.wiring_maps_dir / wire_file

        # Probe subtype update
        self.param.probe_subtype.objects = backend.PROBE_TYPE_MAP[self.probe_type]
        self.probe_subtype = self.param.probe_subtype.objects[0]
        
        # Load data
        self.positions_df = pd.read_csv(self.positions_file)
        self.wiring_df = pd.read_csv(self.wiring_file)
        
        # Update preset options based on probe type
        if self.probe_type in ["1.0", "2.0-1shank"]:
            self.param.preset.objects = backend.SUPPORTED_1shank_PRESETS
            # For single shank probes, only shank 0 is available
            self.param.shank_selector.objects = [0]
            self.shank_selector = 0
        else:
            self.param.preset.objects = backend.SUPPORTED_4shanks_PRESETS
            # For multi-shank probes, all 4 shanks are available
            self.param.shank_selector.objects = [0, 1, 2, 3]
            if self.shank_selector not in [0, 1, 2, 3]:
                self.shank_selector = 0
        self.preset = self.param.preset.objects[0]
            
        # Reset selections
        self.selected_electrodes.clear()
        self.forbidden_electrodes.clear()
        
    def create_electrode_data(self):
        """Create the electrode data for Bokeh visualization"""
        positions = self.positions_df.values
        
        # Calculate electrode positions and colors
        electrode_data = {
            'x': [],
            'y': [],
            'width': [],
            'height': [],
            'shank_id': [],
            'electrode_id': [],
            'color': [],
            'alpha': [],
            'line_color': [],
            'line_width': [],
            'status': []
        }
        
        # Parameters for visualization
        if self.probe_type in ["1.0", "2.0-1shank"]:
            # Single shank
            shank_width = 100
            electrode_width = 15
            electrode_height = 9 if self.probe_type == "2.0-1shank" else 14
            
            for (shank_id, electrode_id, orig_x, y) in positions:
                # Map x position to shank width
                if self.probe_type == "1.0":
                    x_norm = (orig_x - 35) / 24
                    x = x_norm * (shank_width * 0.7) / 2
                else:
                    x_norm = (orig_x - 16) / 16
                    x = x_norm * (shank_width * 0.7) / 2
                
                # Determine electrode status and color
                status, color, alpha, line_color, line_width = self.get_electrode_status(shank_id, electrode_id)
                
                electrode_data['x'].append(x)
                electrode_data['y'].append(y)
                electrode_data['width'].append(electrode_width)
                electrode_data['height'].append(electrode_height)
                electrode_data['shank_id'].append(shank_id)
                electrode_data['electrode_id'].append(electrode_id)
                electrode_data['color'].append(color)
                electrode_data['alpha'].append(alpha)
                electrode_data['line_color'].append(line_color)
                electrode_data['line_width'].append(line_width)
                electrode_data['status'].append(status)
                
        else:
            # Multi-shank
            shank_width = 60
            shank_spacing = 150
            electrode_width = 12
            electrode_height = 8
            
            for (shank_id, electrode_id, orig_x, y) in positions:
                # Calculate shank center
                x_center = shank_id * shank_spacing
                
                # Map electrode position within shank
                x_norm = (orig_x - 16) / 16
                x = x_center + x_norm * (shank_width * 0.7) / 2
                
                # Determine electrode status and color
                status, color, alpha, line_color, line_width = self.get_electrode_status(shank_id, electrode_id)
                
                electrode_data['x'].append(x)
                electrode_data['y'].append(y)
                electrode_data['width'].append(electrode_width)
                electrode_data['height'].append(electrode_height)
                electrode_data['shank_id'].append(shank_id)
                electrode_data['electrode_id'].append(electrode_id)
                electrode_data['color'].append(color)
                electrode_data['alpha'].append(alpha)
                electrode_data['line_color'].append(line_color)
                electrode_data['line_width'].append(line_width)
                electrode_data['status'].append(status)
        
        # Create ColumnDataSource
        self.electrode_source = ColumnDataSource(data=electrode_data)
        
    def get_electrode_status(self, shank_id, electrode_id):
        """
        Get electrode appearance based on its status
        status, color, alpha, line_color, line_width
        """
        if (shank_id, electrode_id) in self.selected_electrodes:
            return "Selected", "red", 1.0, "darkred", 0
        elif (shank_id, electrode_id) in self.forbidden_electrodes:
            return "Forbidden", "black", 1.0, "darkgray", 0
        else: # unselected electrodes
            return "Unselected", "lightgray", 0.8, "gray", 0
            
    def setup_electrode_visualization(self):
        """Setup the electrode rectangles in Bokeh"""
        # Draw electrodes as rectangles
        self.electrode_renderer = self.plot.rect(
            x='x', y='y', 
            width='width', height='height',
            fill_color='color', fill_alpha='alpha',
            line_color='line_color', line_width='line_width',
            source=self.electrode_source,
            hover_fill_color="yellow",
            hover_line_color="orange",
            hover_line_width=3
        )
        
        # Add shank outlines and labels
        self.add_shank_outlines()
        
    def add_shank_outlines(self):
        """Add shank outlines and bank labels"""
        positions = self.positions_df.values
        
        if self.probe_type in ["1.0", "2.0-1shank"]:
            # Single shank outline
            shank_width = 100
            xlim = [-shank_width/2 - 150, shank_width/2 + 150]
            max_y = np.max(positions[:, -1])
            min_y = np.min(positions[:, -1])
            tip_height = (max_y - min_y) * 0.1
            
            # Shank outline
            shank_x = [-shank_width/2, shank_width/2, shank_width/2, 0, -shank_width/2, -shank_width/2]
            shank_y = [max_y + 100, max_y + 100, min_y, min_y - tip_height, min_y, max_y + 100]
            
            self.plot.line(shank_x, shank_y, line_width=3, color="black", alpha=1)
            self.plot.x_range=Range1d(xlim[0], xlim[1])
            
            # Bank labels
            for bank_i in np.arange(0, len(positions), 384):
                if bank_i < len(positions):
                    bank_y = positions[bank_i, -1]
                    self.plot.line([-shank_width/2, shank_width/2], [bank_y, bank_y], 
                                 line_width=2, color="gray", alpha=0.7)
                    self.plot.text([shank_width/2 + 3], [bank_y], 
                                 text=[f'Bank {bank_i//384}'], 
                                 text_font_size="10pt", text_color="gray")
                    
        else:
            # Multi-shank outlines
            shank_width = 60
            shank_spacing = 150
            xlim = [-shank_width/2 - 100, 3 * shank_spacing + shank_width/2 + 100]
            
            for shank_id in range(4):
                shank_mask = positions[:, 0] == shank_id
                x_center = shank_id * shank_spacing
                
                max_y = np.max(positions[shank_mask, -1])
                min_y = np.min(positions[shank_mask, -1])
                tip_height = (max_y - min_y) * 0.08
                
                # Shank outline
                shank_x = [x_center - shank_width/2, x_center + shank_width/2, 
                          x_center + shank_width/2, x_center,
                          x_center - shank_width/2, x_center - shank_width/2]
                shank_y = [max_y + 100, max_y + 100, min_y, min_y - tip_height,
                           min_y, max_y + 100]
                
                self.plot.line(shank_x, shank_y, line_width=3, color="black", alpha=1)
                self.plot.x_range=Range1d(xlim[0], xlim[1])
                
                # Shank label
                self.plot.text([x_center], [min_y - tip_height - 100], 
                             text=[f'Shank {shank_id}'], 
                             text_font_size="12pt", text_color="black", 
                             text_align="center")
                
                # Bank lines and labels
                for bank_i in np.arange(0, len(positions[shank_mask]), 384):
                    if bank_i < len(positions[shank_mask]):
                        bank_y = positions[shank_mask][bank_i, -1]
                        self.plot.line([x_center - shank_width/2, x_center + shank_width/2], 
                                        [bank_y, bank_y], line_width=2, color="gray", alpha=0.7)
                        if shank_id == 3: # Bank labels (only on rightmost shank)
                            self.plot.text([x_center + shank_width/2 + 5], [bank_y], 
                                            text=[f'Bank {bank_i//384}'], 
                                            text_font_size="10pt", text_color="gray")
        
        # Set appropriate axis limits
        self.plot.axis.visible = False
        self.plot.grid.visible = False
    

    #####################################
    ##### Electrode selection logic #####
    #####################################

    def setup_interactions(self):
        """Setup click and selection interactions"""
        # Set up the TapTool to enable single electrode selection
        tap_tool = self.plot.select_one(TapTool)
        if tap_tool:
            tap_tool.callback = CustomJS(args=dict(source=self.electrode_source), code="""
                console.log('Tap tool activated');
                const indices = source.selected.indices;
                console.log('Selected indices:', indices);
            """)
        
        # Python callbacks for interactions - this is the key part
        self.electrode_source.selected.on_change('indices', self.on_electrode_selection)
        
    def get_max_electrodes(self):
        """Get maximum allowed electrodes for current probe type"""
        if self.probe_type in ["1.0", "2.0-1shank", "2.0-4shanks"]:
            return 384
        elif self.probe_type == "NXT":
            return 1536
        else:
            return 384  # Default fallback
    
    def can_select_electrode(self, shank_id, electrode_id):
        """Check if an electrode can be selected"""
        # Don't select if already forbidden
        if (shank_id, electrode_id) in self.forbidden_electrodes:
            return False
        
        # Don't select if we're at the electrode limit and this electrode isn't already selected
        if (shank_id, electrode_id) not in self.selected_electrodes:
            max_electrodes = self.get_max_electrodes()
            if len(self.selected_electrodes) >= max_electrodes:
                print(f"Cannot select more electrodes: limit is {max_electrodes} for {self.probe_type}")
                return False
        
        return True
    
    def on_electrode_selection(self, attr, old, new):
        """Handle electrode selection changes (both tap and box select)"""
        if not new:  # No selection
            return
            
        print(f"Selection changed: {old} -> {new}")
        max_electrodes = self.get_max_electrodes()

        # Identify type of selected box
        self.selectbox_id = "???"
        
        # For single tap (one electrode)
        if len(new) == 1:
            idx = new[0]
            shank_id = self.electrode_source.data['shank_id'][idx]
            electrode_id = self.electrode_source.data['electrode_id'][idx]
            
            print(f"Single tap: electrode {electrode_id} on shank {shank_id}")
            
            # Toggle electrode selection (only if allowed)
            if (shank_id, electrode_id) in self.selected_electrodes:
                # Always allow deselection
                self.selected_electrodes.remove((shank_id, electrode_id))
                print(f"Deselected electrode {electrode_id}")
            else:
                # Check if we can select this electrode
                if self.can_select_electrode(shank_id, electrode_id):
                    self.selected_electrodes.add((shank_id, electrode_id))
                    print(f"Selected electrode {electrode_id}")
                else:
                    print(f"Cannot select electrode {electrode_id} (forbidden or limit reached)")
                    
        # For box select (multiple electrodes)
        elif len(new) > 1:
            # Determine which box tool was used by checking active tool
            # Bokeh JS does not expose selected tools to python API...
            # https://stackoverflow.com/questions/58210752/how-to-get-currently-active-tool-in-bokeh-figure
            # no easy fix
            # self.deselect_mode = False # in the future, use self.selectbox_id
            
            if self.select_mode == "deselect":
                print(f"Box deselect: {len(new)} electrodes")
                
                # Only deselect currently selected (red) electrodes
                valid_deselections = []
                
                for idx in new:
                    shank_id = self.electrode_source.data['shank_id'][idx]
                    electrode_id = self.electrode_source.data['electrode_id'][idx]
                    
                    # Only deselect if it's currently selected (red)
                    if (shank_id, electrode_id) in self.selected_electrodes:
                        valid_deselections.append((shank_id, electrode_id))
                        self.selected_electrodes.remove((shank_id, electrode_id))
                
                print(f"Box deselect: removed {len(valid_deselections)} selected electrodes")
                if len(valid_deselections) < len(new):
                    print(f"Skipped {len(new) - len(valid_deselections)} electrodes (not selected)")
            
            elif self.select_mode == "select":
                print(f"Box select: {len(new)} electrodes")
                
                # Only add electrodes that can be selected (grey/unselected)
                valid_selections = []
                current_count = len(self.selected_electrodes)
                
                for idx in new:
                    shank_id = self.electrode_source.data['shank_id'][idx]
                    electrode_id = self.electrode_source.data['electrode_id'][idx]
                    
                    # Only select if it's currently unselected (grey) and we haven't hit the limit
                    if (shank_id, electrode_id) not in self.selected_electrodes and \
                       (shank_id, electrode_id) not in self.forbidden_electrodes and \
                       current_count < max_electrodes:
                        valid_selections.append((shank_id, electrode_id))
                        self.selected_electrodes.add((shank_id, electrode_id))
                        self.update_forbidden_electrodes()
                        current_count += 1
                
                print(f"Box select: added {len(valid_selections)} valid electrodes")
                if len(valid_selections) < len(new):
                    print(f"Skipped {len(new) - len(valid_selections)} electrodes (forbidden or limit reached)")

            elif self.select_mode == "zigzag_select":
                print(f"Box zigzag select: {len(new)} electrodes")
                # zigzag logic - even electrodes in 1.0, 0, 3, 4, 7, 8... if 2.0
                N_per_shank = backend.PROBE_N[self.probe_type]['N']
                if self.probe_type == "1.0":
                    zigzag_subset = np.arange(0, N_per_shank, 2)
                else:
                    zigzag_subset = []
                    i = 0
                    while i < N_per_shank:  # Limit to 192 channels for bank 0
                        zigzag_subset.append(i)
                        zigzag_subset.append(i + 3)
                        i += 4
                    zigzag_subset = np.array(zigzag_subset)

                # Only add electrodes that can be selected (grey/unselected)
                valid_selections = []
                current_count = len(self.selected_electrodes)
                
                for idx in new:
                    shank_id = self.electrode_source.data['shank_id'][idx]
                    electrode_id = self.electrode_source.data['electrode_id'][idx]
                    
                    # Only select if it's currently unselected (grey) and we haven't hit the limit
                    if (shank_id, electrode_id) not in self.selected_electrodes and \
                       (shank_id, electrode_id) not in self.forbidden_electrodes and \
                       electrode_id in zigzag_subset and \
                       current_count < max_electrodes:
                        valid_selections.append((shank_id, electrode_id))
                        self.selected_electrodes.add((shank_id, electrode_id))
                        self.update_forbidden_electrodes()
                        current_count += 1
                
                print(f"Box select: added {len(valid_selections)} valid electrodes")
                if len(valid_selections) < len(new):
                    print(f"Skipped {len(new) - len(valid_selections)} electrodes (forbidden or limit reached)")
        
        # Update forbidden electrodes and visualization
        self.update_forbidden_electrodes()
        self.update_electrode_colors()
        self.update_electrode_counter()
        
        # Clear the selection to allow for new interactions
        self.electrode_source.selected.indices = []
            
    def update_forbidden_electrodes(self):
        """Update forbidden electrodes based on current selection"""
        if self.selected_electrodes:
            selected_array = np.array(list(self.selected_electrodes))
            forbidden_array = backend.find_forbidden_electrodes(selected_array, self.wiring_df)
            self.forbidden_electrodes = set(map(tuple, forbidden_array))
        else:
            self.forbidden_electrodes.clear()

    #########################################
    ##### Electrode selection esthetics #####
    #########################################

    def update_electrode_colors(self):
        """Update electrode colors in the Bokeh plot"""
        n_electrodes = len(self.electrode_source.data['shank_id'])
        
        colors = []
        alphas = []
        line_colors = []
        line_widths = []
        statuses = []
        
        for i in range(n_electrodes):
            shank_id = self.electrode_source.data['shank_id'][i]
            electrode_id = self.electrode_source.data['electrode_id'][i]
            
            status, color, alpha, line_color, line_width = self.get_electrode_status(shank_id, electrode_id)
            
            colors.append(color)
            alphas.append(alpha)
            line_colors.append(line_color)
            line_widths.append(line_width)
            statuses.append(status)
        
        # Update the data source
        self.electrode_source.data.update({
            'color': colors,
            'alpha': alphas,
            'line_color': line_colors,
            'line_width': line_widths,
            'status': statuses
        })
        
    def apply_preset(self):
        """Apply selected preset configuration"""
        if self.preset:
            try:
                preset_electrodes = backend.get_preset_candidates(
                    self.preset, self.probe_type, self.wiring_df
                )
                self.selected_electrodes = set(map(tuple, preset_electrodes))
                self.update_forbidden_electrodes()
                self.update_electrode_colors()
                self.update_electrode_counter()
            except Exception as e:
                print(f"Error applying preset: {e}")
                
    def parse_electrode_input(self, text):
        """Parse electrode input string like '1,2,3,5,7' or '1-5,7'"""
        if not text.strip():
            return []
            
        electrodes = []
        # Use the selected shank from the shank selector
        shank_id = self.shank_selector
        
        # Split by commas and process each part
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check for range notation (e.g., '1-5' or '1..5')
            range_match = re.match(r'(\d+)[-.]\.?(\d+)', part)
            if range_match:
                start, end = map(int, range_match.groups())
                for e_id in range(start, end + 1):
                    electrodes.append((shank_id, e_id))
            else:
                # Single electrode
                try:
                    e_id = int(part)
                    electrodes.append((shank_id, e_id))
                except ValueError:
                    continue
                    
        return electrodes
    
    def apply_electrode_input(self):
        """Add electrodes from text input to current selection (same logic as box selection)"""
        try:
            text = self.electrode_input.value
            electrodes = self.parse_electrode_input(text)
            
            # Validate electrodes exist in the probe
            max_electrode = self.positions_df['electrode'].max()
            max_allowed = self.get_max_electrodes()
            
            # Use same logic as box selection: only add unselected (grey) electrodes
            added_count = 0
            skipped_forbidden = 0
            skipped_selected = 0
            skipped_limit = 0
            
            for shank_id, electrode_id in electrodes:
                if 0 <= electrode_id <= max_electrode:
                    # Check if electrode is already selected
                    if (shank_id, electrode_id) in self.selected_electrodes:
                        skipped_selected += 1
                        continue
                    
                    # Check if electrode is forbidden
                    if (shank_id, electrode_id) in self.forbidden_electrodes:
                        skipped_forbidden += 1
                        continue
                    
                    # Check if we're at the limit
                    if len(self.selected_electrodes) >= max_allowed:
                        skipped_limit += 1
                        continue
                    
                    # Add the electrode (it's grey/unselected)
                    self.selected_electrodes.add((shank_id, electrode_id))
                    added_count += 1
                    
            # Update forbidden electrodes and visualization
            if added_count > 0:
                self.update_forbidden_electrodes()
                self.update_electrode_colors()
                self.update_electrode_counter()
                
            # Provide feedback
            print(f"Text input on shank {self.shank_selector}:")
            print(f"  Added: {added_count} electrodes")
            if skipped_selected > 0:
                print(f"  Skipped: {skipped_selected} already selected")
            if skipped_forbidden > 0:
                print(f"  Skipped: {skipped_forbidden} forbidden")
            if skipped_limit > 0:
                print(f"  Skipped: {skipped_limit} due to limit ({max_allowed})")
                
        except Exception as e:
            print(f"Error parsing electrode input: {e}")

    def clear_selection(self):
        """Clear all selected electrodes"""
        self.selected_electrodes.clear()
        self.forbidden_electrodes.clear()
        self.update_electrode_colors()
        self.update_electrode_counter()

    ###############################
    ##### IMRO and PDF output #####
    ###############################

    def ready_to_download(self):
        # Check number of selected electrodes
        if not self.selected_electrodes:
            print("No electrodes selected")
            return False
        n_selected = len(self.selected_electrodes)
        max_allowed = self.get_max_electrodes()
        if n_selected < max_allowed:
            print(f"IMRO file NOT generated - you still have {max_allowed - n_selected} electrodes to select!")
            return False
        return True

    def update_filename(self, reset=False):
        if reset:
            self.filename_input.value = f"manual_selection_{self.probe_type}"
        self.filename_input.value = self.filename_input.value.replace(".imro", "").replace(".pdf", "")
        self.download_imro_button.filename = f"{self.filename_input.value}.imro"
        self.download_pdf_button.filename = f"{self.filename_input.value}.pdf"

    def generate_imro(self):
        """Generate IMRO file from current selection"""

        # Convert selection to array format
        selected_array = np.array(list(self.selected_electrodes))
        
        # Generate IMRO list
        self.imro_list = backend.generate_imro_channelmap(
            probe_type = self.probe_type,
            custom_electrodes = selected_array,
            wiring_file = self.wiring_file,
            layout_preset = None,
            reference_id = self.reference_id,
            probe_subtype = self.probe_subtype,
            ap_gain = self.ap_gain_input.value,
            lf_gain = self.lf_gain_input.value,
            hp_filter = self.hardware_hp_filter_on)
        
    def generate_imro_content(self):

        if not self.ready_to_download():
            return

        # Update filename and generate IMRO list
        self.update_filename()
        self.generate_imro()

        # Convert list to string
        content_lines = []
        header = self.imro_list[0]
        content_lines.append(f"({header[0]},{header[1]})")
        
        for entry in self.imro_list[1:]:
            entry_str = " ".join(str(x) for x in entry)
            content_lines.append(f"({entry_str})")
        
        content = "".join(content_lines)
        return StringIO(content)

    def generate_pdf_content(self):

        if not self.ready_to_download():
            return

        # Update filename and generate IMRO list
        self.update_filename()
        self.generate_imro()

        # Create memory buffer
        buffer = BytesIO()
        
        # Make figure
        title = f"Custom channelmap\n{self.probe_type}"
        backend.plot_probe_layout(self.probe_type,
                    self.imro_list,
                    self.positions_file,
                    self.wiring_file,
                    title,
                    figsize=(2, 30),
                    save_plot=False)
        
        # Save current figure to buffer
        plt.savefig(buffer, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        return buffer

    def apply_uploaded_imro(self):
        if not any(self.imro_file_loader.value):
            print("You must upload an imro file before applying it to the selection.")
            return
        # imro_file_content = list(self.imro_file_loader.value.values())[0] # for FileDropper() widget
        imro_file_content = self.imro_file_loader.value
        if isinstance(imro_file_content, bytes):
            imro_file_content = imro_file_content.decode('utf-8')
        imro_list = backend.parse_imro_file(imro_file_content.strip())
        (selected_electrodes,
        self.probe_type, # probe_type value is a monitored param - simply setting its value will update the plot
        self.probe_subtype,
        self.reference_id,
        self.ap_gain_input.value,
        self.lf_gain_input.value,
        self.hardware_hp_filter_on,
        ) = backend.parse_imro_list(imro_list)

        self.selected_electrodes = set(map(tuple, selected_electrodes))
        self.update_forbidden_electrodes()
        self.update_electrode_colors()
        self.update_electrode_counter()

    ######################
    ##### GUI layout #####
    ######################

    def setup_bokeh_plot(self):
        """Setup the Bokeh plot for electrode visualization"""
        # Create custom box select tools with visual distinction
        self.select_mode = None

        select_box_string = "Select Electrodes"
        deselect_box_string = "Deselect Electrodes"
        zigzagselect_box_string = "Zigzag-select Electrodes"
        
        self.box_select_tool = BoxSelectTool(
            description=select_box_string,
            icon=str(GUI_ASSETS_DIR / "selector.png")
        )
        
        self.box_deselect_tool = BoxSelectTool(
            description=deselect_box_string,
            icon=str(GUI_ASSETS_DIR / "deselector.png")
        )

        self.box_zigzagselect_tool = BoxSelectTool(
            description=zigzagselect_box_string,
            icon=str(GUI_ASSETS_DIR / "zigzag_selector.png")
        )
        
        # Create figure with proper tools
        tools = [
            PanTool(),
            WheelZoomTool(),
            self.box_select_tool,
            self.box_deselect_tool,
            self.box_zigzagselect_tool,
            TapTool(),
            ResetTool(),
            HoverTool(tooltips=[
                ("Electrode", "@electrode_id"),
                ("Shank", "@shank_id"),
                ("Position", "(@x, @y)"),
                ("Status", "@status")
            ])
        ]
            
        self.plot = figure(
            width=self.probe_plot_width,
            height=self.probe_plot_height,
            tools=tools,
            title=f"Neuropixels {self.probe_type} Electrode Layout",
            toolbar_location="right",
        )
        
        # Create electrode data and visualization
        self.create_electrode_data()
        self.setup_electrode_visualization()
        self.setup_interactions() # Only necessary for the tap tool

        # Hidden data source for tool state communication and CustomJS to monitor tool changes
        self.tool_state_source = ColumnDataSource(data={'active_tool': ['']})
        self.setup_tool_monitoring(select_box_string,
                                   deselect_box_string,
                                   zigzagselect_box_string)

    def setup_tool_monitoring(self,
                              select_box_string,
                              deselect_box_string,
                              zigzagselect_box_string):
        """JS callback to expose active tools in toolbar"""
        js_code = """
        let tools_info = 'Could not check tools';
        const tools = cb_obj.origin.toolbar.tools;
        
        if (tools && tools.length > 0) {
            // Find our specific tools
            let select_tool = null;
            let deselect_tool = null;
            let zigzagselect_tool = null;
            
            tools.forEach((tool, index) => {
                if (tool.description === 'select_box_string') {
                    select_tool = tool;
                }
                if (tool.description === 'deselect_box_string') {
                    deselect_tool = tool;
                }
                if (tool.description === 'zigzagselect_box_string') {
                    zigzagselect_tool = tool;
                }
            });
            
            if (select_tool && deselect_tool && zigzagselect_tool) {
                if (select_tool.active) {
                    tools_info = 'select';
                } else if (deselect_tool.active) {
                    tools_info = 'deselect';
                } else if (zigzagselect_tool.active) {
                    tools_info = 'zigzag_select';
                } else {
                    tools_info = 'neither_active';
                }
            } else {
                tools_info = 'tools_not_found';
            }
        }
        
        tool_state.data = {active_tool: [tools_info]};
        """
        js_code = js_code.replace('zigzagselect_box_string', zigzagselect_box_string)
        js_code = js_code.replace('deselect_box_string', deselect_box_string)
        js_code = js_code.replace('select_box_string', select_box_string)
        
        JS_selection_monitor = CustomJS(
            args=dict(tool_state=self.tool_state_source),
            code=js_code
        )
        
        # Hack - monitor random GUI events to trigger javascript fetch of selection box type
        self.plot.js_on_event('selectiongeometry', JS_selection_monitor) # Selection events
        self.plot.js_on_event('tap', JS_selection_monitor)               # Plot clicks
        self.plot.js_on_event(events.MouseMove, JS_selection_monitor)    # Mouse move
        self.tool_state_source.on_change('data', self.on_tool_state_change)

    def on_tool_state_change(self, attr, old, new):
        """Handle tool detection during selection"""
        if new.get('active_tool'):
            active_tool = new['active_tool'][0]
            print(f"Selection made with tool: {active_tool}")
            
            if active_tool == 'select':
                self.select_mode = "select"
                print("→ SELECT box activated")
            elif active_tool == 'deselect':
                self.select_mode = "deselect"
                print("→ DESELECT box activated")
            elif active_tool == 'zigzag_select':
                self.select_mode = "zigzag_select"
                print("→ ZIGZAG-SELECT box activated")
            else:
                self.select_mode = "select"
                print(f"→ Unexpected result: {active_tool}, defaulting to SELECT box")

    def create_widgets(self):
        """Create Panel widgets"""
        # Probe type selector
        self.probe_type_selector = pn.Param(
            self, parameters=['probe_type'],
            widgets={'probe_type': {'type': pn.widgets.Select, 'width': 300, 'margin': 0}},
            show_name=False,
        )
        self.probe_subtype_selector = pn.Param(
            self, parameters=['probe_subtype'],
            widgets={'probe_subtype': {'type': pn.widgets.Select, 'width': 140, 'margin': 0}},
            show_name=False,
        )
        
        # Preset selector
        self.preset_selector = pn.Param(
            self, parameters=['preset'],
            widgets={'preset': {'type': pn.widgets.Select, 'margin': 0}},
            show_name=False,
        )
        
        # Apply preset button
        self.apply_button = pn.widgets.Button(
            name="Apply Preset", 
            button_type="primary",
            width=250
        )
        self.apply_button.on_click(lambda event: self.apply_preset())
        
        # Clear selection button (moved to top, orange styling)
        self.clear_button = pn.widgets.Button(
            name="Clear Selection",
            button_type="danger",
            width=120,
            margin=(0, 10, 10, 10),
            align='center',
        )
        self.clear_button.on_click(lambda event: self.clear_selection())
        
        # Shank selector for text input
        self.shank_selector_widget = pn.Param(
            self, parameters=['shank_selector'],
            widgets={'shank_selector': {'type': pn.widgets.Select, 'margin': 0}},
            show_name=False,
        )
        
        # Electrode input
        self.electrode_input = pn.widgets.TextInput(
            name="Electrode Selection",
            placeholder="e.g., 1,2,3,5,7 or 1-5,7",
            width=300
        )
        
        # Apply electrode input button
        self.apply_input_button = pn.widgets.Button(
            name="Add Electrodes to Selection",
            button_type="primary",
            width=250
        )
        self.apply_input_button.on_click(lambda event: self.apply_electrode_input())
        
        # IMRO directory and filename inputs (split into two)
        self.filename_input = pn.widgets.TextInput(
            name="IMRO or PDF filename (omit extension)",
            value=f"manual_selection_{self.probe_type}", # default filename defined here
            width=300
        )
        
        self.download_imro_button = pn.widgets.FileDownload(
            callback=self.generate_imro_content,
            filename=f"{self.filename_input.value}.imro",
            button_type="success",
            width=140,
            icon='file-text',
            label="Download IMRO ⬇",
        )

        self.download_pdf_button = pn.widgets.FileDownload(
            callback=self.generate_pdf_content,
            filename=f"{self.filename_input.value}.pdf", 
            button_type="success",
            width=140,
            icon='file-type-pdf',
            label="Download PDF ⬇"
        )
        
        # Prominent electrode counter (moved to top)
        self.electrode_counter = pn.pane.HTML(
            """
            <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px; 
                        padding: 12px; text-align: center; font-size: 16px; font-weight: bold; 
                        color: #2c3e50;">
                Selected Electrodes: 0/384
            </div>
            """,
            width=300,
            margin=(10, 10),
            align='center',
        )
        
        # Bokeh pane with proper scrolling - larger viewport for better visibility
        self.plot_pane = pn.pane.Bokeh(
            self.plot,
            width=self.probe_plot_width,
            height=self.probe_plot_height,  # Larger viewport height for better electrode visibility
        )

        # Reference and gain inputs
        self.reference_selector_widget = pn.Param(
            self, parameters=['reference_id'],
            widgets={'reference_id': {'type': pn.widgets.Select,  'width': 140, 'margin': 0}},
            show_name=False,
        )

        self.ap_gain_input = pn.widgets.FloatInput(
            name="High-pass 'ap' gain",
            value=self.ap_gain_default,
            step=1e-1, start=0, end=5000,
            width=140
        )

        self.lf_gain_input = pn.widgets.FloatInput(
            name="Low-pass 'lf' gain",
            value=self.lf_gain_default,
            step=1e-1, start=0, end=5000,
            width=140
        )

        self.hardware_hp_filter_on_selector_widget= pn.Param(
            self, parameters=['hardware_hp_filter_on'],
            widgets={'hardware_hp_filter_on': {'type': pn.widgets.Select,  'width': 300, 'margin': 0}},
            show_name=False,
        )

        # IMRO file dropper
        self.imro_file_loader = pn.widgets.FileInput(accept=".imro", width = 300)
        self.apply_uploaded_imro_button = pn.widgets.Button(
            name="Apply uploaded IMRO file to selection ⬆",
            button_type="primary",
            width=250
        )
        self.apply_uploaded_imro_button.on_click(lambda event: self.apply_uploaded_imro())

        
    def create_layout(self):
        """Create the main Panel layout"""

        # Controls panel (fixed on left)
        controls = pn.Column(
            # Prominent electrode counter at top
            self.electrode_counter,
            self.clear_button,

            pn.Column(
                pn.pane.Markdown("## Probe and recording metadata", margin=(-5, 0, 0, 10)),
                pn.pane.Markdown("(see <a href='https://billkarsh.github.io/SpikeGLX/help/imroTables' target='_blank'>IMRO table anatomy</a>)",
                                margin=(-15, 0, -5, 10)),
                
                self.probe_type_selector,
                pn.Row(
                    self.probe_subtype_selector,
                    self.reference_selector_widget,
                    sizing_mode='stretch_width',
                ),
                pn.pane.Markdown("<b>For 1.0 only (2.0 gains not in channelmap):</b>",
                                margin=(0, 0, -15, 10)),
                self.hardware_hp_filter_on_selector_widget,
                pn.Row(
                    self.ap_gain_input,
                    self.lf_gain_input,
                    sizing_mode='stretch_width',
                ),
                styles={'background': "#e6e6e6", 'padding': '10px', 'border-radius': '5px'},
                margin=(0, 5, 0, 5),
            ),

            pn.pane.Markdown("## Preset Selection", margin=(10, 0, -5, 10)),
            self.preset_selector,
            self.apply_button,
            
            pn.pane.Markdown("## Textual Selection", margin=(10, 0, -5, 10)),
            self.shank_selector_widget,
            self.electrode_input,
            self.apply_input_button,

            pn.pane.Markdown("## Selection from IMRO file",
                             margin=(10, 0, -5, 10)),
            self.imro_file_loader,
            # pn.Spacer(height=30),
            self.apply_uploaded_imro_button,
            
            pn.pane.Markdown("## Export Channelmap", margin=(10, 0, -5, 10)),
            self.filename_input,
            pn.Row(
                self.download_imro_button,
                self.download_pdf_button,
                sizing_mode='stretch_width',
                ),

            pn.pane.Markdown("## Instructions", margin=(10, 0, -5, 10)),
            pn.pane.HTML("""
            <div style="font-size: 13px; line-height: 1.4; text-align: justify;">
            <b>Neuropixels hardware Constraints:</b><br>
            Neuropixels electrodes are <a href='https://www.neuropixels.org/support' target='_blank'>hardwired</a> to specific ADCs in the probe's head. When you select an electrode, others become unavailable because they share the same recording lines.
            This GUI allows you to build a channelmap around those constraints: when you select channels, they turn <font color="#c00000"><b>red</b></font>, and those that become unavailable because they share the same lines turn <b>black</b>.<br><br>
            
            <b>You can mix and match four selection methods:</b><br>
            • <b>Presets:</b> Pre-configured channelmaps that respect wiring constraints<br>
            • <b>Textual selection:</b> Type electrode ranges (e.g., "1-10,20-25") to add to the current selection<br>
            • <b>Interactive:</b> Click electrodes directly or drag boxes (selection, deselection, or "zigzag selection") to maually select multiple sites<br>
            • <b>Selection from pre-existing IMRO file</b>: you can pre-load an IMRO file as a starting point before doing any of the above.<br><br>

            Once you reach the <b>target number of electrodes</b> for the selected probe type (384 or 1536), you can <b>download your channelmap</b> as an IMRO file alongside a PDF rendering to easily remember what your channelmap looks like.
            </div>
            """),
            styles={
                'position': 'fixed',
                'top': '0px',
                'left': '0px',
                'height': '100vh',
                'overflow-y': 'auto',  # Allow scrolling within controls if needed
                'z-index': '1000',
                'background': 'white',
                'border-right': '1px solid #ddd',
            },
            width=370,
            scroll=False
        )
        
        # Main layout with properly scrollable plot container
        plot_container = pn.Column(
            self.plot_pane,
            width=self.probe_plot_width,
            height=self.probe_plot_height,  # Match the plot pane height
            scroll=True,  # Enable scrolling for the plot container
        )

        layout = pn.Row(
            controls,
            pn.Spacer(width=370),  # Slightly wider than controls (350 + margin)
            plot_container,
            sizing_mode='stretch_width'
        )
        
        return layout
    
    ################################
    ##### Probe version update #####
    ################################

    @param.depends('probe_type', watch=True)
    def on_probe_type_change(self):
        """
        Handle probe type changes, which require to reset the whole plot.
        Param module monitors value changes of probe_type.
        """
        self.load_probe_data()
        self.setup_bokeh_plot()
        self.update_electrode_counter()
        self.update_filename(reset=True)
        
        # Update the plot pane
        # ("object" attr of plot_pane is the bokeh plot self.plot)
        if hasattr(self, 'plot_pane'):
            self.plot_pane.object = self.plot
    
    def update_electrode_counter(self):
        """Update status information"""
        n_selected = len(self.selected_electrodes)
        # n_forbidden = len(self.forbidden_electrodes)
        max_allowed = self.get_max_electrodes()
        # n_remaining = max_allowed - n_selected
        
        # Update electrode counter
        if hasattr(self, 'electrode_counter'):
            if n_selected < max_allowed:
                counter_html = f"""
                <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px; 
                            padding: 12px; text-align: center; font-size: 16px; font-weight: bold; 
                            color: #AA4A44;">
                    Selected Electrodes: {n_selected}/{max_allowed}
                </div>
                """
            else:
                counter_html = f"""
                <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px; 
                            padding: 12px; text-align: center; font-size: 16px; font-weight: bold; 
                            color: #008000;">
                    Selected Electrodes: {n_selected}/{max_allowed}<br>
                    Ready for IMRO file generation!
                </div>
                """
            self.electrode_counter.object = counter_html


## App creation utilities

def find_free_port(start_port=5008):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports found")


def create_app():
    """Create and configure the Panel app"""
    gui = ChannelmapGUIBokeh()
    layout = gui.create_layout()
    
    # Update status initially
    gui.update_electrode_counter()
    
    return layout


def main():
    print(GUI_ASSETS_DIR)

    # Create app
    app = create_app()
    
    # Serve the app
    port = find_free_port(5008)
    pn.serve(app, port=port, show=True, title="Neuropixels Channelmap Generator (Bokeh)")

if __name__ == "__main__":
    main()
    