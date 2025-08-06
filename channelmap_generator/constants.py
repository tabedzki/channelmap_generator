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

