import numpy as np
import awkward as ak
from glob import glob
import lh5
import matplotlib

matplotlib.use("Agg")
from reboost_functions import align_detectors, build_hardware_triggers, build_hits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
import imageio.v2 as imageio
from tqdm import tqdm


def create_time_bins(
    times,
    n_frames=None,
    log=False,
    tmin=None,
    tmax=None,
    t_short_min=None,
    t_short_max=None,
    n_short_frames=None,
):
    """
    Create time bins and assign each event to a frame index.

    Parameters
    ----------
    times : array-like
        Event times
    n_frames : int, optional
        Number of frames (overrides duration)
    log : bool
        Use logarithmic time binning
    tmin, tmax : float, optional
        Time range (defaults to data min/max)
    t_short_min, t_short_max : float, optional
        Short time range for better resolution of early events. Will be logscale.
    n_short_frames : int, optional
        Number of frames for the short time range.

    Returns
    -------
    frame_idx : np.ndarray
        Frame index for each event. -1 means the event is before the first bin, n_frames means it's after the last bin.
        These cases need to be filtered out afterwards.
    bins : np.ndarray
        Bin edges
    """

    times = np.asarray(times)

    # Set time range
    if tmin is None:
        tmin = times.min()
    if tmax is None:
        tmax = times.max()

    # Create bins
    if log:
        if tmin <= 0:
            raise ValueError("Log binning requires tmin > 0")
        bins = np.logspace(np.log10(tmin), np.log10(tmax), n_frames + 1)
    else:
        bins = np.linspace(tmin, tmax, n_frames + 1)

    if t_short_min is not None:
        if t_short_min <= 0:
            raise ValueError("Log binning requires t_short_min > 0")
        if t_short_max is None:
            raise ValueError("t_short_max must be specified when t_short_min is set")
        if n_short_frames is None:
            raise ValueError("n_short_frames must be specified when t_short_min is set")
        short_bins = np.logspace(
            np.log10(t_short_min), np.log10(t_short_max), n_short_frames + 1
        )
        bins = np.unique(np.concatenate((short_bins, bins)))

    return bins


def build_frames_x(
    times,
    x,
    z,
    particles,
    bins,
    times_scintillator=None,
    x_scintillator=None,
    z_scintillator=None,
    particles_scintillator=None,
    optical_triggers=None,
    optical_hits=None,
):
    """
    Convert (x,y,z,time) → awkward array of shape:
        n_frames * var * {x, z}

    Parameters
    ----------
    times, x, y, z : array-like
    particles : array-like
        PDG codes
    bins : array-like
        Time bin edges (length = n_frames + 1)
    times_scintillator, x_scintillator, y_scintillator, z_scintillator : array-like, optional
        Arguments for scintillator energy deps (if applicable)
    times_argon, x_argon, y_argon, z_argon : array-like, optional
        Arguments for argon energy deps (if applicable)

    argon_values : optional
    if given also adds r and z positions for the argon values to the output array.

    Returns
    -------
    ak.Record
        Awkward record with frame-binned data:

        tracks : ak.Array
            Particle tracks per time frame with fields:
            birth_frame, x, z, particle.

        scintillator : ak.Array, optional
            Same structure as tracks for scintillator inputs
            (if provided).

        optical : ak.Array, optional
            Detector hits per frame with fields:
            birth_frame, det_uid
            (if optical_triggers provided).

        All outputs are sorted by frame and restricted to valid bins.
    """

    n_frames = len(bins) - 1

    # --- binning ---
    birth_frame = np.digitize(times, bins) - 1

    # --- remove out-of-range ---
    valid = (birth_frame >= 0) & (birth_frame < n_frames)

    birth_frame = birth_frame[valid]
    x = x[valid]
    z = z[valid]
    particles = particles[valid]

    # --- sort by frame (Option B core idea) ---
    order = np.argsort(birth_frame)

    birth_frame = birth_frame[order]
    x = x[order]
    z = z[order]
    particles = particles[order]

    # --- build awkward array ---
    frames = ak.zip({"birth_frame": birth_frame, "x": x, "z": z, "particle": particles})

    if times_scintillator is not None:
        birth_frame_scintillator = np.digitize(times_scintillator, bins) - 1
        valid_scintillator = (birth_frame_scintillator >= 0) & (
            birth_frame_scintillator < n_frames
        )

        birth_frame_scintillator = birth_frame_scintillator[valid_scintillator]
        x_scintillator = x_scintillator[valid_scintillator]
        z_scintillator = z_scintillator[valid_scintillator]
        particles_scintillator = particles_scintillator[valid_scintillator]

        order_scintillator = np.argsort(birth_frame_scintillator)

        birth_frame_scintillator = birth_frame_scintillator[order_scintillator]
        x_scintillator = x_scintillator[order_scintillator]
        z_scintillator = z_scintillator[order_scintillator]
        particles_scintillator = particles_scintillator[order_scintillator]

        frames_scintillator = ak.zip(
            {
                "birth_frame": birth_frame_scintillator,
                "x": x_scintillator,
                "z": z_scintillator,
                "particle": particles_scintillator,
            }
        )

        if optical_triggers is None:
            frames = ak.Record(
                {
                    "tracks": frames,
                    "scintillator": frames_scintillator,
                }
            )

    if optical_triggers is not None:
        birth_frame_optical = np.digitize(optical_triggers, bins) - 1
        valid_optical = (birth_frame_optical >= 0) & (birth_frame_optical < n_frames)

        birth_frame_optical = birth_frame_optical[valid_optical]
        optical_hits = optical_hits[valid_optical]

        # detector IDs: [0, 1, 2, ..., 313]
        det_uid = np.arange(optical_hits.shape[1])

        duplicate_hits = False  # Hard-coded option if relevant later.

        if duplicate_hits:
            # Repeat entries according to hit multiplicity
            counts = optical_hits.astype(int)

            pmt_birth_frames = np.repeat(birth_frame_optical, counts.sum(axis=1))

            pmt_uids = np.repeat(
                np.tile(det_uid, len(birth_frame_optical)), counts.ravel()
            )

        else:
            # Only care whether detector had >=1 hit
            mask = optical_hits > 0

            trigger_idx, det_idx = np.nonzero(mask)

            pmt_birth_frames = birth_frame_optical[trigger_idx]
            pmt_uids = det_uid[det_idx]

        order = np.argsort(pmt_birth_frames)
        pmt_birth_frames = pmt_birth_frames[order]
        pmt_uids = pmt_uids[order]

        frames_optical = ak.zip(
            {
                "birth_frame": pmt_birth_frames,
                "det_uid": pmt_uids,
            }
        )

        frames = ak.Record(
            {
                "tracks": frames,
                "scintillator": frames_scintillator,
                "optical": frames_optical,
            }
        )

    return frames


# Legacy function for old r-projection
def build_frames_r(
    times,
    x,
    y,
    z,
    particles,
    bins,
    times_scintillator=None,
    x_scintillator=None,
    y_scintillator=None,
    z_scintillator=None,
    particles_scintillator=None,
    times_argon=None,
    x_argon=None,
    y_argon=None,
    z_argon=None,
    particles_argon=None,
    optical_triggers=None,
    optical_hits=None,
):
    """
    Convert (x,y,z,time) → awkward array of shape:
        n_frames * var * {r, z}

    Parameters
    ----------
    times, x, y, z : array-like
    particles : array-like
        PDG codes
    bins : array-like
        Time bin edges (length = n_frames + 1)
    times_scintillator, x_scintillator, y_scintillator, z_scintillator : array-like, optional
        Arguments for scintillator energy deps (if applicable)
    times_argon, x_argon, y_argon, z_argon : array-like, optional
        Arguments for argon energy deps (if applicable)

    argon_values : optional
    if given also adds r and z positions for the argon values to the output array.

    Returns
    -------
    ak.Record
        Awkward record with frame-binned data:

        tracks : ak.Array
            Particle tracks per time frame with fields:
            birth_frame, r, z, particle.

        scintillator : ak.Array, optional
            Same structure as tracks for scintillator inputs
            (if provided).

        optical : ak.Array, optional
            Detector hits per frame with fields:
            birth_frame, det_uid
            (if optical_triggers provided).

        All outputs are sorted by frame and restricted to valid bins.
    """
    n_frames = len(bins) - 1

    # --- binning ---
    birth_frame = np.digitize(times, bins) - 1

    # --- remove out-of-range ---
    valid = (birth_frame >= 0) & (birth_frame < n_frames)

    birth_frame = birth_frame[valid]
    x = x[valid]
    y = y[valid]
    z = z[valid]
    particles = particles[valid]
    # --- cylindrical transform ---
    r = np.sqrt(x**2 + y**2)

    # --- sort by frame (Option B core idea) ---
    order = np.argsort(birth_frame)

    birth_frame = birth_frame[order]
    r = r[order]
    z = z[order]
    particles = particles[order]

    # --- build awkward array ---
    frames = ak.zip({"birth_frame": birth_frame, "r": r, "z": z, "particle": particles})

    if times_scintillator is not None:
        birth_frame_scintillator = np.digitize(times_scintillator, bins) - 1
        valid_scintillator = (birth_frame_scintillator >= 0) & (
            birth_frame_scintillator < n_frames
        )

        birth_frame_scintillator = birth_frame_scintillator[valid_scintillator]
        x_scintillator = x_scintillator[valid_scintillator]
        y_scintillator = y_scintillator[valid_scintillator]
        z_scintillator = z_scintillator[valid_scintillator]
        particles_scintillator = particles_scintillator[valid_scintillator]

        r_scintillator = np.sqrt(x_scintillator**2 + y_scintillator**2)

        order_scintillator = np.argsort(birth_frame_scintillator)

        birth_frame_scintillator = birth_frame_scintillator[order_scintillator]
        r_scintillator = r_scintillator[order_scintillator]
        z_scintillator = z_scintillator[order_scintillator]
        particles_scintillator = particles_scintillator[order_scintillator]

        frames_scintillator = ak.zip(
            {
                "birth_frame": birth_frame_scintillator,
                "r": r_scintillator,
                "z": z_scintillator,
                "particle": particles_scintillator,
            }
        )

        if optical_triggers is None:
            frames = ak.Record(
                {
                    "tracks": frames,
                    "scintillator": frames_scintillator,
                }
            )

    if optical_triggers is not None:
        birth_frame_optical = np.digitize(optical_triggers, bins) - 1
        valid_optical = (birth_frame_optical >= 0) & (birth_frame_optical < n_frames)

        birth_frame_optical = birth_frame_optical[valid_optical]
        optical_hits = optical_hits[valid_optical]

        # detector IDs: [0, 1, 2, ..., 313]
        det_uid = np.arange(optical_hits.shape[1])

        duplicate_hits = False  # Hard-coded option if relevant later.

        if duplicate_hits:
            # Repeat entries according to hit multiplicity
            counts = optical_hits.astype(int)

            pmt_birth_frames = np.repeat(birth_frame_optical, counts.sum(axis=1))

            pmt_uids = np.repeat(
                np.tile(det_uid, len(birth_frame_optical)), counts.ravel()
            )

        else:
            # Only care whether detector had >=1 hit
            mask = optical_hits > 0

            trigger_idx, det_idx = np.nonzero(mask)

            pmt_birth_frames = birth_frame_optical[trigger_idx]
            pmt_uids = det_uid[det_idx]

        order = np.argsort(pmt_birth_frames)
        pmt_birth_frames = pmt_birth_frames[order]
        pmt_uids = pmt_uids[order]

        frames_optical = ak.zip(
            {
                "birth_frame": pmt_birth_frames,
                "det_uid": pmt_uids,
            }
        )

        frames = ak.Record(
            {
                "tracks": frames,
                "scintillator": frames_scintillator,
                "optical": frames_optical,
            }
        )

    return frames


# Static definitions:

tank_shift = -5238.800000000000182 / 1000
r_water_tank = [
    0.0,
    5000.0,
    5000.0,
    6000.0,
    6000.0,
    1665.0,
    1665.0,
    1135.0,
    1135.0,
    0.0,
]
z_water_tank = [
    0.0,
    0.0,
    800.0,
    800.0,
    9677.6,
    10209.8,
    10040.599999,
    10040.599999,
    10290.599999999999,
    10290.599999999999,
]

r_o_cryo = np.array(
    [
        0.0,
        210.89,
        395.33,
        606.22,
        817.1,
        935.77,
        1094.11,
        1239.23,
        1410.44,
        1581.66,
        1753.22,
        1845.44,
        1937.66,
        2109.23,
        2280.44,
        2372.66,
        2478.11,
        2649.67,
        2741.89,
        2834.11,
        2992.1,
        3124.0,
        3242.66,
        3334.88,
        3414.23,
        3480.0,
        3480.0,
        3466.4,
        3438.96,
        3387.37,
        3339.53,
        3281.14,
        3202.35,
        3120.04,
        3027.41,
        2927.99,
        2832.08,
        2729.13,
        2629.7,
        2519.96,
        2406.93,
        2300.7,
        2173.83,
        2057.29,
        1933.71,
        1817.16,
        1693.81,
        1570.47,
        1440.08,
        1302.9,
        1176.04,
        1135.0,
        1135.0,
        0.0,
    ]
)
z_o_cryo = np.array(
    [
        -4235.0,
        -4232.82,
        -4220.09,
        -4190.26,
        -4160.44,
        -4130.61,
        -4100.79,
        -4070.97,
        -4026.23,
        -3981.67,
        -3922.02,
        -3877.29,
        -3847.46,
        -3772.9,
        -3698.34,
        -3638.69,
        -3593.96,
        -3489.75,
        -3430.1,
        -3385.37,
        -3280.98,
        -3161.68,
        -3027.65,
        -2893.44,
        -2714.5,
        -2416.435,
        1751.435,
        1752.34,
        1943.84,
        2067.5,
        2187.16,
        2302.82,
        2414.48,
        2522.14,
        2605.8,
        2693.63,
        2773.47,
        2825.3,
        2900.95,
        2964.78,
        3024.61,
        3088.44,
        3144.27,
        3208.11,
        3263.94,
        3331.77,
        3359.59,
        3411.42,
        3459.25,
        3519.08,
        3539.08,
        3570.0,
        5019.99,
        5020.0,
    ]
)
r_i_cryo = np.array(
    [
        0.0,
        196.34,
        368.06,
        564.41,
        760.75,
        871.24,
        1018.66,
        1153.76,
        1313.17,
        1472.58,
        1632.31,
        1718.17,
        1804.03,
        1963.76,
        2123.17,
        2209.03,
        2307.2,
        2466.94,
        2552.8,
        2638.66,
        2785.75,
        2908.55,
        3019.03,
        3104.89,
        3178.76,
        3240.0,
        3240.0,
        3227.18,
        3201.32,
        3152.7,
        3107.62,
        3052.59,
        2978.34,
        2900.76,
        2813.47,
        2719.77,
        2629.38,
        2532.36,
        2438.65,
        2335.23,
        2228.7,
        2128.59,
        2009.03,
        1899.19,
        1782.73,
        1672.89,
        1556.64,
        1440.4,
        1317.52,
        1188.24,
        1068.67,
        1030.0,
        1030.0,
        0.0,
    ]
)
z_i_cryo = np.array(
    [
        -4025.0,
        -4022.91,
        -4010.69,
        -3982.07,
        -3953.45,
        -3924.83,
        -3896.21,
        -3867.59,
        -3824.65,
        -3781.9,
        -3724.66,
        -3681.73,
        -3653.1,
        -3581.55,
        -3510.0,
        -3452.76,
        -3409.83,
        -3309.83,
        -3252.59,
        -3209.66,
        -3109.48,
        -2995.0,
        -2866.38,
        -2737.59,
        -2565.86,
        -2279.83,
        1719.83,
        1720.7,
        1904.47,
        2023.14,
        2137.97,
        2248.97,
        2356.12,
        2459.43,
        2539.71,
        2624.0,
        2700.62,
        2750.35,
        2822.95,
        2884.21,
        2941.62,
        3002.88,
        3056.46,
        3117.71,
        3171.29,
        3236.38,
        3263.08,
        3312.82,
        3358.72,
        3416.14,
        3435.33,
        3465.0,
        4959.99,
        4960.0,
    ]
)
r_nmod = np.array([932.99999999, 1750.0, 1750.0, 0.0])
z_nmod = np.array([3200.0, 3200.0, 0.0, 0.0])
r_reentrance = np.array(
    [
        0.0,
        1.5248270270237512,
        3.0496540540475023,
        4.574481081071253,
        6.099308108095005,
        7.624135135118755,
        9.148962162142507,
        10.673789189166257,
        12.19861621619001,
        13.72344324321376,
        15.24827027023751,
        16.773097297261263,
        18.297924324285013,
        19.822751351308764,
        21.347578378332514,
        22.872405405356265,
        24.39723243238002,
        25.92205945940377,
        27.44688648642752,
        28.971713513451267,
        30.49654054047502,
        32.02136756749877,
        33.546194594522525,
        35.07102162154627,
        36.595848648570026,
        38.12067567559377,
        39.64550270261753,
        41.170329729641274,
        42.69515675666503,
        44.219983783688775,
        45.74481081071253,
        47.26963783773628,
        48.79446486476004,
        50.319291891783784,
        51.84411891880754,
        53.368945945831285,
        54.89377297285504,
        56.418599999878786,
        105.76159999977273,
        162.18019999965162,
        218.5987999995304,
        250.34589999946223,
        292.70639999937123,
        331.5290999992878,
        377.3342999991894,
        423.139499999091,
        469.03779999899245,
        493.7092999989394,
        518.3807999988863,
        564.2790999987878,
        610.0842999986894,
        634.7557999986363,
        662.9650999985757,
        708.8633999984772,
        733.5348999984243,
        758.2063999983712,
        800.4737999982805,
        835.7586999982045,
        867.5057999981364,
        892.1772999980834,
        913.4040999980379,
        930.999999998,
        0.0,
    ]
)
z_reentrance = np.array(
    [
        -1327.0,
        -1326.9898697297297,
        -1326.9797394594593,
        -1326.9696091891892,
        -1326.9594789189189,
        -1326.9493486486485,
        -1326.9392183783782,
        -1326.929088108108,
        -1326.9189578378378,
        -1326.9088275675674,
        -1326.898697297297,
        -1326.888567027027,
        -1326.8784367567566,
        -1326.8683064864863,
        -1326.858176216216,
        -1326.8480459459458,
        -1326.8379156756755,
        -1326.8277854054052,
        -1326.8176551351348,
        -1326.8075248648647,
        -1326.7973945945944,
        -1326.787264324324,
        -1326.7771340540537,
        -1326.7670037837836,
        -1326.7568735135133,
        -1326.746743243243,
        -1326.7366129729726,
        -1326.7264827027025,
        -1326.7163524324321,
        -1326.7062221621618,
        -1326.6960918918915,
        -1326.6859616216213,
        -1326.675831351351,
        -1326.6657010810807,
        -1326.6555708108103,
        -1326.6454405405402,
        -1326.6353102702699,
        -1326.6251799999995,
        -1324.4387299999999,
        -1319.3161899999996,
        -1314.1936499999997,
        -1309.0711099999999,
        -1303.9485699999996,
        -1298.8260299999997,
        -1291.1422199999997,
        -1283.4896449999997,
        -1273.2445649999995,
        -1265.5607549999995,
        -1260.4382149999997,
        -1247.6318649999998,
        -1234.8255149999995,
        -1224.5804349999999,
        -1216.8966249999994,
        -1198.9989699999996,
        -1188.7538899999995,
        -1181.0700799999995,
        -1163.1411899999998,
        -1142.6510299999995,
        -1119.6308349999995,
        -1096.5794049999995,
        -1065.8441649999995,
        4919,
        4919.0,
    ]
)
r_skirt = np.array([3440.0, 3440.0])
z_skirt = np.array([-2463.5649999999996 / 2, 2463.5649999999996 / 2])

r_tyvek = np.array([4000.0, 4000.0])
z_tyvek = np.array([0.0, 9843.0])

cryo_shift = 5238.800000000000182 / 1000 + tank_shift
nmod_shift = cryo_shift - 2397.000000000000000 / 1000
skirt_shift = tank_shift + 1291.782500000999789 / 1000

r_o_cryo = r_o_cryo / 1000
z_o_cryo = z_o_cryo / 1000 + cryo_shift
r_i_cryo = r_i_cryo / 1000
z_i_cryo = z_i_cryo / 1000 + cryo_shift
r_nmod = r_nmod / 1000
z_nmod = z_nmod / 1000 + nmod_shift
r_reentrance = r_reentrance / 1000
z_reentrance = z_reentrance / 1000 + cryo_shift
r_skirt = r_skirt / 1000
z_skirt = z_skirt / 1000 + skirt_shift

r_tyvek = r_tyvek / 1000
z_tyvek = z_tyvek / 1000 + tank_shift

r_water_tank = np.array(r_water_tank) / 1000
z_water_tank = np.array(z_water_tank) / 1000 + tank_shift


# PMT uid mapping.
uids_rows = [
    np.arange(0, 50),  # outer floor
    np.arange(50, 80),  # inner floor
    np.arange(80, 95),  # inner inner floor
    np.arange(95, 103),  # inner inner inner floor
    np.arange(103, 104),  # cherry bottom
    np.arange(104, 139),  # lowest wall
    np.arange(139, 174),  # 2nd lowest wall
    np.arange(174, 209),  # 3rd lowest wall
    np.arange(209, 244),  # 4th lowest wall
    np.arange(244, 279),  # 5th lowest wall
    np.arange(279, 314),  # upper wall
]
# PMT row positions:
pmt_shift = 0.4  # Move them a little bit outward so the object is centered
wall_r = 4.05 + pmt_shift
row_r = np.array([3.75, 3.0, 1.8, 0.8, 0.0] + [wall_r] * 6)
row_z = (
    np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.811,
            1.811 + 0.960,
            1.811 + 0.960 * 2,
            1.811 + 0.960 * 3,
            1.811 + 0.960 * 4,
            1.811 + 0.960 * 5,
        ]
    )
    + tank_shift
)
row_phi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0])


def _unpack_gif_config(gif_config, max_ge_energy):
    filename = gif_config.get("filename", "out.gif")
    fps = gif_config.get("fps", 20)
    zlim = gif_config.get("zlim", (np.min(z_water_tank), np.max(z_water_tank) + 0.1))
    linger = gif_config.get("linger", 0)
    hightlight_ge77 = gif_config.get("hightlight_ge77", False)
    add_scintillator = gif_config.get("add_scintillator", False)
    add_pmts = gif_config.get("add_pmts", False)
    neutron_popups = gif_config.get("neutron_popups", False)
    add_info_text = gif_config.get("add_info_text", False)

    dots_colors = gif_config.get("colors", None)
    colors_strings = gif_config.get("colors_strings", None)

    tracks_fadeout_mult = gif_config.get("tracks_fadeout_mult", 1.0)
    n_scint_fadeout_mult = gif_config.get("n_scint_fadeout_mult", 2.0)
    pmt_fadeout_mult = gif_config.get("pmt_fadeout_mult", 2.0)

    dangerous_ge_threshold = gif_config.get("dangerous_ge_threshold", 25)
    ge77_veto_threshold = gif_config.get("ge77_veto_threshold", 6)

    show_x = gif_config.get("show_x", True)
    field_name = "x" if show_x else "r"
    xlim = gif_config.get(
        "xlim",
        ((-np.max(r_water_tank) - 0.1) if show_x else 0, np.max(r_water_tank) + 0.1),
    )

    dangerous_muon = max_ge_energy > dangerous_ge_threshold

    if dots_colors is None:
        dots_colors = (
            "blue",
            "darkred",
            "cyan",
            "green",
            "red",
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    if colors_strings is None:
        colors_strings = ("Blue:", "Darkred:", "Cyan:", "Green:", "Red:", "Yellow:", "")

    return (
        filename,
        fps,
        xlim,
        zlim,
        linger,
        hightlight_ge77,
        add_scintillator,
        add_pmts,
        neutron_popups,
        add_info_text,
        dots_colors,
        colors_strings,
        tracks_fadeout_mult,
        n_scint_fadeout_mult,
        pmt_fadeout_mult,
        ge77_veto_threshold,
        show_x,
        field_name,
        dangerous_muon,
    )


def _setup_figure_and_axes(fig, xlim, zlim, field_name):
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.03, top=0.97)

    # Calculate aspect ratio
    dx = xlim[1] - xlim[0]
    dz = zlim[1] - zlim[0]
    data_aspect = dx / dz

    sidebar_factor = 0.64  # how much of the figure width to dedicate to the sidebar
    lowerbar_factor = 0.36  # how much of the figure height to dedicate to the lower bar

    main_height = 1
    main_width = main_height * data_aspect
    sidebar_width = main_width * sidebar_factor / (1 - sidebar_factor)
    bottom_height = main_height * lowerbar_factor / (1 - lowerbar_factor)

    # --- Setup Grid ---
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[main_width, sidebar_width],
        height_ratios=[main_height, bottom_height],
        wspace=0.035,
        hspace=0.08,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_wall = fig.add_subplot(gs[0, 1])
    ax_floor = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    ax_wall.set_facecolor("#A1E0FF")
    ax_floor.set_facecolor("#A1E0FF")

    ax_floor.set_aspect("equal")
    ax_wall.set_aspect("auto")
    # Anchor the floor axis to the bottom left.
    ax_floor.set_anchor("SW")
    ax_info.set_anchor("SW")
    ax_main.set_aspect("equal", adjustable="box")

    # Remove the ticks for the everything but the main
    for ax in [ax_wall, ax_floor, ax_info]:
        ax.set_xticks([])
        ax.set_yticks([])

    # --- axes ---
    if xlim:
        ax_main.set_xlim(*xlim)
    if zlim:
        ax_main.set_ylim(*zlim)
    ax_main.set_xlabel(field_name)
    ax_main.set_ylabel("z")
    # Set the wall axis ticks myself
    ax_wall.set_xlim(0, 365)
    ax_wall.set_xticks([0, 90, 180, 270, 360])
    ax_wall.set_xticklabels(["0°", "90°", "180°", "270°", "360°"])
    ax_wall.tick_params(axis="x", bottom=True, labelbottom=True)
    ax_wall.set_ylim(*zlim)

    # Set the floor axis ticks up
    phi_ticks = np.arange(0, 360, 45)
    for phi in phi_ticks:
        theta = np.deg2rad(phi)

        # radial guide line
        ax_floor.plot(
            [0, 1.1 * np.sin(theta)],
            [0, 1.1 * np.cos(theta)],
            color="gray",
            lw=0.5,
            alpha=0.5,
        )

        # label position
        label_r = 1.18

        ax_floor.text(
            label_r * np.sin(theta),
            label_r * np.cos(theta),
            f"{phi}°",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax_floor.set_xlim(-1.2, 1.2)
    ax_floor.set_ylim(-1.2, 1.2)

    return fig, ax_main, ax_wall, ax_floor, ax_info


def _setup_static_geometry(ax_main, show_x):
    static_lines = [
        (r_water_tank, z_water_tank),
        (r_o_cryo, z_o_cryo),
        (r_i_cryo, z_i_cryo),
        (r_nmod, z_nmod),
        (r_reentrance, z_reentrance),
        (r_skirt, z_skirt),
        (r_tyvek, z_tyvek),
    ]

    for r_line, z_line in static_lines:
        ax_main.plot(r_line, z_line, color="gray", zorder=-10)
        if show_x:
            ax_main.plot(-r_line, z_line, color="gray", zorder=-10)


def _setup_floor_rollout_geometry(ax_floor, uids_rows, dots_colors):
    # --- Create floor rollout geometry ---
    floor_x = []
    floor_y = []

    floor_radii = [1.0, 0.75, 0.5, 0.25, 0.0]

    for radius, uids in zip(floor_radii, uids_rows[:5]):
        n = len(uids)

        if n == 1:
            theta = np.array([0.0])
        else:
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Mirror the angle because PMTs are distributed counter-clockwise
        x = -radius * np.sin(theta)
        y = radius * np.cos(theta)

        floor_x.append(x)
        floor_y.append(y)

    floor_x = np.concatenate(floor_x)
    floor_y = np.concatenate(floor_y)

    floor_colors = np.zeros((len(floor_x), 4))
    floor_colors[:] = (*dots_colors[6], 1.0)

    floor_scatter = ax_floor.scatter(
        floor_x, floor_y, s=120, facecolors=floor_colors, edgecolors="black"
    )
    return floor_scatter, floor_colors


def _setup_wall_rollout_geometry(ax_wall, uids_rows, dots_colors):
    # --- Create wall rollout geometry ---
    wall_x = []
    wall_y = []

    for i, (uids, z_loc) in enumerate(zip(uids_rows[5:], row_z[5:])):  # wall rows only
        n = len(uids)
        # Mirror the angle also here
        x = (
            360 - np.linspace(0, 360, n, endpoint=False) - 1
        )  # -1 to center the PMT a little better

        y = np.full(n, z_loc)

        wall_x.append(x)
        wall_y.append(y)

    wall_x = np.concatenate(wall_x)
    wall_y = np.concatenate(wall_y)

    wall_colors = np.zeros((len(wall_x), 4))
    wall_colors[:] = (*dots_colors[6], 1.0)

    wall_scatter = ax_wall.scatter(
        wall_x, wall_y, s=120, facecolors=wall_colors, edgecolors="black"
    )

    return wall_scatter, wall_colors


def _setup_tracks(ax_main, frames, dots_colors, hightlight_ge77):
    xz_tracks = frames["tracks"]
    ge77_mask = np.asarray((xz_tracks["particle"] // 10) == 100032077)
    neutron_mask = np.asarray((xz_tracks["particle"] >= 1000000000))

    track_scatter = ax_main.scatter(
        [], [], s=15, color=dots_colors[0], zorder=20, edgecolors="none"
    )  # 20 to be in front of almost everything
    if hightlight_ge77:
        ge77_scatter = ax_main.scatter(
            [], [], s=10, color=dots_colors[1], zorder=0
        )  # 0 to be behind the regular tracks

    return (
        xz_tracks,
        track_scatter,
        ge77_scatter if hightlight_ge77 else None,
        ge77_mask,
        neutron_mask,
    )


def _setup_scintillator_tracks(ax_main, frames, dots_colors):
    xz_scintillator = frames["scintillator"]
    xz_scintillator_neutrons = xz_scintillator[
        (xz_scintillator["particle"] == 2112) | (xz_scintillator["particle"] == -2112)
    ]
    xz_scintillator_muons = xz_scintillator[
        (xz_scintillator["particle"] == 13) | (xz_scintillator["particle"] == -13)
    ]

    scintillator_neutron_scatter = ax_main.scatter(
        [], [], s=4, color=dots_colors[2], zorder=1, edgecolors="none"
    )  # 1 to be in front of the static geometry but behind the regular tracks
    scintillator_muon_scatter = ax_main.scatter(
        [], [], s=5, color=dots_colors[3], zorder=2
    )
    scintillator_current_muon_scatter = ax_main.scatter(
        [], [], s=5, color=dots_colors[4], zorder=21
    )  # 21 to be in front of everything

    return (
        scintillator_neutron_scatter,
        scintillator_muon_scatter,
        scintillator_current_muon_scatter,
        xz_scintillator_neutrons,
        xz_scintillator_muons,
    )


def _setup_pmts(ax_main, frames, dots_colors):
    optical = frames["optical"]
    pmt_birth_frames = optical["birth_frame"]
    pmt_uids = optical["det_uid"]

    # Create the PMT polygons just once and only update color later.
    pmt_polys = []
    # Convert the PMT uids, which go from 0 to 313 for each PMT into row indices going from 0 to len(uids_rows) - 1
    pmt_row_uids = np.empty(len(pmt_uids), dtype=int)
    i = 0  # do the index the good old way for nostalgia reason
    # Do all in the same loop. As when in rome might as well... idk how the saying goes
    for r, z, phi, uids in zip(row_r, row_z, row_phi, uids_rows):
        mask = np.isin(pmt_uids, uids)
        pmt_row_uids[mask] = i
        i += 1

        base = np.array([[0.0, 0.0], [-0.6, 1.6], [0.6, 1.6]]) * 0.3

        c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
        rot = np.array([[c, -s], [s, c]])

        pts = base @ rot.T
        pts[:, 0] += r
        pts[:, 1] += z

        pmt_polys.append(pts)

    pmt_row_colors = np.zeros((len(pmt_polys), 4))
    pmt_row_colors[:] = (*dots_colors[6], 1.0)  # default blue

    # Here we do not have a scatter object but permanent objects that change color.
    pmt_collection = PolyCollection(
        pmt_polys,
        facecolors=pmt_row_colors,
        edgecolors="black",
        zorder=-5,  # -5 is in front of the static geometry but behind the tracks
    )
    ax_main.add_collection(pmt_collection)

    return pmt_birth_frames, pmt_uids, pmt_collection, pmt_row_colors, pmt_row_uids


def _setup_info_text(
    ax_info, add_info_text, colors_strings, dots_colors, dangerous_muon
):
    # --- Setup info text ---
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    if add_info_text:
        # --- Static legend ---
        legend_lines = [
            (dots_colors[0], f"{colors_strings[0]} Neutron captures"),
            (dots_colors[1], f"{colors_strings[1]} Ge-77 captures"),
            (dots_colors[2], f"{colors_strings[2]} Moving neutrons"),
            (dots_colors[3], f"{colors_strings[3]} Muon path"),
            (dots_colors[4], f"{colors_strings[4]} Current muon position"),
            (dots_colors[5], f"{colors_strings[5]} Recent PMT hits"),
        ]
        left_shift = 0.04
        y = 0.95
        for color, text in legend_lines:
            ax_info.scatter(
                0.07 - left_shift,
                y - 0.015,
                s=120,
                marker="s",
                color=color,
                edgecolors="black",
                transform=ax_info.transAxes,
                clip_on=False,
            )

            # draw label text
            ax_info.text(
                0.12 - left_shift,
                y,
                text,
                fontsize=11,
                va="top",
                transform=ax_info.transAxes,
            )
            y -= 0.08

        stats_text = ax_info.text(
            0.05 - left_shift, 0.37, "", fontsize=11, va="top", family="monospace"
        )
        time_text = ax_info.text(
            0.05 - left_shift, 0.11, "", fontsize=11, va="top", family="monospace"
        )
        # Box green if vetoed correctly, else red
        muon_box_color = "green" if dangerous_muon else "red"
        muon_veto_text = ax_info.text(
            0.65 - left_shift,
            0.8,
            "",
            fontsize=20,
            va="top",
            family="monospace",
            color=muon_box_color,
            fontweight="bold",
            zorder=1,  # behind main text
        )
        muon_veto_text.set_bbox(
            dict(
                facecolor="white",
                edgecolor=muon_box_color,
                linewidth=3,
                boxstyle="round,pad=0.4",
            )
        )

        ge77_veto_text = ax_info.text(
            0.65 - left_shift,
            0.5,
            "",
            fontsize=20,
            va="top",
            family="monospace",
            color="red",
            fontweight="bold",
            zorder=1,  # behind main text
        )
        ge77_veto_text.set_bbox(
            dict(
                facecolor="white",
                edgecolor="red",
                linewidth=3,
                boxstyle="round,pad=0.4",
            )
        )
        return stats_text, time_text, muon_veto_text, ge77_veto_text


def _setup_neutron_popups(ax_main, dots_colors):
    MAX_POPUPS = 20

    popup_texts = []
    popup_state = []  # (x, z, age, active)

    for _ in range(MAX_POPUPS):
        t = ax_main.text(
            0,
            0,
            "+ neutron detected",
            color="darkred",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.0,  # start invisible
            visible=True,
        )
        popup_texts.append(t)
        popup_state.append({"age": 999, "active": False})

    return popup_texts, popup_state, MAX_POPUPS


def make_gif_full(frames, bins, max_ge_energy, gif_config):
    (
        filename,
        fps,
        xlim,
        zlim,
        linger,
        hightlight_ge77,
        add_scintillator,
        add_pmts,
        neutron_popups,
        add_info_text,
        dots_colors,
        colors_strings,
        tracks_fadeout_mult,
        n_scint_fadeout_mult,
        pmt_fadeout_mult,
        ge77_veto_threshold,
        show_x,
        field_name,
        dangerous_muon,
    ) = _unpack_gif_config(gif_config, max_ge_energy)

    # Number of frames to generate
    n_frames = len(bins) - 1

    # --- Prepare figure and axes ---
    # Create one figure for everyone to live on in harmony
    fig = plt.figure(figsize=(10, 10), dpi=102.4)  # so 1024 x 1024 pixels.
    fig, ax_main, ax_wall, ax_floor, ax_info = _setup_figure_and_axes(
        fig, xlim, zlim, field_name
    )

    title = ax_main.set_title("")

    # --- setup static geometry ---
    _setup_static_geometry(ax_main, show_x)

    # --- setup info text ---
    stats_text, time_text, muon_veto_text, ge77_veto_text = _setup_info_text(
        ax_info, add_info_text, colors_strings, dots_colors, dangerous_muon
    )

    # --- setup neutron popups ---
    if neutron_popups:
        popup_texts, popup_state, MAX_POPUPS = _setup_neutron_popups(
            ax_main, dots_colors
        )
        active_popup_indices = []

    # --- BLITTING SETUP ---
    fig.canvas.draw()  # IMPORTANT: initial full draw

    background = fig.canvas.copy_from_bbox(fig.bbox)
    # --- setup tracks ---
    xz_tracks, track_scatter, ge77_scatter, ge77_mask, neutron_mask = _setup_tracks(
        ax_main, frames, dots_colors, hightlight_ge77
    )

    # --- setup scintillator tracks if requested ---
    if add_scintillator:
        (
            scintillator_neutron_scatter,
            scintillator_muon_scatter,
            scintillator_current_muon_scatter,
            xz_scintillator_neutrons,
            xz_scintillator_muons,
        ) = _setup_scintillator_tracks(ax_main, frames, dots_colors)

    # --- setup PMTs ---
    pmt_birth_frames, pmt_uids, pmt_collection, pmt_row_colors, pmt_row_uids = (
        _setup_pmts(ax_main, frames, dots_colors)
    )

    floor_scatter, floor_colors = _setup_floor_rollout_geometry(
        ax_floor, uids_rows, dots_colors
    )
    n_floor_pmts = len(floor_colors)

    wall_scatter, wall_colors = _setup_wall_rollout_geometry(
        ax_wall, uids_rows, dots_colors
    )
    pmt_all_colors = np.concatenate([floor_colors, wall_colors])

    # Counters
    detected_neutrons = 0
    captured_neutrons = 0
    ge_77_captures = 0
    primary_muon_vetoed = False

    # store the coordinates of the last neutron capture.
    x_last = 0.0
    z_last = 0.0

    writer = imageio.get_writer(
        filename,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-preset", "ultrafast"],
    )

    # --- Find last frame after which nothing happens ---
    # This can be optimized even more.
    track_birth = xz_tracks["birth_frame"]
    track_max = (
        np.max(track_birth)
        if len(track_birth) > 0
        else 0 + int(linger / n_scint_fadeout_mult) + 1
    )
    pmt_max = (
        np.max(pmt_birth_frames)
        if len(pmt_birth_frames) > 0
        else 0 + int(linger / pmt_fadeout_mult) + 1
    )
    scintillator_max = (
        np.max(xz_scintillator_neutrons["birth_frame"])
        if len(xz_scintillator_neutrons["birth_frame"]) > 0
        else 0 + int(linger / n_scint_fadeout_mult) + 1
    )
    last_frame_index = (
        max(track_max, pmt_max, scintillator_max) + 30
    )  # add some buffer due to the neutron popups to be safe

    # --- generate the frames ---
    for f in tqdm(range(n_frames), desc="Rendering frames"):
        t1 = bins[f + 1]
        if f < last_frame_index:
            # --- update tracks ---
            age = np.asarray(tracks_fadeout_mult * (f - xz_tracks["birth_frame"]))
            mask = (age >= 0) & (age < linger)
            ge77_mask_frame = ge77_mask & (age >= 0)
            x_frame = np.asarray(xz_tracks[field_name][mask])
            z_frame = np.asarray(xz_tracks["z"][mask])
            x_ge77_frame = np.asarray(xz_tracks[field_name][ge77_mask_frame])
            z_ge77_frame = np.asarray(xz_tracks["z"][ge77_mask_frame])

            alpha_frame = 1.0 - age[mask] / linger

            x_last = x_frame[-1] if len(x_frame) > 0 else x_last
            z_last = z_frame[-1] if len(z_frame) > 0 else z_last

            # --- update track scatter ---
            track_scatter.set_offsets(np.column_stack([x_frame, z_frame]))
            track_scatter.set_alpha(None)
            track_scatter.set_facecolors(
                np.column_stack(
                    [
                        np.zeros_like(alpha_frame),
                        np.zeros_like(alpha_frame),
                        np.ones_like(alpha_frame),
                        alpha_frame,
                    ]
                )
            )

            # Stats update
            neutrons_mask_current_frame = neutron_mask & (age == 0)
            captured_neutrons += np.sum(neutrons_mask_current_frame)
            ge_77_captures += np.sum(ge77_mask_frame & (age == 0))

            # --- highlight germanium track permanently if requested ---
            if hightlight_ge77 and len(x_ge77_frame) > 0:
                ge77_scatter.set_offsets(np.column_stack([x_ge77_frame, z_ge77_frame]))
                ge77_scatter.set_alpha(1.0)  # Always fully opaque
                ge77_scatter.set_facecolors("darkred")

            # --- draw scintillator tracks if requested ---
            if add_scintillator:
                age_neutrons = np.asarray(
                    n_scint_fadeout_mult * (f - xz_scintillator_neutrons["birth_frame"])
                )
                age_muons = np.asarray(f - xz_scintillator_muons["birth_frame"])

                mask_neutrons = (age_neutrons >= 0) & (age_neutrons < linger)
                mask_muons = age_muons >= 0  # Never fade out muons.
                mask_current_muons = (
                    age_muons == 0
                )  # But highlight current muon position

                x_scint_neutrons_frame = np.asarray(
                    xz_scintillator_neutrons[field_name][mask_neutrons]
                )
                z_scint_neutrons_frame = np.asarray(
                    xz_scintillator_neutrons["z"][mask_neutrons]
                )
                alpha_scint_neutrons_frame = 1.0 - age_neutrons[mask_neutrons] / linger

                x_scint_muons_frame = np.asarray(
                    xz_scintillator_muons[field_name][mask_muons]
                )
                z_scint_muons_frame = np.asarray(xz_scintillator_muons["z"][mask_muons])
                x_scint_current_muon_frame = np.asarray(
                    xz_scintillator_muons[field_name][mask_current_muons]
                )
                z_scint_current_muon_frame = np.asarray(
                    xz_scintillator_muons["z"][mask_current_muons]
                )

                # --- update neutron scatter ---
                scintillator_neutron_scatter.set_offsets(
                    np.column_stack([x_scint_neutrons_frame, z_scint_neutrons_frame])
                )
                scintillator_neutron_scatter.set_alpha(None)
                scintillator_neutron_scatter.set_facecolors(
                    np.column_stack(
                        [
                            np.zeros_like(alpha_scint_neutrons_frame),
                            np.ones_like(alpha_scint_neutrons_frame),
                            np.ones_like(alpha_scint_neutrons_frame),
                            alpha_scint_neutrons_frame,
                        ]
                    )
                )

                # --- update muon scatter ---
                scintillator_muon_scatter.set_offsets(
                    np.column_stack([x_scint_muons_frame, z_scint_muons_frame])
                )
                scintillator_muon_scatter.set_alpha(None)
                scintillator_muon_scatter.set_facecolors("green")

                # --- update highlighted current muon scatter ---
                scintillator_current_muon_scatter.set_offsets(
                    np.column_stack(
                        [x_scint_current_muon_frame, z_scint_current_muon_frame]
                    )
                )
                scintillator_current_muon_scatter.set_alpha(None)
                scintillator_current_muon_scatter.set_facecolors("red")

            # --- draw PMTs if requested ---
            if add_pmts:
                pmt_row_colors[:] = (*dots_colors[6], 1.0)  # default

                age_pmts = np.asarray(pmt_fadeout_mult * (f - pmt_birth_frames))
                mask_pmts = (age_pmts >= 0) & (age_pmts < linger)

                if np.any(mask_pmts):
                    pmt_uids_frame = pmt_uids[mask_pmts]
                    pmt_row_uids_frame = pmt_row_uids[mask_pmts]
                    alpha_pmts_frame = 1.0 - age_pmts[mask_pmts] / linger

                    # The alpha here is not a real alpha but a transition between color (hit) and color (no hit)
                    rgba = np.column_stack(
                        [
                            alpha_pmts_frame * dots_colors[5][0]
                            + (1 - alpha_pmts_frame) * dots_colors[6][0],
                            alpha_pmts_frame * dots_colors[5][1]
                            + (1 - alpha_pmts_frame) * dots_colors[6][1],
                            alpha_pmts_frame * dots_colors[5][2]
                            + (1 - alpha_pmts_frame) * dots_colors[6][2],
                            np.ones_like(alpha_pmts_frame),
                        ]
                    )
                    # because the uids are already sorted by birth_frame this already implicitly has a "newest hit wins" behaviour.
                    pmt_row_colors[pmt_row_uids_frame] = rgba

                    pmt_collection.set_facecolors(pmt_row_colors)

                    # --- update wall and floor scatter ---
                    pmt_all_colors[:] = (*dots_colors[6], 1.0)

                    # Test display to see PMT alignment
                    # test_indices= []
                    # for arr in uids_rows:
                    #    try:
                    #        progress = f % 50
                    #
                    #        test_indices.append(arr[int(progress / (50 / len(arr)))])
                    #    except IndexError:
                    #        pass
                    # pmt_all_colors[test_indices] = (1.0, 0.0, 0.0, 1.0)
                    # End of test stuff

                    pmt_all_colors[pmt_uids_frame] = rgba

                    floor_scatter.set_facecolors(pmt_all_colors[:n_floor_pmts])
                    wall_scatter.set_facecolors(pmt_all_colors[n_floor_pmts:])

                    # --- Roughly check if this muon would be vetoed ---
                    if not primary_muon_vetoed:
                        multiplicity = len(np.unique(pmt_uids_frame))
                        if (multiplicity > 40) & (
                            t1 < 300
                        ):  # only veto the first 300ns, otherwise bins get too large.
                            primary_muon_vetoed = True

                    # --- Now also add the + neutron detected pop-up on the main plot ---
                    if (
                        neutron_popups and t1 > 1000
                    ):  # Only start counting neutrons after 1 microsecond
                        mask_current_pmts = (
                            age_pmts == 0
                        )  # Only PMTs that are hit in the current frame
                        pmt_uids_current_frame = np.unique(pmt_uids[mask_current_pmts])
                        if len(pmt_uids_current_frame) > 5:
                            detected_neutrons += 1
                            # find free slot
                            for i in range(MAX_POPUPS):
                                if (
                                    popup_state[i]["age"] > 30
                                    or not popup_state[i]["active"]
                                ):
                                    popup_state[i] = {
                                        "age": 0,
                                        "active": True,
                                        "x": x_last,
                                        "z": z_last,
                                    }
                                    active_popup_indices.append(i)
                                    break

                # --- Age it and make it fade-out ---
                if neutron_popups:
                    T = 30

                    for i in active_popup_indices[
                        :
                    ]:  # iterate copy so we can remove safely
                        state = popup_state[i]
                        txt = popup_texts[i]

                        age = state["age"]

                        if age > T:
                            state["active"] = False
                            txt.set_alpha(0.0)
                            active_popup_indices.remove(i)
                            continue

                        txt.set_position((state["x"], state["z"] + 0.01 * age))
                        txt.set_alpha(1.0 - age / T)

                        state["age"] += 1

        # --- time label ---
        title.set_text(f"t = {t1:.2f} ns" if t1 < 1000 else f"t = {t1 / 1000:.2f} us")

        # --- update stats text ---
        if add_info_text:
            status = ""
            if dangerous_muon:
                status = "   >>> DANGEROUS <<<"
            stats_text.set_text(
                f"Detected neutrons:            {detected_neutrons}\n"
                f"Captured neutrons:            {captured_neutrons}\n"
                f"Ge-77 events:                 {ge_77_captures}\n"
                f"Max edep in one Ge-detector:  {max_ge_energy:.2f} keV{status}"
            )

            time_str = f"t = {t1:.2f} ns" if t1 < 1000 else f"t = {t1 / 1000:.2f} µs"

            time_text.set_text(f"Frame: {f}/{n_frames}\n{time_str}")
            if detected_neutrons > ge77_veto_threshold:
                ge77_veto_text.set_text("Ge77 VETOED!")
                # Box green if we vetoed correctly, else it stays red
                if len(x_ge77_frame) > 0:
                    ge77_veto_text.set_color("green")
                    bbox = ge77_veto_text.get_bbox_patch()
                    bbox.set_edgecolor("green")
            if primary_muon_vetoed:
                muon_veto_text.set_text("MUON VETOED!")

        # --- BLIT RESTORE BACKGROUND ---
        fig.canvas.restore_region(background)

        # --- redraw only changed artists ---
        ax_main.draw_artist(track_scatter)

        if hightlight_ge77:
            ax_main.draw_artist(ge77_scatter)

        if add_scintillator:
            ax_main.draw_artist(scintillator_neutron_scatter)
            ax_main.draw_artist(scintillator_muon_scatter)
            ax_main.draw_artist(scintillator_current_muon_scatter)

        if add_pmts:
            ax_floor.draw_artist(floor_scatter)
            ax_wall.draw_artist(wall_scatter)
            ax_main.draw_artist(pmt_collection)

        # text overlays (IMPORTANT: must be drawn manually)
        if add_info_text:
            ax_info.draw_artist(stats_text)
            ax_info.draw_artist(time_text)
            ax_info.draw_artist(muon_veto_text)
            ax_info.draw_artist(ge77_veto_text)

        if neutron_popups:
            for i in active_popup_indices:
                ax_main.draw_artist(popup_texts[i])

        ax_main.draw_artist(title)

        # --- blit to screen buffer ---
        fig.canvas.blit(fig.bbox)

        # --- capture frame for video ---
        frame = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(frame)

    plt.close(fig)

    writer.close()
    return (
        dangerous_muon,
        primary_muon_vetoed,
        detected_neutrons,
        captured_neutrons,
        ge_77_captures,
    )


def read_files(file_pattern):
    files = glob(file_pattern)
    tracks_list = []
    scintillator_edep_list = []
    optical_list = []
    germanium_list = []
    for file in files:
        tracks_list.append(lh5.read_as("tracks", file, "ak"))
        scintillator_edep_list.append(lh5.read_as("stp/scintillator", file, "ak"))
        optical_list.append(lh5.read_as("stp/optical", file, "ak"))
        germanium_list.append(lh5.read_as("stp/germanium", file, "ak"))

    tracks_array = ak.concatenate(tracks_list, axis=0)
    scintillator_edep_array = ak.concatenate(scintillator_edep_list, axis=0)
    optical_array = ak.concatenate(optical_list, axis=0)
    germanium_array = ak.concatenate(germanium_list, axis=0)
    return tracks_array, scintillator_edep_array, optical_array, germanium_array


def render_gif(
    tracks_array, scintillator_edep_array, optical_array, germanium_array, gif_config
):
    event = gif_config.get("event", 0)
    number_of_frames = gif_config.get("number_of_frames", 1000)
    n_short_frames = gif_config.get("n_short_frames", 200)
    # select event
    event_ids_scint = np.unique(scintillator_edep_array["evtid"])
    tracks_one_event = tracks_array[tracks_array["evtid"] == event_ids_scint[event]]
    tracks_one_event = tracks_one_event[
        tracks_one_event["particle"] >= 100000
    ]  # Only count isotopes
    scintillator_edep_one_event = scintillator_edep_array[
        scintillator_edep_array["evtid"] == event_ids_scint[event]
    ]
    optical_one_event = optical_array[optical_array["evtid"] == event_ids_scint[event]]
    germanium_one_event = germanium_array[
        germanium_array["evtid"] == event_ids_scint[event]
    ]

    GREEN = "\033[32m"
    RESET = "\033[0m"

    print(
        f"\n{GREEN}Rendering event {event_ids_scint[event]} with {len(tracks_one_event)} nCaptures: {RESET}"
    )

    # Get the maximum germanium energy deposited in a single detector this event.
    max_ge_energy = 0.0
    if len(germanium_one_event) > 0:
        ge_det_uids = germanium_one_event["det_uid"]
        for det in np.unique(ge_det_uids):
            energy = np.sum(germanium_one_event["edep"][ge_det_uids == det])
            if energy > max_ge_energy:
                max_ge_energy = energy

    # do pmt processing
    all_detectors = np.arange(6000, 6314)

    grouped = ak.Array(
        [
            optical_one_event[optical_one_event["det_uid"] == det]
            for det in all_detectors
        ]
    )

    optical_grouped = align_detectors(grouped)
    optical_triggers = build_hardware_triggers(
        optical_grouped, multiplicity_threshold=1, timegate=1, trigger_deadtime=4
    )
    optical_hits = build_hits(optical_grouped, optical_triggers[0], 4, 4, 0)
    optical_hits = np.swapaxes(np.squeeze(ak.to_numpy(optical_hits), axis=1), 0, 1)
    optical_triggers = optical_triggers[0][0]
    time = tracks_one_event["time"]
    bins = create_time_bins(
        time,
        n_frames=number_of_frames,
        tmin=5000,
        tmax=200000,
        t_short_min=10,
        t_short_max=5000,
        n_short_frames=n_short_frames,
    )
    x_tracks = tracks_one_event["xloc"]
    y_tracks = tracks_one_event["yloc"]
    z_tracks = tracks_one_event["zloc"]
    particles_tracks = tracks_one_event["particle"]

    x_scinti = scintillator_edep_one_event["xloc"]
    y_scinti = scintillator_edep_one_event["yloc"]
    z_scinti = scintillator_edep_one_event["zloc"]
    times_scinti = scintillator_edep_one_event["time"]
    particles_scinti = scintillator_edep_one_event["particle"]

    mask = (
        (particles_scinti == 13)
        | (particles_scinti == -13)
        | (particles_scinti == 2112)
        | (particles_scinti == -2112)
    )
    x_scinti = x_scinti[mask]
    y_scinti = y_scinti[mask]
    z_scinti = z_scinti[mask]
    times_scinti = times_scinti[mask]
    particles_scinti = particles_scinti[mask]
    show_x = gif_config.get("show_x", True)
    if show_x:
        frames = build_frames_x(
            time,
            x_tracks,
            z_tracks,
            particles_tracks,
            bins,
            times_scintillator=times_scinti,
            x_scintillator=x_scinti,
            z_scintillator=z_scinti,
            particles_scintillator=particles_scinti,
            optical_triggers=optical_triggers,
            optical_hits=optical_hits,
        )
    else:
        frames = build_frames_r(
            time,
            x_tracks,
            y_tracks,
            z_tracks,
            particles_tracks,
            bins,
            times_scintillator=times_scinti,
            x_scintillator=x_scinti,
            y_scintillator=y_scinti,
            z_scintillator=z_scinti,
            particles_scintillator=particles_scinti,
            optical_triggers=optical_triggers,
            optical_hits=optical_hits,
        )

    (
        dangerous_muon,
        primary_muon_vetoed,
        detected_neutrons,
        captured_neutrons,
        ge_77_captures,
    ) = make_gif_full(frames, bins, max_ge_energy=max_ge_energy, gif_config=gif_config)

    number_of_optical_hits = len(optical_hits)

    return (
        dangerous_muon,
        primary_muon_vetoed,
        max_ge_energy,
        detected_neutrons,
        captured_neutrons,
        ge_77_captures,
        number_of_optical_hits,
    )
