import argparse
import os
import sys

sys.path.append("processing")
from gif_functions import read_files, render_gif
from sim_functions import select_random_muons, run_sim
import time
from datetime import datetime
import yaml
from pathlib import Path
import fcntl

GLOBAL_METADATA_FILE = "out/global_metadata.yaml"


def update_global_metadata(new_data):
    with open(GLOBAL_METADATA_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            metadata = yaml.safe_load(f) or {}

            for key, value in new_data.items():
                metadata[key] = metadata.get(key, 0) + value

            f.seek(0)
            f.truncate()
            yaml.safe_dump(metadata, f, sort_keys=False)

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_gif(input_files_string, n_random_events, job_id=None):
    # --- Setup paths and filenames ---
    ORIGINAL_CWD = os.getcwd()
    SCRIPT_DIR = Path(__file__).resolve().parent
    os.chdir(SCRIPT_DIR)

    try:
        gdml_file = "gdml/new_baseline.gdml"
        if job_id is not None:
            outfile = f"out_{job_id}.lh5"
        else:
            outfile = "out.lh5"
        day_stamp = datetime.now().strftime("%Y-%m-%d")
        base_folder = f"out/{day_stamp}"
        os.makedirs(base_folder, exist_ok=True)
        unix_ts = int(time.time() * 1e6)

        # --- Setup color prints ---
        GREEN = "\033[32m"
        RESET = "\033[0m"

        # --- Select random muons ---
        print(
            f"{GREEN}Selecting {n_random_events} random muon(s) using ray-tracing:{RESET}"
        )
        muon_file = f".selected_muons_{job_id}.dat"
        sampled_muons = select_random_muons(
            input_files_string, muon_file, n_random_events
        )

        # --- Run simulation ---
        print(f"\n{GREEN}Running simulation with selected muon(s):{RESET}")
        run_sim(muon_file, gdml_file, n_random_events, outfile)
        # cleanup
        os.remove(muon_file)

        # --- Make GIF ---
        tracks_array, scintillator_edep_array, optical_array, germanium_array = (
            read_files(outfile)
        )

        # Setup the global metadata file to update stats.

        global_metadata = {
            "events_simulated": 0,
            "muons_sampled": 0,
            "muons_vetoed": 0,
            "dangerous_muons": 0,
            "dangerous_muons_vetoed": 0,
            "ge_77_vetoed": 0,
            "ge_77_creating_muons": 0,
            "ge_77_creating_muons_vetoed": 0,
            "total_detected_neutrons": 0,
            "total_captured_neutrons": 0,
        }

        generated_files = []
        for i in range(n_random_events):
            ge77_veto_threshold = 6  # if more than this neutrons have been detected count it as ge77 vetoed
            GIF_config = {
                "event": i,
                "number_of_frames": 1000,
                "n_short_frames": 200,
                "filename": f"{base_folder}/event_{i}_{unix_ts}.mp4",
                "fps": 20,
                "linger": 20,
                "hightlight_ge77": True,
                "add_scintillator": True,
                "add_pmts": True,
                "neutron_popups": True,
                "add_info_text": True,
                "n_scint_fadeout_mult": 1,
                "ge77_veto_threshold": ge77_veto_threshold,
                "show_x": False,
            }
            (
                dangerous_muon,
                primary_muon_vetoed,
                max_ge_energy,
                detected_neutrons,
                captured_neutrons,
                ge_77_captures,
                number_of_optical_hits,
            ) = render_gif(
                tracks_array,
                scintillator_edep_array,
                optical_array,
                germanium_array,
                GIF_config,
            )
            generated_files.append(GIF_config["filename"])
            print(f"\n{GREEN}GIF generation finished. Updating metadata...{RESET}")
            metadata = {
                "event": int(i),
                "unix_timestamp": int(unix_ts),
                "sampled_muons": int(sampled_muons),
                "dangerous_muon": bool(dangerous_muon),
                "primary_muon_vetoed": bool(primary_muon_vetoed),
                "max_ge_energy": float(max_ge_energy),
                "detected_neutrons": int(detected_neutrons),
                "captured_neutrons": int(captured_neutrons),
                "ge_77_captures": int(ge_77_captures),
                "number_of_optical_hits": int(number_of_optical_hits),
                "gif_shown": False,
            }
            metadata_filename = f"{base_folder}/event_{i}_{unix_ts}.yaml"

            with open(metadata_filename, "w") as f:
                yaml.dump(metadata, f, sort_keys=False)

            # --- update global counters ---
            global_metadata["events_simulated"] += 1

            global_metadata["muons_vetoed"] += int(primary_muon_vetoed)
            global_metadata["dangerous_muons"] += int(dangerous_muon)
            global_metadata["dangerous_muons_vetoed"] += int(
                dangerous_muon and primary_muon_vetoed
            )
            global_metadata["ge_77_vetoed"] += int(
                (detected_neutrons > ge77_veto_threshold)
            )
            global_metadata["ge_77_creating_muons"] += int(ge_77_captures > 0)
            global_metadata["ge_77_creating_muons_vetoed"] += int(
                (ge_77_captures > 0) and (detected_neutrons > ge77_veto_threshold)
            )
            global_metadata["total_detected_neutrons"] += int(detected_neutrons)
            global_metadata["total_captured_neutrons"] += int(captured_neutrons)

        # Updates outside of the loop, to avoid double counting.
        global_metadata["muons_sampled"] += int(sampled_muons)
        update_global_metadata(global_metadata)

        os.remove(outfile)
    finally:
        # Always restore original cwd even if simulation crashes
        os.chdir(ORIGINAL_CWD)

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random muons from input files")

    parser.add_argument(
        "-i",
        dest="input_files_string",
        required=True,
        help="Glob pattern for muon input files. A random file matching this pattern will be selected.",
    )

    parser.add_argument(
        "-n",
        dest="n_random_events",
        required=True,
        type=int,
        help="Number of random events to select from the selected input file.",
    )

    args = parser.parse_args()

    run_gif(args.input_files_string, args.n_random_events)
