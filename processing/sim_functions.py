import random
import numpy as np
from glob import glob
from remage import remage_run


CYL_RADIUS = 600.0
CYL_Z_MIN = -410.0
CYL_Z_MAX = 410.0


# Do some ray tracing
def ray_hits_cylinder(x0, y0, z0, dx, dy, dz, radius, zmin, zmax):

    # -----------------------
    # Side wall intersection
    # -----------------------
    a = dx**2 + dy**2
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0**2 + y0**2 - radius**2

    # if ray is parallel to the cylinder axis, it can not hit the side wall
    if not np.isclose(a, 0.0):
        disc = b**2 - 4 * a * c

        # discriminant of quadratic formula. If negative, no intersection with infinite cylinder
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)

            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)

            for t in (t1, t2):
                if t < 0:
                    continue

                z_hit = z0 + t * dz

                if zmin <= z_hit <= zmax:
                    return True

    # -----------------------
    # Cap intersections
    # -----------------------

    for zcap in (zmin, zmax):
        if np.isclose(dz, 0.0):
            continue  # ray parallel to caps

        # point of intersection with plane of cap
        t = (zcap - z0) / dz
        # if t < 0, the intersection is behind the ray origin
        if t < 0:
            continue

        x_hit = x0 + t * dx
        y_hit = y0 + t * dy

        # check if the hit point is within the radius of the cap
        if x_hit**2 + y_hit**2 <= radius**2:
            return True

    return False


# Build line offsets to read in lines fast
def build_line_offsets(filename):

    offsets = []

    with open(filename, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()

            if not line:
                break

            offsets.append(pos)

    return offsets


def select_random_muons(input_files_string, output_file, n_random_events):
    input_files = glob(input_files_string)

    if not input_files:
        print("No input files found")
        return

    input_file = random.choice(input_files)
    offsets = build_line_offsets(input_file)

    selected = 0
    used_indices = set()

    with open(output_file, "w") as out:
        while selected < n_random_events:
            idx = random.randrange(len(offsets))

            if idx in used_indices:
                continue

            used_indices.add(idx)

            with open(input_file, "r") as f:
                f.seek(offsets[idx])

                line = f.readline().strip()

            if not line:
                continue

            cols = line.split()

            if len(cols) < 9:
                continue

            try:
                x0 = float(cols[3])
                y0 = float(cols[4])
                z0 = float(cols[5])

                dx = float(cols[6])
                dy = float(cols[7])
                dz = float(cols[8])

            except ValueError:
                continue

            # normalize direction
            norm = np.sqrt(dx**2 + dy**2 + dz**2)

            if norm == 0:
                continue

            dx /= norm
            dy /= norm
            dz /= norm

            if ray_hits_cylinder(
                x0, y0, z0, dx, dy, dz, CYL_RADIUS, CYL_Z_MIN, CYL_Z_MAX
            ):
                out.write(line + "\n")

                selected += 1

    print(f"Finished selecting {n_random_events} muon(s).")
    print(
        f"{len(used_indices) - n_random_events} sampled muon(s) barely missed the experiment setup."
    )
    print(
        f"This corresponds to {len(used_indices) * 60 / 514:.2f} minutes experiment lifetime."
    )
    return len(used_indices)


def run_sim(muon_file, gdml_file, n_events, output_file):
    macro_lines = [
        "/RMG/Manager/Logging/LogLevel summary",
        "/RMG/Geometry/GDMLDisableOverlapCheck true",
        "/RMG/Processes/HadronicPhysics Shielding",
        "/RMG/Processes/OpticalPhysics true",
        "/RMG/Geometry/RegisterDetectorsFromGDML Germanium",
        "/RMG/Geometry/RegisterDetectorsFromGDML Optical",
        "/RMG/Geometry/RegisterDetector Scintillator atmosphericlar 12000",
        "/RMG/Geometry/RegisterDetector Scintillator water 12001",
        "/RMG/Geometry/RegisterDetector Scintillator skirt 12002",
        "/RMG/Geometry/RegisterDetector Scintillator foot 12003",
        "/RMG/Geometry/RegisterDetector Scintillator outercryostat 12004",
        "/RMG/Geometry/RegisterDetector Scintillator vacuumgap 12005",
        "/RMG/Geometry/RegisterDetector Scintillator innercryostat 12006",
        "/RMG/Geometry/RegisterDetector Scintillator reentrancetube 12007",
        "/RMG/Geometry/RegisterDetector Scintillator neutronmoderator 12008",
        "/RMG/Geometry/RegisterDetector Scintillator undergroundlar 12009",
        "/RMG/Output/ActivateOutputScheme Track",
        "/RMG/Processes/DefaultProductionCut 1 mm",
        "/RMG/Processes/SensitiveProductionCut 1 mm",
        "/RMG/Processes/UseGrabmayrsGammaCascades true",
        "/RMG/GrabmayrGammaCascades/SetGammaCascadeFile 1 1 simfiles/water_cascades.txt",  # Geant4.11.03 is cooked
        "/run/initialize",
        "/RMG/Output/Scintillator/EdepCutLow 25 keV",  # Only store events that do something
        "/RMG/Output/Scintillator/Cluster/PreClusterOutputs false",  # might be possible to play with this
        "/RMG/Output/NtuplePerDetector false",
        "/RMG/Output/Scintillator/DiscardZeroEnergyHits false",  # in order to visualize neutron movement.
        "/RMG/Output/Track/AddProcessFilter RMGnCapture",
        "/RMG/Processes/Stepping/DaughterNucleusMaxLifetime 1 hour",
        "/RMG/Generator/Confine UnConfined",
        "/RMG/Generator/Select MUSUNCosmicMuons",
        f"/RMG/Generator/MUSUNCosmicMuons/MUSUNFile {muon_file}",
        f"/run/beamOn {n_events}",
    ]

    try:
        # Run remage
        remage_run(
            macros=macro_lines,
            gdml_files=gdml_file,
            output=output_file,
            overwrite_output=True,
            flat_output=True,
        )
    except Exception as e:
        # Catch any error that remage_run might raise
        print(f"❌ Simulation failed: {e}")  #
