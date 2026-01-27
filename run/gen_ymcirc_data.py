"""
This script generates plaquette state
and Wilson loop matrix element data
in the form of JSON files which are
ingestible by ymcirc.

To add additional lattice cases, follow
the pattern of data already present in
the list lattice_cases below.

See su_n_wilson_loop.py for more detailed
information about various script options.
"""
from pyclebsch.matrix_elements.lattice_data import *
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements
from pathlib import Path
import json

if __name__ == "__main__":
    # Filesystem stuff
    output_mat_elem_json = True
    output_plaquette_states_json = True
    work_dir = Path("./out")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set up cases to generate data for, and configure options.
    # Note that when saving to disk, plane tuples will be saved
    # as string literals since JSON doesn't preserve this type
    # info otherwise.
    FORDER = [1, 2, 3, -1, -2, -3]
    EPS = 1e-10
    N_colors = 3
    PRES = 10
    lattice_cases = [
        {
            "dim": "d=3/2",
            "truncation_mode": "T",
            "num_sites": [3, 2, 1],
            "PBCs": [True, False, False],
            "cutoff": 1,
            "planes": [(1, 2)],
            "file_path_state_data": work_dir / "T1_dim(3_2)_plaquette_states.json",
            "file_path_mat_elem_data": work_dir / "T1_dim(3_2)_magnetic_hamiltonian.json"
        },
        {
            "dim": "d=3/2",
            "truncation_mode": "T",
            "num_sites": [3, 2, 1],
            "PBCs": [True, False, False],
            "cutoff": 2,
            "planes": [(1, 2)],
            "file_path_state_data": work_dir / "T2_dim(3_2)_plaquette_states.json",
            "file_path_mat_elem_data": work_dir / "T2_dim(3_2)_magnetic_hamiltonian.json"
        },
        {
            "dim": "d=2",
            "truncation_mode": "T",
            "num_sites": [3, 3, 1],
            "PBCs": [True, True, False],
            "cutoff": 1,
            "planes": [(1, 2)],
            "file_path_state_data": work_dir / "T1_dim(2)_plaquette_states.json",
            "file_path_mat_elem_data": work_dir / "T1_dim(2)_magnetic_hamiltonian.json"
        },
        {
            "dim": "d=3",
            "truncation_mode": "T",
            "num_sites": [3, 3, 3],
            "PBCs": [False, False, False],
            "cutoff": 1,
            "planes": [(1, 2), (2, 3), (1, 3)],
            "file_path_state_data": work_dir / "T1_dim(3)_OBC_plaquette_states.json",
            "file_path_mat_elem_data": work_dir / "T1_dim(3)_OBC_magnetic_hamiltonian.json"
        },
        # {
        #     "dim": "d=3",
        #     "truncation_mode": "T",
        #     "num_sites": [2, 2, 2],
        #     "PBCs": [True, True, True],
        #     "cutoff": 1,
        #     "planes": [(1, 2), (2, 3), (1, 3)],
        #     "file_path_state_data": work_dir / "T1_dim(3)_cube_PBC_plaquette_states.json",
        #     "file_path_mat_elem_data": work_dir / "T1_dim(3)_cube_PBC_magnetic_hamiltonian.json"
        # },
    ]
    parallelize = True         # May cause EOFError. Rerun if this happens.

    # Data generation.
    # Note: tuple data are converted to string types to prevent the JSON file
    # writes from failing.
    for lattice_case in lattice_cases:
        num_sites = lattice_case["num_sites"]
        PBCs = lattice_case["PBCs"]
        truncation_mode = lattice_case["truncation_mode"]
        cutoff = lattice_case["cutoff"]

        sites, links, plaquettes = sites_links_and_plaquettes(lattice_case["num_sites"], lattice_case["PBCs"], FORDER)
        truncation_irreps, singlets, conj_dict = irreps_and_singlets(N_colors, sites, lattice_case["truncation_mode"], lattice_case["cutoff"])
        lattice_origin = (0, 0, 0)

        metadata_dict = {
                    "dim": lattice_case["dim"],
                    "truncation_mode": lattice_case["truncation_mode"],
                    "num_sites": lattice_case["num_sites"],
                    "PBCs": lattice_case["PBCs"],
                    "cutoff": lattice_case["cutoff"],
                    "planes": list(map(str, lattice_case["planes"])), # type: ignore
                    "f_order": FORDER
                }

        if output_plaquette_states_json is True:
            print(f"Generating {lattice_case['file_path_state_data']}")
            plaq_states_result_dict = {
                "data": [],
                "metadata": metadata_dict
            }
            # Compute plaquette states for each plane, and aggregate.
            plaq_states = []
            for current_plane in lattice_case["planes"]: # type: ignore
                plaq_site_plane = (lattice_origin, current_plane)
                plaq_states += list(map(str, physical_plaquette_states(plaq_site_plane, sites, plaquettes, singlets, FORDER))) # For later JSON encoding.

            # Remove duplicates from the list of plaquette states.
            plaq_states = list(set(plaq_states))

            # Construct json file
            plaq_states_result_dict["data"] = plaq_states

            # save to disk
            with lattice_case["file_path_state_data"].open("w", encoding="utf-8") as f: # type: ignore
                json.dump(plaq_states_result_dict, f)

        if output_mat_elem_json is True:
            print(f"Generating {lattice_case['file_path_mat_elem_data']}")
            mat_elem_result_dict = {
                "data": {},
                "metadata": metadata_dict
            }
            # Compute matrix elements for each plane
            mat_elems_by_plane = {}
            for current_plane in lattice_case["planes"]: # type: ignore
                plaq_site_plane = (lattice_origin, current_plane)
                mat_elems_by_plane[current_plane] = calc_plaquette_elements(N_colors, plaq_site_plane, sites, plaquettes, truncation_irreps, singlets, conj_dict, FORDER, EPS, PRES, parallelize)

            # Merge into single dict
            mat_elems_merged = {}
            for current_plane, mat_elem_data in mat_elems_by_plane.items():
                for (Pf, Pi), mat_elem_val in mat_elem_data.items():
                    mat_elems_merged.setdefault((Pf, Pi), {})[current_plane] = mat_elem_val

            # Collapse plane info if a matrix element has the same value in all planes
            mat_elems_collapsed = {}
            for (Pf, Pi), plane_to_val_map in mat_elems_merged.items():
                current_mat_elem_key_as_str = str((Pf, Pi)) # For later JSON encoding
                values = list(plane_to_val_map.values())
                if np.allclose(values, values[0]) is True:
                    mat_elems_collapsed[current_mat_elem_key_as_str] = np.mean(values)
                else:
                    plane_to_val_map_keys_as_strings = {str(current_plane): mat_elem_val for current_plane, mat_elem_val in plane_to_val_map.items()} # For later JSON encoding
                    mat_elems_collapsed[current_mat_elem_key_as_str] = plane_to_val_map_keys_as_strings

            # Construct json file
            mat_elem_result_dict["data"] = mat_elems_collapsed

            # save to disk
            with lattice_case["file_path_mat_elem_data"].open("w", encoding="utf-8") as f: # type: ignore
                json.dump(mat_elem_result_dict, f)
