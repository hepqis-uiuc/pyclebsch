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
import copy
from tqdm import tqdm

import gzip
import json
from pathlib import Path

import numpy as np

from pyclebsch.matrix_elements.lattice_data import (
    ActiveLink,
    PlaquetteState,
    SiteControlLinks,
    SiteMultiplicityIndex,
    irreps_and_singlets,
    physical_plaquette_states,
    sites_links_and_plaquettes, LatticeDef,
    compute_plaquette_signature
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements

type SiteMultiplicitiesGroup = tuple[
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
]
type ActiveLinksGroup = tuple[ActiveLink, ActiveLink, ActiveLink, ActiveLink]
type ControlLinksGroup = tuple[
    SiteControlLinks, SiteControlLinks, SiteControlLinks, SiteControlLinks
]
type PlaquetteStateYmcircFmt = tuple[
    SiteMultiplicitiesGroup, ActiveLinksGroup, ControlLinksGroup
]


def plaq_state_pyclebsch_to_ymcirc_format(
    plaq_state: PlaquetteState,
) -> PlaquetteStateYmcircFmt:
    """
    Convert a single plaquette state from pyclebsch's format to ymcirc's format.
    """
    a_links = tuple(plaq_state[:4])
    c_links = tuple(plaq_state[4:8])
    vertices = tuple(plaq_state[8:])
    if len(vertices) != 4:
        raise ValueError(f"Expected 4 sites. Site count: '{len(vertices)}'.")

    return (vertices, a_links, c_links)


if __name__ == "__main__":
    # Filesystem stuff
    output_mat_elem_json = True
    output_plaquette_states_json = True
    check_mat_elems_against_plaquette_states = True # Run some validation before writing files.
    work_dir = Path("./out")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set up cases to generate data for, and configure options.
    # Note that when saving to disk, plane tuples will be saved
    # as string literals since JSON doesn't preserve this type
    # info otherwise.
    # NOTE: Only site coordinates where distinct lattice signatures
    # can occur need to be provided when specifying a lattice case.
    # For fully periodic lattices, that means only the origin is needed.
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
            "merge_close_mat_elems": True,
            "site_coords_for_comp": [(0, 0, 0)],
            "file_path_state_data": work_dir / "T1_dim(3_2)_plaquette_states.json.gz",
            "file_path_mat_elem_data": work_dir
            / "T1_dim(3_2)_magnetic_hamiltonian.json.gz",
        },
        {
            "dim": "d=3/2",
            "truncation_mode": "T",
            "num_sites": [3, 2, 1],
            "PBCs": [True, False, False],
            "cutoff": 2,
            "merge_close_mat_elems": True,
            "site_coords_for_comp": [(0, 0, 0)],
            "file_path_state_data": work_dir / "T2_dim(3_2)_plaquette_states.json.gz",
            "file_path_mat_elem_data": work_dir
            / "T2_dim(3_2)_magnetic_hamiltonian.json.gz",
        },
        {
            "dim": "d=2",
            "truncation_mode": "T",
            "num_sites": [3, 3, 1],
            "PBCs": [True, True, False],
            "cutoff": 1,
            "merge_close_mat_elems": True,
            "site_coords_for_comp": [(0, 0, 0)],
            "file_path_state_data": work_dir / "T1_dim(2)_plaquette_states.json.gz",
            "file_path_mat_elem_data": work_dir
            / "T1_dim(2)_magnetic_hamiltonian.json.gz",
        },
        {
            "dim": "d=3",       # This case is a single "cube" of links.
            "truncation_mode": "T",
            "num_sites": [2, 2, 2],
            "PBCs": [False, False, False],
            "cutoff": 1,
            "merge_close_mat_elems": True,
            "site_coords_for_comp": [
                (0, 0, 0),      # bottom, front, left side
                (1, 0, 0),      # right side
                (0, 1, 0),      # back
                (1, 1, 1)       # top
            ],
            "file_path_state_data": work_dir / "T1_dim(3)_OBC_plaquette_states.json.gz",
            "file_path_mat_elem_data": work_dir
            / "T1_dim(3)_OBC_magnetic_hamiltonian.json.gz",
        },
        # {
        #     "dim": "d=3",
        #     "truncation_mode": "T",
        #     "num_sites": [2, 2, 2],
        #     "PBCs": [True, True, True],
        #     "cutoff": 1,
        #     "merge_close_mat_elems": True,
        #     "site_coords_for_comp": [(0, 0, 0)],
        #     "file_path_state_data": work_dir / "T1_dim(3)_cube_PBC_plaquette_states.json.gz",
        #     "file_path_mat_elem_data": work_dir / "T1_dim(3)_cube_PBC_magnetic_hamiltonian.json.gz"
        # },
    ]
    parallelize = True  # May cause EOFError. Rerun if this happens.

    # Data generation.
    # Note: tuple data are converted to string types to prevent the JSON file
    # writes from failing.
    for lattice_case in lattice_cases:
        # Initial config for current lattice case.
        if check_mat_elems_against_plaquette_states is True:
            plaquette_states_in_mat_elem_data = []
            plaquette_states_in_plaquette_state_data = []
        lattice = LatticeDef(lattice_case["num_sites"], lattice_case["PBCs"], FORDER)
        truncation_mode = lattice_case["truncation_mode"]
        cutoff = lattice_case["cutoff"]
        print(f"Current lattice: {lattice}")

        # Compute information needed to obtain plaquette states.
        sites, links, plaquettes = sites_links_and_plaquettes(
            lattice_case["num_sites"], lattice_case["PBCs"], FORDER
        )
        truncation_irreps, singlets, conj_dict = irreps_and_singlets(
            N_colors, sites, lattice_case["truncation_mode"], lattice_case["cutoff"]
        )

        metadata_dict = {
            "dim": lattice_case["dim"],
            "truncation_mode": lattice_case["truncation_mode"],
            "num_sites": lattice_case["num_sites"],
            "PBCs": lattice_case["PBCs"],
            "cutoff": lattice_case["cutoff"],
            "planes": list(map(str, lattice.planes)),
            "site_coords_for_comp": lattice_case["site_coords_for_comp"],
            "close_mat_elems_merged": lattice_case["merge_close_mat_elems"],
            "f_order": FORDER,
        }

        if output_plaquette_states_json is True:
            print(f"Generating data for {lattice_case['file_path_state_data']}")
            plaq_states_result_dict = {"data": [], "metadata": metadata_dict}
            # Compute plaquette states for each plane, and aggregate.
            plaq_states = []
            for current_plane in tqdm(lattice.planes, desc="Plane iteration for plaq states data"):
                for site_coordinate in tqdm(lattice_case["site_coords_for_comp"], desc=f"Site iteration for plaq states data, plane={current_plane}"):
                    plaquette_address = (site_coordinate, current_plane)
                    if plaquette_address not in lattice.plaquettes.keys():
                        continue # Skip plaquette addresses that don't actually exist.
                    plaq_states_current_plane = []
                    for plaq_state in tqdm(physical_plaquette_states(
                            plaquette_address, sites, plaquettes, singlets, FORDER
                        ), desc=f"Enumerate plaquette states at address {plaquette_address}"):
                        plaq_states_current_plane += [str(plaq_state_pyclebsch_to_ymcirc_format(plaq_state))]
                        if check_mat_elems_against_plaquette_states is True:
                            plaquette_states_in_plaquette_state_data += (plaq_state_pyclebsch_to_ymcirc_format(plaq_state),)
                    plaq_states += plaq_states_current_plane

            # Remove duplicates from the list of plaquette states.
            plaq_states = list(set(plaq_states))

            # Construct data for JSON file
            plaq_states_result_dict["data"] = plaq_states


        if output_mat_elem_json is True:
            print(f"Generating data for {lattice_case['file_path_mat_elem_data']}")
            mat_elem_result_dict = {"data": {}, "metadata": metadata_dict}
            # Iterate over planes, compute all mat elems in that plane, then
            # store in nested dicts with following key hierarchy:
            # <Pf|Pi> -> plane -> site half links
            # NOTE: this key hierarchy will be "rolled up" as much as
            # possible if the option "merge_close_mat_elems" is True.
            for current_plane in tqdm(lattice.planes, desc="Plane iteration for mat elem data"):
                for site_coordinate in tqdm(lattice_case["site_coords_for_comp"], desc=f"Site iteration for mat elem data, plane={current_plane}"):
                    plaquette_address = (site_coordinate, current_plane)
                    if plaquette_address not in lattice.plaquettes.keys():
                        continue # Skip plaquette addresses that don't actually exist.
                    _, plaquette_site_half_links = compute_plaquette_signature(plaquette_address, lattice)
                    mat_elems_current_plane = calc_plaquette_elements(
                        N_colors,
                        plaquette_address,
                        sites,
                        plaquettes,
                        truncation_irreps,
                        singlets,
                        conj_dict,
                        FORDER,
                        EPS,
                        PRES,
                        parallelize,
                    )
                    for (Pf, Pi), mat_elem_value in tqdm(mat_elems_current_plane.items(), desc=f"Enumerate mat elems at plaquette address {plaquette_address}"):
                        # Construct the state transition key.
                        Pf_ymcirc_format = plaq_state_pyclebsch_to_ymcirc_format(Pf)
                        Pi_ymcirc_format = plaq_state_pyclebsch_to_ymcirc_format(Pi)
                        Pf_Pi_key = (Pf_ymcirc_format, Pi_ymcirc_format)

                        if check_mat_elems_against_plaquette_states is True:
                            if Pf_ymcirc_format not in plaquette_states_in_mat_elem_data:
                                plaquette_states_in_mat_elem_data.append(Pf_ymcirc_format)
                            if Pi not in plaquette_states_in_mat_elem_data:
                                plaquette_states_in_mat_elem_data.append(Pi_ymcirc_format)


                        # Ensure hierarchical key structure exists before setting matrix element value.
                        # NOTE: Keys cast as strings to preserve tuple type when saved as JSON.
                        Pf_Pi_key_str = str(Pf_Pi_key)
                        current_plane_str = str(current_plane)
                        plaquette_site_half_links_str = str(plaquette_site_half_links)
                        if Pf_Pi_key_str not in mat_elem_result_dict["data"].keys():
                            mat_elem_result_dict["data"][Pf_Pi_key_str] = {}
                        if current_plane not in mat_elem_result_dict["data"][Pf_Pi_key_str]:
                            mat_elem_result_dict["data"][Pf_Pi_key_str][current_plane_str] = {}

                        # Set the matrix element value.
                        mat_elem_result_dict["data"][Pf_Pi_key_str][current_plane_str][plaquette_site_half_links_str] = mat_elem_value

            # Now see how much merging we can do. If all the matrix elements
            # at a lowest level of the key hierarchy are identical, remove that
            # level in the key hierarchy.
            if lattice_case["merge_close_mat_elems"] is True:
                mat_elem_data_merged_if_possible = copy.deepcopy(mat_elem_result_dict["data"])
                for Pf_Pi_key, mat_elems_Pf_Pi in tqdm(mat_elem_result_dict["data"].items(), desc="Attempting to merge mat elems"):
                    mat_elem_data_merged_if_possible[Pf_Pi_key] = {}
                    for current_plane, mat_elems_Pf_Pi_current_plane in mat_elems_Pf_Pi.items():
                        mat_elem_values = list(mat_elems_Pf_Pi_current_plane.values())
                        if np.allclose(mat_elem_values, mat_elem_values[0]) is True:
                            mat_elem_data_merged_if_possible[Pf_Pi_key][current_plane] = mat_elem_values[0]
                    all_signatures_merged_current_plane = all(isinstance(val, float) or isinstance(val, int) for val in mat_elem_data_merged_if_possible[Pf_Pi_key].values())
                    if all_signatures_merged_current_plane is True:
                        mat_elem_values = list(mat_elem_data_merged_if_possible[Pf_Pi_key].values())
                        if np.allclose(mat_elem_values, mat_elem_values[0]):
                            mat_elem_data_merged_if_possible[Pf_Pi_key] = mat_elem_values[0]

                mat_elem_result_dict["data"] = mat_elem_data_merged_if_possible


        # Sanity check before doing file writes.
        # There should be no state labels on the matrix elements
        # that don't appear in the set of all physical plaquette states.
        # Subset check instead of equality check because vanishing matrix
        # elements not included in data file.
        if check_mat_elems_against_plaquette_states is True:
            print("Checking that state labels are consistent between files...")
            assert len(plaq_states_result_dict["data"]) == len(set(plaq_states_result_dict["data"]))
            plaquette_states_in_mat_elem_data = set(plaquette_states_in_mat_elem_data)
            plaquette_states_in_plaquette_state_data = set(plaquette_states_in_plaquette_state_data)
            assert plaquette_states_in_mat_elem_data.issubset(plaquette_states_in_plaquette_state_data)
            print("Done.\n")

        # save plaquette states data to disk
        with gzip.open(
            lattice_case["file_path_state_data"], "wt", encoding="utf-8"
        ) as f:
            json.dump(plaq_states_result_dict, f)

        # save matrix element data to disk
        with gzip.open(
            lattice_case["file_path_mat_elem_data"], "wt", encoding="utf-8"
        ) as f:
            json.dump(mat_elem_result_dict, f)
