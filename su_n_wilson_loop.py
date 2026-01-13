"""
MAIN
"""

if __name__ == "__main__":

    ### IMPORTS ###

    from pyclebsch.matrix_elements.lattice_data import *
    from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements

    '''
    There are 11 variables in this script that you can modify. After setting
    them, all lattice data (site/link/plaquette coordinates and truncated
    irrep/singlet lists) will be created, and matrix elements and/or physical
    plaquette states will be generated.

    FORDER: This assigns a lattice link to every irrep appearing in a singlet
            CGC. Any permutation of this list is valid. Although you can change
            this list, it is recommended that you stick with a choice when
            computing matrix elements for different plaquettes.
    
    EPS: All numbers x with |x| < EPS are treated as zero.

    PRES: Matrix element values are rounded to this many decimal digits.

    N: The number of colors; AKA the N in SU(N). N must be greater than 1.

    nums_sites: The number of sites along each axis of the lattice. Each axis
                must have at least 1 site. Here are some examples.
                -- [2,2,1] with PBCs=[False,False,False] is a lattice with a
                   single plaquette.
                   o -- o
                   |    |
                   o -- o
                -- [2,2,1] with PBCs=[True,False,False] is a lattice with two
                   adjacent plaquettes. The plaquettes are made of the same sites.
                   o -- o --
                   |    |
                   o -- o --
                -- [3,2,1] with PBCs=[False,False,False] is a lattice with two
                   adjacent plaquettes. The plaquettes have some sites not in common.
                   o -- o -- o
                   |    |    |
                   o -- o -- o
                -- [2,2,1] with PBCs=[True,True,False] is a lattice with four
                   plaquettes. All plaquettes have the same sites.
                   |    |
                   o -- o --
                   |    |
                   o -- o --
                -- [2,2,2] with PBCs=[True,True,True] is the smallest 3d lattice
                   that can be made with full periodic boundary conditions.
                -- [2,2,2] with PBCs=[False,False,False] is the smallest 3d lattice
                   (a cube) that can be made with full open boundary conditions.

    PBCs: Tells whether a given lattice axis has periodic boundary conditions.
          True means periodic boundary conditions, False means open boundary
          conditions. An axis with open boundary conditions can have 1 site.
          However, an axis with periodic boundary conditions must have at least
          2 sites. (Each link must connect two distinct sites.)

    truncation_mode: The kind of Hilbert space truncation to use ('T', 'C', or 'B').
                     'T' sets the maximum value the first i-weight entry can have.
                     'C' sets the maximum quadratic Casimir value a link irrep can have.
                     'B' sets the maximum sum of quadratic Casimirs any site singlet can have.

    cutoff: The truncation value for a given mode. It can be a positive integer
            for 'T'-type truncaitons. It can be a positive float for 'C'- and
            'B'-type truncations.

    parallelize: Whether to use parallelization in calculating matrix elements.
                 Note that if Clebsch-Gordan coefficients are calculated for the
                 first time, an EOFError may appear when parallelizing the site
                 factor calculations, and the script will have to be ran again to
                 calculate the matrix elements.

    generate_plaq_states: Whether to print out the physical plaquette states.

    plaq_site_plane: The plaquette to generate the matrix elements and/or
                     physical plaquette states for. A plaquette is specified
                     by a site coordinate (x,y,z) and a lattice plane (i,j),
                     where i<j. x,y,z = 0,1,2,... while i,j = 1,2,3. These
                     specifications should be consistent with the lattice set
                     up with num_sites and PBCs.
    '''

    ### SETUP ###

    FORDER = [1, 2, 3, -1, -2, -3]
    EPS = 1e-10
    PRES = 10

    N = 3
    num_sites = [3,2,1]
    PBCs = [True, False, False]

    truncation_mode = 'T'
    cutoff = 1

    parallelize = True
    generate_plaq_states = False

    plaq_site_plane = ((0,0,0), (1,2))

    ### RUN CODE ###

    sites, links, plaquettes = sites_links_and_plaquettes(num_sites, PBCs, FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(N, sites, truncation_mode, cutoff)

    if generate_plaq_states:
        plaq_states = physical_plaquette_states(plaq_site_plane, sites, plaquettes, singlets, FORDER)
        for state in plaq_states:
            print(state[0:4])
            print(state[4], state[5], state[6], state[7])
            print(state[-4:])
            print()
        print('# Plaquette States:', len(plaq_states))
        print('\n' + '='*100 + '\n')

    matrix_elems = calc_plaquette_elements(N, plaq_site_plane, sites, plaquettes, truncation_irreps, singlets, conj_dict, FORDER, EPS, PRES, parallelize)
    for Pf,Pi in matrix_elems:
        val = matrix_elems[(Pf,Pi)]
        print(Pf[0:4])
        print(Pi[0:4])
        print(Pf[4],Pf[5],Pf[6],Pf[7])
        print(Pf[-4:])
        print(Pi[-4:])
        print(val)
        print()
    print('# Matrix Elements:', len(matrix_elems))
