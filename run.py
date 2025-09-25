import pyclebsch.su_n_operators as ops
import pyclebsch.cgc as cgc

# i-weights for "3", "3bar", and "8" in SU(3)
irrep_3 = (1,0,0)
irrep_3bar = (1,1,0)
irrep_8 = (2,1,0)

"""
EXAMPLES FOR pyclebsch.su_n_operators
"""

# Dimension of SU(N) irrep
print(ops.calc_dimension(irrep_8))

# GT-patterns which label basis states of SU(N) irrep
for gt in ops.find_gt_patterns(irrep_8):
    print(gt, ops.calc_weight(gt, 'z')) # GT-pattern with generalized Jz eigenvalues
    print(f"{ops.ladder_op([gt], 1, '+')}\n") # GT-pattern raised with 1st generalized raising (ladder) operator

# Find su(N) algebra basis (generators of SU(N) group elements)
for T in ops.find_suN_basis(irrep_8):
    print(f"{T.toarray()}\n")

# Find direct sum decomposition of direct product of SU(N) irreps
# Shows dictionary whose keys are direct sum irreps and whose values are their multiplicities
print(f"{ops.find_direct_sum([irrep_8, irrep_8, irrep_3, irrep_3bar])}\n")

# Show symmetric group details in direct sum decomposition
# Shows tuple:
#
# First element is a dictionary whose keys are direct sum irreps and whose
# values are themselves dictionaries with keys that are tuples of Sn irreps labeled
# by partitions of n, and with values of multiplicity
#
# Second element is a list of lists which tell which input irreps were repeated
print(f"{ops.find_symmetry_direct_sum([irrep_8, irrep_8, irrep_3, irrep_3bar])}\n")

# First argument is SU(N) irrep, the second argument is how many times it is repeated
# Keys of returned dictionary are direct sum irreps. Values are also dictionaries
# whose keys are irreps of Sn labeled by partitions of n, and whose values are multiplicities.
# This is a special case of find_symmetry_direct_sum when all irreps are the same
print(f"{ops.find_plethysms(irrep_8, 3)}\n")

"""
EXAMPLES FOR pyclebsch.cgc
(THIS WILL CAUSE WRITES TO YOUR FILE SYSTEM
IN A FOLDER CALLED CGC_Data WHICH IS IN THE PARENT FOLDER OF pyclebsch)
"""

# This is the dictionary of the CGC data, and it is written into the file system
# All calculations are done where the CGCs are in an irrep of the symmetric group
cgc_dict = cgc.calc_cgcs([irrep_8, irrep_8])

# This function does the same thing as above except it prints the CGC data
# instead of returning it as a dictionary
cgc.print_cgcs([irrep_8, irrep_8])

# This is a check that the CGCs satisfy orthogonality conditions
# This should always return True
print(cgc.check_cgcs([irrep_8, irrep_8]))
