# DistortPerovskite

This is a stand-alone script that can generate distorted perovskite structures only from the Glazer tilting pattern [1] and corresponding angles. 

According the paper by Howard and Stokes [3], there are **15** tilting patterns of perovskite ABX3 which can be physically realized. Which tilting pattern is most stable is not easy to answer as it depends on the constituent element (A, B, X) as well as the external condition (temperature, pressure). Recently, the prediction accuracy of density functional theory (DFT) calculation is reasonably high, so it may be possible to predict (meta-)stability of hypothetical perovskites using DFT. If you are interested in such calculations, this script may be helpful.

This script does the (at least partly) similar thing as [POTATO](https://www.unf.edu/~michael.lufaso/spuds/potato.html) [4], I guess.

## Usage

From the command line, please issue, for example
```bash
> python distort_perovskite.py --elements="La Ni O" --bond_length=1.93 --angles="1 1 1" --tilt_pattern="+++" -o a+a+a+.POSCAR.vasp
```
This will generate the "a+a+a+" structure (Glazer notation) of LaNiO3 with the bond-length of Ni-O being constrained to 1.93 Å. The distorted structure is saved in ``a+a+a+.POSCAR.vasp``.

**Important**

The structure created by this script is slightly perturbed from the cubic ABX3 structure. The A site positions as well as the angles of the octahedral rotation are not optimized. Therefore, the energy might be rather high. Please use the generated structure as input and optimize the internal coordinate (and lattice constant if you need) by DFT calculation.

## Dependencies
- Numpy
- spglib

## References

[1]. A. M. Glazer, "The classification of tilted octahedra in perovskites", Acta Cryst. B28, 3384-3392 (1972).

[2]. P. M. Woodward, "Octahedral Tilting in Perovskites. I. Geometrical Considerations", Acta Cryst. B53, 32 (1997).

[3]. C. J. Howard and H. T. Stokes, "Group-Theoretical Analysis of Octahedral Tilting in Perovskites", Acta Cryst. B54, 782 (1998).

[4]. P. M. Woodward, "POTATO - a program for generating perovskite structures distorted by tilting of rigid octahedra", J. Appl. Cryst. 30, 206-207 (1997).
