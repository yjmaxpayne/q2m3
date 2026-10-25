# Input attribution

`water-dimer.xyz` copies the [tutorial geometry](../../../doc/source/tutorials/water-dimer.xyz)
without changes. The tutorial identifies it as S22 water dimer, system 2.
The XYZ comment retains the system name, Angstrom unit, and fragment indices.
These examples include no S22 benchmark energies.

`cod_2010678.cif` copies the repository's existing CIF fixture without changes.
The fixture came from the [Crystallography Open Database](https://www.crystallography.net/cod/2010678.html).
The CIF header records its source and attribution terms.
The header credits IUCr Journals as the original data provider.

The article is Zhi Min Jin, Yuan Jiang Pan, Duan Jun Xu, and Yuan Zhi Xu,
“The 1:1 complex of 4-nitrophenol and 4-methylpyridine,”
*Acta Crystallographica Section C* **56**, e69–e70 (2000),
[doi:10.1107/S0108270100001074](https://doi.org/10.1107/S0108270100001074).
The repository fixture removed trailing spaces on two comment lines.
It changed no data values.

Runtime code reads only the files in this directory.
It does not import test fixtures or download coordinates.
The structure API reads the asymmetric unit without symmetry expansion or periodic images.

See the [fragment guide](../README.md) or [中文说明](../README_CN.md) for normalization and calculation settings.
