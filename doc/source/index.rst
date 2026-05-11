What is q2m3?
=============

.. image:: _static/logo.svg
   :alt: q2m3 logo
   :class: q2m3-home-logo

q2m3 is a hybrid quantum-classical QM/MM framework for small-molecule
quantum chemistry workflows that combine PySCF, PennyLane QPE circuits,
explicit MM point charges, Monte Carlo solvation, and EFTQC resource
estimation.

The documentation emphasizes lightweight H2 examples first. Catalyst, H3O+,
8-bit QPE, and dynamic-Trotter diagnostics are documented as optional paths
because they can require substantially more memory and compile time.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting-started
   core-concepts
   architecture
   tutorials/index
   development
   glossary

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api-reference/index

Project Links
-------------

* Source repository: https://github.com/yjmaxpayne/q2m3
* Issue tracker: https://github.com/yjmaxpayne/q2m3/issues
* Zenodo archive (concept DOI): https://doi.org/10.5281/zenodo.20114945

Citation
--------

If you use q2m3 in research, please cite the Zenodo archive. The concept
DOI `10.5281/zenodo.20114945 <https://doi.org/10.5281/zenodo.20114945>`_
always resolves to the latest published release. Use a version-specific
DOI (for example
`10.5281/zenodo.20114946 <https://doi.org/10.5281/zenodo.20114946>`_ for
``v0.1.1``) when you need to reference the exact code behind a result.
``CITATION.cff`` in the repository root lists both DOIs and powers
GitHub's "Cite this repository" button.

.. code-block:: bibtex

   @software{q2m3_2026,
     title     = {q2m3: A Hybrid Quantum-Classical Framework for QM/MM Simulations},
     author    = {Ye Jun},
     year      = {2026},
     publisher = {Zenodo},
     doi       = {10.5281/zenodo.20114945},
     url       = {https://doi.org/10.5281/zenodo.20114945}
   }
