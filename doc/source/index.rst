What is q2m3?
=============

.. image:: _static/logo.svg
   :alt: q2m3 logo
   :class: q2m3-home-logo

.. Badge block — keep in sync with README.md (manual sync).

.. |badge-python| image:: https://img.shields.io/badge/python-3.11%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. |badge-license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. |badge-docs-ci| image:: https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml
   :alt: Documentation

.. |badge-docs-site| image:: https://img.shields.io/badge/docs-latest-blue
   :target: https://yjmaxpayne.github.io/q2m3/
   :alt: Docs latest

.. |badge-codecov| image:: https://codecov.io/gh/yjmaxpayne/q2m3/graph/badge.svg
   :target: https://codecov.io/gh/yjmaxpayne/q2m3
   :alt: codecov

.. |badge-doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20114945.svg
   :target: https://doi.org/10.5281/zenodo.20114945
   :alt: DOI

.. |badge-pennylane| image:: https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982
   :target: https://pennylane.ai/
   :alt: PennyLane

.. |badge-catalyst| image:: https://img.shields.io/badge/Catalyst-%3E%3D0.14.0-01A982
   :target: https://github.com/PennyLaneAI/catalyst
   :alt: Catalyst

.. |badge-pyscf| image:: https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue
   :target: https://pyscf.org/
   :alt: PySCF

|badge-python| |badge-license| |badge-docs-ci| |badge-docs-site| |badge-codecov| |badge-doi| |badge-pennylane| |badge-catalyst| |badge-pyscf|

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
