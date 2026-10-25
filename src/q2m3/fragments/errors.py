"""Typed failures raised by the fragment interaction-energy workflow."""


class FragmentWorkflowError(RuntimeError):
    """The fragment workflow cannot produce a valid result for its input."""


class ActiveSpaceMismatchError(FragmentWorkflowError):
    """Monomer active spaces do not correspond to the dimer active space."""


class ReferenceImportError(ValueError):
    """An imported reference table violates the required format."""


class ExternalProgramError(FragmentWorkflowError):
    """An external quantum-chemistry program is not available or did not complete."""
