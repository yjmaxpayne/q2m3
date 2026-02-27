import os
from unittest.mock import MagicMock, patch

from q2m3.profiling.catalyst_ir import (
    COMPILATION_STAGES,
    analyze_ir_stages,
    ir_output_dir,
)


def test_compilation_stages_is_list():
    assert isinstance(COMPILATION_STAGES, list)
    assert len(COMPILATION_STAGES) >= 1
    assert all(isinstance(s, str) for s in COMPILATION_STAGES)


def test_ir_output_dir_creates_and_restores_cwd():
    original_cwd = os.getcwd()
    with ir_output_dir() as ir_dir:
        assert os.path.isdir(ir_dir)
        # cwd should have changed
        assert os.getcwd() == ir_dir
    # After context exit, cwd should be restored
    assert os.getcwd() == original_cwd


def test_ir_output_dir_with_explicit_path(tmp_path):
    target = str(tmp_path / "ir_test")
    os.makedirs(target, exist_ok=True)
    original_cwd = os.getcwd()
    with ir_output_dir(path=target) as ir_dir:
        assert ir_dir == target
        assert os.getcwd() == target
    assert os.getcwd() == original_cwd


def test_ir_output_dir_restores_cwd_on_exception():
    original_cwd = os.getcwd()
    try:
        with ir_output_dir():
            raise RuntimeError("test exception")
    except RuntimeError:
        pass
    # cwd should still be restored even after exception
    assert os.getcwd() == original_cwd


def test_analyze_ir_stages_with_mock():
    """Test with mocked get_compilation_stage to avoid Catalyst dependency."""
    mock_fn = MagicMock()
    mock_ir_text = "func @main() {\n  %0 = qreg(2)\n}\n"

    with patch(
        "q2m3.profiling.catalyst_ir.get_compilation_stage",
        return_value=mock_ir_text,
    ):
        result = analyze_ir_stages(mock_fn, stages=["mlir"])

    assert len(result) == 1
    stage_name, size_kb, n_lines = result[0]
    assert stage_name == "mlir"
    assert isinstance(size_kb, float)
    assert n_lines == 3  # 3 lines in mock_ir_text


def test_analyze_ir_stages_empty_on_unavailable():
    """When get_compilation_stage raises, return empty list."""
    mock_fn = MagicMock()

    with patch(
        "q2m3.profiling.catalyst_ir.get_compilation_stage",
        side_effect=Exception("Catalyst not available"),
    ):
        result = analyze_ir_stages(mock_fn)

    assert result == []
