"""Tests for q2m3.utils.plotting module."""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class TestPlotEnergyComparison:
    """Smoke tests for plot_energy_comparison."""

    def test_returns_figure(self):
        from q2m3.utils.plotting import plot_energy_comparison

        hf = [-1.0, -1.1, -1.2, -0.9, -1.05]
        quantum = [-1.01, -1.12, -1.19, -0.91, -1.06]
        fig = plot_energy_comparison(hf, quantum, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_subplots(self):
        from q2m3.utils.plotting import plot_energy_comparison

        hf = [-1.0, -1.1, -1.2]
        quantum = [-1.01, -1.12, -1.19]
        fig = plot_energy_comparison(hf, quantum, show=False)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        from q2m3.utils.plotting import plot_energy_comparison

        out = tmp_path / "test.png"
        hf = [-1.0, -1.1, -1.2]
        quantum = [-1.01, -1.12, -1.19]
        plot_energy_comparison(hf, quantum, output_path=out, show=False)
        assert out.exists()
        plt.close("all")

    def test_custom_title(self):
        from q2m3.utils.plotting import plot_energy_comparison

        hf = [-1.0, -1.1]
        quantum = [-1.01, -1.12]
        fig = plot_energy_comparison(hf, quantum, title="Custom", show=False)
        assert fig._suptitle.get_text() == "Custom"
        plt.close(fig)
