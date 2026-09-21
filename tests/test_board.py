"""The arithmetic that turns what the board reports into summary quantities."""
import board as B


def test_energy_is_volts_times_milliamps_times_milliseconds_in_microjoules():
    assert abs(B.energy_per_inference_uj(2.0, 5.0, 3.3) - 33.0) < 1e-9


def test_tool_footprint_is_compared_with_the_same_widths():
    c = B.compare_footprint((8, 16, 16), 4, 2 * 2228, 600)
    assert abs(c["flash_ratio"] - 2.0) < 1e-9 and abs(c["ram_ratio"] - 1.0) < 1e-9
