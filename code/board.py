"""What the board reports, set beside what the arithmetic predicted.

Nothing here measures anything. The measurements come from the vendor tool running
the model on the microcontroller; this module turns them into summary quantities
and checks them against the analytic footprint of models.py.

Energy per inference follows the vendor's documented method: the run-mode supply
current from the datasheet at the configured clock, times the supply voltage, times
the measured time per inference. It is a derived figure, not a measured one.
"""
import numpy as np
import torch
import models as M


def energy_per_inference_uj(latency_ms, current_ma, voltage_v):
    """Energy for one inference in microjoules: volts x milliamps x milliseconds."""
    return float(voltage_v) * float(current_ma) * float(latency_ms)


def compare_footprint(widths, n_classes, tool_flash_bytes, tool_ram_bytes):
    """Tool-reported flash and RAM divided by the analytic values for the same widths.

    A ratio far from one means the deployed graph is not the one that was evaluated,
    or that the tool counts something the arithmetic leaves out (biases held at 32
    bits, runtime tables, an activation arena).
    """
    fp = M.AccNet(widths, n_classes).footprint_bytes()
    return dict(flash_ratio=tool_flash_bytes / fp["flash"], ram_ratio=tool_ram_bytes / fp["ram"],
                analytic_flash=fp["flash"], analytic_ram=fp["ram"])


def summarise(measurements, n_classes=4):
    """Turn a list of per-model board measurements into summary quantities.

    Each measurement is a dict with widths (tuple or its string), route, tool_flash_bytes,
    tool_ram_bytes, latency_ms, current_ma and voltage_v. Returns the number of models,
    the worst flash and RAM ratios against the analytic footprint, latency and derived
    energy at the largest and smallest widths measured, and, where two routes were
    measured at the same widths, the ratio of their latencies.
    """
    rows = []
    for m in measurements:
        w = tuple(m["widths"]) if not isinstance(m["widths"], str) else \
            tuple(int(v) for v in m["widths"].strip("()").split(","))
        c = compare_footprint(w, n_classes, m["tool_flash_bytes"], m["tool_ram_bytes"])
        rows.append(dict(widths=w, route=m["route"], latency_ms=float(m["latency_ms"]),
                         energy_uj=energy_per_inference_uj(m["latency_ms"], m["current_ma"],
                                                           m["voltage_v"]), **c))
    if not rows:
        return {}
    size = lambda r: M.AccNet(r["widths"], n_classes).n_params()
    big, small = max(rows, key=size), min(rows, key=size)
    same = {}
    for r in rows:
        same.setdefault(r["widths"], []).append(r["latency_ms"])
    ratios = [max(v) / min(v) for v in same.values() if len(v) > 1]
    return dict(models_measured=len(rows),
                flash_tool_over_analytic_max=max(r["flash_ratio"] for r in rows),
                ram_tool_over_analytic_max=max(r["ram_ratio"] for r in rows),
                latency_ms_largest=big["latency_ms"], latency_ms_smallest=small["latency_ms"],
                energy_uj_smallest=small["energy_uj"],
                same_widths_latency_ratio=max(ratios) if ratios else None,
                per_model=[dict(r, widths=str(r["widths"])) for r in rows])


def export_onnx(model, path, win=M.WIN):
    """Export a trained network for the vendor tool, one window of three channels as input.

    Needs the onnx and onnxscript packages, which only the export uses. Quantise
    inside the tool, from the floating-point export, so the deployed graph is
    the tool's int8 graph. The tool's accuracy report can then be compared with the
    int8 weight-only accuracy computed here.
    """
    model.eval()
    torch.onnx.export(model, torch.zeros(1, M.CH, win), path, input_names=["window"],
                      output_names=["logits"], opset_version=13)
    return path
