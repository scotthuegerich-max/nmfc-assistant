"""
density.py

Deterministic layer of the pipeline: no AI involved. Computes density from
shipment dimensions/weight, then resolves the applicable freight class for
a given commodity entry (density-based or fixed-class).
"""

from dataclasses import dataclass


@dataclass
class DensityResult:
    density_pcf: float
    cubic_feet: float


def compute_density_pcf(
    length_in: float, width_in: float, height_in: float, weight_lbs: float
) -> DensityResult:
    """
    Standard NMFC density formula: pounds per cubic foot.
    cubic_feet = (L * W * H in inches) / 1728
    density_pcf = weight_lbs / cubic_feet
    """
    if length_in <= 0 or width_in <= 0 or height_in <= 0:
        raise ValueError("Dimensions must be positive numbers")
    if weight_lbs <= 0:
        raise ValueError("Weight must be a positive number")

    cubic_feet = (length_in * width_in * height_in) / 1728
    density_pcf = weight_lbs / cubic_feet

    return DensityResult(density_pcf=round(density_pcf, 2), cubic_feet=round(cubic_feet, 4))


def resolve_class(entry: dict, density_pcf: float) -> dict:
    """
    Given a commodity entry (from sample_nmfc_dataset.json) and a computed
    density, return the applicable class plus which rule produced it.

    Fixed-class commodities ignore density entirely. Density-based
    commodities walk the density_class_table and find the matching band.
    """
    if not entry["density_based"]:
        return {
            "class": entry["fixed_class"],
            "rule": "fixed_class",
            "matched_band": None,
        }

    for band in entry["density_class_table"]:
        min_d = band["min_density_pcf"]
        max_d = band["max_density_pcf"]
        if density_pcf >= min_d and (max_d is None or density_pcf < max_d):
            return {
                "class": band["class"],
                "rule": "density_band",
                "matched_band": band,
            }

    # Shouldn't happen if tables are well-formed (last band should have max=null),
    # but fail loudly rather than silently returning nothing.
    raise ValueError(
        f"No matching density band found for {entry['item_id']} at density {density_pcf} pcf — "
        "check that the table's bands are contiguous and the last band has max_density_pcf: null"
    )


if __name__ == "__main__":
    # Quick sanity check against SAMPLE-00101 (wooden furniture, unassembled, boxed)
    import json
    from pathlib import Path

    with open(Path(__file__).parent / "sample_nmfc_dataset.json") as f:
        data = json.load(f)

    entry = next(c for c in data["commodities"] if c["item_id"] == "SAMPLE-00101")

    # A 48x30x12in box weighing 65 lbs
    result = compute_density_pcf(length_in=48, width_in=30, height_in=12, weight_lbs=65)
    print(f"Cubic feet: {result.cubic_feet}, Density: {result.density_pcf} pcf")

    classification = resolve_class(entry, result.density_pcf)
    print(f"Resolved class: {classification['class']} (via {classification['rule']})")
