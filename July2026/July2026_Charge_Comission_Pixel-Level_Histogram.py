#!/usr/bin/env python3
"""
Plot raw ADC/dataword histograms for all LArPix pixels in a data file.

Outputs:
    * Histrograms depciting the distribution of the raw ADC of each LArPix pixel

Notes:
  * Selection follows the same packet convention as the original plotting code:
    packet_type == 0 and valid_parity == 1 are used for ADC/dataword packets.
  * YAML geometry is not needed unless you later want the per-channel plots
    placed at physical pixel positions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


DEFAULT_CHUNK_SIZE = 2_000_000
N_CHANNELS = 64


def require_fields(dtype: np.dtype, fields: list[str]) -> None:
    missing = [field for field in fields if dtype.names is None or field not in dtype.names]
    if missing:
        raise KeyError("packets dataset is missing required field(s): " + ", ".join(missing))


def update_minmax(old_min: Optional[float], old_max: Optional[float], values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if values.size == 0:
        return old_min, old_max
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if old_min is None or vmin < old_min:
        old_min = vmin
    if old_max is None or vmax > old_max:
        old_max = vmax
    return old_min, old_max


def read_all_channel_datawords(
    filename,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_selected_packets=None,
):
    """
    Read one HDF5 packet file exactly once and return ADC values grouped by

        (io_group, io_channel, chip_id, channel_id)

    Returns
    -------
    data : dict
        key -> numpy array of ADC values

    livetime : float or None
    """

    filename = Path(filename)

    pieces = defaultdict(list)

    ts_min = None
    ts_max = None

    selected_total = 0

    with h5py.File(filename, "r") as h5:

        packets = h5["packets"]

        require_fields(
            packets.dtype,
            [
                "packet_type",
                "valid_parity",
                "io_group",
                "io_channel",
                "chip_id",
                "channel_id",
                "dataword",
                "timestamp",
            ],
        )

        n_packets = packets.shape[0]

        for start in range(0, n_packets, chunk_size):

            stop = min(start + chunk_size, n_packets)

            p = packets[start:stop]

            ##############################################
            # Timestamp packets
            ##############################################

            ts_mask = p["packet_type"] == 4

            if np.any(ts_mask):

                ts = p["timestamp"][ts_mask].astype(np.float64)

                ts_min, ts_max = update_minmax(
                    ts_min,
                    ts_max,
                    ts,
                )

            ##############################################
            # ADC packets
            ##############################################

            mask = (
                (p["packet_type"] == 0)
                &
                (p["valid_parity"] == 1)
            )

            if not np.any(mask):
                continue

            io_groups = p["io_group"][mask]
            io_channels = p["io_channel"][mask]
            chip_ids = p["chip_id"][mask]
            channel_ids = p["channel_id"][mask]
            adcs = p["dataword"][mask].astype(np.float64)

            if max_selected_packets is not None:

                remaining = max_selected_packets - selected_total

                if remaining <= 0:
                    break

                io_groups = io_groups[:remaining]
                io_channels = io_channels[:remaining]
                chip_ids = chip_ids[:remaining]
                channel_ids = channel_ids[:remaining]
                adcs = adcs[:remaining]

            ##############################################
            # Group by detector element
            ##############################################

            ##########################################################
            # Sort once
            ##########################################################

            order = np.lexsort(
                (
                    channel_ids,
                    chip_ids,
                    io_channels,
                    io_groups,
                )
            )

            io_groups   = io_groups[order]
            io_channels = io_channels[order]
            chip_ids    = chip_ids[order]
            channel_ids = channel_ids[order]
            adcs        = adcs[order]

            ##########################################################
            # Find detector boundaries
            ##########################################################

            keys = np.column_stack(
                (
                    io_groups,
                    io_channels,
                    chip_ids,
                    channel_ids,
                )
            )

            change = np.any(
                keys[1:] != keys[:-1],
                axis=1,
            )

            starts = np.concatenate(
                (
                    [0],
                    np.where(change)[0] + 1,
                )
            )

            ends = np.concatenate(
                (
                    starts[1:],
                    [len(adcs)],
                )
            )

            ##########################################################
            # Store arrays
            ##########################################################

            for start, end in zip(starts, ends):

                key = (
                    int(io_groups[start]),
                    int(io_channels[start]),
                    int(chip_ids[start]),
                    int(channel_ids[start]),
                )

                pieces[key].append(adcs[start:end])

            ##############################################

    data = {
        key: np.concatenate(value)
        for key, value in pieces.items()
    }

    livetime = None

    if (
        ts_min is not None
        and
        ts_max is not None
        and
        ts_max > ts_min
    ):
        livetime = ts_max - ts_min

    return data, livetime

def generate_all_channel_histograms(
    data,
    output_dir,
    log_y=False,
    adc_min=None,
    adc_max=None,
    bin_width=1,
):
    """
    Generate one histogram for every channel.
    """

    output_dir = Path(output_dir)

    for (
        io_group,
        io_channel,
        chip_id,
        channel_id,
    ), values in data.items():

        if values.size == 0:
            continue

        bins = make_integerish_bins(
            values,
            adc_min,
            adc_max,
            bin_width,
        )

        folder = (
            output_dir
            / f"io_group{io_group}"
            / f"io_channel{io_channel}"
            / f"chip{chip_id}"
        )

        filename = folder / f"channel{channel_id:02d}.png"

        title = (
            f"io_group={io_group}, "
            f"io_channel={io_channel}, "
            f"chip={chip_id}, "
            f"channel={channel_id}"
        )

        plot_single_channel_hist(
            values=values,
            bins=bins,
            output_png=filename,
            title=title,
            log_y=log_y,
        )

def generate_chip_summary_csvs(
    data,
    livetime,
    output_dir,
):
    """
    Produce one summary CSV for every chip.
    """

    chips = defaultdict(dict)

    for (
        io_group,
        io_channel,
        chip_id,
        channel_id,
    ), values in data.items():

        chips[
            (
                io_group,
                io_channel,
                chip_id,
            )
        ][channel_id] = values

    output_dir = Path(output_dir)

    for (
        io_group,
        io_channel,
        chip_id,
    ), channels in chips.items():

        data_by_channel = {
            ch: channels.get(
                ch,
                np.array([], dtype=np.float64),
            )
            for ch in range(64)
        }

        rows = channel_summary_rows(
            data_by_channel,
            livetime,
        )

        folder = (
            output_dir
            / f"io_group{io_group}"
            / f"io_channel{io_channel}"
            / f"chip{chip_id}"
        )

        csv_name = folder / "summary.csv"

        save_summary_csv(
            rows,
            csv_name,
        )

def all_selected_values(data_by_channel: dict[int, np.ndarray]) -> np.ndarray:
    parts = [v for v in data_by_channel.values() if v.size]
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts)


def make_integerish_bins(
    values: np.ndarray,
    adc_min: Optional[float],
    adc_max: Optional[float],
    bin_width: float,
) -> np.ndarray:
    if values.size == 0:
        raise ValueError("No selected packets found; cannot make histogram bins.")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    lo = math.floor(float(np.min(values))) if adc_min is None else float(adc_min)
    hi = math.ceil(float(np.max(values))) if adc_max is None else float(adc_max)
    if hi < lo:
        raise ValueError("adc_max must be >= adc_min")
    if hi == lo:
        lo -= bin_width
        hi += bin_width

    # Center integer ADC values in bins when bin_width=1.
    return np.arange(lo - 0.5 * bin_width, hi + 1.5 * bin_width, bin_width)


def channel_summary_rows(data_by_channel: dict[int, np.ndarray], livetime: Optional[float]) -> list[dict[str, float | int | str]]:
    rows = []
    for ch in range(N_CHANNELS):
        values = data_by_channel[ch]
        n = int(values.size)
        if n:
            row = {
                "channel_id": ch,
                "n_packets": n,
                "adc_mean": float(np.mean(values)),
                "adc_std": float(np.std(values)),
                "adc_min": float(np.min(values)),
                "adc_p01": float(np.percentile(values, 1)),
                "adc_p05": float(np.percentile(values, 5)),
                "adc_median": float(np.percentile(values, 50)),
                "adc_p95": float(np.percentile(values, 95)),
                "adc_p99": float(np.percentile(values, 99)),
                "adc_max": float(np.max(values)),
                "rate_like_original": "" if livetime is None else float(n / (livetime + 1e-9)),
            }
        else:
            row = {
                "channel_id": ch,
                "n_packets": 0,
                "adc_mean": "",
                "adc_std": "",
                "adc_min": "",
                "adc_p01": "",
                "adc_p05": "",
                "adc_median": "",
                "adc_p95": "",
                "adc_p99": "",
                "adc_max": "",
                "rate_like_original": "",
            }
        rows.append(row)
    return rows


def save_summary_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel_id",
        "n_packets",
        "adc_mean",
        "adc_std",
        "adc_min",
        "adc_p01",
        "adc_p05",
        "adc_median",
        "adc_p95",
        "adc_p99",
        "adc_max",
        "rate_like_original",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_single_channel_hist(
    values: np.ndarray,
    bins: np.ndarray,
    output_png: Path,
    title: str,
    log_y: bool,
) -> None:
    if values.size == 0:
        raise ValueError("No packets found for the selected channel.")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins)
    ax.set_xlabel("ADC dataword")
    ax.set_ylabel("Packet count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")

    text = (
        f"N = {values.size}\n"
        f"mean = {np.mean(values):.3f}\n"
        f"std = {np.std(values):.3f}\n"
        f"median = {np.median(values):.3f}"
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round", alpha=0.15))
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_combined_chip_hist(
    data_by_channel: dict[int, np.ndarray],
    bins: np.ndarray,
    output_png: Path,
    title: str,
    log_y: bool,
    overlay_channels: bool,
) -> None:
    values = all_selected_values(data_by_channel)
    if values.size == 0:
        raise ValueError("No selected chip packets found.")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.35, label="All selected chip packets")

    if overlay_channels:
        for ch in range(N_CHANNELS):
            ch_values = data_by_channel[ch]
            if ch_values.size:
                ax.hist(ch_values, bins=bins, histtype="step", linewidth=0.7, alpha=0.35)

    ax.set_xlabel("ADC dataword")
    ax.set_ylabel("Packet count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best")

    active_channels = sum(1 for v in data_by_channel.values() if v.size)
    text = (
        f"N = {values.size}\n"
        f"active channels = {active_channels}\n"
        f"mean = {np.mean(values):.3f}\n"
        f"std = {np.std(values):.3f}"
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round", alpha=0.15))
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_channel_grid(
    data_by_channel: dict[int, np.ndarray],
    bins: np.ndarray,
    output_png: Path,
    title: str,
    log_y: bool,
    max_title_channels: int = N_CHANNELS,
) -> None:
    values = all_selected_values(data_by_channel)
    if values.size == 0:
        raise ValueError("No selected chip packets found.")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(8, 8, figsize=(22, 18), sharex=True, sharey=False)

    for ch in range(N_CHANNELS):
        ax = axes[ch // 8, ch % 8]
        ch_values = data_by_channel[ch]
        if ch_values.size:
            ax.hist(ch_values, bins=bins)
            if log_y:
                ax.set_yscale("log")
            mean = np.mean(ch_values)
            std = np.std(ch_values)
            ax.set_title(f"ch {ch}\nN={ch_values.size}, μ={mean:.1f}, σ={std:.1f}", fontsize=8)
        else:
            ax.set_title(f"ch {ch}\nempty", fontsize=8)
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title + "\nSubplot positions are channel_id order, not physical XY geometry.", fontsize=16)
    fig.supxlabel("ADC dataword")
    fig.supylabel("Packet count")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_channel_summary(rows: list[dict[str, float | int | str]], output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    ch = np.array([int(r["channel_id"]) for r in rows])
    n = np.array([int(r["n_packets"]) for r in rows])

    def numeric(field: str) -> np.ndarray:
        out = []
        for r in rows:
            value = r[field]
            out.append(np.nan if value == "" else float(value))
        return np.array(out, dtype=float)

    mean = numeric("adc_mean")
    std = numeric("adc_std")
    median = numeric("adc_median")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(ch, n, marker=".", linestyle="none")
    axes[0].set_ylabel("Packets")
    axes[0].set_yscale("log" if np.nanmax(n) > 0 else "linear")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ch, mean, marker=".", linestyle="none", label="mean")
    axes[1].plot(ch, median, marker="x", linestyle="none", label="median")
    axes[1].set_ylabel("ADC")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ch, std, marker=".", linestyle="none")
    axes[2].set_xlabel("channel_id")
    axes[2].set_ylabel("ADC std")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw dataword histograms for the entire 2x2."
    )
    parser.add_argument("--filename", required=True, help="Input HDF5 file containing a packets dataset")
    parser.add_argument("--output_dir", default="channel_histograms")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max_selected_packets", type=int, default=-1, help="Optional cap after packet selection")
    parser.add_argument("--adc_min", type=float, default=None, help="Lower ADC value for histogram range")
    parser.add_argument("--adc_max", type=float, default=None, help="Upper ADC value for histogram range")
    parser.add_argument("--bin_width", type=float, default=1.0, help="Histogram bin width in ADC units")
    parser.add_argument("--log_y", action="store_true", help="Use log scale for histogram y axes")
    parser.add_argument("--summary_csv", action="store_true", help="Write one summary CSV per chip.")
    args = parser.parse_args()

    #######################################################
    # Read detector once
    #######################################################

    print("Reading HDF5 file...")

    max_selected = (
        None
        if args.max_selected_packets < 0
        else args.max_selected_packets
    )

    data, livetime = read_all_channel_datawords(
        filename=args.filename,
        chunk_size=args.chunk_size,
        max_selected_packets=max_selected,
    )

    print(f"Found {len(data)} active channels.")

    #######################################################
    # Generate CSV summaries
    #######################################################

    if args.summary_csv:

        print("Writing summary CSV files...")

        generate_chip_summary_csvs(
            data=data,
            livetime=livetime,
            output_dir=args.output_dir,
        )

    #######################################################
    # Generate histograms
    #######################################################

    print("Generating histograms...")

    generate_all_channel_histograms(
        data=data,
        output_dir=args.output_dir,
        log_y=args.log_y,
        adc_min=args.adc_min,
        adc_max=args.adc_max,
        bin_width=args.bin_width,
    )

    #######################################################
    # Final statistics
    #######################################################

    total_packets = sum(
        values.size
        for values in data.values()
    )

    print()

    print("Done.")
    print(f"Active channels : {len(data)}")
    print(f"Selected packets: {total_packets}")

    if livetime is not None:
        print(f"Livetime = {livetime}")

    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()