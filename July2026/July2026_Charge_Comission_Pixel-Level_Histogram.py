#!/usr/bin/env python3
"""
Per-chip grids of per-pixel raw ADC/dataword histograms for LArPix packet
data, aggregated over one or many HDF5 files.

Requirements
------------
    Python >= 3.10 (written for 3.13), numpy (1.x or 2.x), h5py >= 3.0,
    matplotlib >= 3.4

Inputs
------
    --input takes any mix of HDF5 files and directories. Directories are
    searched for --pattern (default "*.h5"), and --recursive also searches
    subdirectories. Every file of a dataset is summed together, so each
    histogram is the distribution of that pixel's datawords across ALL files.

    --input2 (optional) gives a second dataset; each pixel is then drawn as
    two overlapping histograms on common binning.

Output
------
    One PNG per chip with an 8x8 grid of per-channel histograms:
        <output_dir>/io_group<G>/tile<T>/chip<CCC>.png
    With --summary_csv, one CSV of per-channel statistics per chip:
        <output_dir>/[dataset<N>/]io_group<G>/tile<T>/chip<CCC>_summary.csv

Selection
---------
    (nothing)                              entire TPC
    --io_group G                           one io_group
    --io_group G --tile T                  one tile on that io_group
    --io_group G --tile T --chip C         one chip on that tile
    Each option accepts several values; every combination is selected.

Conventions
-----------
    * ADC packets:        packet_type == 0 and valid_parity == 1
    * Timestamp packets:  packet_type == 4, used for livetime
                          (livetime = sum over files of max - min timestamp)
    * Chip identity:      (io_group, tile, chip_id). A chip is located by
                          its io_group, then its tile, then its chip_id.
                          io_channel is never part of a chip's identity:
                          packets from one chip arriving on different
                          io_channels of its tile are combined.
    * Tile:               packets carry no tile field, so the tile comes
                          from one of:
                            --tile_map FILE   CSV mapping (io_group,
                                              io_channel) -> tile. This is
                                              the ONLY use of io_channel.
                            --tile_field NAME a per-packet tile column in the
                                              packets dataset (e.g. tile_id);
                                              io_channel is then not read.
                          Packets on an (io_group, io_channel) missing from
                          the tile map are dropped and reported.

Tile map format
---------------
    CSV with a header row containing io_group, io_channel and tile (any
    column order, extra columns ignored, lines starting with # ignored):

        io_group,io_channel,tile
        1,1,1
        1,2,1
        1,3,1
        1,4,1
        1,5,2
        ...

    Several io_channels may belong to the same tile, but each
    (io_group, io_channel) may belong to only ONE tile; conflicting entries
    stop the script with a list of the conflicts.

Memory
------
    Datawords are 8-bit, so instead of storing every value the script keeps a
    64 x 256 count histogram per chip (~64 kB per chip per dataset). Memory
    stays fixed however many files are read, and the summary statistics
    (including percentiles) are still exact.

Usage
-----
python July2026_Charge_Comission_Pixel-Level_Histogram.py \
    --input /path/to/run_before \
    --input2 /path/to/run_after \
    --label1 "Before" --label2 "After" \
    --tile_map tile_map.csv \
    --io_group 1 --tile 3 \
    --workers 8 \
    --output_dir comparison_histograms
"""

import sys

# Checked before anything else, and written with %-formatting, so an old
# interpreter prints a clear message instead of an unrelated ImportError.
if sys.version_info < (3, 10):
    sys.exit(
        "ERROR: this script needs Python 3.10 or newer (3.13 recommended).\n"
        "       Running: Python %s (%s)\n"
        "       Activate a newer environment or set PYTHON in the run script."
        % (sys.version.split()[0], sys.executable)
    )

import argparse
import csv
import math
import multiprocessing
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if not hasattr(plt.Axes, "stairs"):
    sys.exit(f"ERROR: matplotlib >= 3.4 is required (found {matplotlib.__version__}).")
if not hasattr(h5py.Dataset, "fields"):
    sys.exit(f"ERROR: h5py >= 3.0 is required (found {h5py.__version__}).")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 2_000_000
N_CHANNELS = 64
N_ADC = 256                                  # LArPix datawords are 8-bit
BINS_PER_CHIP = N_CHANNELS * N_ADC
ADC_VALUES = np.arange(N_ADC, dtype=np.float64)

# uint32 holds > 4e9 packets per ADC bin per pixel (far beyond any real
# dataset) at half the memory of int64.
COUNT_DTYPE = np.uint32

PACKET_FIELDS = [
    "packet_type", "valid_parity", "io_group",
    "chip_id", "channel_id", "dataword", "timestamp",
]

CSV_FIELDS = [
    "channel_id", "n_packets", "adc_mean", "adc_std", "adc_min",
    "adc_p01", "adc_p05", "adc_median", "adc_p95", "adc_p99", "adc_max",
    "rate_like_original",
]

# (io_group, tile, chip_id)
ChipKey = tuple[int, int, int]
ChipHists = dict[ChipKey, np.ndarray]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    io_groups: tuple[int, ...] | None = None
    tiles: tuple[int, ...] | None = None
    chips: tuple[int, ...] | None = None

    def mask(self, io_group: np.ndarray, tile: np.ndarray | None, chip: np.ndarray) -> np.ndarray:
        """Packets passing the selection. tile=None skips the tile requirement."""
        m = np.ones(io_group.shape, dtype=bool)
        if self.io_groups is not None:
            m &= np.isin(io_group, self.io_groups)
        if self.tiles is not None and tile is not None:
            m &= np.isin(tile, self.tiles)
        if self.chips is not None:
            m &= np.isin(chip, self.chips)
        return m

    def describe(self) -> str:
        if self.io_groups is None:
            return "entire TPC"
        parts = [f"io_group {list(self.io_groups)}"]
        if self.tiles is not None:
            parts.append(f"tile {list(self.tiles)}")
        if self.chips is not None:
            parts.append(f"chip {list(self.chips)}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_h5_files(paths: Sequence[str], pattern: str, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found = path.rglob(pattern) if recursive else path.glob(pattern)
            files.extend(sorted(f for f in found if f.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    # De-duplicate while preserving order.
    unique = list({f.resolve(): f for f in files}.values())
    if not unique:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {', '.join(paths)}")
    return unique


# ---------------------------------------------------------------------------
# Reading / accumulation
# ---------------------------------------------------------------------------

def require_fields(dtype: np.dtype, fields: Iterable[str]) -> None:
    names = dtype.names or ()
    missing = [f for f in fields if f not in names]
    if missing:
        raise KeyError("packets dataset is missing required field(s): " + ", ".join(missing))


def load_tile_map(path: Path) -> tuple[np.ndarray, int, int]:
    """
    Read a CSV tile map (io_group, io_channel, tile) into a lookup table
    lut[io_group, io_channel] -> tile, with -1 for io_channels not in the map.

    Returns (lut, n_io_channels, n_tiles).
    """
    lut = np.full((256, 256), -1, dtype=np.int64)
    owner: dict[tuple[int, int], tuple[int, int]] = {}      # (g, io_ch) -> (tile, entry no.)
    conflicts: list[str] = []

    with path.open(newline="") as f:
        lines = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError(f"tile map {path} is empty")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [c for c in ("io_group", "io_channel", "tile") if c not in columns]
        if missing:
            raise ValueError(f"tile map {path} is missing column(s): {', '.join(missing)}")

        for entry, row in enumerate(reader, start=1):
            try:
                g, ioc, t = (int(row[columns[k]]) for k in ("io_group", "io_channel", "tile"))
            except (TypeError, ValueError):
                raise ValueError(f"tile map {path}, entry {entry}: non-integer value in {dict(row)}") from None
            if not all(0 <= v < 256 for v in (g, ioc, t)):
                raise ValueError(f"tile map {path}, entry {entry}: values must be 0-255, got {(g, ioc, t)}")

            previous = owner.get((g, ioc))
            if previous is not None and previous[0] != t:
                conflicts.append(f"io_group {g}, io_channel {ioc}: tile {previous[0]} "
                                 f"(entry {previous[1]}) and tile {t} (entry {entry})")
                continue
            owner[(g, ioc)] = (t, entry)
            lut[g, ioc] = t

    if conflicts:
        shown = "\n  ".join(conflicts[:10])
        more = f"\n  ... and {len(conflicts) - 10} more" if len(conflicts) > 10 else ""
        raise ValueError(f"tile map {path}: {len(conflicts)} io_channel(s) are assigned to more "
                         f"than one tile:\n  {shown}{more}")
    if not owner:
        raise ValueError(f"tile map {path} has no entries")
    n_tiles = len({(g, t) for (g, _), (t, _) in owner.items()})
    return lut, len(owner), n_tiles


@dataclass(frozen=True)
class TileResolver:
    """Finds each packet's tile: from (io_group, io_channel) via a tile map, or a tile column."""
    lut: np.ndarray | None = None
    tile_field: str | None = None

    def fields(self) -> list[str]:
        """Extra packet columns needed; io_channel is read only in tile-map mode."""
        return [self.tile_field] if self.tile_field else ["io_channel"]

    def resolve(self, p: np.ndarray, mask: np.ndarray, io_group: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Tile per selected packet (int64; -1 = could not be located), plus the
        io_channels used for the lookup (tile-map mode only, for reporting).
        io_channel is used here and nowhere else.
        """
        if self.tile_field is not None:
            return p[self.tile_field][mask].astype(np.int64), None
        assert self.lut is not None
        io_channel = p["io_channel"][mask].astype(np.int64)
        ok = (io_group >= 0) & (io_group < 256) & (io_channel >= 0) & (io_channel < 256)
        tile = np.full(io_group.shape, -1, dtype=np.int64)
        tile[ok] = self.lut[io_group[ok], io_channel[ok]]
        return tile, io_channel

    def describe(self) -> str:
        return f"tile column '{self.tile_field}'" if self.tile_field else "tile map (io_group, io_channel) -> tile"


def accumulate_chunk(
    hists: ChipHists,
    io_group: np.ndarray,
    tile: np.ndarray,
    chip: np.ndarray,
    channel: np.ndarray,
    adc: np.ndarray,
) -> None:
    """Add one chunk of selected packets (all int64 arrays) to per-chip histograms."""
    chip_code = (io_group << 16) | (tile << 8) | chip
    flat = chip_code * BINS_PER_CHIP + channel * N_ADC + adc

    uniq, counts = np.unique(flat, return_counts=True)
    codes = uniq // BINS_PER_CHIP
    local = uniq % BINS_PER_CHIP

    # uniq is sorted, so each chip occupies one contiguous block.
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [uniq.size]))

    for s, e in zip(starts, ends):
        code = int(codes[s])
        key = (code >> 16, (code >> 8) & 0xFF, code & 0xFF)
        h = hists.get(key)
        if h is None:
            h = hists[key] = np.zeros((N_CHANNELS, N_ADC), dtype=COUNT_DTYPE)
        h.reshape(-1)[local[s:e]] += counts[s:e].astype(COUNT_DTYPE)


def accumulate_file(
    filename: Path,
    hists: ChipHists,
    selection: Selection,
    chunk_size: int,
    resolver: TileResolver,
    unmapped: dict[tuple[int, int], int],
    max_packets: int | None = None,
) -> tuple[int, float | None, int]:
    """
    Read one HDF5 packet file and add its selected packets to `hists`.

    Packets that cannot be located on a tile are counted in `unmapped`
    (keyed by (io_group, io_channel) in tile-map mode, (io_group, tile value)
    in tile-field mode) and dropped.

    Returns (n_selected_packets, livetime_or_None, n_out_of_range_packets).
    """
    n_selected = 0
    n_out_of_range = 0
    fields = PACKET_FIELDS + resolver.fields()
    ts_min: float | None = None
    ts_max: float | None = None

    with h5py.File(filename, "r") as h5:
        if "packets" not in h5:
            raise KeyError("no 'packets' dataset")
        packets = h5["packets"]
        require_fields(packets.dtype, fields)

        # Read only the needed columns; much less I/O than whole rows.
        reader = packets.fields(fields)
        n_packets = packets.shape[0]

        for start in range(0, n_packets, chunk_size):
            if max_packets is not None and n_selected >= max_packets:
                break

            p = reader[start:min(start + chunk_size, n_packets)]
            ptype = p["packet_type"]

            # Timestamp packets -> livetime
            ts_mask = ptype == 4
            if np.any(ts_mask):
                ts = p["timestamp"][ts_mask]
                lo, hi = float(ts.min()), float(ts.max())
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)

            # ADC packets
            mask = (ptype == 0) & (p["valid_parity"] == 1)
            if not np.any(mask):
                continue

            io_group = p["io_group"][mask].astype(np.int64)
            chip = p["chip_id"][mask].astype(np.int64)
            channel = p["channel_id"][mask].astype(np.int64)
            adc = p["dataword"][mask].astype(np.int64)

            # Locate each chip: io_group -> tile -> chip_id.
            tile, io_channel = resolver.resolve(p, mask, io_group)
            lost = (tile < 0) | (tile > 255)
            if np.any(lost):
                # Report only packets that could have been selected (tile unknown).
                report = lost & selection.mask(io_group, None, chip)
                second = io_channel if io_channel is not None else tile
                pairs, n = np.unique(np.stack((io_group[report], second[report])), axis=1, return_counts=True)
                for (g, x), k in zip(pairs.T.tolist(), n.tolist()):
                    unmapped[(g, x)] = unmapped.get((g, x), 0) + k
                keep_located = ~lost
                io_group, chip, channel, adc, tile = (
                    a[keep_located] for a in (io_group, chip, channel, adc, tile)
                )

            keep = selection.mask(io_group, tile, chip)
            in_range = (
                (io_group >= 0) & (io_group < 256)
                & (tile >= 0) & (tile < 256)
                & (chip >= 0) & (chip < 256)
                & (channel >= 0) & (channel < N_CHANNELS)
                & (adc >= 0) & (adc < N_ADC)
            )
            n_out_of_range += int(np.count_nonzero(keep & ~in_range))
            keep &= in_range
            if not np.any(keep):
                continue

            arrays = [a[keep] for a in (io_group, tile, chip, channel, adc)]
            if max_packets is not None:
                remaining = max_packets - n_selected
                arrays = [a[:remaining] for a in arrays]

            n_selected += arrays[-1].size
            accumulate_chunk(hists, *arrays)

    livetime = None
    if ts_min is not None and ts_max is not None and ts_max > ts_min:
        livetime = ts_max - ts_min
    return n_selected, livetime, n_out_of_range


def read_dataset(
    files: Sequence[Path],
    selection: Selection,
    chunk_size: int,
    resolver: TileResolver,
    max_selected_packets: int | None,
    name: str,
) -> tuple[ChipHists, float | None]:
    """
    Accumulate per-chip histograms over every file of one dataset.

    Livetime is the sum of per-file livetimes (None if no file had timestamp
    packets). max_selected_packets caps the dataset as a whole.
    """
    hists: ChipHists = {}
    total_selected = 0
    total_livetime: float | None = None
    total_out_of_range = 0
    n_skipped = 0
    unmapped: dict[tuple[int, int], int] = {}

    for i, filename in enumerate(files, start=1):
        remaining = None
        if max_selected_packets is not None:
            remaining = max_selected_packets - total_selected
            if remaining <= 0:
                print(f"  [{name}] packet cap reached; skipping remaining {len(files) - i + 1} file(s).")
                break

        t0 = time.perf_counter()
        try:
            n_sel, livetime, n_oor = accumulate_file(
                filename, hists, selection, chunk_size, resolver, unmapped, remaining
            )
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            # Corrupt or truncated files: report and continue with the rest.
            n_skipped += 1
            print(f"  [{name}] [{i}/{len(files)}] WARNING: skipping {filename}: {exc}")
            continue

        total_selected += n_sel
        total_out_of_range += n_oor
        if livetime is not None:
            total_livetime = livetime + (total_livetime or 0.0)

        print(f"  [{name}] [{i}/{len(files)}] {filename.name}: "
              f"{n_sel} selected packets ({time.perf_counter() - t0:.1f} s)")

    if n_skipped:
        print(f"  [{name}] WARNING: {n_skipped} file(s) could not be read.")
    if unmapped:
        what = "io_channel" if resolver.tile_field is None else resolver.tile_field
        worst = sorted(unmapped.items(), key=lambda kv: -kv[1])[:10]
        listing = ", ".join(f"(io_group {g}, {what} {x}): {n}" for (g, x), n in worst)
        print(f"  [{name}] WARNING: dropped {sum(unmapped.values())} packets that could not be "
              f"located on a tile ({resolver.describe()}). Largest: {listing}")
    if total_out_of_range:
        print(f"  [{name}] WARNING: dropped {total_out_of_range} packets with "
              f"out-of-range ids/channel_id/dataword.")
    return hists, total_livetime


# ---------------------------------------------------------------------------
# Statistics from count histograms
# ---------------------------------------------------------------------------

def counts_stats(counts: np.ndarray) -> dict[str, float] | None:
    n = int(counts.sum())
    if n == 0:
        return None
    c = counts.astype(np.float64)
    mean = float((c * ADC_VALUES).sum() / n)
    std = math.sqrt(float((c * (ADC_VALUES - mean) ** 2).sum()) / n)
    nz = np.flatnonzero(counts)
    return {"n": n, "mean": mean, "std": std, "min": float(nz[0]), "max": float(nz[-1])}


def percentile_from_counts(counts: np.ndarray, q: float) -> float:
    """Same result as np.percentile (linear interpolation) on the raw values."""
    cum = np.cumsum(counts, dtype=np.int64)
    n = int(cum[-1])
    pos = q / 100.0 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    v_lo = int(np.searchsorted(cum, lo, side="right"))
    v_hi = int(np.searchsorted(cum, hi, side="right"))
    return float(v_lo + (v_hi - v_lo) * (pos - lo))


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------

def channel_summary_rows(chip_counts: np.ndarray, livetime: float | None) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for ch in range(N_CHANNELS):
        counts = chip_counts[ch]
        st = counts_stats(counts)
        if st is None:
            row: dict[str, float | int | str] = dict.fromkeys(CSV_FIELDS, "")
            row["channel_id"] = ch
            row["n_packets"] = 0
        else:
            row = {
                "channel_id": ch,
                "n_packets": st["n"],
                "adc_mean": st["mean"],
                "adc_std": st["std"],
                "adc_min": st["min"],
                "adc_p01": percentile_from_counts(counts, 1),
                "adc_p05": percentile_from_counts(counts, 5),
                "adc_median": percentile_from_counts(counts, 50),
                "adc_p95": percentile_from_counts(counts, 95),
                "adc_p99": percentile_from_counts(counts, 99),
                "adc_max": st["max"],
                "rate_like_original": "" if livetime is None else st["n"] / (livetime + 1e-9),
            }
        rows.append(row)
    return rows


def generate_chip_summary_csvs(hists: ChipHists, livetime: float | None, output_dir: Path) -> None:
    for (io_group, tile, chip_id), chip_counts in sorted(hists.items()):
        csv_path = output_dir / f"io_group{io_group}" / f"tile{tile}" / f"chip{chip_id:03d}_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(channel_summary_rows(chip_counts, livetime))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_bin_edges(
    chip_counts_list: Sequence[np.ndarray | None],
    adc_min: float | None,
    adc_max: float | None,
    bin_width: float,
) -> np.ndarray:
    """Common bin edges for all pixels of one chip (and both datasets)."""
    nonzero = [np.flatnonzero(c.sum(axis=0)) for c in chip_counts_list if c is not None]
    nonzero = [nz for nz in nonzero if nz.size]
    if not nonzero and (adc_min is None or adc_max is None):
        raise ValueError("No selected packets for this chip; cannot make histogram bins.")

    lo = float(min(nz[0] for nz in nonzero)) if adc_min is None else float(adc_min)
    hi = float(max(nz[-1] for nz in nonzero)) if adc_max is None else float(adc_max)
    if hi < lo:
        raise ValueError("adc_max must be >= adc_min")
    if hi == lo:
        lo -= bin_width
        hi += bin_width

    # Integer ADC values sit at bin centres when bin_width = 1.
    return np.arange(lo - 0.5 * bin_width, hi + 1.5 * bin_width, bin_width)


@dataclass
class PlotJob:
    key: ChipKey
    chip_counts_list: list[np.ndarray | None]
    labels: list[str]
    edges: np.ndarray
    output_png: Path
    title: str
    log_y: bool
    dpi: int


def plot_chip_grid(job: PlotJob) -> ChipKey:
    """One figure per chip: 8x8 grid of per-pixel histograms (1 or 2 datasets)."""
    job.output_png.parent.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dual = len(job.chip_counts_list) > 1
    edges = job.edges

    fig, axes = plt.subplots(8, 8, figsize=(22, 18), sharex=True, sharey=False)

    for ch in range(N_CHANNELS):
        ax = axes[ch // 8, ch % 8]
        lines: list[tuple[str, str]] = []
        plotted = False

        for i, chip_counts in enumerate(job.chip_counts_list):
            color = colors[i % len(colors)]
            ch_counts = None if chip_counts is None else chip_counts[ch]
            st = None if ch_counts is None else counts_stats(ch_counts)
            if st is None:
                lines.append(("N=0", color))
                continue

            h, _ = np.histogram(ADC_VALUES, bins=edges, weights=ch_counts)
            if dual:
                ax.stairs(h, edges, fill=True, alpha=0.35, color=color)
                ax.stairs(h, edges, color=color, linewidth=0.8)
            else:
                ax.stairs(h, edges, fill=True, color=color)
            plotted = True
            lines.append((f"N={st['n']}, μ={st['mean']:.1f}, σ={st['std']:.1f}", color))

        ax.set_title(f"ch {ch}", fontsize=8)
        if plotted:
            if job.log_y:
                ax.set_yscale("log")
            for j, (text, color) in enumerate(lines):
                ax.text(0.98, 0.97 - 0.11 * j, text, transform=ax.transAxes,
                        ha="right", va="top", fontsize=6,
                        color=color if dual else "black")
        else:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center", fontsize=8)

        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    if dual:
        handles = [Patch(facecolor=colors[i % len(colors)], alpha=0.5, label=label)
                   for i, label in enumerate(job.labels)]
        fig.legend(handles=handles, loc="upper right", fontsize=12)

    fig.suptitle(job.title + "\nSubplot positions are channel_id order, not physical XY geometry.",
                 fontsize=16)
    fig.supxlabel("ADC dataword")
    fig.supylabel("Packet count")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(job.output_png, dpi=job.dpi)
    plt.close(fig)
    return job.key


def build_plot_jobs(
    hists_list: Sequence[ChipHists],
    labels: list[str],
    output_dir: Path,
    log_y: bool,
    adc_min: float | None,
    adc_max: float | None,
    bin_width: float,
    n_files: Sequence[int],
    dpi: int,
) -> list[PlotJob]:
    all_keys = sorted(set().union(*(h.keys() for h in hists_list)))
    file_note = " / ".join(f"{label}: {n} file(s)" for label, n in zip(labels, n_files))

    jobs = []
    for key in all_keys:
        io_group, tile, chip_id = key
        chip_counts_list = [h.get(key) for h in hists_list]
        jobs.append(PlotJob(
            key=key,
            chip_counts_list=chip_counts_list,
            labels=labels,
            edges=make_bin_edges(chip_counts_list, adc_min, adc_max, bin_width),
            output_png=output_dir / f"io_group{io_group}" / f"tile{tile}" / f"chip{chip_id:03d}.png",
            title=f"io_group={io_group}, tile={tile}, chip={chip_id}   ({file_note})",
            log_y=log_y,
            dpi=dpi,
        ))
    return jobs


def run_plot_jobs(jobs: list[PlotJob], workers: int) -> None:
    total = len(jobs)
    report_every = max(1, min(25, total // 10 or 1))

    if workers <= 1 or total <= 1:
        for i, job in enumerate(jobs, start=1):
            plot_chip_grid(job)
            if i % report_every == 0 or i == total:
                print(f"  plotted {i}/{total} chips")
        return

    # "spawn" is safe on every platform and avoids fork-with-threads issues.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, _ in enumerate(pool.map(plot_chip_grid, jobs, chunksize=4), start=1):
            if i % report_every == 0 or i == total:
                print(f"  plotted {i}/{total} chips")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, Selection]:
    parser = argparse.ArgumentParser(
        description="Per-chip grids of per-pixel raw dataword histograms, aggregated over many files."
    )

    g = parser.add_argument_group("inputs")
    g.add_argument("--input", "--filename", dest="input", nargs="+", required=True,
                   help="HDF5 file(s) and/or directories for dataset 1")
    g.add_argument("--input2", "--filename2", dest="input2", nargs="+", default=None,
                   help="Optional HDF5 file(s) and/or directories for dataset 2 (comparison mode)")
    g.add_argument("--pattern", default="*.h5", help="Glob pattern inside directories (default: *.h5)")
    g.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    g.add_argument("--label1", default="Dataset 1", help="Legend label for dataset 1")
    g.add_argument("--label2", default="Dataset 2", help="Legend label for dataset 2")

    g = parser.add_argument_group("selection (omit all for the entire TPC)")
    g.add_argument("--io_group", type=int, nargs="+", default=None, help="io_group(s)")
    g.add_argument("--tile", type=int, nargs="+", default=None, help="tile(s) on the io_group(s); needs --io_group")
    g.add_argument("--chip", type=int, nargs="+", default=None, help="chip_id(s) on the tile(s); needs --tile")

    g = parser.add_argument_group("tile location (one is required)")
    tile_src = g.add_mutually_exclusive_group(required=True)
    tile_src.add_argument("--tile_map", default=None,
                          help="CSV with columns io_group, io_channel, tile (the only use of io_channel)")
    tile_src.add_argument("--tile_field", default=None,
                          help="Name of a per-packet tile column in the packets dataset (e.g. tile_id)")

    g = parser.add_argument_group("reading / plotting")
    g.add_argument("--output_dir", default="chip_histograms")
    g.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    g.add_argument("--max_selected_packets", type=int, default=-1,
                   help="Cap on selected packets per dataset (whole dataset, not per file)")
    g.add_argument("--adc_min", type=float, default=None, help="Lower ADC value for histogram range")
    g.add_argument("--adc_max", type=float, default=None, help="Upper ADC value for histogram range")
    g.add_argument("--bin_width", type=float, default=1.0, help="Histogram bin width in ADC units")
    g.add_argument("--log_y", action="store_true", help="Log-scale y axes")
    g.add_argument("--dpi", type=int, default=120, help="Output image resolution")
    g.add_argument("--workers", type=int, default=1, help="Parallel processes for plotting (default: 1)")
    g.add_argument("--summary_csv", action="store_true",
                   help="Write one summary CSV per chip (per dataset in comparison mode)")

    args = parser.parse_args()

    if args.tile is not None and args.io_group is None:
        parser.error("--tile requires --io_group")
    if args.chip is not None and args.tile is None:
        parser.error("--chip requires --tile (and --io_group)")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if args.bin_width <= 0:
        parser.error("--bin_width must be positive")
    if args.adc_min is not None and args.adc_max is not None and args.adc_max < args.adc_min:
        parser.error("--adc_max must be >= --adc_min")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selection = Selection(
        io_groups=None if args.io_group is None else tuple(args.io_group),
        tiles=None if args.tile is None else tuple(args.tile),
        chips=None if args.chip is None else tuple(args.chip),
    )
    return args, selection


def main() -> int:
    args, selection = parse_args()
    max_selected = None if args.max_selected_packets < 0 else args.max_selected_packets
    output_dir = Path(args.output_dir).expanduser()
    t_start = time.perf_counter()

    print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, "
          f"h5py {h5py.__version__}, matplotlib {matplotlib.__version__}")
    if args.tile_map is not None:
        try:
            lut, n_ioc, n_tiles = load_tile_map(Path(args.tile_map).expanduser())
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        resolver = TileResolver(lut=lut)
        print(f"Tile map: {n_ioc} io_channels on {n_tiles} tiles from {args.tile_map}")
    else:
        resolver = TileResolver(tile_field=args.tile_field)

    print(f"Selection: {selection.describe()}")
    print(f"Tile location: {resolver.describe()}")

    datasets = [(args.input, args.label1)]
    if args.input2 is not None:
        datasets.append((args.input2, args.label2))

    hists_list: list[ChipHists] = []
    livetimes: list[float | None] = []
    n_files: list[int] = []
    labels: list[str] = []

    for paths, label in datasets:
        try:
            files = find_h5_files(paths, args.pattern, args.recursive)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Reading {len(files)} file(s) for {label}...")
        hists, livetime = read_dataset(files, selection, args.chunk_size,
                                       resolver, max_selected, label)
        print(f"Found {len(hists)} active chips in {label}.")
        hists_list.append(hists)
        livetimes.append(livetime)
        n_files.append(len(files))
        labels.append(label)

    if not any(hists_list):
        print("ERROR: no selected packets found in any dataset for this selection.", file=sys.stderr)
        return 1

    if args.summary_csv:
        print("Writing summary CSV files...")
        if len(hists_list) == 1:
            generate_chip_summary_csvs(hists_list[0], livetimes[0], output_dir)
        else:
            for i, (hists, livetime) in enumerate(zip(hists_list, livetimes), start=1):
                generate_chip_summary_csvs(hists, livetime, output_dir / f"dataset{i}")

    jobs = build_plot_jobs(hists_list, labels, output_dir, args.log_y,
                           args.adc_min, args.adc_max, args.bin_width, n_files, args.dpi)
    print(f"Plotting {len(jobs)} chips with {args.workers} worker(s)...")
    run_plot_jobs(jobs, args.workers)

    print()
    print("Done.")
    print(f"Selection: {selection.describe()}")
    for label, hists, livetime, nf in zip(labels, hists_list, livetimes, n_files):
        total_packets = sum(int(h.sum()) for h in hists.values())
        active_pixels = sum(int(np.count_nonzero(h.sum(axis=1))) for h in hists.values())
        print(f"{label}: {nf} file(s), {len(hists)} active chips, "
              f"{active_pixels} active pixels, {total_packets} selected packets")
        if livetime is not None:
            print(f"{label} livetime (summed over files) = {livetime}")
    print(f"Chip images written: {len(jobs)}")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())