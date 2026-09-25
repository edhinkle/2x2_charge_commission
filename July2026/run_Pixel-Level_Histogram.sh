#!/usr/bin/env bash
#
# run_chip_histograms.sh
#
# Edit the settings below, then run:
#   ./run_chip_histograms.sh                  # uses INPUT1 / INPUT2 below
#   ./run_chip_histograms.sh DIR1             # overrides INPUT1
#   ./run_chip_histograms.sh DIR1 DIR2        # overrides INPUT1 and INPUT2
#
set -uo pipefail

# ============================== SETTINGS ===================================
PYTHON=python3                 # needs Python >= 3.10, e.g. a 3.13 env
PY_SCRIPT="July2026_Charge_Comission_Pixel-Level_Histogram.py"

INPUT1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/ColdOperations/data/2025_Operations_Cold/Pedestal_1002_Nominal_Configuration/"  # file or directory
INPUT2="/global/cfs/cdirs/dune/www/data/2x2/nearline_run3/packet/ColdCommissioning/Pedestal_prc4/"                                   # leave empty for single-dataset mode
LABEL1="Run 2 Nominal"
LABEL2="Run 3 Nominal (PRC 4)"
RECURSIVE=false                # search subdirectories

# Tile location (set ONE). io_channel is used only to find each chip's tile.
TILE_MAP="trial_tile_map.csv"   # CSV: io_group,io_channel,tile
TILE_FIELD=""                  # or: name of a tile column in the packets dataset

# Selection: leave empty for the entire TPC. Space-separate multiple values.
IO_GROUP=""                    # e.g. "1"
TILE=""                        # e.g. "2"   (needs IO_GROUP)
CHIP=""                        # e.g. "12"  (needs TILE)

OUTPUT_DIR="chip_histograms"
BIN_WIDTH=1
ADC_MIN=""
ADC_MAX=""
LOG_Y=false
SUMMARY_CSV=false
DPI=120
WORKERS=1                      # parallel plotting processes
# ===========================================================================

INPUT1="${1:-$INPUT1}"
INPUT2="${2:-$INPUT2}"

CMD=("$PYTHON" "$PY_SCRIPT" --input "$INPUT1" --label1 "$LABEL1"
     --output_dir "$OUTPUT_DIR" --bin_width "$BIN_WIDTH" --dpi "$DPI"
     --workers "$WORKERS")

[[ -n "$INPUT2"     ]] && CMD+=(--input2 "$INPUT2" --label2 "$LABEL2")
if [[ -n "$TILE_FIELD" ]]; then CMD+=(--tile_field "$TILE_FIELD"); else CMD+=(--tile_map "$TILE_MAP"); fi
[[ -n "$IO_GROUP" ]] && CMD+=(--io_group $IO_GROUP)
[[ -n "$TILE"     ]] && CMD+=(--tile $TILE)
[[ -n "$CHIP"     ]] && CMD+=(--chip $CHIP)
[[ -n "$ADC_MIN"  ]] && CMD+=(--adc_min "$ADC_MIN")
[[ -n "$ADC_MAX"  ]] && CMD+=(--adc_max "$ADC_MAX")
$RECURSIVE   && CMD+=(--recursive)
$LOG_Y       && CMD+=(--log_y)
$SUMMARY_CSV && CMD+=(--summary_csv)

mkdir -p "$OUTPUT_DIR"
echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
exit "${PIPESTATUS[0]}"