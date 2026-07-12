# Migration from the Colab notebook

The original notebook mixed package installation, Google Drive extraction,
training, topology generation, inference, rendering, metrics, and videos in one
kernel. The server version separates these concerns.

| Notebook behavior | Server replacement |
|---|---|
| `%cd /content/PhysTwin` | `--phystwin-root /real/server/path/PhysTwin` |
| `!pip install ...` | Create the Conda environment once |
| Google Drive mount/copy | Place data directly under the server PhysTwin tree |
| variables shared across cells | explicit command-line arguments and saved NPZ/JSON files |
| embedded script strings | normal files in `scripts/` |
| manual cell ordering | `run_strategy2_pipeline.py` |
| inline display tables | CSV and JSON result files |

The original notebook is intentionally not used as the main entry point.
