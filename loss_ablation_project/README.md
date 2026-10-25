# Loss-function ablation — Hierarchical Bengali meme aggression (multimodal)

Same pipeline as `hierarchical-mimosa-multimodal.ipynb` (split, models, optimizer, epochs, metrics).
Only the training loss changes: `ce` (baseline), `weighted_ce`, `label_smooth_ce`, `focal`, `weighted_focal`.
Default model: `late_xlmr_swin_t` (edit `ABLATION_MODELS` in `loss_ablation.py` to add more).

## Folder layout
```
loss_ablation_project\
    loss_ablation.py      main script (the ablation)
    setup.ps1             one-time install (.venv + PyTorch CUDA + libraries)
    run.ps1               runs the ablation, saves a log
    requirements.txt
    .vscode\              VS Code tasks + interpreter settings
```
Dataset (from Google Drive), anywhere, e.g.:
```
D:\datasets\hierarchical-mimosa\
    hierarchical_aggression.csv
    Img\Img\*.jpg
```

## Run in VS Code
1. File -> Open Folder -> `loss_ablation_project`
2. Terminal -> New Terminal, then:
   ```
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
   (once; ~15-30 min. Must end with `CUDA available: True`)
3. Ctrl+Shift+P -> `Python: Select Interpreter` -> `.venv`
4. Run:
   ```
   powershell -ExecutionPolicy Bypass -File .\run.ps1 -DataRoot "D:\datasets\hierarchical-mimosa"
   ```
   (~2-2.5 h on RTX 4090)

Or use the menu: Terminal -> Run Task -> `1. Setup environment`, then `2. Run loss ablation`.

## Output
`loss_ablation_outputs\`
- `loss_ablation_results.csv` — main table (l1_f1, l1_macro_f1, l1_nonagg_recall, l2_f1, l2_macro_f1 per loss)
- `loss_ablation_partial.csv` — saved after every run
- `ablation_late_xlmr_swin_t.png` — bar chart
- `cm_l1_*.png`, `cm_l2_*.png`, `loss_*.png` — confusion matrices and loss curves

`ablation_log_*.txt` in the project folder contains everything printed.

## Troubleshooting
- `CUDA available: False` -> update NVIDIA driver, delete `.venv`, run setup again.
- DataLoader / worker / spawn error -> add `-NoWorkers` to the run command.
