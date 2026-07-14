# SPEOS RAWXM3 Postprocess

Utilities for post-processing SPEOS RAWXM3 outputs and visualizing VTP results.

## Main scripts

- `batch_post.py`: batch post-processing workflow.
- `sunburn_batch_post.py`: sunburn-specific batch post-processing workflow.
- `visualize_vtp.py`: VTP visualization helper.
- `rawxm3_vtp_pyvista.py`: RAWXM3 VTP processing with PyVista.

## Local usage

Run scripts directly from the repository root.

```powershell
python batch_post.py
python sunburn_batch_post.py
python visualize_vtp.py
```

## Build artifact policy

Generated artifacts must not be committed to git.

- Ignored paths include `build/`, `dist/`, and Python cache directories.
- Binary outputs such as `*.exe`, `*.7z`, and `*.zip` are ignored.
- If executables are needed for distribution, publish them as GitHub Release assets (or use Git LFS if versioning binaries is required).

## Git note

If a push fails with non-fast-forward:

```powershell
git fetch origin
git pull --rebase origin main
git push origin main
```
