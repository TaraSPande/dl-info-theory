# neural-net

Self-contained PyTorch MLPs for **regression** and **classification**.

## Files

- `models.py`: small/large MLP builders
- `trainer.py`: minimal training loops
- `demo.py`: synthetic-data sanity check

## Run demo

```bash
python neural-net/demo.py
```

## Notes

This directory name contains a hyphen (`neural-net/`), so it is not a standard
Python package import name. The demo uses local imports that work when executed
as a script.
