# How to run

From the project directory:

```bash
venv/bin/python extract.py --limit 3    # test
venv/bin/python extract.py --resume    # full run
venv/bin/python analyze.py
venv/bin/python generate_validation_sample.py
```

If Terminal can’t find the project path (e.g. after moving the folder), open the project in Finder, drag the **folder** into the Terminal window to paste its path, type `cd ` in front, press Enter, then run the commands above.

Ensure `.env` contains `ANTHROPIC_API_KEY=sk-ant-...`.
