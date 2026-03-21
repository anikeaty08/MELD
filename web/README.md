# Web Dashboard

This folder keeps the optional MELD dashboard code separate from the core
framework package.

It is intentionally not part of the `meld-framework` wheel or sdist. The core
framework ships only the Python API, CLI, datasets, and training logic.

## Install

From the repo root:

```bash
python -m pip install -r web/requirements.txt
```

## Run

From the repo root:

```bash
python -m web.server
```

If you want to rebuild the React assets:

```bash
cd web/frontend
npm install
npm run build
```
