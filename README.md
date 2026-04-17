# LTGCN-GTGCN

## Quick Start

Clone the repository:

```bash
git clone https://github.com/mobyous/LTGCN-GTGCN.git
cd LTGCN-GTGCN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: fetch the large Brazil combined dataset explicitly:

```bash
python scripts/fetch_data.py
```

Run examples:

```bash
python main.py --experiment spain_persistence
python main.py --experiment brazil_persistence_top40
python main.py --all-spain
python main.py --all-brazil
```

Rolling-origin CV example:

```bash
python main.py --all-spain --rolling-cv-folds 3
```

Notes:

- Spain data files are stored directly in `data/`.
- The large Brazil file `data/covid_brazil_combined.csv` is not tracked in git.
- If it is missing, the loader will try to download it from the latest GitHub release asset named `covid_brazil_combined.csv`.
