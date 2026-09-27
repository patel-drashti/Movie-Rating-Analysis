# Movie Rating Analysis — VS Code Setup Guide (Full Project)

Follow these steps exactly, in order. This gives you one working project folder
on your Desktop with everything: EDA, database, ML model, and dashboard.

---

## STEP 1 — Create the project folder

On your **Desktop**, create a folder named:

```
Movie_Rating_Project
```

Inside it, put these 2 files (given to you separately):

```
Movie_Rating_Project/
├── app.py
└── requirements.txt
```

(Don't rename these files — `app.py` must be named exactly that.)

---

## STEP 2 — Open the folder in VS Code

- Open VS Code
- File → Open Folder → select `Movie_Rating_Project`
- You should see `app.py` and `requirements.txt` in the Explorer panel on the left

---

## STEP 3 — Open a terminal inside VS Code

- Menu: Terminal → New Terminal (or press `` Ctrl+` ``)
- It should open already inside your `Movie_Rating_Project` folder. Confirm with:
  ```powershell
  dir
  ```
  You should see `app.py` and `requirements.txt` listed.

---

## STEP 4 — Create a virtual environment (only needed once)

```powershell
python -m venv venv
```

This creates a `venv` folder inside `Movie_Rating_Project`. Your folder now looks like:

```
Movie_Rating_Project/
├── app.py
├── requirements.txt
└── venv/
```

---

## STEP 5 — Activate the virtual environment

```powershell
venv\Scripts\activate
```

If you get a "running scripts is disabled" error, run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Type `Y` and press Enter, then try activating again.

You'll know it worked when your terminal shows `(venv)` at the start of the line.

---

## STEP 6 — Install all required packages

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: streamlit, pandas, numpy, plotly, scikit-learn — everything the project needs, in one go.

---

## STEP 7 — Run the dashboard

```powershell
streamlit run app.py
```

- It will automatically open in your browser at `http://localhost:8501`
- **First run only:** it downloads the dataset automatically — takes 10-20 seconds. You'll see "Downloading dataset..." spinner in the browser.
- After that, it's instant every time you rerun it.

---

## STEP 8 — Using the dashboard

You'll see a sidebar with 5 sections:

| Section | What it shows |
|---|---|
| **Overview** | Project summary, key numbers |
| **EDA Dashboard** | All charts: rating distribution, top movies, genres, yearly trend, correlation |
| **Database (SQL)** | Real SQL queries run on a SQLite database, pick from dropdown |
| **Prediction Model** | ML model (Random Forest) predicting movie ratings — try your own inputs |
| **About** | Tools used, project pipeline |

This single dashboard covers **Part 1 (EDA) + Part 2 (Database & ML) + Part 3 (Dashboard)** — your entire project.

---

## STEP 9 — Stopping the app

Go back to the VS Code terminal and press `Ctrl+C`.

---

## Running it again later

Every time you come back to work on this:

```powershell
cd Desktop\Movie_Rating_Project
venv\Scripts\activate
streamlit run app.py
```

That's it — 3 commands, no need to reinstall anything again.

---

## Final folder structure (after everything runs once)

```
Movie_Rating_Project/
├── app.py                  ← the dashboard code
├── requirements.txt        ← list of packages needed
├── venv/                   ← Python virtual environment (auto-created)
├── ml-latest-small/        ← dataset (auto-downloaded on first run)
├── ml-latest-small.zip     ← downloaded zip (auto-created)
└── movies.db               ← SQLite database (auto-created on first run)
```

You don't need to create the last 3 — the app creates them automatically the first time you run it.
