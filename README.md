# Smart Shelf Market Basket Analysis
*"Beers and Diapers: The Impossible Correlation"* - IVP + AI team project

> 👉 **New here? Read [TEAM_GUIDE.md](TEAM_GUIDE.md)** - step-by-step setup and tasks for each team member.

A camera watches a 3×3 shelf. **IVP** (ArUco rectification, CLAHE, colour differencing) + a **CNN** (YOLOv8)
work out which products a customer removed. Those removals become **baskets**, which **Apriori** mines for
correlations and **KNN** uses to recommend the next item.

```
shelf photo ─► ArUco rectify ─► preprocess ─► YOLOv8 / colour-diff ─► removed items ─► basket
                                                                                      │
                                         Apriori rules  ◄─── binary matrix ◄──────────┤
                                         KNN next-item  ◄──────────────────────────────┘
```

---

## Repo layout

| Path | What | Owner |
|---|---|---|
| `notebooks/01_preprocessing.ipynb` | Preprocessing steps, ArUco + grid, **Exp 2** preprocessing ablation | P2 |
| `notebooks/02_yolov8_training.ipynb` | Dataset download, YOLOv8 training/eval, **Exp 1** domain gap | P1 |
| `notebooks/03_change_detection.ipynb` | Before/after → removed items, **Exp 3**, exports baskets | P2 |
| `notebooks/04_apriori_knn.ipynb` | Planted-rule simulation, Apriori, **Exp 4** noise sweep, **Exp 5** KNN | P3 |
| `notebooks/05_groceries_real.ipynb` | Apriori + KNN on real Kaggle Groceries data, **Exp 5b** | P3 |
| `src/` | Shared code imported by all notebooks - edit here, not in notebooks | all |
| `tools/aruco_markers.pdf` | Print at 100 % and stick on the shelf corners | P4 |
| `tools/set_repo_url.py` | Writes the GitHub URL into every notebook (run once) | Setup lead |
| `TEAM_GUIDE.md` | Detailed per-person setup + task guide | all |
| `paper/outline.md` | Paper structure, result tables to fill, references | P4 |
| `prototype/` | The first 3 prototype notebooks (kept for reference) | - |

Every notebook **runs without the real data** (it generates demo data and says `DEMO MODE`), so everyone can
start immediately. Demo numbers are *not* results - don't put them in the paper.

---

## How the team shares work

- **Code → GitHub.** Everyone clones the same repo. Change `src/` or notebooks → commit → push.
- **Data, weights, results → one Google Drive folder `product-identification-correlation_data/`.** Every notebook reads and writes there,
  on every machine. A model trained on Colab today can be evaluated on a Mac tomorrow.
- **Same notebook everywhere.** The first cell detects Colab / NVIDIA / Apple GPU / CPU and sets paths.

| Machine | Use it for |
|---|---|
| **Google Colab (T4 GPU)** | Main YOLO training - anyone can run it from any laptop |
| **Lenovo LOQ (RTX 3050)** | Second training job in parallel (Exp 1 / extra runs), no Colab time limits |
| **M5 MacBook Air (24 GB)** | IVP notebooks 01 + 03; YOLO training also works (Apple GPU, ~2× slower than Colab) |
| **M1 MacBook Air (8 GB)** | AI notebooks 04 + 05 (CPU, 1-2 min each), paper |
| **Asus Vivobook** | Roboflow labelling, paper, AI notebooks |

**Training on different devices:** runs are saved to `product-identification-correlation_data/runs/<experiment>_<machine>/`.
To continue a run elsewhere, set `START_WEIGHTS = detection.latest_run_weights("public_only")` in notebook 02.
If Colab disconnects mid-training, `detection.resume(run_dir)` resumes it exactly (Colab only, same paths).
Two people can train two *different* experiments at the same time (e.g. Colab + LOQ).

---

## One-time setup

### Step 1 - GitHub repo (one person, ~5 min)
```bash
cd product-identification-correlation
git init -b main && git add . && git commit -m "Initial project structure"
# create an empty repo on github.com (public is easiest for Colab), then:
git remote add origin https://github.com/<you>/product-identification-correlation.git
git push -u origin main
```
Before the first commit, put the URL into every notebook (Colab needs it):
`python tools/set_repo_url.py https://github.com/<you>/product-identification-correlation.git`. Add the other three as collaborators.

### Step 2 - Shared Google Drive folder (one person, ~2 min)
Create `product-identification-correlation_data` in **My Drive** and share it with the team (Editor).
Each teammate: open "Shared with me" → right-click the folder → **Organise → Add shortcut → My Drive**.
Now it's at `MyDrive/product-identification-correlation_data` for everyone - that's where Colab looks.

### Step 3 - API keys
- **Roboflow** (P1, P4): sign up at roboflow.com → Settings → API key.
- **Colab:** key icon in the left sidebar → add secret `ROBOFLOW_API_KEY` → enable notebook access.
- **Laptops:** copy `local_config.example.json` → `local_config.json` and paste the key (this file is gitignored).

### Step 4 - Per-laptop setup

**Mac (M1 / M5):**
```bash
git clone https://github.com/<you>/product-identification-correlation.git && cd product-identification-correlation
python3.12 -m venv .venv          # brew install python@3.12 if missing
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows + NVIDIA (Lenovo LOQ):** install Python 3.12 from python.org, then in PowerShell:
```powershell
git clone https://github.com/<you>/product-identification-correlation.git; cd product-identification-correlation
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build FIRST
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"                          # must print True
```
(If `cu128` fails, pick the right command at pytorch.org → Get Started.)

**Windows without NVIDIA (Vivobook):** same as above but skip the `torch` line.

**Colab:** nothing to install. Open a notebook from GitHub (File → Open notebook → GitHub tab), set
*Runtime → Change runtime type → T4 GPU*, run the first cell, and allow Drive access.

### Step 5 - Point laptops at the shared Drive folder (optional but recommended)
Install **Google Drive for Desktop**, then in `local_config.json` set `data_dir` to the synced folder, e.g.
- Mac: `~/Library/CloudStorage/GoogleDrive-<email>/My Drive/product-identification-correlation_data`
- Windows: `G:/My Drive/product-identification-correlation_data`

Right-click the folder → *Available offline* so training isn't slowed by streaming. Without this step, data
goes to a local `data/` folder in the repo (fine for solo work, but results won't be shared).

### Step 6 - VS Code
Install the **Python** and **Jupyter** extensions → open the repo folder → open a notebook → *Select Kernel*
(top right) → **Python Environments → .venv**. Run the first cell - it prints the device and data folder.

---

## First update (2 days) - who does what

| Person | Tasks (~2-3 h) | Where | Show the teacher |
|---|---|---|---|
| **P1 Vision** | Notebook 02 on the real dataset: download, explore, train 50 epochs, evaluate | Colab (or LOQ) | Class distribution, mAP table, training curves, confusion matrix, predictions |
| **P2 IVP** | Notebook 01 sections 1-3 on real images; notebook 03 on demo sessions. Take 3-4 phone before/after photos of any products on a table with the printed markers and try them | M5 Air | Pipeline step images, ArUco rectification, change-detection outputs |
| **P3 AI** | Notebooks 04 + 05. Choose planted rules, read the noise sweep, compare KNN metrics | M1 / Vivobook | Top rules, noise-sweep figure, KNN table, Groceries rules |
| **P4 Paper + shelf** | `paper/outline.md` → draft Intro (fix wording), Related Work (verify every reference), Methodology + pipeline diagram. Print ArUco markers, plan the shelf build and product purchase | Vivobook | Paper draft, shelf plan |

Order: P1 runs 02 first (Day 1). After that, P2 can run the ablation (01 §4) and P3 picks up the real class
names and detector error profile automatically.

---

## Real shelf sessions (after the shelf is built)

1. Fix the phone/webcam on a tripod facing the shelf; stick ArUco markers 0-3 (TL, TR, BR, BL) with each
   marker's **centre on a corner of the 3×3 area**. Keep lighting the same for before/after.
2. Write `product-identification-correlation_data/sessions/planogram.json` - which product is in each cell:
   ```json
   {"0,0": "bournvita", "0,1": "nescafe", "0,2": "...", "1,0": "...", "...": "..."}
   ```
   Names must match the dataset class names exactly (printed by notebook 02).
3. For each customer interaction: `sessions/s001/before.jpg`, take items, `sessions/s001/after.jpg`, restock.
4. Log ground truth in `sessions/sessions.csv`:
   ```
   session_id,removed
   s001,nescafe;nivea_lotion
   s002,bournvita
   ```
5. Aim for **40-60 sessions**. Also photograph the full shelf ~150 times (different arrangements and lighting)
   and label those in Roboflow **with the same class names** for Experiment 1.

---

## Experiments → paper tables

| # | Question | Notebook | Output in `product-identification-correlation_data/results/` |
|---|---|---|---|
| 1 | Do own shelf photos close the domain gap? | 02 §5 | `exp1_domain_gap.csv` |
| 2 | Does preprocessing help in bad lighting? | 01 §4 | `ablation_preprocessing.csv/.png` |
| 3 | YOLO vs classical change detection | 03 | `exp3_change_detection.csv`, `exp3_threshold_sweep.png` |
| 4 | How accurate must detection be for Apriori to find the true correlations? | 04 §3 | `exp4_noise_sweep.csv/.png` |
| 5 | Does KNN beat popularity for next-item prediction? | 04 §4, 05 §3 | `exp5_knn.csv`, `exp5b_knn_groceries.csv` |

## Troubleshooting
- **`ModuleNotFoundError`** in VS Code → the kernel isn't `.venv`. Re-select it (top right).
- **Colab can't find `requirements.txt`** → `REPO_URL` is empty and the code isn't in `MyDrive/product-identification-correlation`.
- **Roboflow download fails** → check `RF_VERSION` on the dataset page, and that the API key secret is enabled.
- **CUDA out of memory on the 3050** → `batch=8` in `detection.train(...)`.
- **Windows path problems** → avoid folder names ending in a space (e.g. `College Projects `).
