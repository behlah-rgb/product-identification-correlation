# Team Guide - Smart Shelf Market Basket Analysis

**How to use this guide**
1. **Everyone** reads **Part 1** (10 min): the plan, who does what, and the teamwork rules.
2. The **Setup Lead (P3)** does **Part 2** first. Everyone else waits for the "setup done" message.
3. **Everyone** does **Part 3** on their own laptop.
4. Then go to **your own part**: P1 → Part 4 · P2 → Part 5 · P3 → Part 6 · P4 → Part 7.
5. **Part 8** is the first teacher update. **Part 9** is what happens after it. **Part 10** is for when something breaks.

Fill in the names once:

| Role | Name | Laptop | Main tools |
|---|---|---|---|
| **P1 - Vision (CNN)** | ________ | Lenovo LOQ (RTX 3050) | Google Colab + LOQ, notebook 02 |
| **P2 - IVP** | ________ | M5 MacBook Air (24 GB) | Notebooks 01 + 03 |
| **P3 - AI + Setup Lead** | ________ | M1 MacBook Air (8 GB), which already has the project folder | Notebooks 04 + 05, GitHub/Drive setup |
| **P4 - Paper + Shelf** | ________ | Asus Vivobook | Paper, shelf build, Roboflow labelling |

---

# Part 1 - Overview (everyone)

## 1.1 What we're building, in one paragraph
A phone camera photographs a 3×3 shelf **before** and **after** a customer takes items. Image processing
straightens and cleans the photo (IVP), and a CNN (YOLOv8) recognises the products. Comparing before and after tells us
**which products were taken**, and each customer's items become one **basket**. **Apriori** finds products bought
together (the "beers and diapers" correlations) and **KNN** predicts the next item a customer might want.

## 1.2 Where things live

| What | Where | Why |
|---|---|---|
| Code (notebooks, `src/`) | **GitHub** repo `IVP_Project` | Everyone has the same code |
| Datasets, photos, trained models, results | Google Drive folder **`IVP_Project_data`** | Big files, shared between laptops and Colab |
| Paper | **Overleaf** (P4 shares the link) | Everyone can write at once |

Every notebook finds the right place by itself (Colab, Mac or Windows). If the real data isn't there yet, it
runs on **demo data** and prints `DEMO MODE`. Demo numbers are **not** results, so never put them in the paper.

## 1.3 The two-day plan

```
DAY 1                                                    DAY 2
Setup Lead (P3): GitHub + Drive  (~30 min)
Everyone: laptop setup           (~30-45 min)
P1: download dataset + start training ─────────────────► evaluate, share class names + results
P2: notebook 01 §1-3, notebook 03 demo ────────────────► 01 §4 (needs P1's model), phone trial photos
P3: notebooks 04 + 05 on demo/real data ───────────────► re-run 04 with real class names + P1's errors
P4: paper outline, references, shelf plan ─────────────► paper draft + update slides
```

**Dependencies (who waits for whom):**
- P2's ablation (notebook 01 §4) and P3's re-run of notebook 04 **need P1's trained model**, so P1 starts training first thing on Day 1.
- P4 needs the **class names** from P1 to buy products and plan the shelf.

## 1.4 Teamwork rules (read these)
1. **Each notebook has one owner.** Only the owner edits and commits it. P1 owns 02, P2 owns 01 and 03, P3 owns 04 and 05.
   If you want a change in someone else's notebook, message them.
2. **Changes to `src/` (shared code):** tell the group before pushing, because it affects everyone.
3. **Always `git pull` before you start working, and push when you finish** (see Part 10.1).
4. **Never commit data, photos, model weights or `local_config.json`.** The `.gitignore` already blocks them, so don't force it.
5. Post in the group chat when you finish something another person depends on. Examples: "Model trained, results in Drive",
   "Class names: …".

---

# Part 2 - Setup Lead (P3): one-time setup, ~30 min

Do this on the M1 MacBook Air, which already has the project folder. Everyone else waits for your "setup done" message.

## 2.1 Accounts
- A **GitHub** account for all 4 people (github.com → Sign up).
- A **Google** account for all 4 (for Drive and Colab).

## 2.2 Decide: public or private repo
- **Public (simplest):** Colab can open and download it with no extra steps.
- **Private (if you're worried about copying):** Colab needs a token (step 2.6). It's slightly more work but fine.

## 2.3 Create the GitHub repo
1. Go to **github.com/new**.
2. Repository name: `IVP_Project`. Choose Public or Private.
3. **Don't** tick "Add a README", ".gitignore" or "license" (we already have them).
4. Click **Create repository** and copy the URL shown, e.g. `https://github.com/<your-username>/IVP_Project.git`.

## 2.4 Push the project
Open **Terminal** and run these one at a time:

```bash
cd "/Users/satvikakshintala/Documents/Claude/College Projects/IVP_Project"

# first time using git on this Mac? set your name + email (use your GitHub email)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# put the repo URL into all notebooks (Colab needs it)
.venv/bin/python tools/set_repo_url.py https://github.com/<your-username>/IVP_Project.git

git init -b main
git add .
git status            # check: NO .venv/, data/, *.pt or local_config.json listed
git commit -m "Initial project structure"
git remote add origin https://github.com/<your-username>/IVP_Project.git
git push -u origin main
```

If `git push` asks for a password: GitHub doesn't accept account passwords here. The easiest fix is to install GitHub CLI
and log in through the browser:
```bash
brew install gh
gh auth login          # choose: GitHub.com → HTTPS → Yes (authenticate git) → Login with a web browser
git push -u origin main
```

## 2.5 Invite the team
On the repo page: **Settings → Collaborators → Add people** → add P1, P2 and P4 by GitHub username. They must
**accept the email invite**.

## 2.6 Private repo only: make a token for Colab
1. GitHub → your profile picture → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Name: `colab-ivp`. Expiration: 90 days. Repository access: **Only select repositories → IVP_Project**.
3. Permissions → Repository permissions → **Contents: Read-only**. Click Generate and **copy the token**.
4. Send it to the team **privately** (not in a public group). Each person adds it in Colab as the secret `GITHUB_TOKEN` (Part 3.4).

## 2.7 Create the shared Drive folder
1. Open **drive.google.com** → **New → New folder** → name it exactly **`IVP_Project_data`**.
2. Right-click it → **Share** → add P1, P2 and P4's Google emails as **Editor** → Send.

## 2.8 Tell the team
Post in the group chat:
> Setup done. Repo: `<URL>` (accept the GitHub invite). Drive folder `IVP_Project_data` is shared with you.
> Follow TEAM_GUIDE.md Part 3.

Then do Part 3 yourself (you only need 3.1, 3.2 Mac steps 5-6, 3.5, 3.6 and 3.7, because the environment already exists on the M1).

---

# Part 3 - Everyone: laptop setup, ~30-45 min

## 3.1 Add the shared Drive folder to your own Drive (everyone, 1 min)
The owner (P3) can skip this.
1. drive.google.com → **Shared with me** → find `IVP_Project_data`.
2. Right-click → **Organise → Add shortcut** → choose **My Drive** → Add.
3. Check: `IVP_Project_data` now appears in **My Drive**. Colab looks for it exactly there.

## 3.2 Mac setup (P2 on the M5; P3 already has it)

1. **Install the basics** (skip anything you already have). Open Terminal:
   ```bash
   xcode-select --install                     # Apple developer tools (git). Click Install in the popup.
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # Homebrew
   brew install python@3.12 gh
   ```
   After installing Homebrew, it prints 2 "Next steps" commands (`echo ... >> ~/.zprofile` and `eval ...`). **Run them.**
2. **Log in to GitHub:** `gh auth login` → GitHub.com → HTTPS → Yes → Login with a web browser.
3. **Download the code:**
   ```bash
   mkdir -p ~/Documents/Projects && cd ~/Documents/Projects
   git clone https://github.com/<username>/IVP_Project.git
   cd IVP_Project
   ```
4. **Create the Python environment and install packages** (~5-10 min; it downloads PyTorch):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate          # the prompt now starts with (.venv)
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Check it works:**
   ```bash
   python -c "import torch, ultralytics, cv2, mlxtend; print('OK, Apple GPU:', torch.backends.mps.is_available())"
   ```
   It should print `OK, Apple GPU: True`.
6. Continue with **3.5 (Drive for Desktop)** and **3.6 (VS Code)**.

## 3.3 Windows setup (P1 on the Lenovo LOQ with NVIDIA; P4 on the Vivobook without)

1. **Install:**
   - **Python 3.12** from python.org/downloads → Windows installer (64-bit). In the installer, **tick "Add python.exe to PATH"** → Install Now.
   - **Git for Windows** from git-scm.com → keep all defaults. It includes the browser login for GitHub.
   - **P1 only:** update the NVIDIA driver (GeForce Experience or nvidia.com/drivers). Then open PowerShell, run `nvidia-smi`, and check that it shows the RTX 3050.
2. **Allow virtual environments to activate** (one time). Open **PowerShell** and run:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned      # type Y and press Enter
   ```
3. **Download the code:**
   ```powershell
   mkdir $HOME\Documents\Projects; cd $HOME\Documents\Projects
   git clone https://github.com/<username>/IVP_Project.git     # a browser window opens to log in to GitHub
   cd IVP_Project
   ```
4. **Create the environment:**
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate                                     # prompt starts with (.venv)
   python -m pip install --upgrade pip
   ```
5. **Install packages:**
   - **P1 (NVIDIA): install the GPU version of PyTorch FIRST**, then the rest:
     ```powershell
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
     pip install -r requirements.txt
     python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
     ```
     It must print `CUDA: True NVIDIA GeForce RTX 3050 ...`. If it prints False, or `cu128` fails, go to pytorch.org →
     *Get Started* → Stable / Windows / Pip / Python / CUDA (latest listed), and run the command it shows.
   - **P4 (no NVIDIA):**
     ```powershell
     pip install -r requirements.txt
     python -c "import torch, ultralytics, mlxtend; print('OK')"
     ```
6. Continue with **3.5** and **3.6**.

## 3.4 Google Colab setup (everyone who uses Colab, especially P1)
1. Go to **colab.research.google.com** → **File → Open notebook → GitHub** tab.
   - Private repo: tick **Include private repos** and authorise GitHub.
   - Paste the repo URL, press Enter, and click the notebook you want.
2. **Secrets** (key icon 🔑 in the left sidebar) → **Add new secret**:
   - `ROBOFLOW_API_KEY` = your Roboflow key (P1 and P4, see Part 4.2). Turn on **Notebook access**.
   - Private repo only: `GITHUB_TOKEN` = the token from P3. Turn on **Notebook access**.
3. For training (P1): **Runtime → Change runtime type → T4 GPU → Save**.
4. Run the first cell (the SETUP cell). Colab asks to **connect to Google Drive**: choose your account and allow everything.
   At the end it prints `Running on : Google Colab` and `DATA_DIR : /content/drive/MyDrive/IVP_Project_data`.

   If it says `requirements.txt not found`, the `REPO_URL` in that cell is empty. Tell P3 (step 2.4 was skipped).

## 3.5 Google Drive for Desktop (P1 on the LOQ, P2, P3; optional for P4)
This lets your laptop's notebooks read and write the **same** shared folder that Colab uses.

1. Install **Google Drive for Desktop** (google.com/drive/download) and sign in.
2. Find the folder:
   - **Mac:** Finder sidebar → Google Drive → My Drive → `IVP_Project_data`. The real path is
     `~/Library/CloudStorage/GoogleDrive-<your-email>/My Drive/IVP_Project_data`.
   - **Windows:** File Explorer → **G:** drive → My Drive → `IVP_Project_data`, so the path is `G:/My Drive/IVP_Project_data`.
3. Right-click `IVP_Project_data` → **Make available offline** (Mac) / **Offline access → Available offline** (Windows).
   Without this, training is slow because files stream from the internet.
4. In the project folder, **copy `local_config.example.json` to `local_config.json`** and edit it:
   ```json
   {
     "data_dir": "~/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/IVP_Project_data",
     "ROBOFLOW_API_KEY": "paste-your-key-or-leave-this"
   }
   ```
   On Windows use forward slashes: `"data_dir": "G:/My Drive/IVP_Project_data"`.
   This file stays on your laptop only and is never committed.

If you skip this step, your notebooks save to a local `data/` folder inside the project. That works, but nobody else sees your results.

## 3.6 VS Code (everyone who runs notebooks locally)
1. Install **VS Code** (code.visualstudio.com).
2. Extensions (square icon on the left) → install **Python** and **Jupyter** (both by Microsoft).
3. **File → Open Folder** → select the `IVP_Project` folder.
4. Open any notebook in `notebooks/` → top right **Select Kernel → Python Environments → `.venv`**.
   (If `.venv` isn't listed: Cmd/Ctrl+Shift+P → "Python: Select Interpreter" → *Enter interpreter path* →
   `.venv/bin/python` on Mac, `.venv\Scripts\python.exe` on Windows.)

## 3.7 Final check (everyone, 2 min)
Open `notebooks/04_apriori_knn.ipynb` and run **only the first cell** (▶ next to it). You should see:
```
Running on : macOS-... / Windows-... / Google Colab
Device     : 0 (NVIDIA) / mps (Mac) / cpu
DATA_DIR   : ...IVP_Project_data      ← the shared folder, not .../IVP_Project/data
```
Post "setup done ✅ (device: …)" in the group chat.

---

# Part 4 - P1: Vision (CNN) lead

**You own:** `notebooks/02_yolov8_training.ipynb` · **Machines:** Google Colab (main), Lenovo LOQ (backup / second experiment)
**Your goal for the update:** a trained YOLOv8 model on the Indian Grocery dataset, with its accuracy numbers and plots.
**Time:** ~2-3 h across 2 days (most of it is waiting for training).

## 4.1 Understand what you're doing (5 min)
- **YOLOv8** is a CNN that finds objects in an image and draws a labelled box around each one.
- We start from `yolov8n.pt`, which was pre-trained on everyday objects, and **fine-tune** it on 403 labelled photos of 10 Indian
  products. This is called transfer learning.
- **mAP@50** is the main accuracy score (0 to 1, higher is better). It measures how well the predicted boxes match the true boxes.
  **Precision** asks: of the boxes it drew, how many were right? **Recall** asks: of the real products, how many did it find?

## 4.2 Get a Roboflow API key (5 min)
1. Sign up at **roboflow.com** (free, "Sign in with Google" works).
2. Open **app.roboflow.com** → your workspace → **Settings** (left sidebar, bottom) → **API Keys** → copy the **Private API Key**.
3. Add it to Colab (Part 3.4, secret `ROBOFLOW_API_KEY`). If you'll train on the LOQ too, also add it to `local_config.json`.

## 4.3 Find the dataset version number (2 min)
1. Open https://universe.roboflow.com/agentsk47/indian-grocery-object-detection-mfsnx
2. Click the **Dataset** tab and note the **latest version number** (v1, v2, v3…).
3. Optional check: **Download Dataset → YOLOv8 → Show download code**. The code shows `workspace("agentsk47")`,
   `project("indian-grocery-object-detection-mfsnx")` and `version(N)`. Close it without downloading.

## 4.4 Day 1: download and explore the dataset in Colab (15 min)
1. Open `02_yolov8_training.ipynb` in Colab (Part 3.4), and set **Runtime → T4 GPU**.
2. Run the **SETUP** cell and allow Drive access.
3. In the next cell (dataset config), set `RF_VERSION = <number from 4.3>`. Leave `DEMO = False`. Run it.
4. **Section 1 (Get the dataset):** run it. It downloads about 400 images into `IVP_Project_data/datasets/indian_grocery/`
   and prints the **10 class names**.
   → **Post the class names in the group chat now.** P4 needs them to buy products, and P3 and P2 need them later.
5. **Section 2 (Explore):** run both cells. You'll see images per split, a bar chart of boxes per class, and 4 sample
   images with their boxes. **Check:** are any classes very small (under ~30 boxes)? Note it, because it goes in the paper.

## 4.5 Day 1: train (5 min of work + ~15 min waiting)
1. **Section 3:** in the config cell, leave `RUN_NAME = "public_only"`, `MODEL = "yolov8n.pt"` and `EPOCHS = 50`. Run the training cell.
2. You'll see a progress bar per epoch. It takes about **10-15 min on a T4**.
   **Keep the Colab tab open and your laptop awake.** Free Colab disconnects if the tab idles too long.
3. When it finishes it prints `Training took X min` and the run folder, e.g. `IVP_Project_data/runs/public_only_cuda-Tesla_T4/`.
   Because this is in Drive, **it's already shared with the team**.

**If Colab disconnects mid-training:** reconnect, run SETUP and the config and dataset cells again, then run the resume cell with
the right folder name: `run_dir = detection.resume(RUNS_DIR / "public_only_cuda-Tesla_T4")`, then `best = run_dir / "weights" / "best.pt"`.

**If Colab says GPU limit reached:** train on the LOQ instead (4.8). Free Colab GPU time resets after about a day.

## 4.6 Day 1-2: evaluate (10 min)
1. **Section 4:** run all cells in order.
   - **Evaluate:** prints mAP@50, mAP@50-95, precision, recall and a **per-class table**, and saves
     `IVP_Project_data/results/yolo_public_only.json`. **P3's notebook reads this file automatically.**
   - **Training curves:** loss should go down and mAP up. Saved as `results/training_curves_public_only.png`.
   - **Confusion matrix + PR curve:** shows which products get confused with each other.
   - **Predictions:** 4 test images with predicted boxes.
2. **What's a good result?** For about 400 images and 10 classes, mAP@50 of **0.7-0.9** is typical after 50 epochs. If it's much lower:
   - Try `EPOCHS = 100`, or
   - Try `MODEL = "yolov8s.pt"` (bigger and more accurate, about 2× slower). Use `RUN_NAME = "public_only_s"` so it doesn't overwrite.
3. **Post in the group chat:** "Model trained ✅ mAP@50 = X. Results in Drive." P2 and P3 are waiting for this.

## 4.7 Save your notebook to GitHub
Colab edits are **not** saved to GitHub automatically.
- **File → Save a copy in GitHub** → repo `IVP_Project`, branch `main`, path `notebooks/02_yolov8_training.ipynb` →
  commit message like "P1: trained public_only model" → OK.
- Or, on the LOQ: `git pull` → edit → `git add notebooks/02_yolov8_training.ipynb` → `git commit -m "..."` → `git push`.

## 4.8 Training on the LOQ instead of Colab
1. Complete Part 3.3 (with CUDA), 3.5 (Drive for Desktop plus `local_config.json` with your Roboflow key) and 3.6.
2. Open `notebooks/02_yolov8_training.ipynb` in VS Code and select the `.venv` kernel.
3. Run the cells exactly as in 4.4-4.6. The first cell should say `Device : 0 (cuda-NVIDIA_GeForce_RTX_3050...)`.
4. The run folder is named `public_only_cuda-NVIDIA_GeForce_RTX_3050...`, so it's clear which machine trained it.
5. If you get **CUDA out of memory**, change the training line to add `batch=8`:
   `detection.train(data_yaml, RUN_NAME, weights=START_WEIGHTS, epochs=EPOCHS, imgsz=IMGSZ, batch=8)`.
6. **Continue a Colab run on the LOQ:** set `START_WEIGHTS = detection.latest_run_weights("public_only")` and train again.

## 4.9 Your deliverables for the update
Save these to a Drive folder `IVP_Project_data/update1/P1/`, as files or screenshots (Mac: Cmd+Shift+4, Windows: Win+Shift+S):
- [ ] Class distribution chart (`results/dataset_class_distribution.png`)
- [ ] mAP / precision / recall numbers plus the per-class table
- [ ] Training curves (`results/training_curves_public_only.png`)
- [ ] Confusion matrix and 4 prediction images
- [ ] 3-4 sentences: dataset, model, epochs, device, main result. P4 uses this for the slides and the paper.

---

# Part 5 - P2: IVP lead

**You own:** `notebooks/01_preprocessing.ipynb`, `notebooks/03_change_detection.ipynb` · **Machine:** M5 MacBook Air
**Your goal for the update:** show the image-processing pipeline working (step-by-step images, perspective correction,
change detection), plus a first test on real phone photos.
**Time:** ~3 h across 2 days.

## 5.1 Understand what you're doing (10 min)
Read `src/preprocess.py` and `src/change_detect.py` (both are commented). The techniques, which you'll describe in the paper:
- **ArUco markers:** printed black-and-white squares that OpenCV finds automatically. With 4 of them at the shelf corners, we
  compute a **homography** (perspective transform) that turns a tilted phone photo into a flat, front-on view of the shelf.
- **Gray-world white balance:** removes colour casts (e.g. yellow tube lights).
- **Bilateral filter:** removes noise but keeps edges sharp.
- **CLAHE:** boosts contrast locally, so dark corners of the shelf become visible.
- **Change detection (classical):** after straightening both photos, compute the colour difference (CIELAB ΔE), threshold it, and
  clean it up with morphology (opening and closing). Cells where many pixels changed = product removed. The **planogram**
  (a map of which product sits in which cell) tells us which product it was.
- **Change detection (YOLO):** count products before and after, then subtract.

## 5.2 Day 1: notebook 01, sections 1-3 (45 min)
1. `git pull`, open `notebooks/01_preprocessing.ipynb` in VS Code, and select the `.venv` kernel.
2. Run cells **top to bottom, one at a time** (Shift+Enter).
   - If P1 has already downloaded the dataset and your `local_config.json` points to the Drive folder, you'll see
     `Using REAL dataset images`. Otherwise it's `DEMO MODE`, which is fine for now. Re-run once P1 posts.
3. **Section 1:** the 4-panel figure shows original → white balance → denoise → CLAHE, plus the brightness histograms.
   Try other images: change `sample_paths[0]` to `sample_paths[3]` etc. and pick the one that shows the effect best.
4. **Section 2:** marker detection and rectification on a shelf photo (demo until you have real ones), with the 3×3 grid and
   the 9 cropped cells.
5. **Section 3:** each degradation (dim, uneven light, noise, combined) next to its preprocessed version, plus a table of
   brightness, contrast and sharpness.
6. Don't run **section 4** yet. It needs P1's trained model.

**Experiment (optional, good for the paper):** in `src/preprocess.py`, try `apply_clahe(img, clip_limit=3.0)` or
`denoise(img, "nlm")` and compare. Keep notes of what looks better and why.

## 5.3 Day 1: notebook 03 on demo sessions (30 min)
1. Open `notebooks/03_change_detection.ipynb` and run top to bottom.
2. **Section 1:** a demo session shown before and after, raw and rectified.
3. **Section 2:** the **change mask** (white = changed pixels) and a heatmap of changed-pixel fraction per cell. The
   threshold sweep picks the best threshold. Expect **F1 ≈ 0.9** on demo data.
   → Look at which sessions it gets wrong and why. On demo data, a light-grey product on a light-grey background is invisible to
   colour differencing. That's a real limitation to discuss.
4. **Section 3 (YOLO):** on demo data the YOLO method finds nothing, because the model wasn't trained on these drawn products.
   That's expected. It works properly with real shelf photos and P1's real model.
5. **Section 4:** comparison table, and it saves `results/detected_baskets.csv` (P3 uses this).

## 5.4 Day 2: preprocessing ablation, notebook 01 section 4 (20 min, after P1 posts "model trained")
1. Re-run notebook 01 from the top (so it uses real images), then run **section 4**.
2. It copies the test images 10 times (5 conditions × with/without preprocessing), evaluates P1's model on each,
   and saves `results/ablation_preprocessing.csv` and `.png`.
3. **How to read it:** under `dim`, `uneven` and `combined`, does "+ preprocess" give a higher mAP than "raw"? That's the
   IVP contribution. On `clean` images preprocessing may *lower* mAP slightly, because the model was trained on raw images.
   That's expected and worth reporting honestly.

## 5.5 Day 2: first real photos with a phone (45 min)
This proves the pipeline works on real photos before the shelf exists.

1. **iPhone users: switch to JPG first:** Settings → Camera → Formats → **Most Compatible**. (HEIC photos won't open.)
2. **Print the markers:** `tools/aruco_markers.pdf`, printed at **100% / Actual size** (not "Fit to page"). Measure one with
   a ruler: it should be **6 cm**. Cut out the 4 markers, leaving a white border around each.
3. **Set up a fake shelf on a table or the floor:** tape the markers in a square about 60×60 cm: **ID 0 top-left, 1 top-right,
   2 bottom-right, 3 bottom-left** (as seen from the camera). Place **9 items** (anything, e.g. bottles, boxes, soap) in a 3×3
   layout inside the square, one per cell.
4. **Take photos:** hold the phone steady above or in front (propping it on books is better than holding it). All 4 markers must be visible. No
   flash, and keep hands out of the frame.
   - `before.jpg` → remove 1-2 items → `after.jpg` → put the items back. Repeat for 4 sessions.
5. **Put them in Drive** (Google Drive app on the phone, or upload from the laptop):
   ```
   IVP_Project_data/sessions/
       planogram.json
       sessions.csv
       s001/before.jpg   s001/after.jpg
       s002/before.jpg   s002/after.jpg   ... up to s004
   ```
   `planogram.json`: which item is in each cell, as `"row,col"` with row 0 at the top. You choose the names:
   ```json
   {"0,0": "shampoo", "0,1": "soap", "0,2": "coffee",
    "1,0": "oil",     "1,1": "cream", "1,2": "toothpaste",
    "2,0": "biscuit", "2,1": "noodles", "2,2": "lotion"}
   ```
   `sessions.csv` (make it in any text editor, and save the file as `.csv`):
   ```
   session_id,removed
   s001,soap
   s002,coffee;oil
   s003,cream
   s004,noodles;lotion
   ```
6. **Run notebook 03 again** (and notebook 01 section 2). They now say `REAL` and use your photos.
   The **classical method** result is your first real number. YOLO won't recognise household items, which is expected.
7. **After the update, rename this folder** from `sessions` to `sessions_trial`, so the real shelf sessions later start clean.

**If rectification fails** (`Not all 4 markers visible`): make sure the markers are flat, not glossy (no glare), fully in
frame, and not too small. Each marker should be at least ~40 px wide in the photo.

## 5.6 Your deliverables for the update
Save to `IVP_Project_data/update1/P2/`:
- [ ] Preprocessing step figure plus histograms (notebook 01 §1)
- [ ] ArUco detection → rectified shelf → 9 cells (a real phone photo if you have one)
- [ ] Degradation vs recovery figure (§3) and the ablation chart (§4)
- [ ] Change mask plus cell heatmap, and the Experiment 3 table (demo and your 4 real sessions)
- [ ] 3-4 sentences on the methods and first results for P4

---

# Part 6 - P3: AI lead (and Setup Lead)

**You own:** `notebooks/04_apriori_knn.ipynb`, `notebooks/05_groceries_real.ipynb` · **Machine:** M1 MacBook Air
**Your goal for the update:** show Apriori finding correlations (simulated and real data), the detection-error experiment,
and KNN recommendation results.
**Time:** ~2-3 h (after Part 2).

## 6.1 Understand what you're doing (15 min)
- **Basket** = the set of items one customer took. The **one-hot matrix** has one row per basket and one column per product,
  with 1 meaning "bought".
- **Support(A)** = fraction of baskets containing A.
- **Confidence(A→B)** = P(B | A) = support(A,B) / support(A).
- **Lift(A→B)** = confidence(A→B) / support(B).
  - **Lift > 1:** A and B are bought together more than chance, i.e. a real correlation.
  - **Lift = 1:** no relation.
  - **Lift < 1:** they avoid each other.
- **Apriori** efficiently finds all item combinations above a minimum support, and then builds rules from them.
- **Planted rules:** the shelf can't generate thousands of real baskets, so we **simulate** them and hide ("plant") known
  correlations, e.g. "customers who take Nescafe take Nivea 50% of the time". That lets us *measure* whether Apriori finds them.
- **KNN recommendation:** find the k past baskets most similar to the current one and recommend what those customers had.
  - **Jaccard distance** only looks at items that are present.
  - **Hamming distance** also counts shared *absences*, which is bad for sparse baskets. We test both.
- **Hit@3:** hide one item from a basket, recommend 3 items, and count it as a hit if the hidden one is among them.

## 6.2 Day 1: notebook 04 (45 min)
1. `git pull`, open `notebooks/04_apriori_knn.ipynb` in VS Code, select `.venv`, and run top to bottom (about 1-2 min total).
2. **Section 1:** prints the product names (demo names until P1 downloads the dataset) and the **4 planted rules**, plus
   charts of basket sizes and item popularity.
3. **Section 2 (Apriori, clean baskets):** you should see **"Planted rules recovered: 100%"** and the top-10 rules by lift.
   Check that the top rules involve the planted pairs.
4. **Section 3 (Experiment 4):** this is **the key result of the whole paper**:
   - Table: clean baskets vs baskets that went through the detector's mistakes (missed items, wrong items).
   - **Noise sweep figure:** how many planted rules Apriori still finds as the detector's recall drops from 1.0 to 0.5.
     In the demo run, all rules are found down to a recall of about 0.85 and then start disappearing. **That tells the vision team how
     accurate the detector must be for the correlations to survive.**
5. **Section 4 (Experiment 5, KNN):** table of Hit@1 and Hit@3 for popularity vs KNN-Hamming vs KNN-Jaccard at
   different k. Expect Jaccard ≥ popularity ≥ Hamming, and larger k to help.
6. The worked example shows a sample current basket, its 3 most similar past baskets, and the recommendations.

## 6.3 Day 1: notebook 05, real Groceries data (20 min)
1. Open `notebooks/05_groceries_real.ipynb` and run top to bottom. It downloads the Kaggle dataset automatically (no login
   needed). If the download fails, get `Groceries_dataset.csv` from
   kaggle.com/datasets/heeraldedhia/groceries-dataset and put it in `IVP_Project_data/datasets/groceries/`.
2. You'll see 14,963 baskets and 167 items (average 2.5 items per basket), the **top 15 rules by lift**, the
   **"surprising" pairs** (strong lift between items that aren't obviously related, which is the beers-and-diapers angle), and a
   lift heatmap.
3. **KNN on real data does worse than the popularity baseline** (Hit@3 ≈ 0.13 vs 0.165). Real baskets are tiny, so there's little
   to match on. That's an honest finding. Think about why, and write it down for the Discussion section.

## 6.4 Day 2: re-run with the real products and detector (30 min, after P1 posts)
1. Re-run **notebook 04** from the top. It now:
   - uses the **real 10 class names** from the dataset, and
   - uses **P1's real per-class recall and precision** as the detector error profile (it prints
     `Using detector error profile from yolo_public_only.json`).
2. **Choose your planted rules** in section 1. Mix *obvious* pairs (e.g. shampoo → conditioner of the same brand) with
   *"impossible"* ones (e.g. coffee → body lotion). Write them with the exact class names:
   ```python
   PLANTED = [
       bs.PlantedRule(("<class A>",), "<class B>", 0.55),              # obvious
       bs.PlantedRule(("<class C>",), "<class D>", 0.50),              # "impossible"
       bs.PlantedRule(("<class E>", "<class F>"), "<class G>", 0.60),  # 2-item trigger
       bs.PlantedRule(("<class H>",), "<class I>", 0.45),              # "impossible"
   ]
   ```
   Run everything again and check the clean recovery is still 100%.
3. **Tweak and understand:** try `N_BASKETS = 500` vs `5000`, or weaker strengths (0.3). When does Apriori start missing
   rules? Note the pattern for the paper.
4. Save the notebook and push:
   ```bash
   git pull
   git add notebooks/04_apriori_knn.ipynb notebooks/05_groceries_real.ipynb
   git commit -m "P3: planted rules with real classes, results"
   git push
   ```

## 6.5 Your deliverables for the update
Save to `IVP_Project_data/update1/P3/`:
- [ ] Planted rules plus the top-10 rules table (simulated)
- [ ] Clean vs through-detector recovery table, and the **noise sweep figure** (`results/exp4_noise_sweep.png`)
- [ ] KNN table (`results/exp5_knn.csv`) plus the worked example
- [ ] Groceries: top rules, surprising pairs, heatmap, KNN table
- [ ] 3-4 sentences on the methods and results for P4, plus the support/confidence/lift formulas for the paper

---

# Part 7 - P4: Paper + Shelf lead

**You own:** the paper (Overleaf), the physical shelf, the product purchase, and Roboflow labelling (after the update)
**Machine:** Asus Vivobook
**Your goal for the update:** a paper draft (Intro, Related Work, Methodology skeleton, pipeline diagram), the update slides,
and a concrete shelf plan.
**Time:** ~3 h across 2 days.

## 7.1 Day 1: set up the paper (20 min)
1. Go to **overleaf.com** and sign up (free). **New Project → Templates →** search "IEEE Conference Template" → *Open as Template*.
2. **Share** (top right) → add P1, P2 and P3 as editors.
3. Create the section headings from `paper/outline.md` (Introduction, Related Work, Methodology with subsections 3.1-3.7,
   Experimental Setup, Results, Discussion, Conclusion).
4. Paste the introduction in. **Make these edits:**
   - Replace "Our ipyn files/notebook, for example, uses a one-hot transaction representation…" with:
     *"The implementation represents transactions as one-hot vectors for Apriori and uses Jaccard and Hamming distance for
     nearest-neighbour basket search."*
   - Add a **contributions list** at the end of the intro (4 bullets, copy them from `paper/outline.md` §1).

## 7.2 Day 1: Related Work (60-90 min)
1. For **each reference** in `paper/outline.md`, search the title on **scholar.google.com**:
   - Confirm it exists and the authors and year match (fix anything that doesn't).
   - Click **Cite (″) → BibTeX** and paste it into a `references.bib` file in Overleaf.
   - Read the **abstract** and write one sentence on what it did.
2. Write **3 paragraphs** (about 150 words each):
   - **Retail product detection:** SKU-110K, RP2K, HoloSelecta, Locount, YOLO. What they do, and that they stop at detection.
   - **Market Basket Analysis:** Agrawal et al. (Apriori), support/confidence/lift, grocery studies. They assume transactions already exist.
   - **Recommendation / KNN:** Cover & Hart, item-based collaborative filtering.
   - Final sentence: *"To our knowledge, few works connect shelf-level visual observation to transaction mining in a single pipeline."*
3. Cite with `\cite{key}`, and check that the PDF compiles (green Recompile button).

## 7.3 Day 1-2: pipeline diagram (30 min)
Use **draw.io** (app.diagrams.net) or Excalidraw. Make boxes left to right:
`Shelf + camera (ArUco markers)` → `Rectify + preprocess (WB, bilateral, CLAHE)` → `YOLOv8 detection` →
`Before/after comparison` → `Basket (one-hot vector)` → splits into `Apriori → association rules` and
`KNN → next-item recommendation`. Add a small dashed box labelled "Classical change detection (colour diff + planogram)" parallel
to YOLO. Export as **PNG and PDF** and upload to Overleaf (Methodology 3.1).

## 7.4 Day 1: shelf plan (30 min)
Wait for P1's **class names** (posted on Day 1). Then:
1. **The shelf.** The easiest option is to buy or borrow a **3×3 cube organiser** (fabric or wooden cubby shelf), or build one from foam board /
   cardboard: 3 rows × 3 columns, each cell about **25×25 cm**. A plain, matte, light-coloured back works best (not glossy, and no
   pattern).
2. **Camera mount:** a phone tripod or a clamp mount, fixed about 1-1.5 m in front of the shelf so the whole grid and the
   markers are in frame. **It must not move between photos.** Mark the tripod position on the floor with tape.
3. **Lighting:** a fixed desk lamp or LED strip. Avoid windows, because daylight changes through the day.
4. **Markers:** print `tools/aruco_markers.pdf` at **100% scale** and check with a ruler that each is 6 cm. Stick them with the
   **centre of each marker on the outer corner of the 3×3 grid**: ID 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
5. **Products:** one of each of the **10 dataset classes**, small sizes are fine. Nine go on the shelf and one is spare for variety.
   Keep the receipt list for the paper.
6. Write the plan in a short doc for the group: shelf type, size, cost, who buys what, and when it will be ready.

## 7.5 Day 2: collect everyone's text and make the update slides (60 min)
1. Collect the 3-4 sentence summaries and figures from `IVP_Project_data/update1/P1…P3/`.
2. Paste each person's summary into the right Methodology subsection. Rough is fine; the owners will polish it later.
3. Make the slides (Google Slides, 7-8 slides, see Part 8).

## 7.6 Your deliverables for the update
- [ ] Overleaf draft: Intro (edited), Related Work (3 paragraphs plus BibTeX), Methodology headings with rough text
- [ ] Pipeline diagram
- [ ] Shelf plan plus product list
- [ ] Update slides

---

# Part 8 - The first teacher update (Day 2)

**Slides (P4 assembles, each person presents their own):**
1. **Title + problem.** "Beers and Diapers": connecting what customers pick up to what they buy together.
2. **Pipeline diagram.** The whole system on one slide.
3. **Syllabus fit.** **CNN** (YOLOv8) for product detection, **KNN** for next-item prediction, and Apriori as the
   data-mining step that measures correlations. *Ask the teacher whether Apriori is acceptable.* If not, we replace it with a
   pairwise lift matrix, and nothing else changes.
4. **P1:** dataset, training setup, mAP, confusion matrix, predictions.
5. **P2:** preprocessing steps, ArUco rectification, change detection plus first real-photo result, ablation chart.
6. **P3:** Apriori rules, **noise-sweep figure** (the key IVP→AI link), KNN table, Groceries rules.
7. **Status and next steps.** About 50% done. Next: build the shelf, record real sessions, label shelf photos, run the remaining experiments, write up results.
8. **Questions for the teacher.** Is Apriori OK? Target venue or format for the paper? Deadline?

**Be honest about demo vs real data.** Label which numbers are real (P1's detector, Groceries) and which are simulated
(planted-rule baskets), and say why simulation is needed (you can't collect thousands of baskets from a 3×3 shelf).

---

# Part 9 - After the update (phase 2)

| Step | Who | What |
|---|---|---|
| 1 | P4 (+ all) | Build the shelf, stick the markers, fix the camera and lighting |
| 2 | P4 + P2 | Photograph the full shelf ~150 times (different arrangements, some items missing, lamp on/off, slight camera moves). Keep these separate from the sessions |
| 3 | P4 (+ P1) | **Roboflow labelling:** app.roboflow.com → New Project → *Object Detection* → name `shelf_photos` → upload the ~150 photos → *Annotate* → draw a box around every product using **exactly the dataset class names** → *Generate* → split 70/20/10, **no augmentation** → Create. Split the labelling 4 ways (about 40 photos each) |
| 4 | P1 | Notebook 02 §5: set `SHELF_WORKSPACE/PROJECT/VERSION` → **Experiment 1** (public-only vs public + shelf) |
| 5 | P2 | Write the real `sessions/planogram.json` with class names and record **40-60 sessions** (before → take 1-3 items → after → restock), logging each in `sessions.csv`. Rotate who plays "customer" |
| 6 | P2 | Notebooks 01 and 03 on real sessions → **Experiment 3** real numbers, and the ablation on shelf images |
| 7 | P3 | Notebook 04 §5: recommendations for the real detected baskets; final Experiment 4/5 runs, averaged over 3 seeds |
| 8 | All | Results + Discussion sections (each writes their own), P4 edits the whole paper, then everyone proofreads |

---

# Part 10 - Reference

## 10.1 Git cheat sheet (every work session)
```bash
cd <path>/IVP_Project
git pull                                  # 1. ALWAYS first: get teammates' changes
# ... work ...
git status                                # 2. see what changed
git add notebooks/04_apriori_knn.ipynb    # 3. add ONLY your files
git commit -m "P3: short description"     # 4. commit
git push                                  # 5. share
```
- **`git pull` says conflict in a notebook you don't own:** throw away your copy with `git checkout -- notebooks/<file>.ipynb`, then `git pull` again.
- **Push rejected ("fetch first"):** run `git pull`, then `git push`.
- **Accidentally changed someone's notebook:** `git checkout -- notebooks/<file>.ipynb` before committing.

## 10.2 Where results end up
Everything is saved in `IVP_Project_data/results/`:
- `yolo_public_only.json`: detector metrics (P1)
- `dataset_class_distribution.png`, `training_curves_*.png` (P1)
- `ablation_preprocessing.csv/.png` (P2)
- `exp3_change_detection.csv`, `exp3_threshold_sweep.png`, `detected_baskets.csv` (P2)
- `exp4_noise_sweep.csv/.png`, `exp4_summary.json`, `exp5_knn.csv` (P3)
- `groceries_top_rules.csv`, `groceries_lift_heatmap.png`, `exp5b_knn_groceries.csv` (P3)

Training runs (weights, plots) are in `IVP_Project_data/runs/<experiment>_<machine>/`.

## 10.3 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` in VS Code | Kernel isn't `.venv`. Top right → Select Kernel → `.venv` |
| First cell prints `DATA_DIR: .../IVP_Project/data` instead of the Drive folder | `local_config.json` is missing or has the wrong path (Part 3.5). Check the path exists in Finder / Explorer |
| Colab: `requirements.txt not found` | `REPO_URL` empty in the first cell. P3 runs `tools/set_repo_url.py` and pushes |
| Colab: `git clone` fails (private repo) | Add the `GITHUB_TOKEN` secret with notebook access (Part 2.6 / 3.4) |
| Colab: Drive folder not found | You didn't add the shortcut to My Drive (Part 3.1) |
| Roboflow: `No Roboflow API key` | Add the `ROBOFLOW_API_KEY` secret (Colab) or put it in `local_config.json` |
| Roboflow: download fails | Wrong `RF_VERSION`. Check the Dataset tab on the Roboflow page |
| `CUDA out of memory` (LOQ) | Add `batch=8` to `detection.train(...)` |
| `torch.cuda.is_available()` is False on the LOQ | Reinstall PyTorch with the CUDA command (Part 3.3 step 5) *before* requirements, and update the NVIDIA driver |
| Training very slow on a laptop | Check the first cell's `Device:` line. `cpu` means the GPU isn't being used. Also make the Drive folder available offline |
| `Not all 4 markers visible` | Markers glossy, folded, cut off or too small. Reprint on matte paper and move the camera closer |
| Phone photos won't load | HEIC format. iPhone → Settings → Camera → Formats → Most Compatible |
| `PermissionError` on Windows activate | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (Part 3.3 step 2) |
| Anything else | Copy the **last 10 lines of the error** into the group chat, and say which notebook, which cell and which machine |
