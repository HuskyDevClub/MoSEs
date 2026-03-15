<h1>MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds</h1>

> **Note:** This repository contains the official author implementation of the MoSEs paper ([Wu et al., 2025](https://aclanthology.org/2025.emnlp-main.294/)). The notebook `cse517_reproduction.ipynb` is created for a comprehensive reproducibility study and is **not part of the original paper code**. All other code and resources are provided by the paper authors. Minor optimizations have been made to the original code with the intent to preserve functionality while improving computational efficiency (e.g., parallelization of evaluation loops, caching mechanisms).
>
> **Original Repository:** [creator-xi/MoSEs](https://github.com/creator-xi/MoSEs)
>
> **About the Reproduction Notebook:** The `cse517_reproduction.ipynb` notebook is a reproducibility study that validates all four main claims from the original paper by orchestrating the complete MoSEs pipeline:
>
> **Reproduction Coverage:**
>
> - Tested across three score models: RoBERTa-base, Fast-DetectGPT, and LastDE
> - Eight datasets spanning news, debates, stories, comments, scientific articles, reviews
> - Automated dataset preparation and processing (both main and low-resource configurations)
> - Training Stylistics-Aware Router (SAR) on 8-class Stylistics Reference Repository
> - Conditional Threshold Estimator (CTE) evaluation and metric computation
> - Full ablation studies (SAR effectiveness and conditional feature importance)
> - Additional experiments: reference set size sensitivity analysis and text-length stratified evaluation
> - Colab-optimized with automatic result archiving and caching for efficiency
>

## 🖼️ Framework

The figure below provides an overview of the MoSEs detection framework. First, we build a Stylistics Reference Repository (SRR) annotated with conditional features and semantic embeddings. Then, for a text to be detected, we use a Stylistics-Aware Router (SAR) to activate the most relevant reference samples. Finally, the Conditional Threshold Estimator (CTE) uses these activated samples to dynamically determine a classification threshold and outputs the final prediction with a confidence probability.

![MoSEs Framework](images/framework.png)


---

## 🛠️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HuskyDevClub/MoSEs
    cd MoSEs
    ```

2.  **[Optional] Create a Conda environment and install dependencies:**

    This step is only needed if you plan to run scripts manually or locally. If you're using the reproduction notebook on Google Colab, you can skip this step as Colab handles the environment automatically.

    ```bash
    conda create -n moses python=3.12
    conda activate moses
    pip install -r requirements.txt
    ```

    Alternatively, if you don't use Conda, you can install dependencies directly with:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📓 Running the Reproduction Notebook

The `cse517_reproduction.ipynb` notebook provides the easiest way to reproduce all results from the paper. It orchestrates the entire pipeline in a single, self-contained workflow.

### Quick Start with the Notebook

**Option 1: Run on Google Colab (Recommended)**

Open the notebook directly in Colab by clicking the "Open in Colab" link at the top of the notebook, which will handle GPU allocation and environment setup automatically.

**Option 2: Run Locally**

1. Ensure you have completed the Setup section above (conda environment with dependencies installed)
2. Download or access the `cse517_reproduction.ipynb` notebook
3. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook cse517_reproduction.ipynb
   ```
4. Run all cells sequentially (Kernel → Run All, or manually step through cells)

### What the Notebook Does

The notebook automates the complete MoSEs pipeline:

1. **Data Staging**: Downloads and prepares all datasets (8 datasets, both main and low-resource configurations)
2. **SAR Training**: Trains the Stylistics-Aware Router on the full 8-class Stylistics Reference Repository
3. **CTE Evaluation**: Evaluates all three score models (RoBERTa-base, Fast-DetectGPT, LastDE) using the Conditional Threshold Estimator
4. **Results Generation**: Computes metrics and generates result tables matching the paper:
   - Main results (Table 1, 2, 3) with accuracy and F1-score across all datasets
   - Ablation studies (Table 4: SAR effectiveness, Table 5: conditional feature importance)
   - Additional analyses: reference set size sensitivity and text-length stratified evaluation
5. **Result Export**: Automatically archives and displays all results in a summary table

### Expected Output

Upon completion, the notebook will display:
- A comprehensive metrics table showing ~11% accuracy improvement over static thresholds
- Ablation study results confirming the gains from SAR and conditional features
- Statistical significance testing (paired t-tests) validating results match the original paper

All intermediate results and logs are saved to the `logs/` directory for further inspection.

---

## 🚀 Manual Reproduction (Advanced)

For those who prefer fine-grained control or want to understand each step in detail, this section provides step-by-step instructions for running the pipeline manually.

### Step 1: Prepare Datasets

First, we need to process the raw data. The `split_datasets.py` script reads the source CSV files (e.g., from `data/doc4split/`), calculates semantic embeddings, conditional features (`cond`), and a base discrimination score (`crit`) for each text.

The provided shell scripts in `examples/split_dataset/` automate this process. For example:
```bash
# This script processes the data using the "fast-detect-gpt" method for the 'crit' score.
bash examples/split_dataset/run_split_datasets_fast.sh
```
This will generate a set of JSON files in a new directory (e.g., `data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast/`), where each file corresponds to a specific dataset style (e.g., `cmv_dataset.json`, `xsum_dataset.json`).

### Step 2: Create Train/Test Splits

Next, we split the processed JSON files from the previous step into training and testing sets using `process_csv.py`. The training set will serve as the Stylistics Reference Repository (SRR).

The scripts in `examples/process_csv/` automate this split. For example:
```bash
# This script splits the data generated in the previous step into train and test sets.
bash examples/process_csv/run_process_csv_fast.sh
```
This creates two new directories, such as `..._fast_train` and `..._fast_test`, containing the final data splits.

### Step 3: Train the Stylistics-Aware Router (SAR)

The SAR model is trained to identify the style of an input text and retrieve relevant reference samples from the SRR.

To train the SAR model, run the script in `examples/run_sar/`:
```bash
# This command trains the SAR and saves the model weights and class names.
bash examples/run_sar/run_train.sh
```
This will save the trained model (`subcentroids_head_epochxx.pt`) and the style class names (`class_names.json`) into the `weights/` directory. These artifacts are required for the final evaluation step.

### Step 4: Run Evaluation with the Conditional Threshold Estimator (CTE)

Finally, we evaluate the full MoSEs framework. The `CTE.py` script takes a test file, uses the trained SAR to retrieve relevant data from the SRR (the `..._train` folder), and then trains and evaluates the Conditional Threshold Estimator on the fly for each sample in the test set.

The scripts in `examples/run_cte/` run this final evaluation. For example:
```bash
# This script evaluates the MoSEs framework on all test sets.
# Note: Please check and modify the SAR_PATH in the script to point to your trained model, e.g., subcentroids_head_epoch100.pt
bash examples/run_cte/run_cte_fast_detect.sh
```
The script will iterate through the test files, print the evaluation metrics (Accuracy, F1-score, etc.) for each, and save the detailed results to a JSON file in the `logs/` directory.

---

## ✍️ Citation

If you find this reproduction study or the original work helpful, please cite both the original paper and our reproducibility study:

**Original Paper:**
```bibtex
@inproceedings{wu2025moses,
  title={MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds},
  author={Wu, Junxi and Wang, Jinpeng and Liu, Zheng and Chen, Bin and Hu, Dongjian and Wu, Hao and Xia, Shu-Tao},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

**Reproduction Study:**
```bibtex
@misc{lin2026moses_reproduction,
  title={Reproducibility Study of MoSEs: Uncertainty-Aware AI-Generated Text Detection},
  author={Lin, Wynter and Huang, Ziqian and He, Yuxiang},
  year={2026},
  note={CSE 517 course project, University of Washington},
  url={https://github.com/HuskyDevClub/MoSEs}
}
```