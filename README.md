# Coursework 2 – Aria Gharachorlou

**Original Paper:**  
- KEATS link: https://keats.kcl.ac.uk/pluginfile.php/12677542/mod_resource/content/4/Hacohen2022.pdf  
- Public link: https://proceedings.mlr.press/v162/hacohen22a.html  



# Disclaimer Before Use 
* The notebooks directory has all seperate components available for use - for seameless execution, typiclust_B{values} are placed in the appropriate directory (TPCRP_Algorithm)
* For your *convenience*, I added a *unified_notebook* in root of the repository - please refer to this as the *intended point of entry* for the processing of graphs.
* To train the algorithms and generate results akin to mine, you can run the TPCRP_Algorithm.py or variants*


variants* : Modified_TPCRP_Algorithm, Supervised_TPCRP, Unsupervised_TPCRP


# Approximate Structure
* TPCRP_Algorithm : Is the Self-Supervised algorithm implementation of TPCRP
* Supervised_TPCRP : Is the Supervised algorithm implementation of TPCRP 
This provides the Fully supervised baseline for my Report
* Unsupervised_TPCRP : Is the Unsupervised algorithm implementation of TPCRP 
This provides the unsupervised baseline for my report 
There are no epochs stated in unsupervised - due to the lack of a training (optimisation) loop

* Uncertainity baselines such as Least Confidence, BALD, DBAL, etc... are implemented via the "Uncertainity_Baseline_Implementation.py"

---

# Running the Code and Interpreting Results

### 1. Create and activate a virtual environment
python3 -m venv env
source env/bin/activate



### 2. Install all required dependencies
pip3 install -r requirements.txt



### 3. Running the TPCRP algorithm (self-supervised)
Execute the main TPCRP script:
python TPCRP_Algorithm/Modified_MultiRoundTPCRP.py  // alternatively // please execute it via your IDE 
* If you wish to train the algorithm, please ensure to comment out "run_pipeline" and comment "generate_and_save_typiclust_selections" - whilst having the trained directory in the same directory as the script 

This will:

- Train the self-supervised encoder for 500 epochs  
- Extract features from CIFAR‑10  
- Run the multi‑round TypiClust selection algorithm  
- Save all outputs into a directory named `budget_results/`

In this repository, these directories were manually renamed to:

- `unmodified_results/`  
- `modified_results/`  

This allows the Jupyter notebooks to distinguish between the original TPCRP implementation and the modified version used

### 4. Running the uncertainty baselines and generating plots
To reproduce the baseline comparisons and generate the accuracy curves used, run:
python TPCRP_Algorithm/Uncertainty_Baseline_Implementation




This script:

- Loads CIFAR‑10  
- Trains a lightweight CNN for each uncertainty baseline  
- Evaluates BALD and DBAL using Monte‑Carlo dropout  
- Loads TypiClust selections from `.npy` files  
- Produces accuracy curves for all methods  
- Saves the final comparison plot as `baseline_comparison.png`

All plots and metrics used in the report are generated through this pipeline.

---

# Additional Details

### TPCRP_Algorithm/
Contains the full self-supervised implementation of TPCRP, including:

- ResNet‑18 encoder  
- NT‑Xent contrastive learning  
- Feature extraction  
- Multi‑round TypiClust selection (TPC‑RP)

Both supervised and unsupervised variants are included, following the structure described in the paper.

### unmodified_results/
Contains:

- SSL training loss  
- Extracted features  
- TypiClust selections  
- Plots and metrics for the original TPCRP algorithm  [ including Baseline_Comparisons ] 
- These results replicate the behaviour described in Hacohen et al. (2022)

### modified_results/
Contains:

- SSL loss  
- Modified selection indices  
- Plots and metrics for the modified TPCRP algorithm  
- All modifications are clearly commented in `Modified_MultiRoundTPCRP.py`

The modification uses a weighting parameter λ = 0.01 to balance typicality and diversity.

---

### Modification Summary

The original K‑Means clustering step is replaced with a DBSCAN‑based approach, combined with a diversity‑penalised scoring function:
score(i) = typicality(i) – λ * distance_to_selected(i)




Where:

- `typicality(i)` encourages selecting representative samples  
- `distance_to_selected(i)` penalises redundancy  
- `λ = 0.01` balances typicality and diversity  

### Rationale

DBSCAN identifies sparse and dense regions, clustering them together - these changes theoretically will improve the information of the samples and enhance the performance

All modified regions are clearly annotated in the code for ease of marking.

---





