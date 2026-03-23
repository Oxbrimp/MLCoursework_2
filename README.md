# Coursework 2 - Aria Gharachorlou
* Original Paper : https://keats.kcl.ac.uk/pluginfile.php/12677542/mod_resource/content/4/Hacohen2022.pdf
* Non-KEATS Paper Link : https://proceedings.mlr.press/v162/hacohen22a.html

* TPCRP_Algorithm : Is the self-supervised algorithm implementing TPCRP
* Supervised / Unsupervised TPCRP are variants as shown in the paper 

* Evaluation Metrics Directory : Contains diagram generation relevant to Task 2 

* Unmodified_results = Data, loss and plots for the unmodified TPCRP Algorithm, akin to Hacohen's paper
* modified_results = Data, loss and plots for the modified TPCRP Algorithm, using my intuition & modification
- For your convenience when marking, I've commented regions of improvement in the Modified_MultiRoundTPCRP script
- Lambda was set to 0.01 - for a balanced approach : balancing typicality & distance 



# Modified TPCRP Algorithm
* For the modification of my algorithm, I utilised a DBSCAN clustering algorithm with a diversity penalty
* The diversity penalty functions by taking the score(i) = typicality(i) - lambda (constant) * distance_to_selected(i)
- This ensures high typicality, high diversity and better coverage, which align well with DBSCAN's radius (core / edge ) and neighbour count design 