## Hospital Readmission Prediction Report

### Purpose
Predict whether patients will be readmitted within 30 days after discharge.

### Data and Preprocessing
The dataset has 9 columns, with no missing or duplicated entries. Preprocessing Steps are as follows:
- **Numerical Columns:** Left unchanged, except Logistic Regression uses scaling.
- **Categorical Columns:** One-hot encoding with drop_first=True.
- **Text Column (discharge_note):** Converted to binary indicators for key phrases (e.g. blood pressure, current medication etc) 
  
### Machine Learning Modeling
Models used: **Logistic Regression (LR), Random Forest (RF), XGBoost**  
- **Class weighting** applied to address imbalance.  
- **Model evaluation procedure:** Repeated 10 times with different random seeds. 
  - For each repetition:  
    1. Split dataset into 80% training / 20% test.  
    2. Perform 5-fold cross-validation on the 80% training set to tune hyperparameters.  
    3. Evaluate metrics on the held-out 20% test set.  
  - Final Reported metrics below are the mean and standard deviation across the 10 repetitions.

| model                 | roc_auc_mean | roc_auc_std | f1_mean  | f1_std   | prc_auc_mean | prc_auc_std |
|-----------------------|-------------|------------|----------|---------|--------------|------------|
| LR   | 0.5583    | 0.0622   | 0.3998 | 0.1445 | 0.4073     | 0.0503   |
| RF         | 0.5446    | 0.0592   | 0.3813 | 0.0953| 0.4396     | 0.0887   |
| XGBoost               | 0.4847    | 0.0691   | 0.3659 | 0.0866| 0.3612     | 0.0637   |

- **Important Features**: They are estimated using Shap values based on RF. The top 5 features are num_previous_admissions, age, current_medication_mentioned, length_of_stay and blood pressure_mentioned.

### Medical Note Extraction
Tested models: flan-t5-large (text generation), biomedical-ner-all (NER), roberta-base-squad2 (QA) and bart-large-finetuned-squadv1 (QA) 
- **bart-large-finetuned-squadv1** performed best.  
- Extraction of follow-up actions worked well; other categories inconsistent at low confidence. Results in note_extraction.csv.  
- Extraction features not included in ML models since the performance is not convincing.
   
### Future Work
- Upgrade to Llama-3 on GPU for better extraction.  
- Explore TF-IDF or Bayesian approaches.  
- Integrate external datasets (e.g., SVI, SDOH etc) to improve predictive power.
