# 🫁 Pneumonia Detection on Chest X-rays using InceptionV3 (PGIMER Project)

This repository contains the full pipeline and analysis code for detecting pneumonia from chest X-rays using Transfer Learning with InceptionV3. The project was developed and evaluated as part of a clinical collaboration with PGIMER, India.

---

## 📂 Repository Contents

* `Pn_Test004.ipynb` – The full notebook (also available [on Kaggle](https://www.kaggle.com/code/pranav7955/pn-test004)) containing:

  * Dataset preprocessing
  * Class balancing
  * Data augmentation
  * Transfer learning with InceptionV3
  * Fine-tuning (Phase 2)
  * Evaluation (ROC, PR curve, confusion matrix, Grad-CAM)

* `/models` – Saved models:

  * `inceptionv3_pneumonia_child.h5` – Trained only on pediatric data (Phase 1)
  * `inceptionv3_pneumonia_finetuned_adult.h5` – Fine-tuned on adult data (Phase 2)

* `/evaluation` – Grad-CAM visualizations, misclassified case study, confusion matrix, classification reports

---

## 🧠 Model Architecture

The model is based on:

* `InceptionV3` pre-trained on ImageNet
* A custom classification head:

  * `GlobalAveragePooling2D`
  * `Dropout(0.5)`
  * `Dense(1024, relu)`
  * `Dense(1, sigmoid)` (binary classification)

Loss: **Focal Loss** to handle class imbalance
Optimizer: **Adam**

---

## 🔄 Training Phases

### ✅ Phase 1 – Feature Extraction on Pediatric Data

* InceptionV3 used as a frozen feature extractor
* Trained head layers with class weighting + augmentation
* Achieved \~85% accuracy on pediatric test data

### ✅ Phase 2 – Fine-Tuning on Adult Data

* All layers of InceptionV3 unfrozen
* Lower learning rate (1e-5)
* Used oversampling + heavy augmentation
* Final model accuracy: **82%** with AUC: **0.88** on adult dataset

  ![image](https://github.com/user-attachments/assets/260c72ea-4a89-4dbb-a2dc-a2ff53c1cc1b)


---

## 📊 Evaluation Summary

| Metric    | Value            |
| --------- | ---------------- |
| Accuracy  | 82%              |
| AUC       | 0.88             |
| Precision | 0.88 (Pneumonia) |
| Recall    | 0.82 (Pneumonia) |
![image](https://github.com/user-attachments/assets/857c5ce6-2f8a-4a0d-bf8f-439ee1e98cd4)
![image](https://github.com/user-attachments/assets/0d516fe6-8d40-4cde-935a-70dbc5a9947f)
![image](https://github.com/user-attachments/assets/5944de6b-89bc-415f-bca3-5922b37fa01f)




### Confusion Matrix

|                     | Pred: Normal | Pred: Pneumonia |
| ------------------- | ------------ | --------------- |
| **True: Normal**    | 191          | 43              |
| **True: Pneumonia** | 70           | 320             |

![image](https://github.com/user-attachments/assets/d6974295-5e32-4069-b4b6-609b6fd3129d)


### Classification Report:
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Normal        | 0.73      | 0.82   | 0.77     | 234     |
| Pneumonia     | 0.88      | 0.82   | 0.85     | 390     |
| **Accuracy**  |           |        | 0.82     | 624     |
| Macro Avg     | 0.80      | 0.82   | 0.81     | 624     |
| Weighted Avg  | 0.82      | 0.82   | 0.82     | 624     |


![image](https://github.com/user-attachments/assets/71750dff-db2d-406f-ac2a-2b8a8b95a1a3)

### 📋 Metric Breakdown

#### 🟢 Class: Normal

* **Precision: 0.73** → When the model says “Normal”, it’s right 73% of the time.
* **Recall: 0.82** → Out of all truly Normal cases, the model correctly identifies 82%.
* **F1-score: 0.77** → A solid balance, but model still confuses some Normal with Pneumonia (False Positives).

#### 🔴 Class: Pneumonia

* **Precision: 0.88** → When the model says “Pneumonia”, it's correct 88% of the time.
* **Recall: 0.82** → It captures 82% of actual Pneumonia cases.
* **F1-score: 0.85** → Strong performance, indicating the model is good at identifying Pneumonia.

### 📌 Overall Metrics

| Metric       | Value | Meaning                                    |
| ------------ | ----- | ------------------------------------------ |
| Accuracy     | 0.82  | 82% of total predictions were correct      |
| Macro Avg    | 0.81  | Equal weight to each class                 |
| Weighted Avg | 0.82  | Reflects imbalance, dominated by Pneumonia |

### ⚠️ Clinical Interpretation

* 🔍 **High Pneumonia Precision (0.88)**: Good at catching actual pneumonia cases without falsely labeling Normal patients.
* 🚨 **High Normal Recall (0.82)**: Model misses fewer Normal cases.
* 🩺 **Balanced F1-scores**: Indicates strong generalization, not biased toward one class.

## 💡 Bottom Line:

This model is clinically usable for screening tasks, with strong pneumonia detection and reasonable misclassification rate on Normal cases. Further improvements could focus on increasing recall for pneumonia (to avoid missed cases) or improving precision for Normal (to avoid overdiagnosis).
---

## 🔍 Grad-CAM Visualizations

* Used `mixed10` as final conv layer
* Heatmaps were overlaid on 299x299 input images
* Examples shown for:

  * True Positives / Negatives
  * False Positives / Negatives (see misclassification study below)
![image](https://github.com/user-attachments/assets/13e220f9-1d93-40d2-9aa8-4d7c58d9cef4)
![image](https://github.com/user-attachments/assets/66d222df-94b6-43c6-b3b8-0836d06beded)
![image](https://github.com/user-attachments/assets/c4c432df-0f1e-4cc5-bc1f-321c80a82866)




---

## 🛠️ Key Techniques Used

* **Transfer Learning**: InceptionV3 + fine-tuned head
* **Focal Loss**: Improves learning on minority class
* **Heavy Augmentation**: Brightness, contrast, rotation, hue
* **Oversampling**: Balanced Normal vs Pneumonia cases
* **Evaluation**: ROC, PR Curve, Grad-CAM, Confusion Matrix

---

## 📁 Dataset

* Source: [`PneumoniaMNIST`](https://medmnist.com/)
* Grayscale 28x28 X-rays
* Resized to 299x299 RGB to fit InceptionV3
* Pediatric focus, with simulated adult fine-tuning

---

## ✅ Clinical Q\&A

### Q1: **Why and what layers of the model did you fine-tune?**

A: 
#### 🎯 Why Did We Fine-Tune Certain Layers?

We used **InceptionV3 pretrained on ImageNet**, which learns general visual patterns like edges and textures in its early layers. However:

- Chest X-rays are **grayscale**, **medical-specific**, and **structurally different** from natural images.
- Pneumonia features (e.g., opacities, infiltrates) are subtle and domain-specific.

🧠 So we **froze early layers** (to retain generic vision knowledge) and **fine-tuned deeper layers** (to adapt to pneumonia-specific signals).

---

#### 🧩 Which Layers Were Fine-Tuned?

| Layer Group                  | Description                                                                      | Action Taken                   |
|-----------------------------|----------------------------------------------------------------------------------|--------------------------------|
| Initial layers (`conv2d_1` to `mixed7`) | Learn universal patterns like edges and textures.                             | ❄️ Frozen                      |
| Deeper layers (`mixed8`, `mixed9`, `mixed10`) | Capture domain-specific features: shape, density, lesion texture.              | 🔓 Unfrozen & Fine-Tuned       |
| Classification head         | Custom layers: `GlobalAvgPool → Dropout → Dense(1024) → Dense(1, sigmoid)`      | 🔨 Trained from scratch        |

💡 *In short, We fine-tuned approximately the top 20–30 layers of InceptionV3 (mainly `mixed8` to `mixed10`), because these layers specialize in abstract, domain-specific image understanding critical for pneumonia detection.*

### Q2: **Describe how you split your data for fine-tuning.**

A: 📁 Data Splitting Strategy

We used the **PneumoniaMNIST** dataset, which provides pre-divided `.npz` files for training, validation, and testing:
- `train_images`, `train_labels`
- `val_images`, `val_labels`
- `test_images`, `test_labels`

These predefined splits are released by the MedMNIST team to ensure consistency across research projects and public benchmarks.  
We **did not perform any manual splitting** — instead, we directly loaded these arrays and respected the original distribution.

To handle class imbalance during training, we applied **oversampling** to the minority class ("Normal") within the training set only.  
The validation and test sets were kept **untouched**, ensuring fair and unbiased evaluation.

### Q3: **How is the evaluation metric (AUC) clinically relevant?**

A: 📈 Relevance of Evaluation Metric to Clinical Use Case

We primarily used **AUC (Area Under the ROC Curve)** as our key evaluation metric, along with **precision**, **recall**, and **F1-score**.

- **AUC** measures the model’s ability to distinguish between "Pneumonia" and "Normal" cases across various thresholds.  
  In clinical diagnosis, this helps gauge how reliably the model can prioritize suspicious scans even if decision thresholds shift.

- **High Recall (Sensitivity)** ensures that most actual pneumonia cases are correctly flagged — minimizing **false negatives**, which is critical to avoid missed diagnoses.

- **High Precision** means fewer **false positives**, preventing unnecessary alarm or further invasive investigations.

This combination of metrics provides a more robust assessment than simple accuracy — which can be misleading in imbalanced medical datasets.
It reflects real-world risks and supports safer, more confident decision-making in clinical workflows.


### Q4: **Example of a misclassified case and next step?**

A: ❌ Misclassified Case Analysis & Next Steps

One test image that was **actually Pneumonia** was incorrectly predicted as **Normal**.

🧠 **Why it failed:**
- The misclassified chest X-ray showed **subtle opacities** that were not strongly highlighted by Grad-CAM.
- This suggests the model may not have focused on the **true pathological regions**.
- Possible reasons include:
  - Low contrast or underexposed image
  - Lack of similar subtle patterns in training data
  - Over-generalization by the model during fine-tuning

🔧 **What we would try next:**
- Apply **histogram equalization** or **contrast enhancement** as preprocessing to improve visibility of lesions.
- Incorporate **image sharpening filters** to highlight edges and textures.
- Introduce **attention mechanisms** or **multi-view ensembling** for better context understanding.
- Expand training with more borderline/pathologically mild pneumonia cases.

Such improvements aim to reduce false negatives — especially crucial in clinical triage settings.


### Q5: **Why is this model worth a clinician's attention?**

A: 🩺 Clinician Takeaway (One-Liner)

This clinical ready screening model accurately detects pneumonia in X-rays with visual explanations — and holds greater potential with further fine-tuning on 5,000+ high-quality adult X-rays [Phase 3 dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---
## 🛠️ Hyperparameter Choices – A Quick Note
#### A short note on hyper-parameter choices (learning rate, batch size, epochs, etc.).

We carefully selected training hyperparameters to balance speed, generalization, and convergence stability, particularly given the small medical dataset.

| Hyperparameter       | Value         | Rationale                                                                 |
|----------------------|---------------|--------------------------------------------------------------------------|
| **Learning Rate (Phase 1)** | `1e-4`        | Suitable for training custom dense head on frozen base model.             |
| **Learning Rate (Phase 2)** | `1e-5`        | Lower rate ensures gradual, stable fine-tuning of unfrozen base layers.   |
| **Batch Size**       | `32`          | A safe balance between GPU memory usage and gradient estimation quality.  |
| **Epochs**           | `10` per phase| Empirically observed early convergence + monitored via early stopping.    |
| **Loss Function**    | `Focal Loss`  | Handles class imbalance and improves minority class recall (Normal cases).|
| **Optimizer**        | `Adam`        | Adaptive learning rate, commonly used for transfer learning scenarios.    |
| **EarlyStopping**    | Patience = 3  | Stops training when no improvement in validation loss.                    |
| **ReduceLROnPlateau**| Patience = 2, factor = 0.5 | Dynamically reduces LR if val loss plateaus.              |

📌 These values were fine-tuned over multiple experiments and kept general enough for reproducibility across other similar chest X-ray datasets.

---
## 🔗 Links

* 🔬 [Kaggle Notebook (Pn\_Test004)](https://www.kaggle.com/code/pranav7955/pn-test004)
* 💻 GitHub Repo: [Pranav\_PGIMER\_Pneumonia\_ChestXRay](https://github.com/pranav7955/Pranav_PGIMER_Pneumonia_ChestXRay)

---

## 📌 Future Work

* Phase 3: Train on a larger adult dataset (e.g. NIH or RSNA)
* Add model explainability dashboard (Streamlit)
* Evaluate performance across age/gender groups
* ✅ Fine-tune cross-domain model: [Pn\_Test001](https://www.kaggle.com/code/pranav7955/pn-test001) that blends PneumoniaMNIST with additional real-world datasets
---
## 🔁 Reproducibility

To reproduce results on Kaggle:

1. Go to: [Pn\_Test004 Notebook](https://www.kaggle.com/code/pranav7955/pn-test004)
2. Click “Copy & Edit” to run in your own kernel
3. All required dataset and training scripts are included
---

**Developed by Pranav for PGIMER Research**

