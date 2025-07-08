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
              precision    recall  f1-score   support

      Normal       0.73      0.82      0.77       234
   Pneumonia       0.88      0.82      0.85       390

    accuracy                           0.82       624
   macro avg       0.80      0.82      0.81       624
weighted avg       0.82      0.82      0.82       624

![image](https://github.com/user-attachments/assets/71750dff-db2d-406f-ac2a-2b8a8b95a1a3)

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

A: We fine-tuned **all layers** of InceptionV3 in Phase 2 to adapt the model to domain-specific features present in adult chest X-rays.

### Q2: **Describe how you split your data for fine-tuning.**

A: Pediatric data was used for Phase 1 training. Phase 2 fine-tuning reused the adult subset (split into train/val/test) with random oversampling and strong augmentation.

### Q3: **How is the evaluation metric (AUC) clinically relevant?**

A: AUC reflects the model's ability to balance sensitivity and specificity—critical in pneumonia screening to avoid missed cases (FN) or overdiagnosis (FP).

### Q4: **Example of a misclassified case and next step?**

A: One case was labeled Pneumonia but misclassified as Normal. Grad-CAM showed diffused attention with unclear opacity—likely due to low image contrast. Next, we will test contrast normalization and explore ensemble models.

### Q5: **Why is this model worth a clinician's attention?**

A: "This model offers fast, accurate pneumonia detection with interpretability via heatmaps."

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

**Developed by Pranav for PGIMER Research**
