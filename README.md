# 🔊 Sonar Rock vs Mine Classification using Logistic Regression

This project builds a machine learning model to classify sonar signals as either **Rock (R)** or **Mine (M)** based on 60 numerical features that represent energy in different frequency bands of the sonar signal. It demonstrates a practical use case of **Logistic Regression** for **binary classification** and is ideal for beginners learning machine learning workflows using real-world data.

---

## 📁 Dataset

- **Source**:(https://www.kaggle.com/datasets/mattcarter865/mines-vs-rocks)
- **Description**: This dataset contains 208 instances, each with 60 floating-point attributes derived from sonar signals bounced off various surfaces.
- **Target**:  
  - `R` – Signal reflected by a Rock  
  - `M` – Signal reflected by a Mine  

---

## 📦 Technologies Used

| Tool/Library | Purpose |
|--------------|---------|
| **Python** | Programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations and reshaping |
| **scikit-learn** | Model building, evaluation, and prediction |

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
The dataset is loaded using Pandas, with `header=None` since the file has no predefined column names.

### 2️⃣ Preprocessing
- Input (`X`) = 60 signal features (columns 0 to 59)
- Output (`Y`) = Target label (column 60)

### 3️⃣ Train-Test Split
- The dataset is split using `train_test_split()` from scikit-learn
- `test_size=0.1` → 90% training, 10% testing
- `stratify=Y` → Ensures both classes (Rock, Mine) are proportionally represented in both sets

### 4️⃣ Model Training
- A **Logistic Regression** model is created using `LogisticRegression()`
- Fitted to the training data using `.fit(X_train, Y_train)`

### 5️⃣ Accuracy Evaluation
- Predictions are made for both train and test datasets
- Accuracy is calculated using `accuracy_score()`

### 6️⃣ Custom Prediction
- A new set of 60 features is passed to the model
- The model predicts whether it's a rock or a mine

---

## 📊 Code Snippet

```python
input_data = (0.0453, 0.0523, 0.0843, 0.0689, ..., 0.0052, 0.0044)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)
print(prediction)
