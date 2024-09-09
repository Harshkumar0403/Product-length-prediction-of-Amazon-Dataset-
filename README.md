# Product-length-prediction-of-Amazon-Dataset-

### **Objective**
This project aims to build a machine learning model that predicts the **product length** based on catalog metadata. Product length is a critical attribute for efficient packaging, storage, and customer satisfaction. The dataset comprises product details like title, description, bullet points, product type ID, and product length, spanning a massive 2.2 million records. The ultimate goal is to develop an accurate model that can predict product length to save time and improve operational efficiency in warehouses.

---

## **Prerequisite Knowledge**
To understand and work on this project, the following prerequisites are essential:
- **Python Programming**: Knowledge of Python for data preprocessing and model training.
- **Machine Learning Concepts**: Basic understanding of supervised learning, regression models, and hyperparameter tuning.
- **NLP (Natural Language Processing)**: Familiarity with text preprocessing techniques such as stemming, tokenization, and word embeddings.
- **Libraries**: Experience with libraries such as Pandas, NumPy, Gensim (for word embeddings), Scikit-learn, and LightGBM.

---

## **Approach**

### **1. Data Understanding:**
The dataset consists of 6 columns:
- **Product ID**: A unique identifier for each product (ignored during training).
- **Product Title, Description, and Bullet Points**: These are textual columns that describe the product.
- **Product Type ID**: An integer that represents the type of product.
- **Product Length**: The target variable (continuous), which represents the length of the product.

### **2. Sampling Data**:
Given the large dataset size, we chose to work with a 5% sample of the data to reduce resource consumption and facilitate quicker experimentation. This sampling still ensures a meaningful subset for model training and evaluation.

---

## **Text Preprocessing**

1. **Combining Text Columns**: 
   - A new column `TOTAL_SENTENCE` was created by concatenating the `Title`, `Description`, and `Bullet Points` columns for each product.
   
2. **Text Normalization**:
   - Converted all text to **lowercase** for consistency.
   - Applied the **Snowball Stemmer** to reduce words to their base forms.

3. **Tokenization and Word Embeddings**:
   - Tokenized the text using **Gensim’s Word2Vec**, creating word embeddings for each word in the corpus.
   - Averaged the embeddings for each product to represent the product’s textual data numerically.

---

## **Methods**

### **1. Feature Engineering:**
- The dataset was reduced to 4 columns: `PRODUCT_TYPE_ID`, `PRODUCT_LENGTH`, and the numeric representation of the `TOTAL_SENTENCE`.
- The `PRODUCT_TYPE_ID` was used directly, while the `PRODUCT_LENGTH` was the target for regression.

### **2. Model Selection:**
We experimented with several regression models to find the best one for this task:

1. **Decision Tree Regressor**:
   - Baseline model with minimal hyperparameter tuning.
   - Results: High training error and validation error, indicating overfitting.

2. **Random Forest Regressor**:
   - An ensemble of decision trees with different hyperparameters (number of estimators, max depth, etc.).
   - Better performance than decision tree but still struggled with validation error.

3. **LightGBM Regressor**:
   - A gradient boosting model designed for fast training and high accuracy with large datasets.
   - Tuned hyperparameters such as `num_leaves`, `max_depth`, `learning_rate`, and `n_estimators`.
   - Achieved the best performance compared to previous models.

---

## **Model Training and Visualizing Performance**

### **1. Decision Tree:**
   - **Training Error**: 636.58
   - **Validation Error**: 636.85
   - **RMSE**: 676.51 (baseline)

   The decision tree showed clear signs of overfitting, with nearly identical training and validation errors, indicating a lack of generalization.

### **2. Random Forest:**
   - **Training Error (TE)**: 632.43 (max_depth=5)
   - **Validation Error (VE)**: 624.48
   - The model reduced the training error but still had a high validation error, suggesting some overfitting.

### **3. LightGBM**:
   - **TE**: 506.33 (after hyperparameter tuning)
   - **VE**: 616.19
   - Best tuned configuration: 
     - `learning_rate`: 0.005
     - `max_depth`: 8
     - `num_leaves`: 128
     - `n_estimators`: 1000

   LightGBM achieved the best overall balance between training and validation error, although validation error still remained higher than training, likely due to noise and complexity in the data.

---

## **Summary of Results**
| Model                 | Train RMSE | Validation RMSE |
|-----------------------|------------|-----------------|
| **Baseline (mean)**    | 676.51     | 662.17          |
| **Decision Tree**      | 636.58     | 636.85          |
| **Random Forest**      | 632.43     | 624.48          |
| **LightGBM** (tuned)   | 506.33     | 616.19          |

### **Best Model**:
- **LightGBM** was the most effective model in this task, with its ability to handle complex patterns in the data.
- Despite the better results, there is still a performance gap between training and validation, indicating room for further tuning and noise reduction.

---

## **Conclusion**
The project successfully built models to predict product length based on catalog metadata, with LightGBM outperforming Decision Tree and Random Forest in terms of error metrics. However, due to data noise and possible overfitting, further improvements can be made through more advanced preprocessing, regularization, and possibly incorporating more sophisticated models or text representations like **BERT**.

---

## **Future Work**
- Explore other NLP techniques like **TF-IDF**, **BERT embeddings**, or **LSTM models** to enhance the representation of textual features.
- Experiment with regularization techniques to further mitigate overfitting.
- Implement additional feature engineering to capture more relationships between `PRODUCT_TYPE_ID` and product length.

---

