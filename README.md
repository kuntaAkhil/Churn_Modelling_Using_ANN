# Customer Churn Prediction using Artificial Neural Network (ANN)

## Project Overview

This project predicts whether a customer will leave a bank (Customer Churn Prediction) using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The dataset contains customer information such as credit score, geography, gender, age, balance, salary, and other banking details. The model learns patterns from the data and predicts whether a customer is likely to churn.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Dataset

Dataset used:
- **Churn_Modelling.csv**

Features include:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Target:
- Exited (0 = Customer stays, 1 = Customer leaves)

---

## Project Workflow

### 1. Data Preprocessing

- Loaded the dataset using Pandas.
- Selected input features and target variable.
- Converted categorical columns (Geography and Gender) using One-Hot Encoding.
- Split the dataset into Training and Testing sets.
- Standardized the feature values using StandardScaler.

---

### 2. Building the ANN

Created a Sequential Neural Network with:

- Input Layer
- Hidden Layer 1 - 11 neurons (ReLU)
- Hidden Layer 2 - 7 neurons (ReLU)
- Hidden Layer 3 - 6 neurons (ReLU)
- Output Layer - 1 neuron (Sigmoid)

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

Evaluation Metric:
- Accuracy

---

### 3. Training the Model

The model was trained using:

- 1000 maximum epochs
- Batch size = 10
- Validation Split = 33%

Early Stopping was used to stop training automatically when the validation loss stopped improving.

---

### 4. Model Evaluation

After training:

- Predicted customer churn on the test dataset.
- Converted probabilities into binary predictions using a threshold of 0.5.
- Evaluated the model using:
  - Confusion Matrix
  - Accuracy Score

---

## Training Graphs

The project plots:

- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

These graphs help understand how well the model is learning and whether it is overfitting.

---

## Libraries Used

```python
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

---

## How to Run

1. Clone this repository

```bash
git clone <your-repository-link>
```

2. Install the required packages

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

3. Place the dataset:

```
Churn_Modelling.csv
```

inside the project folder.

4. Run

```bash
python ANN.py
```

---

## Learning Outcomes

Through this project, I learned:

- Data preprocessing using Pandas
- One-Hot Encoding
- Feature Scaling
- Building an Artificial Neural Network
- Using TensorFlow and Keras
- Early Stopping
- Binary Classification
- Model Evaluation using Accuracy and Confusion Matrix
- Visualizing training performance

---

## Future Improvements

- Hyperparameter tuning
- Dropout regularization
- Cross Validation
- Saving and loading trained models
- Deploying the model using Flask or FastAPI
- Building a simple web interface for predictions

---

## Author

**Akhileshwar Kunta**

Learning Deep Learning and AI with hands-on implementation using TensorFlow and Python.