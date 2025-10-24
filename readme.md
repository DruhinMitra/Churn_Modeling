# Customer Churn Prediction using K-Means Clustering

This project applies **K-Means Clustering** on the classic **Churn Modelling dataset** to group customers based on churn behavior.  
It includes automatic detection of the optimal number of clusters using the **Elbow Method** and evaluates the clustering performance against actual churn labels.

---

## 🚀 Features

- Uses **K-Means** from `scikit-learn` for unsupervised clustering  
- Automatically determines the **best number of clusters** using the **Elbow Method** via the `kneed` library  
- Evaluates approximate **accuracy** by mapping clusters to real labels  
- Modular class-based design for easy reuse (`Test` class with methods for cluster selection and model training)  

---

## 🧠 How It Works

1. The dataset is preprocessed (irrelevant columns are dropped, and categorical data is factorized).  
2. The Elbow Method determines the best number of clusters by minimizing inertia.  
3. The K-Means model fits the training data and predicts cluster labels for the test data.  
4. Each cluster is mapped to the most common actual label (`Exited`), allowing an **approximate accuracy** calculation.  

---

## 🧩 Project Structure

📦 churn_model_kmeans
┣ 📜 churn_model_test.py # Main script containing the Test class and K-Means implementation
┣ 📜 Churn_Modelling.csv # Dataset (should be placed in the same folder)
┣ 📜 README.md # Documentation



---

## ⚙️ Installation

Make sure you have Python 3.8+ and install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn kneed scipy

▶️ Usage

Place your Churn_Modelling.csv file in the same directory.

Run the script:

python churn_model_test.py

The output will print the approximate accuracy of the model.
🧰 Tech Stack

Python

NumPy

Pandas

Matplotlib

Scikit-learn

SciPy

Kneed

🧑‍💻 Author

Druhin Mitra
AI & Data Enthusiast


🏁 Future Improvements

Add confusion matrix and visual cluster comparison

Extend to other clustering methods (DBSCAN, Hierarchical)

Deploy model with Flask/FastAPI for online predictions

🪪 License

This project is licensed under the MIT License – feel free to use and modify it for learning or research.