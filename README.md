NutriLens: Intelligent Food Deficiency Predictor Using ML and Clustering

NutriLens is an end-to-end nutrition analytics project powered by machine learning and data science techniques. It enables analysis, prediction, clustering, and anomaly detection of food items based on their nutritional profiles. It also includes a GUI tool for real-time nutritional deficiency prediction based on user inputs.

Project Objectives

* Analyze nutritional data of food items.
* Predict protein content using regression models.
* Detect nutritional deficiencies using classification.
* Group food items by nutritional similarity using clustering.
* Identify anomalous (unusual) food items using Isolation Forest.
* Visualize correlations and patterns among nutrients.
* Perform association rule mining using the Apriori algorithm.
* Provide a user interface to predict deficiencies from custom input.

Technologies & Libraries

* Programming Language: Python
* ML Models: Linear Regression, Random Forest Classifier, KMeans, Isolation Forest
* Visualization: Matplotlib, Seaborn
* Data Manipulation: Pandas, NumPy
* Preprocessing: LabelEncoder, StandardScaler
* Dimensionality Reduction: PCA
* Frequent Pattern Mining: mlxtend (Apriori, Association Rules)
* GUI: Tkinter
* Metrics & Evaluation: Mean Squared Error, Classification Report

Features and Functionalities

1. Data Exploration and Cleaning

* Loads `food_nutrition.csv` dataset
* Shows dataset structure and summary
* Handles missing values using mean imputation

2. Protein Deficiency Detection

* Flags foods with protein < 15g

3. Linear Regression (Protein Prediction)

* Predicts protein using other nutrients
* Uses Mean Squared Error for evaluation
* Displays actual vs predicted graph

4. Classification (Deficiency Detection)

* Labels food as deficient based on low Vitamin A or Protein
* Trains Random Forest Classifier
* Shows precision, recall, and F1-score

5. Clustering with KMeans

* Groups foods by nutrient similarity
* Uses PCA for 2D visualization

6. Anomaly Detection

* Detects unusual foods using Isolation Forest
* Visualizes anomalies based on Protein and Fat content

7. Association Rule Mining

* Uses Apriori algorithm to find frequent nutrient sets
* Generates high-confidence association rules

8. Graphical User Interface (GUI)

* Built with Tkinter
* Accepts user inputs for nutrients
* Predicts deficiency based on trained model
* Displays result in a message box

Getting Started

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

2. Run the script:

```bash
python main.py
```

Ensure `food_nutrition.csv` is in the same folder as `main.py`.
