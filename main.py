import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
file_path = 'food_nutrition.csv'
data = pd.read_csv(file_path)
print("Shape of the dataset:")
print(data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head(5))
print("\nLast 5 rows of the dataset:")
print(data.tail(5))
print("\nBasic information about the dataset:")
print(data.info())
print("\nSummary statistics of the dataset:")
print(data.describe())
print("\nMissing values in the dataset:")
print(data.isnull().sum())
print("\nData Types of each column:")
print(data.dtypes)
print("\nCleaned dataset information:")
data.fillna(data.mean(), inplace=True)
print(data.isnull().sum())
print(data.info())
numeric_columns = data.select_dtypes(include='number').columns.drop('ID')
grouped_data = data.groupby('FoodGroup')[numeric_columns].mean()
print("\nMean values for each nutritional component by Food Group:")
print(grouped_data)
threshold_protein = 15
deficient_protein = data[data['Protein_g'] < threshold_protein]
print(f"\nFoods with protein deficiency (protein < {threshold_protein}):")
print(deficient_protein[['FoodGroup', 'ShortDescrip', 'Protein_g']])
corr_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Nutritional Components')
plt.show()
print("\nLINEAR REGRESSION MODEL")
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

X = data.drop(columns=['FoodGroup', 'ShortDescrip', 'Descrip', 'Protein_g'])
y = data['Protein_g']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='yellow')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
print("\nCLASSIFICATION")
nutrient_columns = [
    'Protein_g', 'Fat_g', 'Carb_g', 'Sugar_g', 'Fiber_g', 'VitA_mcg', 'VitB6_mg', 
    'VitB12_mcg', 'VitC_mg', 'VitE_mg', 'Folate_mcg', 'Niacin_mg', 'Riboflavin_mg', 
    'Thiamin_mg', 'Calcium_mg', 'Iron_mg', 'Magnesium_mg', 'Phosphorus_mg', 'Zinc_mg'
]
data['Deficiency'] = (data['VitA_mcg'] < data['VitA_USRDA']) | (data['Protein_g'] < 15)  
X_class = data[nutrient_columns]
y_class = data['Deficiency'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nCLUSTERING")
scaler = StandardScaler()
X_cluster = scaler.fit_transform(data[nutrient_columns])
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_cluster)

print("\nFirst 5 rows with cluster labels:")
print(data.head(5))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_cluster)
data['PC1'] = principal_components[:, 0]
data['PC2'] = principal_components[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=data, palette='viridis')
plt.title('KMeans Clusters (2D PCA Projection)')
plt.show()

iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(X_cluster)

plt.figure(figsize=(10, 6))
plt.scatter(data['Protein_g'], data['Fat_g'], c=data['Anomaly'], cmap='coolwarm')
plt.xlabel('Protein_g')
plt.ylabel('Fat_g')
plt.title('Anomalies in Food Items Based on Protein and Fat Content')
plt.colorbar(label='Anomaly')
plt.show()
def predict_deficiency():
    try:
        user_input = [float(entry.get()) for entry in entries]
        user_df = pd.DataFrame([user_input], columns=nutrient_columns)
        prediction = clf.predict(user_df)
        result = "Deficiency Detected" if prediction[0] == 1 else "No Deficiency"
        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")




root = Tk()
root.title("Nutrition Deficiency Predictor")


entries = []
for i, col in enumerate(nutrient_columns):
    Label(root, text=col).grid(row=i, column=0, padx=10, pady=5)
    entry_var = StringVar()
    entry = Entry(root, textvariable=entry_var)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)



predict_button = Button(root, text="Predict Deficiency", command=predict_deficiency)
predict_button.grid(row=len(nutrient_columns), columnspan=2, pady=20)


root.mainloop()



subset_data = data.head(20)

nutrient_columns = [
    'Protein_g', 'Fat_g', 'Carb_g', 'Sugar_g', 'Fiber_g', 'VitA_mcg', 'VitB6_mg', 
    'VitB12_mcg', 'VitC_mg', 'VitE_mg', 'Folate_mcg', 'Niacin_mg', 'Riboflavin_mg', 
    'Thiamin_mg', 'Calcium_mg', 'Iron_mg', 'Magnesium_mg', 'Phosphorus_mg', 'Zinc_mg'
]

df_bool = subset_data[nutrient_columns] > 0
print("Boolean DataFrame created.")

frequent_itemsets = apriori(df_bool, min_support=0.1, use_colnames=True)

if frequent_itemsets.empty:
    print("No frequent itemsets found with the given min_support threshold.")
else:
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    if rules.empty:
        print("No association rules found with the given min_threshold for confidence.")
    else:
        print("Association Rules:")
        print(rules)


# In[ ]:
