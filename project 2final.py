import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Create dataset
data = {
    "Glucose": [85,89,120,150,95,140,130,100,160,110,115,125,134,145,155,165,175,185,90,105],
    "BMI": [22,24,30,35,25,33,31,26,38,28,29,32,34,36,39,41,43,45,23,27],
    "Age": [25,30,45,50,28,48,52,35,47,60,40,41,55,71,20,90,31,19,9,39],
    "Diabetes": [0,0,1,1,0,1,1,0,1,0,0,1,1,1,1,1,1,1,0,0]
}

#Convert to DataFrame
df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

print("\nDataset Statistcs:")
print(df.describe())

# Features (input data)
X = df[['Glucose', 'BMI', 'Age']]

#Labels (output)
y=df['Diabetes']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Create AI model
model= DecisionTreeClassifier()

# Trin the model
model.fit(X_train,y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

# Show results
print("Predictions:")
print(predictions)

print("\nActual Values:")
print(y_test.values)

print("\nAccuracy:")
print(accuracy * 100, "%")

# Test with a new patient
new_patient = pd.DataFrame([[145,34,50]],
                           columns=["Glucose", "BMI", "Age"])

prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0]

print("\nNew Patient Prediction:")

if prediction == 1:
    print("Diabetic")
else:
    print("Non-Diabetic")

print("\nPrediction Probability:")
print(probability)


          