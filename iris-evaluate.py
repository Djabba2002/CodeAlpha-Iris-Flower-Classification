import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Précision du modèle : {accuracy:.2f}\n")

print("🧾 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))