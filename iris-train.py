from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Modèle entraîné et sauvegardé dans model.pkl")