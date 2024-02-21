from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dp import x_y_dataset


sns.set_theme()

x, y = x_y_dataset()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=31)


random_forest_model = RandomForestClassifier(n_estimators=100)

random_forest_model.fit(x_train, y_train)


y_pred = random_forest_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Classification Report:", report)
print(f"Accuracy: {accuracy:.4f}")


sns.heatmap(confusion, annot=True, cmap="Blues")
plt.savefig('confusion_matrix.png')
