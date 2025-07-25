import json
import numpy as np

report = json.load(open("evaluation_results/classification_report.json"))
cm = np.load("evaluation_results/confusion_matrix.npy")
class_names = list(report.keys())[:-3]  # remove overall metrics

print("Overall accuracy:", report["accuracy"])
print("Macro F1:", report["macro avg"]["f1-score"])
print("Weighted F1:", report["weighted avg"]["f1-score"])

print("\nConfusion Matrix:")
print(cm)

for idx, name in enumerate(class_names):
    row = cm[idx]
    others = [(class_names[j], row[j]) for j in range(len(class_names)) if j != idx]
    others.sort(key=lambda x: x[1], reverse=True)
    top = others[0]
    print(f"{name} most confused with {top[0]} ({top[1]} images)")
