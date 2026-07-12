import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import _load_classifier, FEATURE_COLUMNS

def verify_semantics():
    df = pd.read_csv("data/training_features.csv")
    
    # Pick one of each class
    targets = {
        "transit": df[df["label"] == "transit"].iloc[0],
        "eclipsing_binary": df[df["label"] == "eclipsing_binary"].iloc[0],
        "blend": df[df["label"] == "blend"].iloc[0],
        "other": df[df["label"] == "other"].iloc[0]
    }
    
    clf = _load_classifier("models/exoplanet_classifier.joblib")
    
    print("\n--- SEMANTIC VERIFICATION ---")
    for true_label, row in targets.items():
        print(f"\n[TRUE LABEL: {true_label.upper()}] (kepoi_name={row.get('kepoi_name', 'Unknown')})")
        
        # Prepare row
        feature_row = pd.DataFrame([row])
        for col in clf.feature_names_:
            if col not in feature_row.columns:
                feature_row[col] = 0.0
        aligned_row = feature_row[clf.feature_names_]
        
        mean_proba, std_proba = clf.predict_proba(aligned_row)
        classes = clf.classes_
        
        # Merge into dict
        probs = {cls: p for cls, p in zip(classes, mean_proba[0])}
        predicted_class = max(probs, key=probs.get)
        
        print(f"Predicted: {predicted_class.upper()} (p={probs[predicted_class]:.4f})")
        for k, v in probs.items():
            print(f"  {k}: {v:.4f}")
            
if __name__ == "__main__":
    verify_semantics()
