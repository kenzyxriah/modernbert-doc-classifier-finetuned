import sys
import os
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.ml.document_classifier.classifier import DocumentClassifier
from lib.ml.pipeline import get_cached_pipeline

def evaluate():
    print("Loading test dataset ...")
    df = pd.read_csv("scratch/classifier_test.csv").dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} test examples.")
    print("\nLoading trained model from C:/Users/ADMIN/Downloads/custom_classifier1...")
    
    try:
        pipe = get_cached_pipeline("text-classification", r"C:\Users\ADMIN\Downloads\custom_classifier1")
        classifier = DocumentClassifier(pipe)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    dummy_definitions = {"Public": "", "Internal": "", "Confidential": "", "Restricted": ""}
    
    correct = 0
    total = len(df)
    
    print("\nRunning evaluation on unseen data...")
    
    for i, row in df.iterrows():
        text = str(row["text"])
        true_label = str(row["label"])
        
        result = classifier.predict(text, dummy_definitions)
        predicted_label = result["category"]
        print(f'Model Predicted: {predicted_label} with confidence {result["confidence"]}')
        print(f'Actual Prediction: {true_label}')
        if predicted_label.lower() == true_label.lower():
            correct += 1
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total}...")
            
    accuracy = (correct / total) * 100
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total Tested: {total}")
    print(f"Total Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 90:
        print("Model is absolutely crushing it! Definitely ready to push!")
    elif accuracy >= 80:
        print("Model looks very solid! Good to push.")
    else:
        print("Accuracy is a bit low. You might want to review the dataset.")

if __name__ == "__main__":
    evaluate()
