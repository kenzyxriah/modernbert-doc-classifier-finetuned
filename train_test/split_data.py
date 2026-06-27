import pandas as pd

def split_data():
    print("Loading original training_data.csv...")
    df = pd.read_csv("scratch/training_data.csv")
    
    print(f"Original dataset size: {len(df)}")
    
    # Split 80% train, 20% test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    train_df.to_csv("scratch/classifier_train.csv", index=False)
    test_df.to_csv("scratch/classifier_test.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    split_data()
