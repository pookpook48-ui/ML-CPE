import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.select_dtypes(include=['number'])
    return X

if __name__ == "__main__":
    X = load_data('data-animal/animal_dataset.csv')
    print(len(X))