import pandas as pd
import numpy as np
import os 
import sys 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




def main():
    df = pd.read_csv("Stars.csv")
    print(df.head())
    print("Shape:", df.shape)

    print("DF, is null", df.isnull())

    print("Null values per column:", df.isnull().sum())

    #train test split, save files
    test = df.sample(frac=0.2, random_state=42)
    train = df.drop(test.index)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()