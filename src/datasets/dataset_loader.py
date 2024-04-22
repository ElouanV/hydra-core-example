import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from sklearn.datasets import fetch_california_housing


class DatasetLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        if self.config.datasets.name == "iris":
            return self.load_iris()
        elif self.config.datasets.name == "wine":
            return self.load_wine()
        elif self.config.datasets.name == "california":
            return self.load_california_housing()
        else:
            raise ValueError("Invalid dataset name")
        
    def load_iris(self):
        iris = load_iris()
        return train_test_split(iris.data, iris.target, test_size=self.config.datasets.test_size, random_state=self.config.random_state)
    
    def load_wine(self):
        wine = load_wine()
        return train_test_split(wine.data, wine.target, test_size=self.config.datasets.test_size, random_state=self.config.random_state)
    
    def load_california_housing(self):
        data = fetch_california_housing()
        return train_test_split(data.data, data.target, test_size=self.config.datasets.test_size, random_state=self.config.random_state)
    
