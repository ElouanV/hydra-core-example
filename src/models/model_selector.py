import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

class ModelSelector:
    """Model selector class to get model based on model name
    """
    def __init__(self, config) -> None:
        """Initialize ModelSelector class

        :param config: Configuration object
        :type config: object
        """
        self.config = config

    def get_model(self):
        """ Get model based on model name

        :raises ValueError: Raise ValueError if model name is invalid
        :return: sklearn-like model
        :rtype: object
        """
        if self.config.models.name == "logistic_regression":
            return self.get_logistic_regression()
        elif self.config.models.name == "random_forest":
            return self.get_random_forest()
        elif self.config.models.name == "xgboost":
            return self.get_xgboost()
        else:
            raise ValueError("Invalid model name")
        
    def get_logistic_regression(self):
        """Get logistic regression model

        :return: Logistic regression model
        :rtype: object
        """
        assert self.config.datasets.task == "classification", "Logistic regression is only for classification task"
        penality = self.config.models.param.penality
        C = self.config.models.param.C
        return LogisticRegression(penalty=penality, C=C, max_iter=1000)
    
    def get_random_forest(self):
        """Get random forest model

        :return: Random forest model
        :rtype: object 
        """
        n_estimators = self.config.models.param.n_estimators
        max_depth = self.config.models.param.max_depth
        if self.config.datasets.task == "classification":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        else: 
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    
    def get_xgboost(self):
        """Get xgboost model

        :raises ValueError: Raise ValueError if task name is invalid
        :return: XGBoost model
        :rtype: object
        """
        if self.config.datasets.task == "classification":
            num_estimators = self.config.models.param.n_estimators
            max_depth = self.config.models.param.max_depth
            return XGBClassifier(n_estimators=num_estimators, max_depth=max_depth)
        elif self.config.datasets.task == "regression":
            num_estimators = self.config.models.param.n_estimators
            max_depth = self.config.models.param.max_depth
            return XGBRegressor(n_estimators=num_estimators, max_depth=max_depth)
        else:
            raise ValueError("Invalid task name")