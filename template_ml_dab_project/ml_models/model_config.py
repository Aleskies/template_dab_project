"""
Model Configuration
Centralized configuration for ML models and training
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import os


@dataclass
class ModelConfig:
    """Configuration for ML model training and deployment"""
    
    # Model metadata
    model_name: str = "my_ml_model"
    model_version: str = "1.0.0"
    description: str = "ML Model Template"
    
    # Data configuration
    catalog: str = "workspace"
    schema: str = os.getenv("USER", "default")
    train_table: str = "train_data"
    test_table: str = "test_data"
    feature_table: str = "features"
    
    # Feature configuration
    target_column: str = "target"
    feature_columns: List[str] = field(default_factory=lambda: [
        "feature1",
        "feature2",
        "feature3"
    ])
    
    # Training configuration
    test_size: float = 0.2
    random_state: int = 42
    
    # Model hyperparameters (ejemplo con RandomForest)
    model_type: str = "random_forest"  # Options: random_forest, xgboost, lightgbm
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42
    })
    
    # MLflow configuration
    experiment_name: str = "/Shared/ml_experiments"
    model_registry_name: str = "my_ml_model"
    model_alias: str = "Production"  # Alias for model versioning
    
    # Unity Catalog paths
    @property
    def train_table_path(self) -> str:
        """Full path to training table"""
        return f"{self.catalog}.{self.schema}.{self.train_table}"
    
    @property
    def test_table_path(self) -> str:
        """Full path to test table"""
        return f"{self.catalog}.{self.schema}.{self.test_table}"
    
    @property
    def feature_table_path(self) -> str:
        """Full path to feature table"""
        return f"{self.catalog}.{self.schema}.{self.feature_table}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters based on model type"""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return self.hyperparameters
        elif self.model_type == "xgboost":
            return {
                "n_estimators": self.hyperparameters.get("n_estimators", 100),
                "max_depth": self.hyperparameters.get("max_depth", 6),
                "learning_rate": self.hyperparameters.get("learning_rate", 0.1),
                "random_state": self.hyperparameters.get("random_state", 42)
            }
        elif self.model_type == "lightgbm":
            return {
                "n_estimators": self.hyperparameters.get("n_estimators", 100),
                "max_depth": self.hyperparameters.get("max_depth", -1),
                "learning_rate": self.hyperparameters.get("learning_rate", 0.1),
                "random_state": self.hyperparameters.get("random_state", 42)
            }
        else:
            return self.hyperparameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "description": self.description,
            "catalog": self.catalog,
            "schema": self.schema,
            "train_table": self.train_table,
            "test_table": self.test_table,
            "feature_table": self.feature_table,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "experiment_name": self.experiment_name,
            "model_registry_name": self.model_registry_name,
            "model_alias": self.model_alias
        }


# Predefined configurations for different scenarios
CLASSIFICATION_CONFIG = ModelConfig(
    model_name="classification_model",
    description="Binary or multiclass classification model",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "criterion": "gini",
        "random_state": 42
    }
)

REGRESSION_CONFIG = ModelConfig(
    model_name="regression_model",
    description="Regression model for continuous predictions",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "criterion": "squared_error",
        "random_state": 42
    }
)

XGBOOST_CONFIG = ModelConfig(
    model_name="xgboost_model",
    description="XGBoost model for gradient boosting",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
)
