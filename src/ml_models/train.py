"""
Model Training Script
Main script for training ML models with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import mlflow
from mlflow.models import infer_signature
import argparse
import sys

from ml_models.model_config import ModelConfig
from ml_models.feature_engineering import FeatureEngineer
from ml_models.mlflow_utils import MLflowTracker


class ModelTrainer:
    """Model training orchestrator"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize trainer
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.mlflow_tracker = MLflowTracker(config.experiment_name)
        self.model = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load training data from Unity Catalog
        
        Returns:
            DataFrame with training data
        """
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.getOrCreate()
        
        # Load from Unity Catalog table
        df = spark.table(self.config.train_table_path).toPandas()
        
        print(f"Loaded {len(df)} rows from {self.config.train_table_path}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X, y)
        """
        # Handle missing values
        df = self.feature_engineer.handle_missing_values(df)
        
        # Extract features and target
        X = df[self.config.feature_columns]
        y = df[self.config.target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str = "classification"
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            task_type: Type of task (classification or regression)
            
        Returns:
            Trained model
        """
        params = self.config.get_model_params()
        
        if task_type == "classification":
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)
        
        print(f"Training {self.config.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "classification"
    ) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            task_type: Type of task (classification or regression)
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {}
        
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            print("\nClassification Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        else:  # regression
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2_score"] = r2_score(y_test, y_pred)
            
            print("\nRegression Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def run_training_pipeline(
        self,
        task_type: str = "classification",
        register_model: bool = True
    ):
        """
        Run complete training pipeline with MLflow tracking
        
        Args:
            task_type: Type of task (classification or regression)
            register_model: Whether to register model in Model Registry
        """
        with self.mlflow_tracker.start_run(run_name=f"{self.config.model_name}_training"):
            
            # Set tags
            self.mlflow_tracker.set_tags({
                "model_type": self.config.model_type,
                "task_type": task_type,
                "training_table": self.config.train_table_path
            })
            
            # Log config parameters
            self.mlflow_tracker.log_params(self.config.to_dict())
            
            # Load data
            print("=" * 50)
            print("Loading data...")
            df = self.load_data()
            
            # Prepare features
            print("=" * 50)
            print("Preparing features...")
            X, y = self.prepare_features(df)
            
            # Split data
            print("=" * 50)
            print("Splitting data...")
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Train model
            print("=" * 50)
            print("Training model...")
            self.train_model(X_train, y_train, task_type)
            
            # Evaluate model
            print("=" * 50)
            print("Evaluating model...")
            metrics = self.evaluate_model(X_test, y_test, task_type)
            
            # Log metrics
            self.mlflow_tracker.log_metrics(metrics)
            
            # Log feature importance
            if hasattr(self.model, "feature_importances_"):
                feature_importance = self.feature_engineer.get_feature_importance(
                    self.model,
                    self.config.feature_columns,
                    top_n=20
                )
                print("\nTop 10 Feature Importances:")
                print(feature_importance.head(10))
                
                # Log as artifact
                self.mlflow_tracker.log_dict(
                    feature_importance.to_dict(),
                    "feature_importance.json"
                )
            
            # Create model signature
            signature = infer_signature(X_train, self.model.predict(X_train))
            input_example = X_train.head(3)
            
            # Log model
            print("=" * 50)
            print("Logging model to MLflow...")
            
            if register_model:
                self.mlflow_tracker.log_model(
                    self.model,
                    artifact_path="model",
                    model_type="sklearn",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=self.config.model_registry_name
                )
                
                # Set model alias
                run = mlflow.active_run()
                model_uri = f"runs:/{run.info.run_id}/model"
                self.mlflow_tracker.register_model(
                    model_uri,
                    self.config.model_registry_name,
                    alias=self.config.model_alias
                )
                
                print(f"Model registered as '{self.config.model_registry_name}' with alias '{self.config.model_alias}'")
            else:
                self.mlflow_tracker.log_model(
                    self.model,
                    artifact_path="model",
                    model_type="sklearn",
                    signature=signature,
                    input_example=input_example
                )
            
            print("=" * 50)
            print("Training pipeline completed successfully!")
            print(f"Run ID: {mlflow.active_run().info.run_id}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--catalog", type=str, default="workspace", help="Catalog name")
    parser.add_argument("--schema", type=str, default="default", help="Schema name")
    parser.add_argument("--train-table", type=str, default="train_data", help="Training table name")
    parser.add_argument("--model-name", type=str, default="my_ml_model", help="Model name")
    parser.add_argument("--task-type", type=str, default="classification", 
                       choices=["classification", "regression"], help="Task type")
    parser.add_argument("--register", action="store_true", help="Register model in Model Registry")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ModelConfig(
        catalog=args.catalog,
        schema=args.schema,
        train_table=args.train_table,
        model_name=args.model_name,
        model_registry_name=args.model_name
    )
    
    # Create trainer and run pipeline
    trainer = ModelTrainer(config)
    trainer.run_training_pipeline(
        task_type=args.task_type,
        register_model=args.register
    )


if __name__ == "__main__":
    main()
