"""
Model Prediction Script
Script for making predictions using trained models
"""
import pandas as pd
import numpy as np
import mlflow
from typing import Union, List, Dict, Any
import argparse

from model_config import ModelConfig
from feature_engineering import FeatureEngineer
from mlflow_utils import MLflowTracker


class ModelPredictor:
    """Model prediction orchestrator"""
    
    def __init__(
        self,
        model_name: str,
        model_version: Union[int, str] = "Production",
        config: ModelConfig = None
    ):
        """
        Initialize predictor
        
        Args:
            model_name: Name of the registered model
            model_version: Version number or alias (e.g., "Production", "Staging", 1, 2)
            config: Model configuration (optional)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.config = config or ModelConfig()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.mlflow_tracker = MLflowTracker(self.config.experiment_name)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load model from MLflow Model Registry"""
        try:
            if isinstance(self.model_version, int):
                model_uri = f"models:/{self.model_name}/{self.model_version}"
            else:
                # Assume it's an alias like "Production" or "Staging"
                model_uri = f"models:/{self.model_name}@{self.model_version}"
            
            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded successfully from {model_uri}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            df: Input dataframe
            
        Returns:
            Processed dataframe ready for prediction
        """
        # Handle missing values
        df = self.feature_engineer.handle_missing_values(df)
        
        # Select only required features
        if self.config.feature_columns:
            df = df[self.config.feature_columns]
        
        return df
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Array of predictions
        """
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Make prediction
        predictions = self.model.predict(df_prepared)
        
        return predictions
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Make probability predictions (for classification models)
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Array of prediction probabilities
        """
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df_prepared)
            return probabilities
        else:
            raise AttributeError("Model does not support probability predictions")
    
    def batch_predict(
        self,
        input_table: str,
        output_table: str,
        batch_size: int = 1000
    ):
        """
        Make predictions on a full table in batches
        
        Args:
            input_table: Full path to input table (catalog.schema.table)
            output_table: Full path to output table (catalog.schema.table)
            batch_size: Number of rows to process at once
        """
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, struct
        
        spark = SparkSession.builder.getOrCreate()
        
        # Load input data
        print(f"Loading data from {input_table}...")
        df_spark = spark.table(input_table)
        total_rows = df_spark.count()
        print(f"Total rows to process: {total_rows}")
        
        # Convert to Pandas for prediction
        df_pandas = df_spark.toPandas()
        
        # Prepare features
        print("Preparing features...")
        df_prepared = self.prepare_features(df_pandas)
        
        # Make predictions in batches
        print(f"Making predictions in batches of {batch_size}...")
        predictions = []
        
        for i in range(0, len(df_prepared), batch_size):
            batch = df_prepared.iloc[i:i+batch_size]
            batch_predictions = self.predict(batch)
            predictions.extend(batch_predictions)
            
            if (i + batch_size) % 10000 == 0:
                print(f"Processed {min(i + batch_size, len(df_prepared))} / {len(df_prepared)} rows")
        
        # Add predictions to original dataframe
        df_pandas['prediction'] = predictions
        
        # Convert back to Spark DataFrame
        df_result = spark.createDataFrame(df_pandas)
        
        # Write to output table
        print(f"Writing results to {output_table}...")
        df_result.write.mode("overwrite").saveAsTable(output_table)
        
        print(f"Predictions saved to {output_table}")
        print(f"Total predictions: {len(predictions)}")
    
    def predict_with_explanation(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Make prediction with feature importance explanation
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with prediction and explanation
        """
        # Make prediction
        prediction = self.predict(data)
        
        # Get feature importance if available
        explanation = {
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        }
        
        # Try to get feature importance from underlying model
        if hasattr(self.model, '_model_impl'):
            underlying_model = self.model._model_impl
            if hasattr(underlying_model, 'feature_importances_'):
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = data
                
                feature_importance = pd.DataFrame({
                    'feature': df.columns,
                    'importance': underlying_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                explanation['feature_importance'] = feature_importance.to_dict('records')
        
        return explanation


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name in registry")
    parser.add_argument("--model-version", type=str, default="Production", 
                       help="Model version or alias (e.g., Production, Staging, 1, 2)")
    parser.add_argument("--input-table", type=str, help="Input table path (catalog.schema.table)")
    parser.add_argument("--output-table", type=str, help="Output table path (catalog.schema.table)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for predictions")
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = ModelPredictor(
        model_name=args.model_name,
        model_version=args.model_version
    )
    
    # Make batch predictions if tables provided
    if args.input_table and args.output_table:
        predictor.batch_predict(
            input_table=args.input_table,
            output_table=args.output_table,
            batch_size=args.batch_size
        )
    else:
        print("No input/output tables provided. Use --input-table and --output-table for batch predictions")
        print("\nFor interactive predictions, use the ModelPredictor class directly:")
        print("  predictor = ModelPredictor('model_name', 'Production')")
        print("  predictions = predictor.predict({'feature1': 1.0, 'feature2': 2.0})")


if __name__ == "__main__":
    main()
