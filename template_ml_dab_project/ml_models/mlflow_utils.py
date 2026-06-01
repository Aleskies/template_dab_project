"""
MLflow Utilities
Helper functions for MLflow experiment tracking and model management
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime


class MLflowTracker:
    """MLflow experiment tracking and model management"""
    
    def __init__(self, experiment_name: str):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Set or create experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error setting experiment: {e}")
            self.experiment = None
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metric
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        model_type: str = "sklearn",
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log model to MLflow
        
        Args:
            model: Trained model object
            artifact_path: Path within run to save model
            model_type: Type of model (sklearn, xgboost, lightgbm)
            signature: Model signature
            input_example: Example input for model
            registered_model_name: Name to register model in Model Registry
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def log_artifact(self, artifact_path: str) -> None:
        """
        Log artifact to MLflow
        
        Args:
            artifact_path: Path to artifact file
        """
        mlflow.log_artifact(artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str) -> None:
        """
        Log dictionary as JSON artifact
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the artifact file
        """
        mlflow.log_dict(dictionary, filename)
    
    def log_figure(self, figure, filename: str) -> None:
        """
        Log matplotlib/plotly figure
        
        Args:
            figure: Matplotlib or Plotly figure
            filename: Name for the artifact file
        """
        mlflow.log_figure(figure, filename)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for current run
        
        Args:
            tags: Dictionary of tags to set
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a specific run
        
        Args:
            run_id: ID of the run
            
        Returns:
            Dictionary with run information
        """
        run = self.client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags
        }
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: List[str] = None
    ) -> pd.DataFrame:
        """
        Search runs in the experiment
        
        Args:
            filter_string: Filter string for runs
            max_results: Maximum number of results to return
            order_by: List of columns to order by
            
        Returns:
            DataFrame with run information
        """
        if order_by is None:
            order_by = ["start_time DESC"]
        
        return mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by
        )
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        alias: Optional[str] = None
    ) -> None:
        """
        Register model in Model Registry
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            alias: Optional alias to assign (e.g., "Production", "Staging")
        """
        result = mlflow.register_model(model_uri, model_name)
        
        if alias:
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=result.version
            )
        
        return result
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str
    ) -> None:
        """
        Transition model to a different stage (deprecated, use aliases instead)
        
        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Stage to transition to (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def set_model_alias(
        self,
        model_name: str,
        alias: str,
        version: int
    ) -> None:
        """
        Set alias for a model version
        
        Args:
            model_name: Name of the registered model
            alias: Alias to set (e.g., "Production", "Staging", "Champion")
            version: Version number to assign alias to
        """
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version
        )
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[int] = None,
        alias: Optional[str] = None
    ) -> Any:
        """
        Get specific version of a registered model
        
        Args:
            model_name: Name of the registered model
            version: Version number (if not using alias)
            alias: Model alias (e.g., "Production")
            
        Returns:
            Model version details
        """
        if alias:
            return self.client.get_model_version_by_alias(model_name, alias)
        elif version:
            return self.client.get_model_version(model_name, str(version))
        else:
            # Get latest version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                return versions[0]
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[int] = None,
        alias: Optional[str] = None
    ):
        """
        Load a model from Model Registry
        
        Args:
            model_name: Name of the registered model
            version: Version number (if not using alias)
            alias: Model alias (e.g., "Production")
            
        Returns:
            Loaded model
        """
        if alias:
            model_uri = f"models:/{model_name}@{alias}"
        elif version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        return mlflow.pyfunc.load_model(model_uri)
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metric_names: List of metric names to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id}
            
            for metric in metric_names:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = False
    ) -> Dict[str, Any]:
        """
        Get the best run based on a metric
        
        Args:
            metric_name: Name of metric to optimize
            ascending: If True, lower is better
            
        Returns:
            Dictionary with best run information
        """
        runs_df = self.search_runs(
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs_df) > 0:
            best_run = runs_df.iloc[0]
            return {
                "run_id": best_run.run_id,
                "metrics": {col: best_run[col] for col in runs_df.columns if col.startswith("metrics.")},
                "params": {col: best_run[col] for col in runs_df.columns if col.startswith("params.")}
            }
        
        return None
