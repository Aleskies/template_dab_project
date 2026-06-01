"""
Feature Engineering Module
Functions for data preprocessing and feature creation
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    """Feature engineering and preprocessing utilities"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = "mean",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in dataframe
        
        Args:
            df: Input dataframe
            strategy: Strategy for imputation (mean, median, most_frequent, constant)
            columns: Specific columns to impute (if None, impute all numeric columns)
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)
                df[col] = self.imputers[col].fit_transform(df[[col]])
            else:
                df[col] = self.imputers[col].transform(df[[col]])
        
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input dataframe
            columns: Columns to scale
            method: Scaling method (standard, minmax)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        for col in columns:
            if col not in self.scalers:
                if method == "standard":
                    self.scalers[col] = StandardScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    self.scalers[col] = MinMaxScaler()
                
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input dataframe
            columns: Categorical columns to encode
            method: Encoding method (onehot, label)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        for col in columns:
            if method == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
            
            elif method == "onehot":
                if col not in self.encoders:
                    # Get dummies and store column names
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    self.encoders[col] = dummies.columns.tolist()
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                else:
                    # Apply same encoding as training
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Ensure same columns exist
                    for enc_col in self.encoders[col]:
                        if enc_col not in dummies.columns:
                            dummies[enc_col] = 0
                    df = pd.concat([df.drop(col, axis=1), dummies[self.encoders[col]]], axis=1)
        
        return df
    
    def create_date_features(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Create features from date column
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            
        Returns:
            DataFrame with additional date features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter
        df[f"{date_column}_is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs
        
        Args:
            df: Input dataframe
            column_pairs: List of tuples with column pairs to interact
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        for col1, col2 in column_pairs:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-10)  # Avoid division by zero
        
        return df
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input dataframe
            columns: Columns to create polynomial features from
            degree: Degree of polynomial
            
        Returns:
            DataFrame with polynomial features
        """
        df = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                df[f"{col}_pow_{d}"] = df[col] ** d
        
        return df
    
    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_column: str,
        agg_columns: List[str],
        agg_functions: List[str] = ["mean", "std", "min", "max"]
    ) -> pd.DataFrame:
        """
        Create aggregation features based on grouping
        
        Args:
            df: Input dataframe
            group_column: Column to group by
            agg_columns: Columns to aggregate
            agg_functions: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregation features
        """
        df = df.copy()
        
        for col in agg_columns:
            for func in agg_functions:
                agg_name = f"{col}_{func}_by_{group_column}"
                df[agg_name] = df.groupby(group_column)[col].transform(func)
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from dataframe
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers
            method: Method to detect outliers (iqr, zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        for col in columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        return df
    
    def get_feature_importance(
        self,
        model,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            })
            importance_df = importance_df.sort_values("importance", ascending=False)
            return importance_df.head(top_n)
        else:
            raise ValueError("Model does not have feature_importances_ attribute")
