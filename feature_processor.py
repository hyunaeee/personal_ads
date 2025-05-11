"""
Feature processing module for personalized ad recommendation system.

This module handles preprocessing of user and ad features.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import local modules
from config import Config

# Try to import SentenceTransformer for text encoding
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Process and transform features for recommendation models"""
    
    def __init__(self, config: Config):
        """Initialize feature processor with configuration"""
        self.config = config
        
        # Feature encoders and scalers
        self.user_cat_encoders = {}
        self.ad_cat_encoders = {}
        self.user_num_scaler = None
        self.ad_num_scaler = None
        
        # Embedding dimensions for each categorical feature
        self.user_cat_dims = {}
        self.ad_cat_dims = {}
        
        # Text encoder for ad content
        self.text_encoder = None
    
    def fit(self, user_df: pd.DataFrame, ad_df: pd.DataFrame):
        """Fit feature processors on training data"""
        logger.info("Fitting feature processors...")
        
        # Process user categorical features
        for feat in self.config.user_categorical_features:
            if feat in user_df.columns:
                encoder = LabelEncoder()
                encoder.fit(user_df[feat].fillna('UNKNOWN'))
                self.user_cat_encoders[feat] = encoder
                self.user_cat_dims[feat] = len(encoder.classes_)
                logger.info(f"User feature {feat}: {len(encoder.classes_)} categories")
        
        # Process ad categorical features
        for feat in self.config.ad_categorical_features:
            if feat in ad_df.columns:
                encoder = LabelEncoder()
                encoder.fit(ad_df[feat].fillna('UNKNOWN'))
                self.ad_cat_encoders[feat] = encoder
                self.ad_cat_dims[feat] = len(encoder.classes_)
                logger.info(f"Ad feature {feat}: {len(encoder.classes_)} categories")
        
        # Process numerical features
        if self.config.user_numerical_features:
            user_num_features = [f for f in self.config.user_numerical_features if f in user_df.columns]
            if user_num_features:
                self.user_num_scaler = StandardScaler()
                self.user_num_scaler.fit(
                    user_df[user_num_features].fillna(0)
                )
        
        if self.config.ad_numerical_features:
            ad_num_features = [f for f in self.config.ad_numerical_features if f in ad_df.columns]
            if ad_num_features:
                self.ad_num_scaler = StandardScaler()
                self.ad_num_scaler.fit(
                    ad_df[ad_num_features].fillna(0)
                )
        
        # Initialize text encoder for ad content
        if 'content' in ad_df.columns and SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                # Initialize with a small batch to warm up the encoder
                sample_content = ad_df['content'].fillna('').head(5).tolist()
                _ = self.text_encoder.encode(sample_content)
                logger.info("Text encoder initialized for ad content")
            except Exception as e:
                logger.warning(f"Could not initialize text encoder: {e}")
                self.text_encoder = None
        
        return self
    
    def transform_user_features(self, user_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform user features for model input"""
        transformed = {}
        
        # Transform categorical features
        for feat in self.config.user_categorical_features:
            if feat in user_df.columns and feat in self.user_cat_encoders:
                encoder = self.user_cat_encoders[feat]
                values = user_df[feat].fillna('UNKNOWN').values
                # Handle unknown categories
                mask = np.isin(values, encoder.classes_)
                encoded_values = np.zeros(len(values), dtype=int)
                encoded_values[mask] = encoder.transform(values[mask])
                # Use default value 0 for unknown categories
                transformed[f"user_{feat}"] = encoded_values
        
        # Transform numerical features
        if self.config.user_numerical_features and self.user_num_scaler:
            user_num_features = [f for f in self.config.user_numerical_features if f in user_df.columns]
            if user_num_features:
                scaled = self.user_num_scaler.transform(
                    user_df[user_num_features].fillna(0)
                )
                for i, feat in enumerate(user_num_features):
                    transformed[f"user_{feat}"] = scaled[:, i]
        
        return transformed
    
    def transform_ad_features(self, ad_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform ad features for model input"""
        transformed = {}
        
        # Transform categorical features
        for feat in self.config.ad_categorical_features:
            if feat in ad_df.columns and feat in self.ad_cat_encoders:
                encoder = self.ad_cat_encoders[feat]
                values = ad_df[feat].fillna('UNKNOWN').values
                # Handle unknown categories
                mask = np.isin(values, encoder.classes_)
                encoded_values = np.zeros(len(values), dtype=int)
                encoded_values[mask] = encoder.transform(values[mask])
                # Use default value 0 for unknown categories
                transformed[f"ad_{feat}"] = encoded_values
        
        # Transform numerical features
        if self.config.ad_numerical_features and self.ad_num_scaler:
            ad_num_features = [f for f in self.config.ad_numerical_features if f in ad_df.columns]
            if ad_num_features:
                scaled = self.ad_num_scaler.transform(
                    ad_df[ad_num_features].fillna(0)
                )
                for i, feat in enumerate(ad_num_features):
                    transformed[f"ad_{feat}"] = scaled[:, i]
        
        # Transform ad content if available
        if 'content' in ad_df.columns and self.text_encoder:
            content = ad_df['content'].fillna('').values
            text_embeddings = self.text_encoder.encode(content.tolist())
            transformed["ad_content_embedding"] = text_embeddings
        
        return transformed
    
    def prepare_context_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare context features for the model"""
        context_features = {}
        
        # Extract hour of day if timestamp is available
        if 'timestamp' in df.columns:
            try:
                # Convert timestamp to hour of day (0-23)
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                context_features['context_hour_input'] = df['hour'].values
                
                # Day of week (0-6, Monday=0)
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                context_features['context_day_input'] = df['day_of_week'].values
            except:
                # Default values if timestamp conversion fails
                context_features['context_hour_input'] = np.zeros(len(df))
                context_features['context_day_input'] = np.zeros(len(df))
        else:
            # Default values if timestamp not available
            context_features['context_hour_input'] = np.zeros(len(df))
            context_features['context_day_input'] = np.zeros(len(df))
        
        # Device type - use from data if available, otherwise default to 0
        if 'device_type' in df.columns and 'device_type' in self.user_cat_encoders:
            encoder = self.user_cat_encoders['device_type']
            values = df['device_type'].fillna('UNKNOWN').values
            # Handle unknown categories
            mask = np.isin(values, encoder.classes_)
            encoded_values = np.zeros(len(values), dtype=int)
            encoded_values[mask] = encoder.transform(values[mask])
            context_features['context_device_input'] = encoded_values
        else:
            context_features['context_device_input'] = np.zeros(len(df))
        
        return context_features
    
    def save(self, path: str):
        """Save feature processor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save without the text encoder (will be reloaded)
        text_encoder_temp = self.text_encoder
        self.text_encoder = None
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Restore text encoder
        self.text_encoder = text_encoder_temp
        logger.info(f"Feature processor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'FeatureProcessor':
        """Load feature processor from disk"""
        with open(path, 'rb') as f:
            processor = pickle.load(f)
        
        # Reinitialize text encoder if needed
        if processor.text_encoder is None and SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                processor.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                logger.info("Text encoder reinitialized")
            except Exception as e:
                logger.warning(f"Could not reinitialize text encoder: {e}")
        
        logger.info(f"Feature processor loaded from {path}")
        return processor
