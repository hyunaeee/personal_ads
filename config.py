
"""
Configuration module for personalized ad recommendation system.

This module defines the configuration settings for the recommendation system.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration for the recommendation system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with default values or from a config file"""
        
        # General settings
        self.random_seed = 42
        self.log_dir = "logs/"
        
        # Data paths
        self.user_data_path = "data/users.parquet"
        self.ad_data_path = "data/ads.parquet"
        self.interaction_data_path = "data/interactions.parquet"
        self.model_save_path = "models/saved/"
        
        # Feature configurations
        self.user_categorical_features = ["gender", "age_group", "country", "device_type"]
        self.user_numerical_features = ["activity_level", "days_since_registration"]
        self.ad_categorical_features = ["campaign_id", "advertiser_id", "category", "language"]
        self.ad_numerical_features = ["bid_amount", "campaign_days_remaining"]
        
        # Embedding dimensions
        self.user_embedding_dim = 128
        self.ad_embedding_dim = 128
        self.text_embedding_dim = 768  # BERT dimension
        
        # Training parameters
        self.batch_size = 512
        self.epochs = 20
        self.learning_rate = 0.001
        self.early_stopping_patience = 3
        
        # Model architecture
        self.tower_hidden_units = [256, 128]
        self.ranking_hidden_units = [256, 128, 64]
        self.dropout_rate = 0.3
        
        # Retrieval parameters
        self.retrieval_candidates = 500
        self.ranking_candidates = 50
        self.final_candidates = 10
        
        # Exploration parameters
        self.exploration_factor = 0.1  # Thompson sampling parameter
        self.alpha_prior = 1.0  # Prior alpha for beta distribution
        self.beta_prior = 1.0   # Prior beta for beta distribution
        
        # Cold start parameters
        self.cold_start_exploration_factor = 0.3
        self.min_interactions_for_personalization = 5
        
        # Serving parameters
        self.serving_threads = 10
        self.max_request_size = 1000
        self.cache_ttl = 300  # seconds
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # Override with config file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Re-seed after loading config
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
    
    def save_config(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
