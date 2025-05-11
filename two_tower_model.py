"""
Two-Tower model for personalized ad recommendation system.

This module implements the two-tower neural network architecture for user and ad embeddings.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from typing import Dict, List, Tuple

# Import local modules
from config import Config
from feature_processor import FeatureProcessor

logger = logging.getLogger(__name__)


class TwoTowerModel:
    """Two-Tower Neural Network for generating user and ad embeddings"""
    
    def __init__(self, config: Config, feature_processor: FeatureProcessor):
        """Initialize the two-tower model"""
        self.config = config
        self.feature_processor = feature_processor
        self.model = None
        self.user_model = None
        self.ad_model = None
    
    def _create_tower(self, input_features, name):
        """Create a single tower of the two-tower model"""
        # Concatenate all input features
        concatenated = layers.Concatenate(name=f"{name}_concatenated")(input_features)
        
        # Hidden layers
        x = concatenated
        for i, units in enumerate(self.config.tower_hidden_units):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"{name}_dense_{i}"
            )(x)
            x = layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
            x = layers.Dropout(self.config.dropout_rate, name=f"{name}_dropout_{i}")(x)
        
        # Final embedding layer
        embedding_dim = (self.config.user_embedding_dim if name == "user" 
                        else self.config.ad_embedding_dim)
        
        tower_output = layers.Dense(
            embedding_dim,
            activation='linear',
            kernel_regularizer=regularizers.l2(1e-6),
            name=f"{name}_embedding"
        )(x)
        
        # L2 normalize embeddings
        tower_output = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name=f"{name}_normalized"
        )(tower_output)
        
        return tower_output
    
    def build(self):
        """Build the two-tower model architecture"""
        # === User Tower Inputs ===
        user_inputs = {}
        user_features = []
        
        # Categorical feature inputs
        for feat in self.config.user_categorical_features:
            if feat in self.feature_processor.user_cat_dims:
                dim = self.feature_processor.user_cat_dims[feat]
                user_inputs[f"user_{feat}"] = layers.Input(
                    shape=(1,), name=f"user_{feat}_input"
                )
                
                # Create embedding layer
                embed_dim = min(600, int(6 * (dim ** 0.25)))
                embedding_layer = layers.Embedding(
                    input_dim=dim,
                    output_dim=embed_dim,
                    embeddings_regularizer=regularizers.l2(1e-5),
                    name=f"user_{feat}_embedding_layer"
                )(user_inputs[f"user_{feat}"])
                
                user_features.append(layers.Flatten(name=f"user_{feat}_flatten")(embedding_layer))
        
        # Numerical feature inputs
        for feat in self.config.user_numerical_features:
            user_inputs[f"user_{feat}"] = layers.Input(
                shape=(1,), name=f"user_{feat}_input"
            )
            user_features.append(user_inputs[f"user_{feat}"])
        
        # === Ad Tower Inputs ===
        ad_inputs = {}
        ad_features = []
        
        # Categorical feature inputs
        for feat in self.config.ad_categorical_features:
            if feat in self.feature_processor.ad_cat_dims:
                dim = self.feature_processor.ad_cat_dims[feat]
                ad_inputs[f"ad_{feat}"] = layers.Input(
                    shape=(1,), name=f"ad_{feat}_input"
                )
                
                # Create embedding layer
                embed_dim = min(600, int(6 * (dim ** 0.25)))
                embedding_layer = layers.Embedding(
                    input_dim=dim,
                    output_dim=embed_dim,
                    embeddings_regularizer=regularizers.l2(1e-5),
                    name=f"ad_{feat}_embedding_layer"
                )(ad_inputs[f"ad_{feat}"])
                
                ad_features.append(layers.Flatten(name=f"ad_{feat}_flatten")(embedding_layer))
        
        # Numerical feature inputs
        for feat in self.config.ad_numerical_features:
            ad_inputs[f"ad_{feat}"] = layers.Input(
                shape=(1,), name=f"ad_{feat}_input"
            )
            ad_features.append(ad_inputs[f"ad_{feat}"])
        
        # Ad content embedding input
        if self.feature_processor.text_encoder is not None:
            ad_inputs["ad_content_embedding"] = layers.Input(
                shape=(self.config.text_embedding_dim,),
                name="ad_content_embedding_input"
            )
            
            # Project content embedding to lower dimension
            content_projection = layers.Dense(
                128,
                activation='relu',
                name="content_projection"
            )(ad_inputs["ad_content_embedding"])
            
            ad_features.append(content_projection)
        
        # === Build Towers ===
        user_embedding = self._create_tower(user_features, "user")
        ad_embedding = self._create_tower(ad_features, "ad")
        
        # === Combine Towers for Training ===
        # Dot product similarity
        dot_product = layers.Dot(axes=1, normalize=True, name="dot_product")([
            user_embedding, ad_embedding
        ])
        
        # Build models
        # Full model for training
        self.model = models.Model(
            inputs=list(user_inputs.values()) + list(ad_inputs.values()),
            outputs=dot_product,
            name="two_tower_model"
        )
        
        # User model for inference
        self.user_model = models.Model(
            inputs=list(user_inputs.values()),
            outputs=user_embedding,
            name="user_tower_model"
        )
        
        # Ad model for inference
        self.ad_model = models.Model(
            inputs=list(ad_inputs.values()),
            outputs=ad_embedding,
            name="ad_tower_model"
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        logger.info(f"Two-tower model built: {len(user_features)} user features, {len(ad_features)} ad features")
        return self.model
    
    def train(self, interactions_df: pd.DataFrame, user_df: pd.DataFrame, ad_df: pd.DataFrame):
        """Train the two-tower model using interaction data"""
        logger.info(f"Training two-tower model with {len(interactions_df)} interactions...")
        
        # Sample negative interactions
        interactions_positive = interactions_df[interactions_df['clicked'] == 1]
        
        # Create training dataset with negative sampling
        train_data = self._generate_training_data(
            interactions_positive, user_df, ad_df
        )
        
        # Create inputs for the model
        user_features = self.feature_processor.transform_user_features(train_data)
        ad_features = self.feature_processor.transform_ad_features(train_data)
        
        # Combine all inputs
        model_inputs = {}
        model_inputs.update(user_features)
        model_inputs.update(ad_features)
        
        # Target values (1 for positive interactions, 0 for negative)
        y = train_data['label'].values
        
        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            callbacks.TensorBoard(
                log_dir=f"{self.config.log_dir}/two_tower/{int(time.time())}",
                histogram_freq=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            model_inputs,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Two-tower model training completed")
        
        # Print final metrics
        val_auc = history.history['val_auc'][-1]
        val_loss = history.history['val_loss'][-1]
        logger.info(f"Final validation metrics - AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")
        
        return history
    
    def _generate_training_data(self, interactions_positive: pd.DataFrame,
                               user_df: pd.DataFrame, ad_df: pd.DataFrame,
                               neg_ratio: int = 4) -> pd.DataFrame:
        """Generate training data with negative sampling"""
        # Merge positive interactions with user and ad features
        positive_samples = interactions_positive.merge(
            user_df, on='user_id', how='left'
        ).merge(
            ad_df, on='ad_id', how='left'
        )
        positive_samples['label'] = 1
        
        # Sample negative interactions
        user_ids = positive_samples['user_id'].unique()
        ad_ids = ad_df['ad_id'].unique()
        
        num_negatives = len(positive_samples) * neg_ratio
        negative_samples_list = []
        
        # For each user, sample random ads they haven't interacted with
        for user_id in user_ids:
            user_pos_ads = interactions_positive[
                interactions_positive['user_id'] == user_id
            ]['ad_id'].values
            
            # Eligible ads for negative sampling
            eligible_ads = np.setdiff1d(ad_ids, user_pos_ads)
            
            if len(eligible_ads) == 0:
                continue
            
            # Calculate number of negatives for this user
            n_samples = min(
                int(num_negatives * len(user_pos_ads) / len(positive_samples)),
                len(eligible_ads)
            )
            
            # Sample random negative ads
            sampled_ads = np.random.choice(eligible_ads, size=n_samples, replace=False)
            
            # Create negative samples
            for ad_id in sampled_ads:
                negative_samples_list.append({
                    'user_id': user_id,
                    'ad_id': ad_id,
                    'clicked': 0,
                    'timestamp': interactions_positive['timestamp'].mean()
                })
        
        # Create negative samples dataframe
        negative_samples = pd.DataFrame(negative_samples_list)
        
        # Merge negative samples with user and ad features
        negative_samples = negative_samples.merge(
            user_df, on='user_id', how='left'
        ).merge(
            ad_df, on='ad_id', how='left'
        )
        negative_samples['label'] = 0
        
        # Combine positive and negative samples
        combined_samples = pd.concat([positive_samples, negative_samples], axis=0)
        combined_samples = combined_samples.sample(frac=1).reset_index(drop=True)
        
        logger.info(f"Generated training data: {len(positive_samples)} positive, {len(negative_samples)} negative")
        return combined_samples
    
    def get_user_embedding(self, user_df: pd.DataFrame) -> np.ndarray:
        """Generate user embeddings for the given users"""
        if self.user_model is None:
            raise ValueError("Model not built or trained yet")
        
        # Transform user features
        user_features = self.feature_processor.transform_user_features(user_df)
        
        # Generate embeddings
        user_embeddings = self.user_model.predict(user_features)
        
        return user_embeddings
    
    def get_ad_embedding(self, ad_df: pd.DataFrame) -> np.ndarray:
        """Generate ad embeddings for the given ads"""
        if self.ad_model is None:
            raise ValueError("Model not built or trained yet")
        
        # Transform ad features
        ad_features = self.feature_processor.transform_ad_features(ad_df)
        
        # Generate embeddings
        ad_embeddings = self.ad_model.predict(ad_features)
        
        return ad_embeddings
    
    def save(self, base_path: str):
        """Save the model to disk"""
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        # Save the full model
        full_model_path = os.path.join(base_path, "full_model")
        self.model.save(full_model_path)
        
        # Save user tower
        user_model_path = os.path.join(base_path, "user_model")
        self.user_model.save(user_model_path)
        
        # Save ad tower
        ad_model_path = os.path.join(base_path, "ad_model")
        self.ad_model.save(ad_model_path)
        
        logger.info(f"Two-tower model saved to {base_path}")
    
    @staticmethod
    def load(base_path: str, config: Config, feature_processor: FeatureProcessor) -> 'TwoTowerModel':
        """Load the model from disk"""
        # Create a new instance
        model = TwoTowerModel(config, feature_processor)
        
        # Load the models
        full_model_path = os.path.join(base_path, "full_model")
        user_model_path = os.path.join(base_path, "user_model")
        ad_model_path = os.path.join(base_path, "ad_model")
        
        model.model = tf.keras.models.load_model(full_model_path)
        model.user_model = tf.keras.models.load_model(user_model_path)
        model.ad_model = tf.keras.models.load_model(ad_model_path)
        
        logger.info(f"Two-tower model loaded from {base_path}")
        return model
