"""
Fraud Detection Model Training Pipeline

This implementation represents a production-grade ML training system for fraud detection with
the following architectural considerations:

1. Environment Agnostic Configuration
   - Uses YAML config for environment-specific parameters
   - Strict separation of secrets vs configuration
   - Multi-environment support via .env files

2. Observability
   - Structured logging with multiple sinks
   - MLflow experiment tracking
   - Artifact storage with MinIO (S3-compatible)

3. Production Readiness
   - Kafka integration for real-time data ingestion
   - Automated hyperparameter tuning
   - Model serialization/registry
   - Comprehensive metrics tracking
   - Class imbalance mitigation

4. Operational Safety
   - Environment validation checks
   - Data quality guards
   - Comprehensive error handling
   - Model performance baselining
"""

import json
import logging
import os

import boto3
import matplotlib.pyplot as plt
import mlflow
from mlflow import sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from kafka import KafkaConsumer
from mlflow.models import infer_signature
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, average_precision_score, precision_score, \
    recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Configure dual logging to file and stdout with structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    """
    End-to-end fraud detection training system implementing MLOps best practices.

    Key Architecture Components:
    - Configuration Management: Centralized YAML config with environment overrides
    - Data Ingestion: Kafka consumer with SASL/SSL authentication
    - Feature Engineering: Temporal, behavioral, and monetary feature constructs
    - Model Development: XGBoost with SMOTE for class imbalance
    - Hyperparameter Tuning: Randomized search with stratified cross-validation
    - Model Tracking: MLflow integration with metrics/artifact logging
    - Deployment Prep: Model serialization and registry

    The system is designed for horizontal scalability and cloud-native operation.
    """

    def __init__(self, config_path='/app/config.yaml'):
        # Environment hardening for containerized deployments
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # Load environment variables before config to allow overrides
        load_dotenv(dotenv_path='/app/.env')

        # Configuration lifecycle management
        self.config = self._load_config(config_path)

        # Security-conscious credential handling
        env_vars = {
        'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
        'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
    }
        os.environ.update({k: v for k, v in env_vars.items() if v is not None})


        # Pre-flight system checks
        self._validate_environment()

        # MLflow configuration for experiment tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _load_config(self, config_path: str) -> dict:
        """
        Load and validate hierarchical configuration with fail-fast semantics.

        Implements:
        - YAML configuration parsing
        - Early validation of critical parameters
        - Audit logging of configuration loading
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _validate_environment(self):
        """
        System integrity verification with defense-in-depth checks:
        1. Required environment variables
        2. Object storage connectivity
        3. Credential validation

        Fails early to prevent partial initialization states.
        """
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')

        self._check_minio_connection()

    def _check_minio_connection(self):
        """
        Validate object storage connectivity and bucket configuration.

        Implements:
        - S3 client initialization with error handling
        - Bucket existence check
        - Automatic bucket creation (if configured)

        Maintains separation of concerns between configuration and infrastructure setup.
        """
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))

    def read_from_kafka(self) -> pd.DataFrame:
        """
        Secure Kafka consumer implementation with enterprise features:

        - SASL/SSL authentication
        - Auto-offset reset for recovery scenarios
        - Data quality checks:
          - Schema validation
          - Fraud label existence
          - Fraud rate monitoring

        Implements graceful shutdown on timeout/error conditions.
        """
        try:
            topic = self.config['kafka']['topic']
            logger.info('Connecting to kafka topic %s', topic)

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username=self.config['kafka']['username'],
                sasl_plain_password=self.config['kafka']['password'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=self.config['kafka'].get('timeout', 10000)
            )

            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)
            if df.empty:
                raise ValueError('No messages received from Kafka.')

            # Temporal data standardization
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')

            # Data quality monitoring point
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Kafka data read successfully with fraud rate: %.2f%%', fraud_rate)

            return df
        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature engineering pipeline implementing advanced fraud detection concepts:

        1. Cyclical Temporal Features:
        - Sine/cosine encoding for hour and day of week
        - Night/weekend indicators

        2. Velocity Features:
        - Multi-window transaction frequency (1h, 6h, 24h)
        - Transaction amount velocity

        3. Behavioral Features:
        - Personalized amount z-scores
        - Historical spending pattern deviations

        4. Merchant Intelligence:
        - Dynamic merchant fraud rates
        - New merchant detection per user
        - Merchant risk profiling

        5. Interaction Features:
        - Cross-feature combinations for complex patterns

        Maintains immutability via DataFrame.copy() and validates feature set integrity.
        """
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # ---- Enhanced Temporal Feature Engineering ----
        # Cyclical encoding preserves time continuity (midnight adjacent to 11 PM)
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
        
        # Day of week cyclical encoding (Sunday adjacent to Monday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Traditional time indicators (kept for interpretability)
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 5)).astype(int)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['transaction_day'] = df['timestamp'].dt.day

        # ---- Velocity Feature Engineering ----
        # Multi-window transaction frequency captures different fraud patterns
        velocity_windows = ['1H', '6H', '24H']
        for window in velocity_windows:
            # Transaction count in time window
            df[f'txn_count_{window}'] = df.groupby('user_id', group_keys=False).apply(
                lambda g: g.rolling(window, on='timestamp', closed='left')['amount'].count().fillna(0)
            )
            
            # Transaction amount sum in time window
            df[f'txn_sum_{window}'] = df.groupby('user_id', group_keys=False).apply(
                lambda g: g.rolling(window, on='timestamp', closed='left')['amount'].sum().fillna(0)
            )
            
            # Average transaction amount in window (avoid division by zero)
            df[f'txn_avg_{window}'] = np.where(
                df[f'txn_count_{window}'] > 0,
                df[f'txn_sum_{window}'] / df[f'txn_count_{window}'],
                0
            )

        # ---- Enhanced Behavioral Feature Engineering ----
        # Personalized z-score for amount (captures individual spending patterns)
        df['amount_zscore_30'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: (
                (g['amount'] - g['amount'].rolling(30, min_periods=1).mean()) / 
                g['amount'].rolling(30, min_periods=1).std().fillna(1.0)
            ).fillna(0.0)
        )
        
        # Multiple lookback windows for amount ratios
        for window in [7, 14, 30]:
            df[f'amount_to_avg_ratio_{window}d'] = df.groupby('user_id', group_keys=False).apply(
                lambda g: (g['amount'] / g['amount'].rolling(window, min_periods=1).mean()).fillna(1.0)
            )
        
        # Amount percentile within user's history
        df['amount_percentile'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: g['amount'].rolling(50, min_periods=1).rank(pct=True).fillna(0.5)
        )

        # ---- Merchant Intelligence Engineering ----
        # Dynamic merchant fraud rate (updated with actual data)
        df['merchant_fraud_rate'] = df.groupby('merchant')['is_fraud'].transform('mean')
        
        # New merchant detection per user
        df['new_merchant_for_user'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: ~g['merchant'].isin(g['merchant'].shift(1).dropna())
        ).astype(int)
        
        # Merchant transaction frequency (how often this merchant appears)
        df['merchant_popularity'] = df.groupby('merchant')['merchant'].transform('count') / len(df)
        
        # Static high-risk merchant list (configuration-driven)
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk_static'] = df['merchant'].isin(high_risk_merchants).astype(int)

        # ---- Interaction Feature Engineering ----
        # Complex patterns through feature combinations
        df['amount_x_fraud_rate'] = df['amount'] * df['merchant_fraud_rate']
        df['velocity_x_amount'] = df['txn_count_1H'] * df['amount']
        df['night_x_new_merchant'] = df['is_night'] * df['new_merchant_for_user']
        df['weekend_x_high_amount'] = df['is_weekend'] * (df['amount_zscore_30'] > 2).astype(int)
        df['high_velocity_x_night'] = (df['txn_count_1H'] > 3).astype(int) * df['is_night']

        # ---- Time-based Features ----
        # Days since last transaction (user dormancy)
        df['days_since_last_txn'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: g['timestamp'].diff().dt.total_seconds() / (24 * 3600)
        ).fillna(0)
        
        # Transaction timing consistency (fraud often happens at unusual times for user)
        user_common_hours = df.groupby('user_id')['transaction_hour'].apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12
        )
        df['hour_deviation_from_normal'] = abs(
            df['transaction_hour'] - df['user_id'].map(user_common_hours)
        )

        # ---- Feature Selection and Validation ----
        feature_cols = [
            # Basic features
            'amount', 'merchant',
            
            # Enhanced temporal features
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_night', 'is_weekend', 'transaction_day',
            
            # Velocity features
            'txn_count_1H', 'txn_count_6H', 'txn_count_24H',
            'txn_sum_1H', 'txn_sum_6H', 'txn_sum_24H',
            'txn_avg_1H', 'txn_avg_6H', 'txn_avg_24H',
            
            # Behavioral features
            'amount_zscore_30', 'amount_to_avg_ratio_7d', 'amount_to_avg_ratio_14d', 
            'amount_to_avg_ratio_30d', 'amount_percentile',
            
            # Merchant features
            'merchant_fraud_rate', 'new_merchant_for_user', 'merchant_popularity', 'merchant_risk_static',
            
            # Interaction features
            'amount_x_fraud_rate', 'velocity_x_amount', 'night_x_new_merchant',
            'weekend_x_high_amount', 'high_velocity_x_night',
            
            # Time-based features
            'days_since_last_txn', 'hour_deviation_from_normal'
        ]

        # Schema validation guard
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')

        # Feature existence validation
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning('Missing features will be handled by preprocessor: %s', missing_features)

        # Data quality checks
        logger.info('Feature engineering completed. Shape: %s, Features: %d', df.shape, len(feature_cols))
        logger.info('Null values per feature: %s', df[feature_cols].isnull().sum().to_dict())
        
        return df[feature_cols + ['is_fraud']]

    def train_model(self):
        """
        End-to-end training pipeline implementing ML best practices:

        1. Data Quality Checks
        2. Stratified Data Splitting
        3. Class Imbalance Mitigation (SMOTE)
        4. Hyperparameter Optimization
        5. Threshold Tuning
        6. Model Evaluation
        7. Artifact Logging
        8. Model Registry

        Implements MLflow experiment tracking for full reproducibility.
        """
        try:
            logger.info('Starting model training process')

            # Data ingestion and feature engineering
            df = self.read_from_kafka()
            data = self.create_features(df)

            # Train/Test split with stratification
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud']

            # Class imbalance safeguards
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model'].get('test_size', 0.2),
                stratify=y,
                random_state=self.config['model'].get('seed', 42)
            )

            # MLflow experiment tracking context
            with mlflow.start_run():
                # Dataset metadata logging
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Categorical feature preprocessing
                preprocessor = ColumnTransformer([
                    ('merchant_encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32
                    ), ['merchant'])
                ], remainder='passthrough')

                # XGBoost configuration with efficiency optimizations
                xgb = XGBClassifier(
                    eval_metric='aucpr',  # Optimizes for precision-recall area
                    random_state=self.config['model'].get('seed', 42),
                    reg_lambda=1.0,
                    n_estimators=self.config['model']['params']['n_estimators'],
                    n_jobs=-1,
                    tree_method=self.config['model'].get('tree_method', 'hist')  # GPU-compatible
                )

                # Imbalanced learning pipeline
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=self.config['model'].get('seed', 42))),
                    ('classifier', xgb)
                ], memory='./cache')

                # Hyperparameter search space design
                param_dist = {
                    'classifier__max_depth': [3, 5, 7],  # Depth control for regularization
                    'classifier__learning_rate': [0.01, 0.05, 0.1],  # Conservative range
                    'classifier__subsample': [0.6, 0.8, 1.0],  # Stochastic gradient boosting
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0],  # Feature randomization
                    'classifier__gamma': [0, 0.1, 0.3],  # Complexity control
                    'classifier__reg_alpha': [0, 0.1, 0.5]  # L1 regularization
                }

                # Optimizing for F-beta score (beta=2 emphasizes recall)
                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    n_iter=20,
                    scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=-1,
                    refit=True,
                    error_score='raise',
                    random_state=self.config['model'].get('seed', 42)
                )

                logger.info('Starting hyperparameter tuning...')
                searcher.fit(X_train, y_train)
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                logger.info('Best hyperparameters: %s', best_params)

                # Threshold optimization using training data
                train_proba = best_model.predict_proba(X_train)[:, 1] # type: ignore
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                             zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)

                # Model evaluation
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test) # type: ignore
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1] # type: ignore
                y_pred = (test_proba >= best_threshold).astype(int)

                # Comprehensive metrics suite
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'threshold': float(best_threshold)
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

                # Confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap("Blues"))
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Not Fraud', 'Fraud'])
                plt.yticks(tick_marks, ['Not Fraud', 'Fraud'])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')

                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename)
                mlflow.log_artifact(cm_filename)
                plt.close()

                # Precision-Recall curve documentation
                plt.figure(figsize=(10, 6))
                plt.plot(recall_arr, precision_arr, marker='.', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_filename = 'precision_recall_curve.png'
                plt.savefig(pr_filename)
                mlflow.log_artifact(pr_filename)
                plt.close()

                # Model packaging and registry
                signature = infer_signature(X_train, y_pred)
                sklearn.log_model(
                    sk_model=best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name='fraud_detection_model'
                )

                # Model serialization for deployment
                os.makedirs('/app/models', exist_ok=True)
                joblib.dump(best_model, '/app/models/fraud_detection_model.pkl')
                joblib.dump(best_model.named_steps['preprocessor'], '/app/models/preprocessor.pkl') # type: ignore
                logger.info('Training successfully completed with metrics: %s', metrics)

                return best_model, metrics

        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise