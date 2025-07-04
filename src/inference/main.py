"""
Batch Fraud Detection Inference Pipeline

This script processes transaction data in batches from Kafka, applies feature engineering,
runs fraud detection using a pre-trained machine learning model, and writes predictions
back to Kafka.
"""

# Standard library imports
import logging
import os
import math

# Third-party imports
import joblib  # For loading serialized ML models
import numpy as np
import pandas as pd
import yaml  # For parsing YAML configuration files
from dotenv import load_dotenv  # For loading environment variables from .env file
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, hour, dayofmonth, dayofweek, when, lit, coalesce,
    sin, cos, count, sum as _sum, avg, stddev, rank, abs as spark_abs,
    unix_timestamp, lag, max as spark_max, min as spark_min, mean as spark_mean,
    desc, asc, collect_list, size, isnan, isnull
)
from pyspark.sql.window import Window
from pyspark.sql.pandas.functions import pandas_udf  # For Pandas vectorized UDFs
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, TimestampType, FloatType)

# Configure logging to track pipeline operations and errors
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO for operational messages
    format="%(asctime)s [%(levelname)s] %(message)s"  # Structured log format
)
logger = logging.getLogger(__name__)  # Create logger instance for the module


class FraudDetectionBatchInference:
    """
    Batch fraud detection inference pipeline that implements comprehensive feature engineering
    aligned with the training pipeline features.
    """

    # Class variables for Kafka configuration
    bootstrap_servers = None
    topic = None
    security_protocol = None
    sasl_mechanism = None
    username = None
    password = None
    sasl_jaas_config = None

    def __init__(self, config_path="/app/config.yaml"):
        """Initialize pipeline with configuration and dependencies"""
        load_dotenv(dotenv_path="/app/.env")
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
        self.model = self._load_model(self.config["model"]["path"])
        self.preprocessor = self._load_preprocessor(self.config["preprocessor"]["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        self.broadcast_preprocessor = self.spark.sparkContext.broadcast(self.preprocessor)
        logger.debug("Environment variables loaded: %s", dict(os.environ))

    def _load_model(self, model_path):
        """Load pre-trained fraud detection model from disk"""
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
            return model
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

    def _load_preprocessor(self, preprocessor_path):
        """Load pre-fitted preprocessor from disk"""
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded from %s", preprocessor_path)
            return preprocessor
        except Exception as e:
            logger.error("Error loading preprocessor: %s", str(e))
            raise

    @staticmethod
    def _load_config(config_path):
        """Load YAML configuration file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _init_spark_session(self):
        """Initialize Spark session with Kafka dependencies"""
        try:
            spark_config = self.config.get("spark", {})
            packages = spark_config.get("packages", "")

            builder = SparkSession.builder.appName("FraudDetectionBatchInference")

            if packages:
                builder = builder.config("spark.jars.packages", packages)

            spark = builder.getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")  # suppress Spark/Kafka INFO logs
            logger.info("SparkSession initialized successfully.")
            return spark

        except Exception as e:
            logger.exception("Failed to initialize SparkSession.")
            raise RuntimeError("SparkSession initialization failed") from e

    def read_batch_from_kafka(self, start_offset="earliest", end_offset="latest"):
        """Read batch data from Kafka topic and parse JSON payload"""
        logger.info("Reading batch data from Kafka topic %s", self.config["kafka"]["topic"])

        kafka_config = self.config["kafka"]
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
        kafka_topic = kafka_config["topic"]
        kafka_security_protocol = kafka_config.get("security_protocol", "SASL_SSL")
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username")
        kafka_password = kafka_config.get("password")

        kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.plain.PlainLoginModule required '
            f'username="{kafka_username}" password="{kafka_password}";'
        )

        # Store configuration for reuse
        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        self.sasl_jaas_config = kafka_sasl_jaas_config

        # Define schema for incoming JSON transaction data
        json_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("location", StringType(), True),
        ])

        # Create batch DataFrame from Kafka source
        df = self.spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("subscribe", kafka_topic) \
            .option("startingOffsets", start_offset) \
            .option("endingOffsets", end_offset) \
            .option("kafka.security.protocol", kafka_security_protocol) \
            .option("kafka.sasl.mechanism", kafka_sasl_mechanism) \
            .option("kafka.sasl.jaas.config", kafka_sasl_jaas_config) \
            .load()

        # Parse JSON payload
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), json_schema).alias("data")) \
            .select("data.*")
                    
        return parsed_df

    def add_features(self, df):
        """
        Enhanced feature engineering aligned with training pipeline.
        Optimized for batch processing with proper window functions.
        """
        # ---- Enhanced Temporal Features ----
        df = df.withColumn("transaction_hour", hour(col("timestamp")))
        
        # Cyclical encoding for temporal features
        df = df.withColumn("hour_sin", 
                          sin(lit(2 * math.pi) * col("transaction_hour") / lit(24)))
        df = df.withColumn("hour_cos", 
                          cos(lit(2 * math.pi) * col("transaction_hour") / lit(24)))
        
        df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
        df = df.withColumn("dow_sin", 
                          sin(lit(2 * math.pi) * col("day_of_week") / lit(7)))
        df = df.withColumn("dow_cos", 
                          cos(lit(2 * math.pi) * col("day_of_week") / lit(7)))
        
        # Traditional time indicators
        df = df.withColumn("is_night",
                          when((col("transaction_hour") >= 22) | (col("transaction_hour") < 5), 1).otherwise(0))
        df = df.withColumn("is_weekend",
                          when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
        df = df.withColumn("transaction_day", dayofmonth(col("timestamp")))

        # ---- Velocity Features (Batch-Optimized) ----
        df = df.withColumn("unix_ts", unix_timestamp("timestamp"))

        # Define time-based windows for velocity calculations
        user_time_window = Window.partitionBy("user_id").orderBy("unix_ts")
        
        # 1-hour window velocity features
        user_1h_window = user_time_window.rangeBetween(-3600, -1)
        df = df.withColumn("txn_count_1H", count("*").over(user_1h_window))
        df = df.withColumn("txn_sum_1H", coalesce(_sum("amount").over(user_1h_window), lit(0.0)))
        df = df.withColumn("txn_avg_1H", 
                          when(col("txn_count_1H") > 0, col("txn_sum_1H") / col("txn_count_1H")).otherwise(0.0))

        # 6-hour window velocity features
        user_6h_window = user_time_window.rangeBetween(-21600, -1)
        df = df.withColumn("txn_count_6H", count("*").over(user_6h_window))
        df = df.withColumn("txn_sum_6H", coalesce(_sum("amount").over(user_6h_window), lit(0.0)))
        df = df.withColumn("txn_avg_6H", 
                          when(col("txn_count_6H") > 0, col("txn_sum_6H") / col("txn_count_6H")).otherwise(0.0))

        # 24-hour window velocity features
        user_24h_window = user_time_window.rangeBetween(-86400, -1)
        df = df.withColumn("txn_count_24H", count("*").over(user_24h_window))
        df = df.withColumn("txn_sum_24H", coalesce(_sum("amount").over(user_24h_window), lit(0.0)))
        df = df.withColumn("txn_avg_24H", 
                          when(col("txn_count_24H") > 0, col("txn_sum_24H") / col("txn_count_24H")).otherwise(0.0))
        
        # ---- Behavioral Features ----
        # 7-day, 14-day, and 30-day windows for behavioral analysis
        user_7d_window = user_time_window.rangeBetween(-604800, -1)
        user_14d_window = user_time_window.rangeBetween(-1209600, -1)
        user_30d_window = user_time_window.rangeBetween(-2592000, -1)

        df = df.withColumn("amount_avg_7d", coalesce(avg("amount").over(user_7d_window), lit(0.0)))
        df = df.withColumn("amount_avg_14d", coalesce(avg("amount").over(user_14d_window), lit(0.0)))
        df = df.withColumn("amount_avg_30d", coalesce(avg("amount").over(user_30d_window), lit(0.0)))
        df = df.withColumn("amount_mean_30", col("amount_avg_30d"))  # Alias for consistency
        df = df.withColumn("amount_std_30", coalesce(stddev("amount").over(user_30d_window), lit(1.0)))

        # Calculate z-score
        df = df.withColumn("amount_zscore_30", 
                          (col("amount") - col("amount_mean_30")) / col("amount_std_30"))

        # Calculate amount ratios
        df = df.withColumn("amount_to_avg_ratio_7d", 
                          coalesce(col("amount") / col("amount_avg_7d"), lit(1.0)))
        df = df.withColumn("amount_to_avg_ratio_14d", 
                          coalesce(col("amount") / col("amount_avg_14d"), lit(1.0)))
        df = df.withColumn("amount_to_avg_ratio_30d", 
                          coalesce(col("amount") / col("amount_avg_30d"), lit(1.0)))

        # Amount percentile calculation
        @pandas_udf(DoubleType()) # type: ignore
        def compute_percentile(amount_list: pd.Series, current_amount: pd.Series) -> pd.Series:
            result = []
            for alist, current in zip(amount_list, current_amount):
                try:
                    alist = list(alist) if alist is not None else []
                    if len(alist) < 2:
                        result.append(0.0)
                    else:
                        result.append(float(np.sum(np.array(alist) <= current) / len(alist) * 100))
                except Exception:
                    result.append(0.0)
            return pd.Series(result)
        
        # Collect historical amounts for percentile calculation
        df = df.withColumn("amount_list", 
                          collect_list("amount").over(user_30d_window))
        df = df.withColumn("amount_percentile", 
                          compute_percentile(col("amount_list"), col("amount"))) # type: ignore
        df = df.drop("amount_list")
        
        # ---- Merchant Intelligence ----
        # Merchant fraud rate (simplified - use static mapping for batch)
        merchant_fraud_rates = self.config.get('merchant_fraud_rates', {})
        default_fraud_rate = self.config.get('default_merchant_fraud_rate', 0.1)
        
        # Create merchant fraud rate mapping
        merchant_mapping = lit(default_fraud_rate)
        for merchant_name, fraud_rate in merchant_fraud_rates.items():
            merchant_mapping = when(col("merchant") == merchant_name, lit(fraud_rate)).otherwise(merchant_mapping)
        
        df = df.withColumn("merchant_fraud_rate", merchant_mapping)
        
        # Merchant popularity (simplified calculation)
        df = df.withColumn("merchant_popularity", lit(0.001))  # Placeholder for batch
        
        # High-risk merchant detection
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df = df.withColumn("merchant_risk_static", 
                          col("merchant").isin(high_risk_merchants).cast("int"))
        
        # New merchant detection for user (simplified)
        df = df.withColumn("new_merchant_for_user", lit(0))  # Simplified for batch

        # ---- Time-based Features ----
        # Days since last transaction
        df = df.withColumn("prev_unix_ts", 
                          lag("unix_ts", 1).over(user_time_window))
        df = df.withColumn("days_since_last_txn", 
                          coalesce((col("unix_ts") - col("prev_unix_ts")) / lit(86400.0), lit(0.0)))
        
        # Hour deviation from user's normal pattern (simplified)
        df = df.withColumn("hour_deviation_from_normal", 
                          spark_abs(col("transaction_hour") - lit(12.0)).cast("double"))

        # ---- Interaction Features ----
        df = df.withColumn("amount_x_fraud_rate", col("amount") * col("merchant_fraud_rate"))
        df = df.withColumn("velocity_x_amount", col("txn_count_1H") * col("amount"))
        df = df.withColumn("night_x_new_merchant", col("is_night") * col("new_merchant_for_user"))
        df = df.withColumn("weekend_x_high_amount", 
                          col("is_weekend") * when(col("amount_zscore_30") > 2, 1).otherwise(0))
        df = df.withColumn("high_velocity_x_night", 
                          when(col("txn_count_1H") > 3, 1).otherwise(0) * col("is_night"))

        # ---- Feature Selection and Type Casting ----
        feature_columns = [
            "amount", "merchant", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night", "is_weekend", 
            "transaction_day", "txn_count_1H", "txn_count_6H", "txn_count_24H", "txn_sum_1H", 
            "txn_sum_6H", "txn_sum_24H", "txn_avg_1H", "txn_avg_6H", "txn_avg_24H", 
            "amount_zscore_30", "amount_to_avg_ratio_7d", "amount_to_avg_ratio_14d", 
            "amount_to_avg_ratio_30d", "amount_percentile", "merchant_fraud_rate", 
            "new_merchant_for_user", "merchant_popularity", "merchant_risk_static",
            "amount_x_fraud_rate", "velocity_x_amount", "night_x_new_merchant",
            "weekend_x_high_amount", "high_velocity_x_night", "days_since_last_txn", 
            "hour_deviation_from_normal"
        ]
        
        # Ensure all features are double type and handle nulls
        for col_name in feature_columns:
            if col_name in [c.name for c in df.schema.fields]:
                df = df.withColumn(col_name, coalesce(col(col_name).cast("double"), lit(0.0)))

        return df

    def run_batch_inference(self, start_offset="earliest", end_offset="latest"):
        """Main pipeline execution flow: process batch and run predictions"""
        import pandas as pd

        # Process batch data from Kafka
        df = self.read_batch_from_kafka(start_offset, end_offset)
        
        if df.count() == 0:
            logger.info("No data found in the specified offset range")
            return
        
        # Add comprehensive features
        feature_df = self.add_features(df)
        
        broadcast_model = self.broadcast_model
        broadcast_preprocessor = self.broadcast_preprocessor

        # Enhanced prediction UDF
        @pandas_udf("double") # type: ignore
        def predict_udf(
                amount: pd.Series, merchant: pd.Series, hour_sin: pd.Series, hour_cos: pd.Series,
                dow_sin: pd.Series, dow_cos: pd.Series, is_night: pd.Series,
                is_weekend: pd.Series, transaction_day: pd.Series,
                txn_count_1H: pd.Series, txn_count_6H: pd.Series, txn_count_24H: pd.Series,
                txn_sum_1H: pd.Series, txn_sum_6H: pd.Series, txn_sum_24H: pd.Series,
                txn_avg_1H: pd.Series, txn_avg_6H: pd.Series, txn_avg_24H: pd.Series,
                amount_zscore_30: pd.Series, amount_to_avg_ratio_7d: pd.Series,
                amount_to_avg_ratio_14d: pd.Series, amount_to_avg_ratio_30d: pd.Series,
                amount_percentile: pd.Series, merchant_fraud_rate: pd.Series,
                new_merchant_for_user: pd.Series, merchant_popularity: pd.Series,
                merchant_risk_static: pd.Series, amount_x_fraud_rate: pd.Series,
                velocity_x_amount: pd.Series, night_x_new_merchant: pd.Series,
                weekend_x_high_amount: pd.Series, high_velocity_x_night: pd.Series,
                days_since_last_txn: pd.Series, hour_deviation_from_normal: pd.Series
        ) -> pd.Series:
            """Enhanced vectorized UDF for fraud prediction"""
            
            # Create feature DataFrame matching training format
            input_df = pd.DataFrame({
                "amount": amount, "merchant": merchant, "hour_sin": hour_sin, "hour_cos": hour_cos,
                "dow_sin": dow_sin, "dow_cos": dow_cos, "is_night": is_night,
                "is_weekend": is_weekend, "transaction_day": transaction_day,
                "txn_count_1H": txn_count_1H, "txn_count_6H": txn_count_6H, 
                "txn_count_24H": txn_count_24H, "txn_sum_1H": txn_sum_1H,
                "txn_sum_6H": txn_sum_6H, "txn_sum_24H": txn_sum_24H,
                "txn_avg_1H": txn_avg_1H, "txn_avg_6H": txn_avg_6H, "txn_avg_24H": txn_avg_24H,
                "amount_zscore_30": amount_zscore_30, 
                "amount_to_avg_ratio_7d": amount_to_avg_ratio_7d,
                "amount_to_avg_ratio_14d": amount_to_avg_ratio_14d,
                "amount_to_avg_ratio_30d": amount_to_avg_ratio_30d,
                "amount_percentile": amount_percentile,
                "merchant_fraud_rate": merchant_fraud_rate,
                "new_merchant_for_user": new_merchant_for_user,
                "merchant_popularity": merchant_popularity,
                "merchant_risk_static": merchant_risk_static,
                "amount_x_fraud_rate": amount_x_fraud_rate,
                "velocity_x_amount": velocity_x_amount,
                "night_x_new_merchant": night_x_new_merchant,
                "weekend_x_high_amount": weekend_x_high_amount,
                "high_velocity_x_night": high_velocity_x_night,
                "days_since_last_txn": days_since_last_txn,
                "hour_deviation_from_normal": hour_deviation_from_normal
            })
            
            # Apply preprocessor to encode merchant column
            try:
                # Apply the preprocessor (expects merchant as a string or object)
                transformed_data = broadcast_preprocessor.value.transform(input_df)
                # Convert to DataFrame with correct column names
                feature_columns = (
                    ['merchant'] +  # OrdinalEncoder outputs a single column for merchant
                    [col for col in input_df.columns if col != 'merchant']
                )
                input_df = pd.DataFrame(transformed_data, columns=feature_columns)
            except Exception as e:
                logger.error(f"Preprocessing error: {str(e)}")
                return pd.Series([0.0] * len(input_df), dtype='float64')
            
            # Handle missing values and ensure proper types
            input_df = input_df.fillna(0.0).astype('float32')
            
            # Log for debugging
            logger.info(f"Input DataFrame dtypes:\n{input_df.dtypes}")
            logger.info(f"Null counts:\n{input_df.isnull().sum()}")
            
            # Get predictions
            try:
                probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
                return pd.Series(probabilities, dtype='float64')
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return pd.Series([0.0] * len(input_df), dtype='float64')

        feature_columns = [
            col("amount"), col("merchant"), col("hour_sin"), col("hour_cos"), col("dow_sin"), col("dow_cos"),
            col("is_night"), col("is_weekend"), col("transaction_day"), col("txn_count_1H"),
            col("txn_count_6H"), col("txn_count_24H"), col("txn_sum_1H"), col("txn_sum_6H"), col("txn_sum_24H"),
            col("txn_avg_1H"), col("txn_avg_6H"), col("txn_avg_24H"), col("amount_zscore_30"),
            col("amount_to_avg_ratio_7d"), col("amount_to_avg_ratio_14d"), col("amount_to_avg_ratio_30d"),
            col("amount_percentile"), col("merchant_fraud_rate"), col("new_merchant_for_user"),
            col("merchant_popularity"), col("merchant_risk_static"), col("amount_x_fraud_rate"),
            col("velocity_x_amount"), col("night_x_new_merchant"), col("weekend_x_high_amount"),
            col("high_velocity_x_night"), col("days_since_last_txn"), col("hour_deviation_from_normal")
        ]
        # Apply predictions
        prediction_df = feature_df.withColumn("fraud_probability", predict_udf(*feature_columns))  # type: ignore
def run_batch_inference(self, start_offset="earliest", end_offset="latest"):
        """Main pipeline execution flow: process batch and run predictions"""
        import pandas as pd

        # Process batch data from Kafka
        df = self.read_batch_from_kafka(start_offset, end_offset)
        
        if df.count() == 0:
            logger.info("No data found in the specified offset range")
            return
        
        # Add comprehensive features
        feature_df = self.add_features(df)
        
        broadcast_model = self.broadcast_model
        broadcast_preprocessor = self.broadcast_preprocessor

        # Enhanced prediction UDF
        @pandas_udf("double") # type: ignore
        def predict_udf(
                amount: pd.Series, merchant: pd.Series, hour_sin: pd.Series, hour_cos: pd.Series,
                dow_sin: pd.Series, dow_cos: pd.Series, is_night: pd.Series,
                is_weekend: pd.Series, transaction_day: pd.Series,
                txn_count_1H: pd.Series, txn_count_6H: pd.Series, txn_count_24H: pd.Series,
                txn_sum_1H: pd.Series, txn_sum_6H: pd.Series, txn_sum_24H: pd.Series,
                txn_avg_1H: pd.Series, txn_avg_6H: pd.Series, txn_avg_24H: pd.Series,
                amount_zscore_30: pd.Series, amount_to_avg_ratio_7d: pd.Series,
                amount_to_avg_ratio_14d: pd.Series, amount_to_avg_ratio_30d: pd.Series,
                amount_percentile: pd.Series, merchant_fraud_rate: pd.Series,
                new_merchant_for_user: pd.Series, merchant_popularity: pd.Series,
                merchant_risk_static: pd.Series, amount_x_fraud_rate: pd.Series,
                velocity_x_amount: pd.Series, night_x_new_merchant: pd.Series,
                weekend_x_high_amount: pd.Series, high_velocity_x_night: pd.Series,
                days_since_last_txn: pd.Series, hour_deviation_from_normal: pd.Series
        ) -> pd.Series:
            """Enhanced vectorized UDF for fraud prediction"""
            
            # Create feature DataFrame matching training format
            input_df = pd.DataFrame({
                "amount": amount, "merchant": merchant, "hour_sin": hour_sin, "hour_cos": hour_cos,
                "dow_sin": dow_sin, "dow_cos": dow_cos, "is_night": is_night,
                "is_weekend": is_weekend, "transaction_day": transaction_day,
                "txn_count_1H": txn_count_1H, "txn_count_6H": txn_count_6H, 
                "txn_count_24H": txn_count_24H, "txn_sum_1H": txn_sum_1H,
                "txn_sum_6H": txn_sum_6H, "txn_sum_24H": txn_sum_24H,
                "txn_avg_1H": txn_avg_1H, "txn_avg_6H": txn_avg_6H, "txn_avg_24H": txn_avg_24H,
                "amount_zscore_30": amount_zscore_30, 
                "amount_to_avg_ratio_7d": amount_to_avg_ratio_7d,
                "amount_to_avg_ratio_14d": amount_to_avg_ratio_14d,
                "amount_to_avg_ratio_30d": amount_to_avg_ratio_30d,
                "amount_percentile": amount_percentile,
                "merchant_fraud_rate": merchant_fraud_rate,
                "new_merchant_for_user": new_merchant_for_user,
                "merchant_popularity": merchant_popularity,
                "merchant_risk_static": merchant_risk_static,
                "amount_x_fraud_rate": amount_x_fraud_rate,
                "velocity_x_amount": velocity_x_amount,
                "night_x_new_merchant": night_x_new_merchant,
                "weekend_x_high_amount": weekend_x_high_amount,
                "high_velocity_x_night": high_velocity_x_night,
                "days_since_last_txn": days_since_last_txn,
                "hour_deviation_from_normal": hour_deviation_from_normal
            })
            
            # Apply preprocessor to encode merchant column
            try:
                # Apply the preprocessor (expects merchant as a string or object)
                transformed_data = broadcast_preprocessor.value.transform(input_df)
                # Convert to DataFrame with correct column names
                feature_columns = (
                    ['merchant'] +  # OrdinalEncoder outputs a single column for merchant
                    [col for col in input_df.columns if col != 'merchant']
                )
                input_df = pd.DataFrame(transformed_data, columns=feature_columns)
            except Exception as e:
                logger.error(f"Preprocessing error: {str(e)}")
                return pd.Series([0.0] * len(input_df), dtype='float64')
            
            # Handle missing values and ensure proper types
            input_df = input_df.fillna(0.0).astype('float32')
            
            # Log for debugging
            logger.info(f"Input DataFrame dtypes:\n{input_df.dtypes}")
            logger.info(f"Null counts:\n{input_df.isnull().sum()}")
            
            # Get predictions
            try:
                probabilities = broadcast_model.value.predict_proba(input_df)[:, 1]
                return pd.Series(probabilities, dtype='float64')
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return pd.Series([0.0] * len(input_df), dtype='float64')

        feature_columns = [
            col("amount"), col("merchant"), col("hour_sin"), col("hour_cos"), col("dow_sin"), col("dow_cos"),
            col("is_night"), col("is_weekend"), col("transaction_day"), col("txn_count_1H"),
            col("txn_count_6H"), col("txn_count_24H"), col("txn_sum_1H"), col("txn_sum_6H"), col("txn_sum_24H"),
            col("txn_avg_1H"), col("txn_avg_6H"), col("txn_avg_24H"), col("amount_zscore_30"),
            col("amount_to_avg_ratio_7d"), col("amount_to_avg_ratio_14d"), col("amount_to_avg_ratio_30d"),
            col("amount_percentile"), col("merchant_fraud_rate"), col("new_merchant_for_user"),
            col("merchant_popularity"), col("merchant_risk_static"), col("amount_x_fraud_rate"),
            col("velocity_x_amount"), col("night_x_new_merchant"), col("weekend_x_high_amount"),
            col("high_velocity_x_night"), col("days_since_last_txn"), col("hour_deviation_from_normal")
        ]
        
        # Apply predictions
        prediction_df = feature_df.withColumn("fraud_probability", predict_udf(*feature_columns)) # type: ignore

        # Apply threshold to get binary prediction
        fraud_threshold = self.config.get('fraud_threshold', 0.45)
        prediction_df = prediction_df.withColumn("prediction", 
                                                when(col("fraud_probability") >= fraud_threshold, 1).otherwise(0))

        # ===== COMPREHENSIVE FRAUD DETECTION STATISTICS =====
        
        # Get total counts
        total_count = prediction_df.count()
        fraud_count = prediction_df.filter(col("prediction") == 1).count()
        legitimate_count = total_count - fraud_count
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0
        
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Transactions Processed: {total_count:,}")
        logger.info(f"Fraudulent Transactions: {fraud_count:,}")
        logger.info(f"Legitimate Transactions: {legitimate_count:,}")
        logger.info(f"Fraud Rate: {fraud_rate:.2f}%")
        logger.info(f"Fraud Threshold Used: {fraud_threshold}")
        logger.info("=" * 60)
        
        # Statistics on fraud probabilities
        logger.info("FRAUD PROBABILITY DISTRIBUTION:")
        logger.info("-" * 40)
        probability_stats = prediction_df.select("fraud_probability").describe()
        
        # Collect stats to log them
        prob_stats_data = probability_stats.collect()
        for row in prob_stats_data:
            logger.info(f"{row['summary']}: {row['fraud_probability']}")
        
        # Fraud probability ranges
        prob_ranges = prediction_df.groupBy(
            when(col("fraud_probability") < 0.1, "Very Low (< 0.1)")
            .when(col("fraud_probability") < 0.3, "Low (0.1 - 0.3)")
            .when(col("fraud_probability") < 0.5, "Medium (0.3 - 0.5)")
            .when(col("fraud_probability") < 0.7, "High (0.5 - 0.7)")
            .when(col("fraud_probability") < 0.9, "Very High (0.7 - 0.9)")
            .otherwise("Extreme (≥ 0.9)").alias("Risk_Level")
        ).count().orderBy(desc("count"))
        
        logger.info("TRANSACTIONS BY RISK LEVEL:")
        logger.info("-" * 40)
        risk_level_data = prob_ranges.collect()
        for row in risk_level_data:
            logger.info(f"{row['Risk_Level']}: {row['count']:,} transactions")
            
             
        # Filter fraud predictions
        fraud_predictions = prediction_df.filter(col("prediction") == 1)
        
        fraud_count = fraud_predictions.count()
        total_count = prediction_df.count()
        
        logger.info(f"Processed {total_count} transactions, found {fraud_count} fraud cases")

        # Write to Kafka
        output_topic = self.config.get('output_topic', 'fraud_predictions')
        
        (fraud_predictions.selectExpr(
            "CAST(transaction_id AS STRING) AS key",
            "to_json(struct(*)) AS value"
        )
         .write
         .format("kafka")
         .option("kafka.bootstrap.servers", self.bootstrap_servers)
         .option("topic", output_topic)
         .option("kafka.security.protocol", self.security_protocol)
         .option("kafka.sasl.mechanism", self.sasl_mechanism)
         .option("kafka.sasl.jaas.config", self.sasl_jaas_config)
         .save())
        
        logger.info(f"Successfully wrote {fraud_count} fraud predictions to topic {output_topic}")


if __name__ == "__main__":
    """Main entry point for the batch inference pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Fraud Detection Inference')
    parser.add_argument('--start-offset', default='earliest', help='Starting offset for batch processing')
    parser.add_argument('--end-offset', default='latest', help='Ending offset for batch processing')
    parser.add_argument('--config', default='./config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    inference = FraudDetectionBatchInference(args.config)
    inference.run_batch_inference(args.start_offset, args.end_offset)