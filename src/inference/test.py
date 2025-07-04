"""
Basic Kafka Data Consumer for Testing Fraud Detection Pipeline

This script consumes transaction data from a Kafka topic, validates the schema
against the producer's TRANSACTION_SCHEMA, and logs sample messages for debugging
and testing the inference pipeline.
"""

import json
import logging
import os
from typing import Dict, Any
from kafka import KafkaConsumer
import yaml
from dotenv import load_dotenv
from jsonschema import validate, ValidationError, FormatChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./test_consumer.log")
    ]
)
logger = logging.getLogger(__name__)

# JSON Schema for transaction validation (aligned with producer)
TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "user_id": {"type": "number", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 100000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "timestamp": {
            "type": "string",
            "format": "date-time"
        },
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1}
    },
    "required": ["transaction_id", "user_id", "amount", "currency", "timestamp", "is_fraud"]
}

class KafkaTestConsumer:
    """Kafka consumer for testing data ingestion from TransactionProducer."""

    def __init__(self, config_path: str = "/app/config.yaml"):
        """Initialize consumer with configuration."""
        load_dotenv(dotenv_path="/app/.env")
        self.config = self._load_config(config_path)
        self.consumer = self._init_consumer()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded from %s", config_path)
            return config
        except Exception as e:
            logger.error("Error loading config: %s", str(e))
            raise

    def _init_consumer(self) -> KafkaConsumer:
        """Initialize Kafka consumer with SASL/SSL settings."""
        try:
            kafka_config = self.config["kafka"]
            consumer = KafkaConsumer(
                kafka_config["topic"],
                bootstrap_servers=kafka_config.get("bootstrap_servers", "localhost:9092").split(","),
                security_protocol=kafka_config.get("security_protocol", "SASL_SSL"),
                sasl_mechanism=kafka_config.get("sasl_mechanism", "PLAIN"),
                sasl_plain_username=kafka_config.get("username"),
                sasl_plain_password=kafka_config.get("password"),
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
                consumer_timeout_ms=kafka_config.get("timeout", 10000),
                group_id="test-consumer-group"
            )
            logger.info("Kafka consumer initialized for topic %s", kafka_config["topic"])
            return consumer
        except Exception as e:
            logger.error("Error initializing Kafka consumer: %s", str(e))
            raise

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message against TRANSACTION_SCHEMA with date-time checking."""
        try:
            validate(
                instance=message,
                schema=TRANSACTION_SCHEMA,
                format_checker=FormatChecker()
            )
            return True
        except ValidationError as e:
            logger.error(f"Invalid transaction: {e.message}")
            return False
        except Exception as e:
            logger.error(f"Error validating message: {str(e)}")
            return False

    def consume(self, max_messages: int = 10):
        """Consume and log messages from Kafka topic."""
        logger.info("Starting to consume messages from Kafka")
        message_count = 0
        fraud_count = 0
        try:
            for message in self.consumer:
                if message_count >= max_messages:
                    logger.info("Reached max messages (%d), stopping", max_messages)
                    logger.info("Fraud rate: %.2f%%", (fraud_count / max_messages * 100) if max_messages > 0 else 0)
                    break
                if self.validate_message(message.value):
                    logger.info("Valid message received: %s", message.value)
                    message_count += 1
                    if message.value.get("is_fraud", 0) == 1:
                        fraud_count += 1
                else:
                    logger.warning("Invalid message skipped: %s", message.value)
        except Exception as e:
            logger.error("Error consuming messages: %s", str(e))
        finally:
            self.consumer.close()
            logger.info("Kafka consumer closed")

if __name__ == "__main__":
    try:
        consumer = KafkaTestConsumer("./config.yaml")
        consumer.consume(max_messages=50000)
    except Exception as e:
        logger.error("Test consumer failed: %s", str(e))    