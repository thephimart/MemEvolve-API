"""Tests for logging infrastructure."""

from utils.logging import (
    OperationLogger,
    setup_logging,
    get_logger,
    configure_from_config,
    StructuredLogger
)
from pathlib import Path
import logging
import json
import tempfile
import pytest
import sys
sys.path.insert(0, 'src')


class TestOperationLogger:
    """Test operation logger."""

    def test_initialization(self):
        logger = OperationLogger()
        assert logger.enable_logging is True
        assert logger.max_entries == 10000
        assert logger.operations == []

    def test_initialization_disabled(self):
        logger = OperationLogger(enable_logging=False)
        assert logger.enable_logging is False

    def test_initialization_custom_max(self):
        logger = OperationLogger(max_entries=100)
        assert logger.max_entries == 100

    def test_log_operation(self):
        logger = OperationLogger()
        logger.log("test_operation", {"key": "value"})
        assert len(logger.operations) == 1
        assert logger.operations[0]["operation"] == "test_operation"
        assert logger.operations[0]["details"]["key"] == "value"

    def test_log_disabled(self):
        logger = OperationLogger(enable_logging=False)
        logger.log("test_operation", {"key": "value"})
        assert len(logger.operations) == 0

    def test_log_with_metadata(self):
        logger = OperationLogger()
        logger.log("test", {"data": "test"}, metadata={"agent_id": "test_1"})
        assert logger.operations[0]["metadata"]["agent_id"] == "test_1"

    def test_max_entries_limit(self):
        logger = OperationLogger(max_entries=5)
        for i in range(10):
            logger.log("test", {"index": i})

        assert len(logger.operations) == 5
        assert logger.operations[0]["details"]["index"] == 5
        assert logger.operations[-1]["details"]["index"] == 9

    def test_get_operations_all(self):
        logger = OperationLogger()
        logger.log("op1", {})
        logger.log("op2", {})
        logger.log("op1", {})

        ops = logger.get_operations()
        assert len(ops) == 3

    def test_get_operations_filtered(self):
        logger = OperationLogger()
        logger.log("encode", {"type": "lesson"})
        logger.log("retrieve", {"query": "test"})
        logger.log("encode", {"type": "skill"})

        encode_ops = logger.get_operations("encode")
        assert len(encode_ops) == 2
        assert all(op["operation"] == "encode" for op in encode_ops)

    def test_get_operations_with_limit(self):
        logger = OperationLogger()
        for i in range(10):
            logger.log("test", {"index": i})

        ops = logger.get_operations(limit=3)
        assert len(ops) == 3
        assert ops[0]["details"]["index"] == 7
        assert ops[-1]["details"]["index"] == 9

    def test_get_operations_filtered_with_limit(self):
        logger = OperationLogger()
        for i in range(10):
            logger.log(f"op_{i % 3}", {"index": i})

        ops = logger.get_operations("op_1", limit=2)
        assert len(ops) == 2
        assert all(op["operation"] == "op_1" for op in ops)

    def test_get_stats_empty(self):
        logger = OperationLogger()
        stats = logger.get_stats()
        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["first_timestamp"] is None
        assert stats["last_timestamp"] is None

    def test_get_stats_with_operations(self):
        logger = OperationLogger()
        logger.log("encode", {})
        logger.log("retrieve", {})
        logger.log("encode", {})

        stats = logger.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["encode"] == 2
        assert stats["by_type"]["retrieve"] == 1
        assert stats["first_timestamp"] is not None
        assert stats["last_timestamp"] is not None

    def test_clear(self):
        logger = OperationLogger()
        logger.log("test", {})
        logger.clear()
        assert len(logger.operations) == 0

    def test_export(self):
        logger = OperationLogger()
        logger.log("test", {"data": "test"})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            temp_path = f.name

        try:
            logger.export(temp_path)
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                exported_data = json.load(f)

            assert len(exported_data) == 1
            assert exported_data[0]["operation"] == "test"
        finally:
            Path(temp_path).unlink()


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_default(self):
        logger = setup_logging()
        assert logger.level == 20
        assert len(logger.handlers) == 1

    def test_setup_with_level(self):
        logger = setup_logging(level="DEBUG")
        assert logger.level == 10

    def test_setup_with_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log',
                                         delete=False) as f:
            temp_path = f.name

        try:
            logger = setup_logging(log_file=temp_path)
            assert Path(temp_path).exists()
            assert len(logger.handlers) == 2

            logger.info("Test message")

            with open(temp_path, 'r') as f:
                content = f.read()

            assert "Test message" in content
        finally:
            Path(temp_path).unlink()

    def test_setup_custom_format(self):
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(log_format=custom_format)
        assert logger.handlers[0].formatter is not None

    def test_setup_invalid_level(self):
        logger = setup_logging(level="INVALID")
        assert logger.level == 20


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_instance(self):
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_same_instance(self):
        logger1 = get_logger("shared_logger")
        logger2 = get_logger("shared_logger")
        assert logger1 is logger2


class TestConfigureFromConfig:
    """Test configure from config dictionary."""

    def test_configure_from_dict(self):
        config = {
            "level": "DEBUG",
            "log_file": None,
            "format": "%(message)s"
        }
        logger = configure_from_config(config)
        assert logger.level == 10

    def test_configure_with_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log',
                                         delete=False) as f:
            temp_path = f.name

        try:
            config = {
                "level": "INFO",
                "log_file": temp_path,
                "format": "%(message)s"
            }
            logger = configure_from_config(config)
            assert len(logger.handlers) == 2
        finally:
            Path(temp_path).unlink()


class TestStructuredLogger:
    """Test structured logger."""

    def test_initialization(self):
        logger = StructuredLogger("test_structured")
        assert logger.logger.name == "test_structured"

    def test_log_with_context(self):
        logger = StructuredLogger("test_structured")
        logger.log("info", "Test message", key="value", num=42)

    def test_debug(self):
        logger = StructuredLogger("test_structured")
        logger.debug("Debug message", context="test")

    def test_info(self):
        logger = StructuredLogger("test_structured")
        logger.info("Info message", user="test")

    def test_warning(self):
        logger = StructuredLogger("test_structured")
        logger.warning("Warning message", severity="high")

    def test_error(self):
        logger = StructuredLogger("test_structured")
        logger.error("Error message", code=500)

    def test_critical(self):
        logger = StructuredLogger("test_structured")
        logger.critical("Critical message", system="down")

    def test_log_no_context(self):
        logger = StructuredLogger("test_structured")
        logger.info("Message without context")

    def test_log_with_context_formats(self):
        logger = StructuredLogger("test_structured")
        logger.info("Test", str_val="text", int_val=123, float_val=3.14)
