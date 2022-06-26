import logging

from colorlog import ColoredFormatter

IRSPACK_LOGGER_NAME = "IRSPACK"
_logger = logging.getLogger(IRSPACK_LOGGER_NAME)
_logger.propagate = False
_formatter = ColoredFormatter(
    "%(log_color)s[IRSPACK:%(levelname)-1.1s %(asctime)s]%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    },
)
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


def get_default_logger() -> logging.Logger:
    return _logger


def disable_default_handler() -> None:
    _logger.removeHandler(_handler)


__all__ = ["get_default_logger", "disable_default_handler"]
