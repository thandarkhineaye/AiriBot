import logging
import logging.config
import os

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '../log/main.log')
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-s %(filename)s#%(funcName)s:%(lineno)-8d %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": LOG_FILE_PATH,
            "encoding": "utf8",
            "when": "D",
            "interval": 1,
            "backupCount": 14
        }
    },
    "loggers": {
        "console": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "file": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}


def set_log_conf():
    """
    set logging conf
    :return:
    """
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logging.config.dictConfig(LOG_CONFIG)
    logging.getLogger('matplotlib.font_manager').disabled = True
