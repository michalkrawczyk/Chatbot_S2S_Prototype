import logging

from general.config import ADDITIONAL_LOGGER_INFO

logger = logging.getLogger("voice_transcriber")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def conditional_logger_info(*args, **kwargs):
    """Log messages conditionally based on ADDITIONAL_LOGGER_INFO setting
    Calling just them on debug would be too verbose (because of information from any other functions)
    This is simple wrapper to avoid writing each time new logger
    """
    if ADDITIONAL_LOGGER_INFO:
        logger.info(*args, **kwargs)
