import logging

from config import DEBUG_INFO

logger = logging.getLogger('voice_transcriber')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def conditional_debug_info(*args, **kwargs):
    if DEBUG_INFO:
        logger.info(*args, **kwargs)