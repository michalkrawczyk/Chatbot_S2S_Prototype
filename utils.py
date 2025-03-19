import logging

logger = logging.getLogger('voice_transcriber')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Settings
DEBUG_INFO = True
RECURSION_LIMIT = 5
AGENT_TRACE = True
AGENT_VERBOSE = True

def conditional_debug_info(*args, **kwargs):
    if DEBUG_INFO:
        logger.info(*args, **kwargs)