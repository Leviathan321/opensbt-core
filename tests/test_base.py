import os
from opensbt.config import LOG_FILE
from opensbt.utils.log_utils import setup_logging

def configure_logging_and_env():    
    os.chmod(os.getcwd(), 0o777)
    setup_logging(LOG_FILE)