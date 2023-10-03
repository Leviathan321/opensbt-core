import logging as log

def setup_logging(log_to):
    # Disable messages from matplot lib.
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Logging setup."
    log_level = log.INFO

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    log.info(start_msg)

def disable_pymoo_warnings():
    from pymoo.config import Config

    Config.warnings['not_compiled'] = False