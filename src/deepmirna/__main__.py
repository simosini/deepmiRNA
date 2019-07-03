import logging
import sys

def setup_logging():
    """
    Setup basic logging
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def main():
    setup_logging()


if __name__ == '__main__':
    main()