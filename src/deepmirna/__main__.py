import logging
import sys
from argparse import ArgumentParser

from deepmirna.train_model import train_model, evaluate_model
from deepmirna.test_model import  test_model
import deepmirna.configurator as config

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"

_logger = logging.getLogger(__name__)

def setup_logging():
    """
    Setup basic logging
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def main():
    setup_logging()
    parser = ArgumentParser(description='DeepMiRNA : a miRNA target prediction tool.')
    parser.add_argument('-c', '--conf', help='allow to specify a configuration file')
    parser.add_argument('option', choices=['test', 'train', 'train_eval'],
                        help='test, train or train and validate the model')
    args = parser.parse_args()

    # default config file location
    config_file = '../../config.ini'
    use_default = True
    if args.conf:
        config_file = args.conf
        use_default = False
    try:
        config.set_global_variables(config_file, use_default_values=use_default)
    except:
        raise FileNotFoundError('Please specify a configuration file')

    if args.option == 'test':
        _logger.info(' Testing the model...')
        test_model()
    elif args.option == 'train':
        _logger.info(' Training the model using all available data')
        _ = train_model()
    else:
        _logger.info(' Model validation started')
        _ = evaluate_model()

    _logger.info(' Process completed with success')


if __name__ == '__main__':
    main()