import logging
import sys
import os
from argparse import ArgumentParser

import datetime
from deepmirna.globs import ROOT_DIR
from deepmirna.train_model import train_model, evaluate_model
from deepmirna.test_model import  test_model
import deepmirna.configurator as config

from deepmirna import __version__

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

def parse_arguments(args):
    """
    Parse command line parameters
    :param args: command line parameters as list of strings
    :return: command line parameters namespace
    """

    parser = ArgumentParser(description='DeepMiRNA : a miRNA target prediction tool.')
    parser.add_argument(
        '-v', '--version', help='show current DeepMiRNA version.',
        action='version', version='deepmiRNA {ver}'.format(ver=__version__))
    parser.add_argument('-c', '--conf', help='allow to specify a configuration file')
    parser.add_argument('option', choices=['test', 'train', 'train_eval'],
                        help='test, train or train and validate the model')
    return parser.parse_args(args)



def main(args):
    setup_logging()
    args = parse_arguments(args)

    # default config file location
    config_file = os.path.join(ROOT_DIR, 'config.ini')
    use_default = True

    if args.conf:
        config_file = args.conf
        use_default = False

    config.set_global_variables(config_file, use_default)

    start_time = datetime.datetime.now()
    if args.option == 'test':
        _logger.info(' Testing the model...')
        test_model()
    elif args.option == 'train':
        _logger.info(' Training the model using all available data')
        _ = train_model()
    else:
        _logger.info(' Model validation started')
        _ = evaluate_model()

    _logger.info(' Process completed with success. Total computation time: {} seconds'
                 .format((datetime.datetime.now() - start_time).seconds))

def run():
    main(sys.argv[1:])

if __name__ == '__main__':
    run()