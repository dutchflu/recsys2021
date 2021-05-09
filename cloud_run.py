"""
Main entry point for cloud run
"""
from argparse import ArgumentParser

from training.config_factory import config_factory
from commons.data_provision import load_config
from commons.log import log

def get_args():
    """Define the task arguments with the default values.
    Returns:
      experiment parameters
    """
    args_parser = ArgumentParser()

    args_parser.add_argument(
        "--config_file", required=False, help="config file path", default="./config.json")

    return args_parser.parse_args()


def main():
    args = get_args()

    # if it is a cloud config file, download and read that file first
    config = load_config(args.config_file, config_factory)
    log.info("config loaded: %s", config)


if __name__ == "__main__":
    main()
