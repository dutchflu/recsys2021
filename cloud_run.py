"""
Main entry point for cloud run
"""
from argparse import ArgumentParser

from training.config_factory import config_factory
from training.pipelines.training_evaluation_pipeline import TrainingEvaluationPipelineConfig, TrainingEvaluationPipeline
from training.data_preparation.data_preparation import DataProvider
from training.pipelines.splitter import RandomSplitter
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


def run_training_evaluation_pipeline(pipeline_config: TrainingEvaluationPipelineConfig):
    data_provider = DataProvider(pipeline_config.data_provider_config)
    splitter = RandomSplitter()

    training_evaluation_pipeline = TrainingEvaluationPipeline(data_provider,
                                                              splitter,
                                                              pipeline_config.train_config)

    training_evaluation_pipeline.run()


def main():
    args = get_args()

    # if it is a cloud config file, add the function to
    # download the file to local first
    pipeline_config = load_config(args.config_file, config_factory)
    log.info("config loaded: %s", pipeline_config)

    if pipeline_config.pipeline_type == "training_evaluation_pipeline":
        run_training_evaluation_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
