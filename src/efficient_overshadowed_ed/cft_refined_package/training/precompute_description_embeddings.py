import os
import argparse

import torch.cuda

from refined_package.inference.processor import Refined
from refined_package.utilities.general_utils import get_logger
from refined_package.offline_data_generation.clean_wikipedia import str2bool

LOG = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=os.path.join(os.path.expanduser('~'), '.cache', 'refined')
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="File path to directory contain model.pt and config.json file of the model.",
    )
    parser.add_argument(
        "--entity_set",
        type=str,
        required=True,
        help="Entity set can be wither 'wikipedia' or 'wikidata'. It determines which set of entities to"
             "precompute description embeddings for. Note that the entity IDs are not the same across entity sets, "
             "which means separate precomputed description embeddings files are needed for Wikipedia and Wikidata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device.",
    )
    parser.add_argument(
        "--download_files",
        type=str2bool,
        help="download_files.",
    )
    args = parser.parse_args()
    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Using device: {device}.")
    refined = Refined.from_pretrained(model_name=args.model_dir,
                                      entity_set=args.entity_set,
                                      data_dir=args.data_dir,
                                      use_precomputed_descriptions=False,
                                      download_files=args.download_files,
                                      device=device)
    LOG.info('Precomputing description embeddings.')
    refined.precompute_description_embeddings()
    LOG.info('Done.')


if __name__ == '__main__':
    main()
