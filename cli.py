import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="model.h5",
        help="Path of trained model"
    )

    return parser.parse_args()
