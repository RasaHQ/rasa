import argparse


def set_deploy_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--provider", type=str, choices=["gcloud"], help="Cloud provider",
    )

    parser.add_argument(
        "--service-account",
        type=str,
        help="Key file for the google cloud service account",
    )

    parser.add_argument(
        "--project-id", type=str, help="ID of the google cloud project",
    )
