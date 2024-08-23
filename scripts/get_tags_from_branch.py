import ast
import os
import subprocess
import sys


def get_tags_from_branches(branches):
    tags = []

    for branch in branches:
        # Strip ".x" suffix from branch name.
        branch_name = branch.rstrip(".x")

        # Try to fetch all git tags for the given branch.
        # `output` will be sorted in the descending order.
        try:
            # file deepcode ignore HandleUnicode: This file is only used in CI/CD pipelines.
            output = subprocess.check_output(
                f'git tag | sort -r -V | grep -E "^{branch_name}.[0-9]+$"',
                shell=True,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError:
            print(f"Failed to retrieve tags for `{branch_name}` branch.")
            continue

        branch_tags = output.strip().split("\n")
        tags.extend(branch_tags)

    return tags


# The list of supported branches from the Rasa Product Release and Maintenance Policy (https://rasa.com/rasa-product-release-and-maintenance-policy/)
input_branches = sys.argv[1]
# The branch that the product you're patching became available on
first_release = sys.argv[2]

# Convert stringified list to a Python list.
branches = ast.literal_eval(input_branches)
tags = get_tags_from_branches(branches)

# Filter out tags lower than the version where the product you're patching became available
first_release_version = tuple(map(int, first_release.split(".")))
filtered_tags = [
    tag for tag in tags if tuple(map(int, tag.split("."))) >= first_release_version
]

print(f"{filtered_tags}")

with open(os.environ["GITHUB_OUTPUT"], "a") as github_output_file:
    print(f"tags={filtered_tags}", file=github_output_file)
