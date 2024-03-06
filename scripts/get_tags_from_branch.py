import ast
import os
import subprocess
import sys

def get_tags_from_branches(branches):
    tags = []

    for branch in branches:
        # Strip ".x" suffix from branch name.
        branch_name = branch.rstrip('.x')

        # Try to fetch all git tags for the given branch.
        # `output` will be sorted in the descending order.
        try:
            output = subprocess.check_output(
                f'git tag | sort -r -V | grep -E "^{branch_name}.[0-9]+$"',
                shell=True,
                universal_newlines=True
            )
        except subprocess.CalledProcessError:
            print(f'Failed to retrieve tags for `{branch_name}` branch.')
            continue

        branch_tags = output.strip().split('\n')
        tags.extend(branch_tags)

    return tags

input_branches = sys.argv[1]

# Convert stringified list to a Python list.
branches = ast.literal_eval(input_branches)
tags = get_tags_from_branches(branches)

# Filter out tags lower than '3.7.8'
filtered_tags = [tag for tag in tags if tuple(map(int, tag.split('.'))) >= (3, 7, 8)]

print(f'{filtered_tags}')

with open(os.environ['GITHUB_OUTPUT'], 'a') as github_output_file:
    print(f'tags={filtered_tags}', file=github_output_file)
