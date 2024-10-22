import os


def find_new_dependencies():
    """Find new dependencies in the PR output that are not in the main output."""
    main_output = os.environ["MAIN_OUTPUT"]
    pr_output = os.environ["PR_OUTPUT"]

    main_dependencies = set(main_output.split(","))
    pr_dependencies = set(pr_output.split(","))

    new_dependencies = pr_dependencies - main_dependencies
    return new_dependencies


if __name__ == '__main__':
    result = find_new_dependencies()
    print(", ".join(list(result)))
