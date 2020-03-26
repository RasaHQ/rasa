Added flag `--dry` for checking running `rasa train --dry` to check if the training is necessary.
It is useful if we don't necessarily want to train new models. This feature helps with rasa CICD integration.
Check data on one platform and then train on another.
