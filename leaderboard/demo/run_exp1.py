import os
import copy
import shutil

from leaderboard.utils import base_experiment
from leaderboard.nlu import exp_1_stratify_intents_and_responses


### Remove results from last test run

tmp_dir = "test2"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)


### Chose Configurations

base_config = exp_1_stratify_intents_and_responses.Config(
    exclusion_percentage=0.0,
    drop_intents_with_less_than=2,
    random_seed=1,
    test_fraction=0.2,
    data=base_experiment.DataConfig(
        name="examples-rules-nlu",
        data_path="../../examples/rules/data/nlu.yml",
        domain_path="../../examples/rules/domain.yml",
    ),
    model=base_experiment.ModelConfig(),
    clear_rasa_cache=True,
)

variations = {
    "interesting-config-name1": "./config2.yml",
    "another-very-interesting-config-because-reasons": "./config1.yml",
}

configs = []
for name, config_path in variations.items():
    config = copy.deepcopy(base_config)
    config.model.name = name
    config.model.config_path = config_path
    configs.append(config)

### Run!

base_experiment.multirun(
    experiment_module=exp_1_stratify_intents_and_responses,
    configs=configs, 
    out_dir=tmp_dir,
)
