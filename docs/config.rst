.. _config:

Configuring polices via a config.yaml
======================

You can set the policies you would like the Core model to use in a YAML file.

For example:

.. code-block:: yaml

  policies:
    - name: KerasPolicy
      max_history: 5
    - name: MemoizationPolicy
      max_history: 5
    - name: FallbackPolicy
      nlu_threshold: 0.4
      core_threshold: 0.3
      fallback_action_name: my_fallback_action
    - name: path.to.your.policy.class
      arg1: ...

Pass the YAML file's name to the train script using the --config argument (or just -c).
If no config.yaml is given, the policies default to KerasPolicy, MemoizationPolicy and FallbackPolicy.

Note that a policy specified higher in the config.yaml will take precedence over
a policy specified lower if the confidences are equal.
