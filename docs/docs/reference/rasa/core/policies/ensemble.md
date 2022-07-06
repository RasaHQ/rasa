---
sidebar_label: rasa.core.policies.ensemble
title: rasa.core.policies.ensemble
---
#### is\_not\_in\_training\_data

```python
def is_not_in_training_data(policy_name: Optional[Text], max_confidence: Optional[float] = None) -> bool
```

Checks whether the prediction is empty or by a policy which did not memoize data.

**Arguments**:

- `policy_name` - The name of the policy.
- `max_confidence` - The max confidence of the policy&#x27;s prediction.
  

**Returns**:

  `False` if and only if an action was predicted (i.e. `max_confidence` &gt; 0) by
  a `MemoizationPolicy`

## InvalidPolicyEnsembleConfig Objects

```python
class InvalidPolicyEnsembleConfig(RasaException)
```

Exception that can be raised when the policy ensemble is not valid.

## PolicyPredictionEnsemble Objects

```python
class PolicyPredictionEnsemble(ABC)
```

Interface for any policy prediction ensemble.

Given a list of predictions from policies, which include some meta data about the
policies themselves, an &quot;ensemble&quot; decides what the final prediction should be, in
the following way:
1. If the previously predicted action was rejected, then the ensemble sets the
   probability for this action to 0.0 (in all given predictions).
2. It combines the information from the single predictions, which include some
   meta data about the policies (e.g. priority), into a final prediction.
3. If the sequence of events given at the time of prediction ends with a user
   utterance, then the ensemble adds a special event to the event-list included in
   the final prediction that indicates whether the final prediction was made based
   on the actual text of that user utterance.

Observe that policies predict &quot;mandatory&quot; as well as &quot;optional&quot;
events. The ensemble decides which of the optional events should
be passed on.

#### combine\_predictions\_from\_kwargs

```python
def combine_predictions_from_kwargs(tracker: DialogueStateTracker, domain: Domain, **kwargs: Any) -> PolicyPrediction
```

Derives a single prediction from predictions given as kwargs.

**Arguments**:

- `tracker` - dialogue state tracker holding the state of the conversation,
  which may influence the combination of predictions as well
- `domain` - the common domain
- `**kwargs` - arbitrary keyword arguments. All policy predictions passed as
  kwargs will be combined.
  

**Returns**:

  a single prediction

#### combine\_predictions

```python
@abstractmethod
def combine_predictions(predictions: List[PolicyPrediction], tracker: DialogueStateTracker, domain: Domain) -> PolicyPrediction
```

Derives a single prediction from the given list of predictions.

**Arguments**:

- `predictions` - a list of policy predictions that include &quot;confidence scores&quot;
  which are non-negative but *do not* necessarily up to 1
- `tracker` - dialogue state tracker holding the state of the conversation,
  which may influence the combination of predictions as well
- `domain` - the common domain
  

**Returns**:

  a single prediction

## DefaultPolicyPredictionEnsemble Objects

```python
class DefaultPolicyPredictionEnsemble(PolicyPredictionEnsemble,  GraphComponent)
```

An ensemble that picks the &quot;best&quot; prediction and combines events from all.

The following rules determine which prediction is the &quot;best&quot;:
1. &quot;No user&quot; predictions overrule all other predictions.

2. End-to-end predictions overrule all other predictions based on
    user input - if and only if *no* &quot;no user&quot; prediction is present in the
    given ensemble.

3. Given two predictions, if the maximum confidence of one prediction is
    strictly larger than that of the other, then the prediction with the
    strictly larger maximum confidence is considered to be &quot;better&quot;.
    The priorities of the policies that made these predictions does not matter.

4. Given two predictions of policies that are equally confident, the
    prediction of the policy with the higher priority is considered to be
    &quot;better&quot;.

Observe that this comparison is *not* symmetric if the priorities are allowed to
coincide (i.e. if we cannot distinguish two predictions using 1.-4., then
the first prediction is considered to be &quot;better&quot;).

The list of events in the final prediction will contain all mandatory
events contained in the given predictions, the optional events given in the
&quot;best&quot; prediction, and `DefinePrevUserUtteredFeaturization` event (if the
prediction was made for a sequence of events ending with a user utterance).

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DefaultPolicyPredictionEnsemble
```

Creates a new instance (see parent class for full docstring).

#### combine\_predictions

```python
def combine_predictions(predictions: List[PolicyPrediction], tracker: DialogueStateTracker, domain: Domain) -> PolicyPrediction
```

Derives a single prediction from the given list of predictions.

Note that you might get unexpected results if the priorities are non-unique.
Moreover, the order of events in the result is determined by the order of the
predictions passed to this method.

**Arguments**:

- `predictions` - a list of policy predictions that include &quot;probabilities&quot;
  which are non-negative but *do not* necessarily up to 1
- `tracker` - dialogue state tracker holding the state of the conversation
- `domain` - the common domain
  

**Returns**:

  The &quot;best&quot; prediction.

