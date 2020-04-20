import tensorflow as tf
from typing import Optional, Text


class WarmupDecayLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies warm up and polynomial decay strategies to the learning rate.

    Args:
        pick_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.
        decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.
        end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
        decay_power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
        cycle: A boolean, whether or not it should cycle beyond decay_steps.
        name: String.  Optional name of the operation.
    """

    def __init__(
        self,
        pick_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        end_learning_rate: float = 0.0001,
        decay_power: float = 1.0,
        cycle: bool = False,
        name: Optional[Text] = None,
    ):
        super().__init__()
        self.pick_learning_rate = pick_learning_rate
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)

        self.decay_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            pick_learning_rate, decay_steps, end_learning_rate, decay_power, cycle
        )
        self.name = name

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.pick_learning_rate * step / self.warmup_steps,
            lambda: self.decay_rate(step - self.warmup_steps),
        )

    def get_config(self):
        return {
            "pick_learning_rate": self.pick_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_rate.decay_steps,
            "end_learning_rate": self.decay_rate.end_learning_rate,
            "power": self.decay_rate.power,
            "cycle": self.decay_rate.cycle,
            "name": self.name,
        }
