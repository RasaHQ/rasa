from collections import namedtuple
import tensorflow as tf


class TimedNTM(object):
    """Timed Neural Turing Machine

    inspired by paper:
        https://arxiv.org/pdf/1410.5401.pdf
    implementation inspired by:
        https://github.com/carpedm20/NTM-tensorflow/blob/master/ntm_cell.py
    """

    def __init__(self, attn_shift_range, sparse_attention, name):
        """Construct the `TimedNTM`.

        Args:
            attn_shift_range: Python int.
                A time range within which to attend to the memory by location
            sparse_attention: Python bool.
                If `True` use sparsemax instead of softmax for probs
            name: Name to use when creating ops.
        """

        # interpolation gate
        self.name = 'timed_ntm_' + name

        self.inter_gate = tf.layers.Dense(
                units=1,
                activation=tf.sigmoid,
                name=self.name + '/inter_gate'
        )
        # if use sparsemax instead of softmax for probs
        self.sparse_attention = sparse_attention

        # shift weighting if range is provided
        if attn_shift_range:
            self.shift_weight = tf.layers.Dense(
                    units=2 * attn_shift_range + 1,
                    activation=tf.nn.softmax,
                    name=self.name + '/shift_weight'
            )
        else:
            self.shift_weight = None

        # sharpening parameter
        self.gamma_sharp = tf.layers.Dense(
                units=1,
                activation=lambda a: tf.nn.softplus(a) + 1,
                bias_initializer=tf.constant_initializer(1),
                name=self.name + '/gamma_sharp'
        )

    def __call__(self, cell_output, scores, scores_state, ignore_mask):
        # apply exponential moving average with interpolation gate weight
        # to scores from previous time which are equal to probs at this point
        # different from original NTM where it is applied after softmax
        i_g = self.inter_gate(cell_output)

        # scores limited by time
        scores = tf.concat([i_g * scores[:, :-1] + (1 - i_g) * scores_state,
                            scores[:, -1:]], 1)
        next_scores_state = scores

        # create probabilities for attention
        if self.sparse_attention:
            probs = tf.contrib.sparsemax.sparsemax(scores)
        else:
            probs = tf.nn.softmax(scores)

        if self.shift_weight is not None:
            s_w = self.shift_weight(cell_output)

            # we want to go back in time during convolution
            conv_probs = tf.reverse(probs, axis=[1])

            # preare probs for tf.nn.depthwise_conv2d
            # [in_width, in_channels=batch]
            conv_probs = tf.transpose(conv_probs, [1, 0])
            # [batch=1, in_height=1, in_width=time+1, in_channels=batch]
            conv_probs = conv_probs[tf.newaxis, tf.newaxis, :, :]

            # [filter_height=1, filter_width=2*attn_shift_range+1,
            #   in_channels=batch, channel_multiplier=1]
            conv_s_w = tf.transpose(s_w, [1, 0])
            conv_s_w = conv_s_w[tf.newaxis, :, :, tf.newaxis]

            # perform 1d convolution
            # [batch=1, out_height=1, out_width=time+1, out_channels=batch]
            conv_probs = tf.nn.depthwise_conv2d_native(conv_probs, conv_s_w,
                                                       [1, 1, 1, 1], 'SAME')
            conv_probs = conv_probs[0, 0, :, :]
            conv_probs = tf.transpose(conv_probs, [1, 0])

            probs = tf.reverse(conv_probs, axis=[1])

        # Sharpening
        g_sh = self.gamma_sharp(cell_output)

        powed_probs = tf.pow(probs, g_sh)
        probs = powed_probs / (
                tf.reduce_sum(powed_probs, 1, keepdims=True) + 1e-32)

        # set probs for no intents and action_listens to zero
        if ignore_mask is not None:
            probs = tf.concat([tf.where(ignore_mask,
                                        tf.zeros_like(probs[:, :-1]),
                                        probs[:, :-1]),
                               probs[:, -1:]], 1)
        return probs, next_scores_state


def _compute_time_attention(attention_mechanism, attn_inputs, attention_state,
                            # time is added to calculate time attention
                            time, timed_ntm, ignore_mask, attention_layer):
    """Computes the attention and alignments limited by time
        for a given attention_mechanism.

        Modified helper method from tensorflow."""

    scores, _ = attention_mechanism(attn_inputs, state=attention_state)

    # take only scores from current and past times
    timed_scores = scores[:, :time + 1]
    timed_scores_state = attention_state[:, :time]
    if ignore_mask is not None:
        timed_ignore_mask = ignore_mask[:, :time]
    else:
        timed_ignore_mask = None

    # pass these scores to NTM
    probs, next_scores_state = timed_ntm(attn_inputs, timed_scores,
                                         timed_scores_state,
                                         timed_ignore_mask)

    # concatenate probs with zeros to get new alignments
    zeros = tf.zeros_like(scores)
    # remove current time from attention
    alignments = tf.concat([probs[:, :-1], zeros[:, time:]], 1)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([attn_inputs, context], 1))
    else:
        attention = context

    # return current time to attention
    alignments = tf.concat([probs, zeros[:, time + 1:]], 1)
    next_attention_state = tf.concat([next_scores_state,
                                      zeros[:, time + 1:]], 1)
    return attention, alignments, next_attention_state


class TimeAttentionWrapperState(
            namedtuple("TimeAttentionWrapperState",
                       tf.contrib.seq2seq.AttentionWrapperState._fields +
                       ("all_cell_states",))):  # added
    """Modified  from tensorflow's tf.contrib.seq2seq.AttentionWrapperState
        see there for description of the parameters

    Additional fields:
        - `all_cell_states`: All states of the wrapped `RNNCell`
           at all the previous time steps.
    """

    def clone(self, **kwargs):
        """Copied  from tensorflow's tf.contrib.seq2seq.AttentionWrapperState
            see there for description of the parameters"""

        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tf.contrib.framework.with_same_shape(old, new)
            return new

        return tf.contrib.framework.nest.map_structure(
                with_same_shape,
                self,
                super(TimeAttentionWrapperState, self)._replace(**kwargs)
        )


class TimeAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    """Custom AttentionWrapper that takes into account time
        when calculating attention.
        Attention is calculated before calling rnn cell.

        Modified from tensorflow's tf.contrib.seq2seq.AttentionWrapper.
    """

    def __init__(self, cell,
                 attention_mechanism,
                 sequence_len,
                 attn_shift_range=0,
                 sparse_attention=False,
                 attention_layer_size=None,
                 alignment_history=False,
                 inputs_and_attn_inputs_fn=None,
                 ignore_mask=None,
                 cell_input_fn=None,
                 index_of_attn_to_copy=None,
                 likelihood_fn=None,
                 tensor_not_to_copy=None,
                 output_attention=False,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None):
        """Construct the `TimeAttentionWrapper`.
            See the super class for the original arguments description.

        Additional args:
            sequence_len: Python integer.
                Maximum length of the sequence, used to create
                appropriate TensorArray for all cell states
                in TimeAttentionWrapperState
            attn_shift_range: Python integer (`0` by default).
                A time range within which to attend to the memory
                by location in Neural Turing Machine.
            sparse_attention: Python bool.
                A flag to use sparsemax (if `True`) instead of
                softmax (if `False`, default) for probabilities
            inputs_and_attn_inputs_fn: (optional) A `callable`.
                A function that creates inputs and attention inputs tensors.
            ignore_mask: (optional) Boolean Tensor.
                Determines which time steps to ignore in attention
            index_of_attn_to_copy: (optional) Python integer.
                An index of attention mechanism that picks
                which part of attention tensor to use for copying to output,
                the default is `None`, which turns off copying mechanism.
                Copy inspired by: https://arxiv.org/pdf/1603.06393.pdf
            likelihood_fn: (optional) A `callable`.
                A method to perform likelihood calculation to
                filter time step in copy mechanism.
                Returns a tuple of binary likelihood and likelihood
            tensor_not_to_copy: (optional) A Tensor.
                A tensor, which shouldn't be copied from previous time steps

        Modified args:
            output_attention: Python bool.  If `True`, the output at each
                time step is the concatenated cell outputs,
                attention values and additional values described in
                `additional_output_size()`, used in copy mechanism.
        """
        super(TimeAttentionWrapper, self).__init__(
                cell,
                attention_mechanism,
                attention_layer_size,
                alignment_history,
                cell_input_fn,
                output_attention,
                initial_cell_state,
                name,
                attention_layer
        )
        self._sequence_len = sequence_len

        if not isinstance(attn_shift_range, list):
            # attn_shift_range might not be a list
            attn_shift_range = [attn_shift_range]
        self._timed_ntms = [TimedNTM(attn_shift_range[0],
                                     sparse_attention,
                                     name='0')]
        if self._is_multi:
            # if there are several attention mechanisms,
            # create additional TimedNTMs for them
            for i in range(1, len(attention_mechanism)):
                if len(attn_shift_range) < i + 1:
                    # original attn_shift_range might not be a list
                    attn_shift_range.append(attn_shift_range[-1])
                self._timed_ntms.append(TimedNTM(attn_shift_range[i],
                                                 sparse_attention,
                                                 name=str(i)))

        if inputs_and_attn_inputs_fn is None:
            inputs_and_attn_inputs_fn = self._default_inputs_and_attn_inputs_fn
        else:
            if not callable(inputs_and_attn_inputs_fn):
                raise TypeError(
                    "inputs_and_attn_inputs_fn must be callable, saw type: {}"
                    "".format(type(inputs_and_attn_inputs_fn).__name__)
                )
        self._inputs_and_attn_inputs_fn = inputs_and_attn_inputs_fn

        if not isinstance(ignore_mask, list):
            self._ignore_mask = [ignore_mask]
        else:
            self._ignore_mask = ignore_mask

        self._index_of_attn_to_copy = index_of_attn_to_copy

        self._likelihood_fn = likelihood_fn
        self._tensor_not_to_copy = tensor_not_to_copy

    @staticmethod
    def _default_inputs_and_attn_inputs_fn(inputs, cell_state):
        if isinstance(cell_state, tf.contrib.rnn.LSTMStateTuple):
            return inputs, tf.concat([inputs, cell_state.h], -1)
        else:
            return inputs, tf.concat([inputs, cell_state], -1)

    @staticmethod
    def additional_output_size():
        """Number of additional outputs:

        likelihoods:
            attn_likelihood, state_likelihood
        debugging info:
            current_time_prob,
            bin_likelihood_not_to_copy, bin_likelihood_to_copy

        **Method should be static**
        """
        return 2 + 3

    @property
    def output_size(self):
        if self._output_attention:
            if self._index_of_attn_to_copy is not None:
                # output both raw rnn cell_output and
                # cell_output with copied attention
                # together with attention vector itself
                # and additional output
                return (2 * self._cell.output_size +
                        self._attention_layer_size +
                        self.additional_output_size())
            else:
                return self._cell.output_size + self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `TimeAttentionWrapper`.
        Returns:
            A `TimeAttentionWrapperState` tuple containing shapes
            used by this object.
        """

        # use AttentionWrapperState from superclass
        state_size = super(TimeAttentionWrapper, self).state_size

        all_cell_states = self._cell.state_size

        return TimeAttentionWrapperState(
                cell_state=state_size.cell_state,
                time=state_size.time,
                attention=state_size.attention,
                alignments=state_size.alignments,
                attention_state=state_size.attention_state,
                alignment_history=state_size.alignment_history,
                all_cell_states=all_cell_states)

    def zero_state(self, batch_size, dtype):
        """Modified  from tensorflow's zero_state
            see there for description of the parameters"""

        # use AttentionWrapperState from superclass
        zero_state = super(TimeAttentionWrapper,
                           self).zero_state(batch_size, dtype)

        with tf.name_scope(type(self).__name__ + "ZeroState",
                           values=[batch_size]):
            # store all cell states into a tensor array to allow
            # copy mechanism to go back in time
            if isinstance(self._cell.state_size,
                          tf.contrib.rnn.LSTMStateTuple):
                all_cell_states = tf.contrib.rnn.LSTMStateTuple(
                        tf.TensorArray(dtype, size=self._sequence_len + 1,
                                       dynamic_size=False,
                                       clear_after_read=False
                                       ).write(0, zero_state.cell_state.c),
                        tf.TensorArray(dtype, size=self._sequence_len + 1,
                                       dynamic_size=False,
                                       clear_after_read=False
                                       ).write(0, zero_state.cell_state.h)
                )
            else:
                all_cell_states = tf.TensorArray(
                        dtype, size=0,
                        dynamic_size=False,
                        clear_after_read=False
                ).write(0, zero_state.cell_state)

            return TimeAttentionWrapperState(
                    cell_state=zero_state.cell_state,
                    time=zero_state.time,
                    attention=zero_state.attention,
                    alignments=zero_state.alignments,
                    attention_state=zero_state.attention_state,
                    alignment_history=zero_state.alignment_history,
                    all_cell_states=all_cell_states
            )

    # helper methods for copy mechanism
    def _get_memory_probs(self, all_alignments, time):
        """Helper method to get memory_probs from all_alignments"""

        memory_probs = tf.stop_gradient(all_alignments[
                self._index_of_attn_to_copy][:, :time])
        # filter memory_probs with ignore_mask
        memory_probs = tf.where(
                self._ignore_mask[self._index_of_attn_to_copy][:, :time],
                tf.zeros_like(memory_probs),
                memory_probs
        )
        # binarize memory_probs only if max value is larger than margin=0.1
        memory_probs_max = tf.reduce_max(memory_probs, axis=1, keepdims=True)
        memory_probs_max = tf.where(memory_probs_max > 0.1,
                                    memory_probs_max, -memory_probs_max)

        return tf.where(tf.equal(memory_probs, memory_probs_max),
                        tf.ones_like(memory_probs),
                        tf.zeros_like(memory_probs))

    @staticmethod
    def _history_alignments(memory_probs):
        """Helper method to apply binary mask to memory_probs"""

        current_time_prob = 1 - tf.reduce_sum(memory_probs, 1, keepdims=True)
        return tf.concat([memory_probs, current_time_prob], 1)

    @staticmethod
    def _apply_alignments_to_history(alignments, history_states, state):
        """Helper method to apply attention probabilities to rnn history

        copied from tf's `_compute_attention(...)`"""

        expanded_alignments = tf.stop_gradient(tf.expand_dims(alignments, 1))

        history_states = tf.concat([history_states,
                                    tf.expand_dims(state, 1)], 1)

        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # expanded_alignments shape is
        #   [batch_size, 1, memory_time]
        # history_states shape is
        #   [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.

        return tf.squeeze(tf.matmul(expanded_alignments, history_states), [1])

    def _prev_output(self, state, alignments, time):
        """Helper method to get previous output from memory"""

        # get all previous outputs from appropriate
        # attention mechanism's memory limited by current time
        prev_outputs = tf.stop_gradient(self._attention_mechanisms[
                self._index_of_attn_to_copy].values[:, :time, :])

        # multiply by alignments to get one vector from one time step
        return self._apply_alignments_to_history(alignments,
                                                 prev_outputs,
                                                 state)

    def _additional_likelihood(self, output, prev_output, current_time_prob):
        """Helper method to create additional likelihood to maximize"""

        _, likelihood = self._likelihood_fn(
                output, tf.stop_gradient(prev_output))
        return tf.where(current_time_prob < 0.5,
                        likelihood, tf.ones_like(likelihood))

    def _new_hidden_state(self, prev_all_cell_states,
                          new_state, alignments, time):
        """Helper method to look into rnn history"""

        # reshape to (batch, time, memory_time) and
        # do not include current time because
        # we do not want to pay attention to it,
        # but we need to read it instead of
        # adding conditional flow if time == 0
        prev_cell_states = tf.transpose(
                prev_all_cell_states.gather(
                        tf.range(0, time + 1)), [1, 0, 2]
        )[:, :-1, :]

        return self._apply_alignments_to_history(alignments,
                                                 prev_cell_states,
                                                 new_state)

    def _new_next_cell_state(self, prev_all_cell_states,
                             next_cell_state, new_cell_output,
                             alignments, time):
        """Helper method to recalculate new next_cell_state"""

        if isinstance(next_cell_state, tf.contrib.rnn.LSTMStateTuple):
            next_cell_state_c = self._new_hidden_state(
                prev_all_cell_states.c,
                next_cell_state.c,
                alignments,
                time
            )
            next_cell_state_h = self._new_hidden_state(
                prev_all_cell_states.h,
                new_cell_output,
                alignments,
                time
            )
            return tf.contrib.rnn.LSTMStateTuple(next_cell_state_c,
                                                 next_cell_state_h)
        else:
            return self._new_hidden_state(prev_all_cell_states,
                                          alignments, new_cell_output, time)

    @staticmethod
    def _all_cell_states(prev_all_cell_states, next_cell_state, time):
        """Helper method to recalculate all_cell_states tensor array"""

        if isinstance(next_cell_state, tf.contrib.rnn.LSTMStateTuple):
            return tf.contrib.rnn.LSTMStateTuple(
                    prev_all_cell_states.c.write(time + 1, next_cell_state.c),
                    prev_all_cell_states.h.write(time + 1, next_cell_state.h)
            )
        else:
            return prev_all_cell_states.write(time + 1, next_cell_state)

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        The order has changed:
        - Step 1: Calculate attention inputs based on the previous cell state
                  and current inputs
        - Step 2: Score the output with `attention_mechanism`.
        - Step 3: Calculate the alignments by passing the score through the
                  `normalizer` and limit them by time.
        - Step 4: Calculate the context vector as the inner product between the
                  alignments and the attention_mechanism's values (memory).
        - Step 5: Calculate the attention output by concatenating
                  the cell output and context through the attention layer
                  (a linear layer with `attention_layer_size` outputs).
        - Step 6: Mix the `inputs` and `attention` output via
                  `cell_input_fn` to get cell inputs.
        - Step 7: Call the wrapped `cell` with these cell inputs and
                  its previous state.
        - Step 8: (optional) Maybe copy output and cell state from history

        Args:
          inputs: (Possibly nested tuple of) Tensor,
                  the input at this time step.
          state: An instance of `TimeAttentionWrapperState`
                 containing tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `TimeAttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of
          `TimeAttentionWrapperState`.
        """
        if not isinstance(state, TimeAttentionWrapperState):
            raise TypeError("Expected state to be instance of "
                            "TimeAttentionWrapperState. "
                            "Received type {} instead.".format(type(state)))

        # Step 1: Calculate attention based on
        #         the previous output and current input
        cell_state = state.cell_state

        inputs, attn_inputs = self._inputs_and_attn_inputs_fn(inputs,
                                                              cell_state)

        cell_batch_size = (
                attn_inputs.shape[0].value or
                tf.shape(attn_inputs)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  "
                "Are you using "
                "the BeamSearchDecoder?  "
                "You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            attn_inputs = tf.identity(
                    attn_inputs, name="checked_attn_inputs")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            # Steps 2 - 5 are performed inside `_compute_time_attention`
            (attention, alignments,
             next_attention_state) = _compute_time_attention(
                    attention_mechanism, attn_inputs,
                    previous_attention_state[i],
                    # time is added to calculate time attention
                    state.time, self._timed_ntms[i],
                    # provide boolean mask, to ignore some time steps
                    self._ignore_mask[i],
                    self._attention_layers[i]
                    if self._attention_layers else None)

            alignment_history = previous_alignment_history[i].write(
                    state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)

        # Step 6: Mix the `inputs` and `attention` output via
        #         `cell_input_fn` to get cell inputs.
        cell_inputs = self._cell_input_fn(inputs, attention)

        # Step 7: Call the wrapped `cell` with these cell inputs and
        #         its previous state.
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        prev_all_cell_states = state.all_cell_states

        if self._index_of_attn_to_copy is not None:
            # Step 8: Maybe copy output and cell state from history

            # get relevant previous outputs from history
            attn_to_copy = all_attentions[self._index_of_attn_to_copy]
            # copy them to current output
            cell_output_with_attn = cell_output + attn_to_copy

            memory_probs = self._get_memory_probs(all_alignments, state.time)

            # check that we do not pay attention to `tensor_not_to_copy`
            bin_likelihood_not_to_copy, _ = self._likelihood_fn(
                    cell_output_with_attn, self._tensor_not_to_copy)
            # recalculate probs
            memory_probs *= 1 - bin_likelihood_not_to_copy

            history_alignments = self._history_alignments(memory_probs)

            # get previous output from the history
            prev_output = self._prev_output(cell_output_with_attn,
                                            history_alignments,
                                            state.time)

            # check that current output is close to
            # the one in the history to which we pay attention to
            bin_likelihood_to_copy, _ = self._likelihood_fn(
                    cell_output_with_attn, prev_output)
            # recalculate probs
            memory_probs *= bin_likelihood_to_copy

            history_alignments = self._history_alignments(memory_probs)
            current_time_prob = history_alignments[:, -1:]

            # create additional likelihoods to maximize
            attn_likelihood = self._additional_likelihood(
                    attn_to_copy,
                    prev_output,
                    current_time_prob
            )
            state_likelihood = self._additional_likelihood(
                    cell_output + tf.stop_gradient(attn_to_copy),
                    prev_output,
                    current_time_prob
            )

            # recalculate new next_cell_state based on history_alignments
            next_cell_state = self._new_next_cell_state(
                    prev_all_cell_states,
                    next_cell_state,
                    cell_output_with_attn,
                    history_alignments,
                    state.time
            )

            all_cell_states = self._all_cell_states(
                    prev_all_cell_states,
                    next_cell_state,
                    state.time
            )

            if self._output_attention:
                # concatenate cell outputs, attention, additional likelihoods
                # and copy_attn_debug
                output = tf.concat([cell_output_with_attn,
                                    cell_output,
                                    attention,
                                    # additional likelihoods
                                    attn_likelihood, state_likelihood,
                                    # copy_attn_debug
                                    bin_likelihood_not_to_copy,
                                    bin_likelihood_to_copy,
                                    current_time_prob], 1)
            else:
                output = cell_output_with_attn

        else:
            # do not waste resources on storing history
            all_cell_states = prev_all_cell_states

            if self._output_attention:
                output = tf.concat([cell_output, attention], 1)
            else:
                output = cell_output

        next_state = TimeAttentionWrapperState(
                time=state.time + 1,
                cell_state=next_cell_state,
                attention=attention,
                attention_state=self._item_or_tuple(all_attention_states),
                alignments=self._item_or_tuple(all_alignments),
                alignment_history=self._item_or_tuple(maybe_all_histories),
                all_cell_states=all_cell_states
        )
        return output, next_state


class ChronoBiasLayerNormBasicLSTMCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """Custom LayerNormBasicLSTMCell that allows chrono initialization
        of gate biases.

        See super class for description."""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 input_bias=0.0,
                 activation=tf.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 dropout_keep_prob=1.0,
                 dropout_prob_seed=None,
                 out_layer_size=None,
                 reuse=None):
        """Initializes the basic LSTM cell

        Additional args:
            input_bias: float, The bias added to input gates.
            out_layer_size: (optional) integer, The number of units in
                the optional additional output layer.
        """
        super(ChronoBiasLayerNormBasicLSTMCell, self).__init__(
                num_units,
                forget_bias=forget_bias,
                activation=activation,
                layer_norm=layer_norm,
                norm_gain=norm_gain,
                norm_shift=norm_shift,
                dropout_keep_prob=dropout_keep_prob,
                dropout_prob_seed=dropout_prob_seed,
                reuse=reuse
        )
        self._input_bias = input_bias
        self._out_layer_size = out_layer_size

    @property
    def output_size(self):
        return self._out_layer_size or self._num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units,
                                             self.output_size)

    @staticmethod
    def _dense_layer(args, layer_size):
        """Optional out projection layer"""
        proj_size = args.get_shape()[-1]
        dtype = args.dtype
        weights = tf.get_variable("kernel",
                                  [proj_size, layer_size],
                                  dtype=dtype)
        bias = tf.get_variable("bias",
                               [layer_size],
                               dtype=dtype)
        out = tf.nn.bias_add(tf.matmul(args, weights), bias)
        return out

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        c, h = state
        args = tf.concat([inputs, h], 1)
        concat = self._linear(args)
        dtype = args.dtype

        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._norm(i, "input", dtype=dtype)
            j = self._norm(j, "transform", dtype=dtype)
            f = self._norm(f, "forget", dtype=dtype)
            o = self._norm(o, "output", dtype=dtype)

        g = self._activation(j)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = tf.nn.dropout(g, self._keep_prob, seed=self._seed)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 g * tf.sigmoid(i + self._input_bias))  # added input_bias

        # do not do layer normalization on the new c,
        # because there are no trainable weights
        # if self._layer_norm:
        #     new_c = self._norm(new_c, "state", dtype=dtype)

        new_h = self._activation(new_c) * tf.sigmoid(o)

        # added dropout to the hidden state h
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            new_h = tf.nn.dropout(new_h, self._keep_prob, seed=self._seed)

        # add postprocessing of the output
        if self._out_layer_size is not None:
            with tf.variable_scope('out_layer'):
                new_h = self._dense_layer(new_h, self._out_layer_size)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state
