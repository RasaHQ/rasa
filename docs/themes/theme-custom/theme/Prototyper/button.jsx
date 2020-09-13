import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleNotch } from '@fortawesome/free-solid-svg-icons';

// FIXME: surely not the right place for this
const COLOR_PURPLE_RASA = '#5a17ee';
const COLOR_DISABLED_GREY = '#bbb';
const COLOR_WHITE = '#fff';

const Button = ({ loading, style, ...props }) => (
  <>
    <button
      {...props}
      style={{
        backgroundColor: props.disabled ? COLOR_DISABLED_GREY : COLOR_PURPLE_RASA,
        border: '1px solid transparent',
        color: COLOR_WHITE,
        borderRadius: 8,
        padding: 12,
        fontSize: 15,
        fontWeight: '600',
        cursor: props.disabled ? 'default' : 'pointer',
        ...style,
      }}
    />
    {loading ? (
      <FontAwesomeIcon icon={faCircleNotch} spin style={styles.spinner} color={COLOR_DISABLED_GREY} />
    ) : undefined}
  </>
);

const styles = {
  spinner: { marginLeft: 8 },
};

export default Button;
