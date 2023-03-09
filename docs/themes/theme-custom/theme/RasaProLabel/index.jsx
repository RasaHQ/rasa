import * as React from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

function RasaProLabel({isLoading, ...props}) {
  return (
      <div className={clsx(styles.label)}>Rasa Pro Only</div>
  )
}

export default RasaProLabel;
