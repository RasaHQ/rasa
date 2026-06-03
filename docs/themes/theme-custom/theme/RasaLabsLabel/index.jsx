import * as React from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

function RasaLabsLabel({isLoading, ...props}) {
  return (
      <div className={clsx(styles.label)}>Rasa Labs</div>
  )
}

export default RasaLabsLabel;
