import * as React from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

function ActionLabel({isLoading, ...props}) {
  return (
      <div className={clsx(styles.label)}>Action Required</div>
  )
}

export default ActionLabel;
