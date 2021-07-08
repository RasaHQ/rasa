import * as React from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {faCircleNotch} from "@fortawesome/free-solid-svg-icons";

function Button({isLoading, ...props}) {
  return (
    <div className={clsx(styles.buttonContainer)}>
      <button
        {...props}
        className={clsx(styles.button)}
      />
      {isLoading && <FontAwesomeIcon icon={faCircleNotch} spin className={clsx(styles.buttonSpinner)} />}
    </div>
  )
}

export default Button;
