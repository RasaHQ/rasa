import * as React from 'react';
import {LiveProvider, LiveEditor, LiveError, LivePreview} from 'react-live';
import clsx from 'clsx';
import ThemeContext from '@theme/_contexts/ThemeContext';

import styles from './styles.module.css';

function Text({children, ...props}) {
  return (
    <div className={clsx(styles.text)}>
      {children}
    </div>
  )
}

function Code({children, ...props}) {
  return (
    <div className={clsx(styles.code)}>
      {children}
    </div>
  )
}

function Section({children, ...props}) {
  return (
    <div className={clsx(styles.section)}>
      {children}
    </div>
  )
}

function Container({children, values, defaultValue, ...props}) {
  const [selectedValue, setSelectedValue] = React.useState(defaultValue);
  const tabRefs = [];

  const changeSelectedValue = (newValue) => {
    setSelectedValue(newValue);
  };

  return (
    <div className={clsx(styles.container)}>
      <ul className={clsx(styles.containerTabs)}>
        {values.map(({value, label}) => (
          <li
            role="tab"
            tabIndex={0}
            className={clsx(styles.tab)}
            aria-selected={selectedValue === value}
            key={value}
            ref={(tabControl) => tabRefs.push(tabControl)}
            onFocus={() => changeSelectedValue(value)}
            onClick={() => {
              changeSelectedValue(value);
            }}
            >
            {label}
          </li>
        ))}
      </ul>
      <div className={clsx(styles.containerSections)}>
        {
          React.Children.toArray(children).filter(
            (child) => child.props.value === selectedValue
          )
        }
      </div>
    </div>
  )
}

export default {
  Container,
  Section,
  Text,
  Code,
}