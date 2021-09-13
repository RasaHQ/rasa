import * as React from 'react';
import clsx from 'clsx';

import RasaButton from "../RasaButton";
import styles from './styles.module.css';

function Text({values, selectedValue, setSelectedValue, children, ...props}) {
  const index = values.findIndex((v) => v.value === selectedValue);

  const onButtonClick = () => {
    if (index < values.length - 1) {
      setSelectedValue(values[index + 1].value);
    }
  };

  return (
    <div className={clsx(styles.text)}>
      <div className={clsx(styles.textTop)}>
        {children}
      </div>

      {
        index !== -1 && index !== values.length - 1 &&
        <div className={clsx(styles.textButtons)}>
          <NextStepButton onClick={onButtonClick}/>
        </div>
      }
    </div>
  )
}

function Code({children}) {
  return (
    <div className={clsx(styles.code)}>
      {children}
    </div>
  )
}

function Section({children, ...props}) {
  return (
    <div className={clsx(styles.section, {[styles.sectionActive]: props.selectedValue === props.value})}>
      {React.Children.toArray(children).map(
        (child) => React.cloneElement(child, props)
      )}
    </div>
  )
}

function NextStepButton(props) {
  return (
    <RasaButton {...props}>
      Next step >>
    </RasaButton>
  )
}

function Container({children, values, defaultValue, ...props}) {
  const [selectedValue, setSelectedValue] = React.useState(defaultValue);
  const tabRefs = [];

  const changeSelectedValue = (newValue) => {
    setSelectedValue(newValue);
  };

  return (
    <div className={clsx(styles.container, 'mdx-box-max')}>
      <ul className={clsx(styles.containerTabs)}>
        {values.map(({value, label}) => (
          <li
            role="tab"
            tabIndex={0}
            className={clsx(styles.tab, {[styles.tabActive]: selectedValue === value})}
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
          React.Children.toArray(children).map(
            (child) => React.cloneElement(child,
              {values, selectedValue, setSelectedValue, ...props}
            )
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
  Code
};
