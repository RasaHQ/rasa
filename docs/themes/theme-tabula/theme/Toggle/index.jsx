/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import React from 'react';
import Toggle from 'react-toggle';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import clsx from 'clsx';

import styles from './styles.module.scss';

const Dark = ({ icon, style }) => (
  <span className={clsx(styles.toggle, styles.dark)} style={style}>
    {icon}
  </span>
);

const Light = ({ icon, style }) => (
  <span className={clsx(styles.toggle, styles.light)} style={style}>
    {icon}
  </span>
);

// eslint-disable-next-line react/display-name
export default function (props) {
  const {
    siteConfig: {
      themeConfig: {
        colorMode: {
          switchConfig: { darkIcon, darkIconStyle, lightIcon, lightIconStyle },
        },
      },
    },
    isClient,
  } = useDocusaurusContext();
  return (
    <Toggle
      disabled={!isClient}
      icons={{
        checked: <Dark icon={darkIcon} style={darkIconStyle} />,
        unchecked: <Light icon={lightIcon} style={lightIconStyle} />,
      }}
      {...props}
    />
  );
}
