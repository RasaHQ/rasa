/* eslint-disable react/display-name */
import React from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
/*
SHORTCODES
*/
import useBaseUrl from '@docusaurus/useBaseUrl';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Prototyper, { DownloadButton, TrainButton } from '@theme/Prototyper';

import styles from './styles.module.scss';

export default {
	code: (props) => {
		const { children } = props;

		if (typeof children === 'string') {
			if (!children.includes('\n')) {
				return <code {...props} />;
			}

			return <CodeBlock {...props} />;
		}

		return children;
	},
	a: (props) => {
		return <Link {...props} />;
	},
	pre: (props) => <div className={styles.mdxCodeBlock} {...props} />,
	h1: Heading('h1'),
	h2: Heading('h2'),
	h3: Heading('h3'),
	h4: Heading('h4'),
	h5: Heading('h5'),
	h6: Heading('h6'),
	/*
	docsaurus
	*/
	useBaseUrl,
	Tabs,
	TabItem,
	/*
	custom, rasa-x
	*/
	Prototyper,
	DownloadButton,
	TrainButton,
	/*
	custom, tabula
	*/
};
