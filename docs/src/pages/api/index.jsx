import React from 'react';
import { RedocStandalone } from 'redoc';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';

import { rawPage } from './styles.module.scss';

const Redoc = (props) => {
	const specUrl = useBaseUrl('/spec/action-server.yml');

	return (
		<Layout>
			<main className={rawPage}>
				<BrowserOnly fallback={<div>Loading...</div>}>
					{() => <RedocStandalone specUrl={specUrl} {...props} />}
				</BrowserOnly>
			</main>
		</Layout>
	);
};

export default Redoc;
