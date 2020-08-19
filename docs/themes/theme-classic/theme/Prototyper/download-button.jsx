import React from 'react';

import Button from './button';
import PrototyperContext from './context';

const DownloadButton = (props) => {
	const prototyperContext = React.useContext(PrototyperContext);

	return (
		<Button
			onClick={prototyperContext.downloadProject}
			disabled={!prototyperContext.hasTrained || !!prototyperContext.isTraining}
			{...props}
		>
			Download project
		</Button>
	);
};

export default DownloadButton;
