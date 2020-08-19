import React from 'react';

import Button from './button';
import PrototyperContext from './context';

const TrainButton = (props) => {
	const prototyperContext = React.useContext(PrototyperContext);

	return (
		<Button
			onClick={prototyperContext.trainModel}
			disabled={!!prototyperContext.isTraining}
			loading={!!prototyperContext.isTraining}
			{...props}
		>
			Train
		</Button>
	);
};

export default TrainButton;
