import React from 'react';

import RasaButton from '../RasaButton';
import PrototyperContext from './context';

const TrainButton = (props) => {
  const prototyperContext = React.useContext(PrototyperContext);

  return (
    <RasaButton
      onClick={prototyperContext.trainModel}
      disabled={!!prototyperContext.isTraining}
      isLoading={!!prototyperContext.isTraining}
      {...props}
    >
      Train
    </RasaButton>
  );
};

export default TrainButton;
