import React from 'react';

import Button from '../Button';
import PrototyperContext from './context';

const TrainButton = (props) => {
  const prototyperContext = React.useContext(PrototyperContext);

  return (
    <Button
      onClick={prototyperContext.trainModel}
      disabled={!!prototyperContext.isTraining}
      isLoading={!!prototyperContext.isTraining}
      {...props}
    >
      Train
    </Button>
  );
};

export default TrainButton;
