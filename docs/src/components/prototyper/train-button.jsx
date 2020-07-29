import React from 'react';
import Button from '@site/src/components/button';


// https://trainer-service.prototyping.rasa.com/trainings

const TrainButton = (props) => (
  <Button
    {...props}
  >
    Train
  </Button>
);

export default TrainButton;
