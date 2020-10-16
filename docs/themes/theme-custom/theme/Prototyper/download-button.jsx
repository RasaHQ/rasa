import React from 'react';

import RasaButton from '../RasaButton';
import PrototyperContext from './context';

const DownloadButton = (props) => {
  const prototyperContext = React.useContext(PrototyperContext);

  return (
    <RasaButton
      onClick={prototyperContext.downloadProject}
      disabled={prototyperContext.chatState !== "ready" && prototyperContext.chatState !== "needs_to_be_retrained"}
      {...props}
    >
      Download project
    </RasaButton>
  );
};

export default DownloadButton;
