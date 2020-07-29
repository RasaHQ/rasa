import React from 'react';
import Button from '@site/src/components/button';
import PrototyperContext from './context';


const DownloadButton = (props) => {
    const prototyperContext = React.useContext(PrototyperContext);

    return (
      <Button
        onClick={prototyperContext.downloadProject}
        {...props}
      >
        Download project
      </Button>
    );
}

export default DownloadButton;
