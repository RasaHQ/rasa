import React from 'react';

import ThemeContext from '@theme/theme-context';

const Prototyper = ({children, startPrototyperApi}) => {
    const [hasStarted, setHasStarted] = React.useState(false);
    const [trainingData, setTrainingData] = React.useState({});

    const onLiveCodeStart = (name, value) => {
        setTrainingData({...trainingData, name: value});
    };

    const onLiveCodeChange = (name, value) => {
        setTrainingData({...trainingData, name: value});
        if (!hasStarted) {
            // track the start here
            setHasStarted(true);
            fetch(startPrototyperApi, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    // FIXME: tracking id
                    tracking_id: 'the-hash',
                    editor: 'main',
                })
            });
        }
    };

    return (
      <ThemeContext.Provider value={{onLiveCodeChange, onLiveCodeStart}}>
        {children}
      </ThemeContext.Provider>
    );
};

export default Prototyper;
