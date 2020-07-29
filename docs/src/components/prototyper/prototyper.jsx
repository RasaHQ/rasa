import React from 'react';

import ThemeContext from '@theme/theme-context';
import PrototyperContext from './context';

// FIXME: spinner states
const Prototyper = ({children, startPrototyperApi, trainModelApi}) => {
    const [trackingId, setTrackingId] = React.useState(null);
    const [hasStarted, setHasStarted] = React.useState(false);
    const [projectDownloadUrl, setProjectDownloadUrl] = React.useState(null);
    const [trainingData, setTrainingData] = React.useState({});

    const onLiveCodeStart = (name, value) => {
        // FIXME: tracking id + no tracking in dev?
        setTrackingId("the-hash");
        setTrainingData((prevTrainingData) => ({...prevTrainingData, [name]: value}));
    };

    const onLiveCodeChange = (name, value) => {
        setTrainingData((prevTrainingData) => ({...prevTrainingData, [name]: value}));
        if (!hasStarted) {
            // track the start here
            setHasStarted(true);
            fetch(startPrototyperApi, {
                method: 'POST',
                headers: {
                  'Accept': 'application/json',
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tracking_id: trackingId,
                    editor: 'main',
                })
            });
        }
    };

    const trainModel = () => {
        fetch(trainModelApi, {
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ tracking_id: trackingId, ...trainingData }),
        })
            .then(response => response.json())
            .then(data => {
                setProjectDownloadUrl(data.project_download_url);
                // if (result['rasa_service_url']) {
                //     const conversationId = uuidv4();
                //     startFetchingTracker(result['rasa_service_url'], chatBlockId, conversationId);
                //   }
            });
    };

    const downloadProject = () => {
        if (projectDownloadUrl) {
            location.href = projectDownloadUrl;
        }
    };

    return (
      <ThemeContext.Provider value={{onLiveCodeChange, onLiveCodeStart}}>
        <PrototyperContext.Provider value={{ trainModel, downloadProject }}>
            {children}
        </PrototyperContext.Provider>
      </ThemeContext.Provider>
    );
};

export default Prototyper;
