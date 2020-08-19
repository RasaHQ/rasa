import React from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import ThemeContext from '@theme/theme-context';
import { isProductionBuild, uuidv4 } from '@theme/utils';

import PrototyperContext from './context';

const jsonHeaders = {
	Accept: 'application/json',
	'Content-Type': 'application/json',
};
const trackerPollingInterval = 2000;

const Prototyper = ({
	children,
	startPrototyperApi,
	trainModelApi,
	chatBlockSelector,
	chatBlockScriptUrl,
}) => {
	const [trackingId, setTrackingId] = React.useState(null);
	const [hasStarted, setHasStarted] = React.useState(false);
	const [projectDownloadUrl, setProjectDownloadUrl] = React.useState(null);
	const [trainingData, setTrainingData] = React.useState({});
	const [pollingIntervalId, setPollingIntervalId] = React.useState(null);
	const [hasTrained, setHasTrained] = React.useState(false);
	const [isTraining, setIsTraining] = React.useState(false);

	// FIXME: once we can use `rasa-ui` outside of `rasa-x`, we can remove this
	const insertChatBlockScript = () => {
		if (ExecutionEnvironment.canUseDOM) {
			const scriptElement = document.createElement('script');
			scriptElement.src = chatBlockScriptUrl;
			document.body.appendChild(scriptElement);
		}
	};

	// update tracking id when component is mounting
	React.useEffect(() => {
		setTrackingId(isProductionBuild() ? uuidv4() : 'the-hash');
		insertChatBlockScript();
	}, []);
	// initialize the chatblock once we have a tracking id
	React.useEffect(() => {
		if (trackingId !== null) {
			updateChatBlock();
		}
	}, [trackingId]);

	const onLiveCodeStart = (name, value) => {
		setTrainingData((prevTrainingData) => ({ ...prevTrainingData, [name]: value }));
	};

	const onLiveCodeChange = (name, value) => {
		setTrainingData((prevTrainingData) => ({ ...prevTrainingData, [name]: value }));
		if (!hasStarted) {
			// track the start here
			setHasStarted(true);
			fetch(startPrototyperApi, {
				method: 'POST',
				headers: jsonHeaders,
				body: JSON.stringify({
					tracking_id: trackingId,
					editor: 'main',
				}),
			});
		}
	};

	const trainModel = () => {
		setIsTraining(true);
		// train the model, resetting the chatblock
		if (pollingIntervalId) {
			updateChatBlock();
			clearInterval(pollingIntervalId);
			setPollingIntervalId(null);
		}

		fetch(trainModelApi, {
			method: 'POST',
			headers: jsonHeaders,
			body: JSON.stringify({ tracking_id: trackingId, ...trainingData }),
		})
			.then((response) => response.json())
			.then((data) => {
				setHasTrained(true);
				setProjectDownloadUrl(data.project_download_url);
				if (data.rasa_service_url) {
					startFetchingTracker(data.rasa_service_url);
				}
			})
			.finally(() => setIsTraining(false));
	};

	const downloadProject = () => {
		if (projectDownloadUrl) {
			location.href = projectDownloadUrl;
		}
	};

	const updateChatBlock = (baseUrl = '', tracker = {}) => {
		if (ExecutionEnvironment.canUseDOM) {
			if (!window.ChatBlock) {
				// FIXME: once we can use `rasa-ui` outside of `rasa-x`, we can remove this
				setTimeout(() => updateChatBlock(baseUrl, tracker), 500);
			} else {
				window.ChatBlock.default.init({
					onSendMessage: (message) => {
						sendMessage(baseUrl, message);
					},
					username: trackingId,
					tracker,
					selector: chatBlockSelector,
				});
			}
		}
	};

	const fetchTracker = (baseUrl) => {
		fetch(`${baseUrl}/conversations/${trackingId}/tracker`, {
			method: 'GET',
			header: 'jsonHeaders',
		})
			.then((response) => response.json())
			.then((tracker) => updateChatBlock(baseUrl, tracker));
	};

	const sendMessage = (baseUrl, message) => {
		fetch(`${baseUrl}/webhooks/rest/webhook`, {
			method: 'POST',
			headers: jsonHeaders,
			body: JSON.stringify({
				sender: trackingId,
				message: message,
			}),
		}).then(() => fetchTracker(baseUrl));
	};

	const startFetchingTracker = (baseUrl) => {
		fetchTracker(baseUrl);

		const updateIntervalId = setInterval(() => {
			fetchTracker(baseUrl);
		}, trackerPollingInterval);

		setPollingIntervalId(updateIntervalId);
	};

	return (
		<ThemeContext.Provider value={{ onLiveCodeChange, onLiveCodeStart }}>
			<PrototyperContext.Provider value={{ trainModel, downloadProject, hasTrained, isTraining }}>
				{children}
			</PrototyperContext.Provider>
		</ThemeContext.Provider>
	);
};

export default Prototyper;
