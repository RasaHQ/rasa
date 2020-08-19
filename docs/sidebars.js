module.exports = {
	someSidebar: [
		'index',
		{
			type: 'category',
			label: 'Building Assistants',
			collapsed: true,
			items: [
				{
					type: 'category',
					label: 'Getting Started',
					collapsed: true,
					items: [
						'prototype-an-assistant',
						'installation',
						// 'cheatsheet',
						'migrate-from',
					],
				},
				{
					type: 'category',
					label: 'Best Practices',
					collapsed: true,
					items: ['conversation-driven-development', 'generating-nlu-data', 'writing-stories'],
				},
				{
					type: 'category',
					label: 'Conversation Patterns',
					collapsed: true,
					items: [
						'chitchat-faqs',
						'business-logic',
						'fallback-handoff',
						'unexpected-input',
						'contextual-conversations',
					],
				},
				{
					type: 'category',
					label: 'Preparing For Production',
					collapsed: true,
					items: [
						'messaging-and-voice-channels',
						'tuning-your-model',
						'testing-your-assistant',
						'setting-up-ci-cd',
						'how-to-deploy',
					],
				},
				{
					type: 'category',
					label: 'Reference',
					collapsed: true,
					items: ['glossary', require('./docs/reference/sidebar.json')],
				},
			],
		},
		{
			type: 'category',
			label: 'Concepts',
			collapsed: true,
			items: [
				{
					type: 'category',
					label: 'Training Data',
					items: ['training-data-format', 'nlu-training-data', 'stories', 'rules'],
				},
				'domain',
				{
					type: 'category',
					label: 'Config',
					items: [
						'model-configuration',
						{
							type: 'category',
							label: 'Pipeline Components',
							items: [
								'components/language-models',
								'components/tokenizers',
								'components/featurizers',
								'components/intent-classifiers',
								'components/entity-extractors',
								'components/selectors',
								'components/custom-nlu-components',
							],
						},
						'policies',
						'training-data-importers',
					],
				},
				{
					type: 'category',
					label: 'Actions',
					items: [
						// 'actions',
						'responses',
						{
							type: 'category',
							label: 'Custom Actions',
							items: [
								'custom-actions',
								'knowledge-bases',
								{
									type: 'category',
									label: 'Rasa SDK',
									collapsed: true,
									items: [
										'running-action-server',
										'tracker-dispatcher',
										// 'events',
										// 'rasa-sdk-changelog'
									],
								},
							],
						},
						'retrieval-actions',
						'forms',
						'reminders-and-external-events',
						'default-actions',
					],
				},
				{
					type: 'category',
					label: 'Channel Connectors',
					items: [
						'connectors/your-own-website',
						'connectors/facebook-messenger',
						'connectors/slack',
						'connectors/telegram',
						'connectors/twilio',
						'connectors/hangouts',
						'connectors/microsoft-bot-framework',
						'connectors/cisco-webex-teams',
						'connectors/rocketchat',
						'connectors/mattermost',
					],
				},
				{
					type: 'category',
					label: 'Architecture', // name still confusing with architecture page elsewhere
					items: ['tracker-stores', 'event-brokers', 'model-storage', 'lock-stores', 'nlg'],
				},
			],
		},
		{
			type: 'category',
			label: 'APIs',
			collapsed: true,
			items: [
				'command-line-interface',
				{
					type: 'category',
					label: 'HTTP API',
					collapsed: true,
					items: ['http-api', 'http-api-spec'],
				},
				'jupyter-notebooks',
			],
		},
		{
			type: 'category',
			label: 'Change Log',
			collapsed: true,
			items: ['changelog', 'migration-guide'],
		},
	],
};
