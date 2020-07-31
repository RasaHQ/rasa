module.exports = {
  someSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: true,
      items: [
        'index',
        'prototype-an-assistant',
        'installation',
        'cheatsheet',
        'migrate-from',
      ],
    },
    {
      type: 'category',
      label: 'Best Practices',
      collapsed: true,
      items: [
        'cdd',
        'generating-nlu-data',
        'writing-stories',
      ],
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
        'tuning-your-model',
        'testing-your-assistant',
        'setting-up-ci-cd',
        'how-to-deploy',
        'messaging-and-voice-channels',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        'glossary',
      ],
    },
    {
      type: 'category',
      label: 'Rasa Open Source',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Training Data',
          items: [
            'training-data-format',
            'stories',
          ],
        },
        'domain',
        {
          type: 'category',
          label: 'Config',
          items: [
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
            {
              type: 'category',
              label: 'Policies',
              items: [
                'policies',
              ],
            },
            'training-data-importers',
          ],
        },
        {
          type: 'category',
          label: 'Actions',
          items: [
            'responses',
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
          items: [
            'tracker-stores',
            'event-brokers',
            'model-storage',
            'lock-stores',
            'nlg',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Rasa SDK',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Custom Actions',
          items: [
            'actions',
            'knowledge-bases',
          ],
        },
        'tracker',
        'events',
        {
          type: 'category',
          label: 'Reference',
          items: [
            'action-server',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'architecture',
        'command-line-interface',
        'http-api',
        'jupyter-notebooks',
        {
          type: 'category',
          label: 'Versioning',
          items: [
            'changelog',
            'migration-guide',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Old Content',
      items: [
        'configuring-http-api',
        'using-nlu-only',
        'slots',
        'interactive-learning',
        'rasa-sdk',
        ]
    },
  ],
};
