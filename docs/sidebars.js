module.exports = {
  someSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
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
      collapsed: false,
      items: [
        'cdd',
        'generating-nlu-data',
        'writing-stories',
      ],
    },
    {
      type: 'category',
      label: 'Preparing For Production',
      collapsed: false,
      items: [
        'testing-your-assistant',
        'setting-up-ci-cd',
        'how-to-deploy',
        'messaging-and-voice-channels',
      ],
    },
    {
      type: 'category',
      label: 'NLU',
      collapsed: false,
      items: [
        'training-data-format',
        {
          type: 'category',
          label: 'Components',
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
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'glossary',
      ],
    },
    {
      type: 'category',
      label: 'Dialogue Management',
      collapsed: false,
      items: [
        'stories',
        'domains',
        {
          type: 'category',
          label: 'Policies',
          items: [
            'policies',
          ],
        },
        {
          type: 'category',
          label: 'Actions',
          items: [
            'retrieval-actions',
            'forms',
            'reminders-and-external-events',
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
            'lock-stores',
            'nlg',
            'cloud-storage',
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
        'building-assistants',
        'configuring-http-api',
        'using-nlu-only',
        'language-support',
        'choosing-a-pipeline',
        'entity-extraction',
        'responses',
        'slots',
        'interactive-learning',
        'fallback-actions',
        'rasa-sdk',
        'training-data-importers',
        'core-featurization',
        'tensorflow_usage',
        ]
    },
  ],
};
