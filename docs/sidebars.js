module.exports = {
  default: [
    'introduction',
    'playground',
    {
      type: 'category',
      label: 'Building Assistants',
      collapsed: false,
      items: [
          'installation',
          'migrate-from',
          'command-line-interface',
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
            'reaching-out-to-user',
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
        "glossary",
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
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
            'components',
            'policies',
            'training-data-importers',
            'language-support',
          ],
        },
        {
          type: 'category',
          label: 'Actions',
          items: [
            'actions',
            'responses',
            'custom-actions',
            'forms',
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
            'connectors/custom-connectors',
          ],
        },
        {
          type: 'category',
          label: 'Architecture', // name still confusing with architecture page elsewhere
          items: [
            'arch-overview',
            'tracker-stores',
            'event-brokers',
            'model-storage',
            'lock-stores',
            'nlu-only',
            'nlg',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'APIs',
      collapsed: true,
      items: [
        'http-api',
        // 'jupyter-notebooks',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        'telemetry/telemetry',
        'telemetry/reference',
        require('./docs/reference/sidebar.json')],
    },
    {
      type: 'category',
      label: 'Change Log',
      collapsed: true,
      items: ['changelog', 'migration-guide'],
    },
  ],
};
