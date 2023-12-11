module.exports = {
  default: [
    "introduction",
    "rasa-pro",
    {
      type: "category",
      label: "Installation",
      collapsed: true,
      items: [
        "installation/environment-set-up",
        "installation/installing-rasa-open-source",
        {
          label: "Installing Rasa Pro",
          collapsed: true,
          type: "category",
          items: [
            "installation/rasa-pro/rasa-pro-artifacts",
            "installation/rasa-pro/installation",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "Building Assistants",
      collapsed: false,
      items: [
        "migrate-from",
        "command-line-interface",
        {
          type: "category",
          label: "Best Practices",
          collapsed: true,
          items: [
            "conversation-driven-development",
            "generating-nlu-data",
            "writing-stories",
          ],
        },
        {
          type: "category",
          label: "Conversation Patterns",
          collapsed: true,
          items: [
            "chitchat-faqs",
            "business-logic",
            "fallback-handoff",
            "unexpected-input",
            "contextual-conversations",
            "reaching-out-to-user",
          ],
        },
        {
          type: "category",
          label: "Preparing For Production",
          collapsed: true,
          items: [
            "messaging-and-voice-channels",
            "tuning-your-model",
            "testing-your-assistant",
            "setting-up-ci-cd",
          ],
        },
        "glossary",
      ],
    },
    {
      type: "category",
      label: "Deploying Assistants",
      collapsed: true,
      items: [
        "deploy/introduction",
        "deploy/deploy-rasa",
        "deploy/deploy-action-server",
        "deploy/deploy-rasa-pro-services",
      ],
    },
    {
      type: "category",
      label: "Monitoring and Analyzing Assistants",
      collapsed: true,
      items: [
        {
          type: "category",
          label: "Analytics",
          collapsed: true,
          items: [
            "monitoring/analytics/getting-started-with-analytics",
            "monitoring/analytics/realtime-markers",
            "monitoring/analytics/example-queries",
            "monitoring/analytics/data-structure-reference",
          ],
        },
        "monitoring/tracing",
        "monitoring/load-testing-guidelines",
      ],
    },
    "pii-management",
    {
      type: "category",
      label: "Concepts",
      collapsed: false,
      items: [
        {
          type: "category",
          label: "Training Data",
          items: [
            "training-data-format",
            "nlu-training-data",
            "stories",
            "rules",
          ],
        },
        "domain",
        {
          type: "category",
          label: "Config",
          items: [
            "model-configuration",
            "components",
            "policies",
            "custom-graph-components",
            "training-data-importers",
            "language-support",
            "graph-recipe",
            "spaces",
          ],
        },
        {
          type: "category",
          label: "Actions",
          items: [
            "actions",
            "responses",
            "custom-actions",
            "forms",
            "default-actions",
            "slot-validation-actions",
          ],
        },
        {
          type: "category",
          label: "Evaluation",
          items: ["markers"],
        },
        {
          type: "category",
          label: "Channel Connectors",
          items: [
            {
              type: "category",
              label: "Text & Chat",
              items: [
                "connectors/facebook-messenger",
                "connectors/slack",
                "connectors/telegram",
                "connectors/twilio",
                "connectors/hangouts",
                "connectors/microsoft-bot-framework",
                "connectors/cisco-webex-teams",
                "connectors/rocketchat",
                "connectors/mattermost",
              ],
            },
            {
              type: "category",
              label: "Voice",
              items: ["connectors/audiocodes-voiceai-connect"],
            },
            "connectors/custom-connectors",
          ],
        },
        {
          type: "category",
          label: "Architecture", // name still confusing with architecture page elsewhere
          items: [
            "arch-overview",
            "tracker-stores",
            "event-brokers",
            "model-storage",
            "lock-stores",
            "secrets-managers",
            "nlu-only",
            "nlg",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "Action Server",
      collapsed: true,
      items: [
        "action-server/index",
        {
          "Action Server Fundamentals": [
            "action-server/actions",
            "action-server/events",
          ],
        },
        {
          "Using the Rasa SDK": [
            "action-server/running-action-server",
            {
              type: "category",
              label: "Writing Custom Actions",
              collapsed: true,
              items: [
                "action-server/sdk-actions",
                "action-server/sdk-tracker",
                "action-server/sdk-dispatcher",
                "action-server/sdk-events",
                {
                  type: "category",
                  label: "Special Action Types",
                  collapsed: true,
                  items: [
                    "action-server/knowledge-bases",
                    "action-server/validation-action",
                  ],
                },
              ],
            },
            "action-server/sanic-extensions",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "APIs",
      collapsed: true,
      items: [
        "http-api",
        "nlu-only-server",
        // 'jupyter-notebooks',
      ],
    },
    {
      type: "category",
      label: "Reference",
      collapsed: true,
      items: [
        "telemetry/telemetry",
        "telemetry/reference",
        require("./docs/reference/sidebar.json"),
      ],
    },
    {
      type: "category",
      label: "Change Log",
      collapsed: true,
      items: [
        "rasa-pro-changelog",
        "changelog",
        "sdk_changelog",
        "compatibility-matrix",
        "migration-guide",
        {
          type: "link",
          label: "Actively Maintained Versions",
          href: "https://rasa.com/rasa-product-release-and-maintenance-policy/",
        },
      ],
    },
  ],
};
