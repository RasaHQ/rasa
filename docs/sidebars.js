module.exports = {
  default: [
    "introduction", // TODO: ENG-516
    {
      type: "category",
      label: "Getting started",
      collapsed: false,
      items: [
        "calm", // TODO: ENG-517
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
        "tutorial",
        "command-line-interface",        
      ],
    },
    {
      type: "category",
      label: "Key concepts",
      collapsed: true,
      items: [
        // TODO: ENG-537
        "concepts/flows",
        "concepts/dialogue-understanding",        
        "concepts/domain",
        "concepts/unhappy-paths",
        {
          type: "category",
          label: "Actions",
          items: [
            "concepts/actions",
            "concepts/custom-actions",
            "concepts/default-actions",
            "concepts/slot-validation-actions",
          ],
        },
        {
          type: "category",
          label: "Responses",
          items: [
            "concepts/responses",
            "concepts/contextual-response-rephraser",            
          ],
        },        
        {
          type: "category",
          label: "Components",
          items: [
            "concepts/components/overview",
            "llms/llm-configuration",
            "llms/llm-custom",
            "concepts/components/custom-graph-components",
            "concepts/components/graph-recipe",
          ],
        },
        "concepts/policies", // TODO: ENG-538
        {
          type: "category",
          label: "Building Classic Assistants",
          collapsed: true,
          items: [
            // TODO: ENG-537
            {
              type: "category",
              label: "Conversation Patterns",
              collapsed: true,
              items: [
                "building-classic-assistants/chitchat-faqs",
                "building-classic-assistants/business-logic",
                "building-classic-assistants/fallback-handoff",
                "building-classic-assistants/unexpected-input",
                "building-classic-assistants/contextual-conversations",
                "building-classic-assistants/reaching-out-to-user",
              ],
            },
            {
              type: "category",
              label: "Config",
              items: [
                "building-classic-assistants/model-configuration",
                "building-classic-assistants/components", // TODO: ENG-538
                "building-classic-assistants/policies", // TODO: ENG-538
                "building-classic-assistants/language-support",
              ],
            },
            "building-classic-assistants/domain",
            "building-classic-assistants/rules",
            "building-classic-assistants/stories",
            "building-classic-assistants/forms",
            "building-classic-assistants/training-data-format",
            "building-classic-assistants/nlu-training-data",
            "building-classic-assistants/tuning-your-model",
            "building-classic-assistants/nlu-only",         
          ],
        },
      ],
    },
    // TODO: ENG-539
    {
      type: "category",
      label: "Testing & Deploying to Production",
      collapsed: true,
      items: [
        {
          type: "category",
          label: "Architecture",
          items: [
            "production/arch-overview",
            "production/tracker-stores",
            "production/event-brokers",
            "production/model-storage",
            "production/lock-stores",
            "production/secrets-managers",
            "production/nlg",
          ],
        },
        // TODO: ENG-540
        "production/testing-your-assistant",
        {
          type: "category",
          label: "Channel Connectors",
          items: [
            "connectors/messaging-and-voice-channels",
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
        "production/setting-up-ci-cd",
        {
          type: "category",
          label: "Evaluation",
          items: ["production/markers"],
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
        "production/load-testing-guidelines",
      ],
    },
    {
      type: "category",
      label: "Security & Compliance",
      collapsed: true,
      items: [
        "security/llm-guardrails", // TODO: ENG-541
        "security/pii-management",
      ],
    },
    {
      type: "category",
      label: "Operating at scale",
      collapsed: true,
      items: [
        "operating/llm-cost-optimizations",
        {
          type: "category",
          label: "Analytics",
          collapsed: true,
          items: [
            "operating/analytics/getting-started-with-analytics",
            "operating/analytics/realtime-markers",
            "operating/analytics/example-queries",
            "operating/analytics/data-structure-reference",
          ],
        },
        "operating/tracing",
        "operating/spaces",        
      ],
    },
    {
      type: "category",
      label: "APIs",
      collapsed: true,
      items: ["http-api", "nlu-only-server"],
    },
    {
      type: "category",
      label: "Reference",
      collapsed: true,
      items: [
        "building-classic-assistants/glossary",        
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
  ]
};
