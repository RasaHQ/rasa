module.exports = {
  default: [
    "introduction", // TODO: ENG-516
    {
      type: "category",
      label: "Getting started",
      collapsed: false,
      items: [
        "conversational-ai-with-rasa", // TODO: create-task
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
        "dm2-vs-intent-based-systems", // TODO: create-task
        "dm2-vs-llm-chaining", // TODO: create-task
      ],
    },
    {
      type: "category",
      label: "Key concepts",
      collapsed: true,
      items: [
        // TODO: create-task spit CLI docs
        "concepts/command-line-interface",
        "concepts/flows",
        "concepts/domain",
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
        "concepts/llm-command-generator",
        "concepts/responses",
        {
          type: "category",
          label: "Config",
          items: [
            "concepts/llm-configuration",
            "concepts/custom-graph-components",
            "concepts/training-data-importers",
            "concepts/graph-recipe",
            "concepts/spaces",
          ],
        },
        "concepts/policies", // TODO: create-task
        // TODO: create-task add how-to-guides
        "concepts/glossary",
        {
          type: "category",
          label: "Building Classic Assistants",
          collapsed: true,
          items: [
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
                "building-classic-assistants/components",
                "building-classic-assistants/policies",
                "building-classic-assistants/language-support",
              ],
            },
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
        // TODO: create-task change content to remove DM1 topics
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
        "security/llm-guardrails", // TODO: create-task
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
  llms: [
    "start-here",
    "tutorial",
    {
      type: "category",
      label: "Getting Started",
      collapsed: false,

      items: [
        "llms/llm-installation", // DOESN'T NEED TO STAY
        "llms/llm-next-gen", // TODO: reuse content
        "llms/llm-configuration", // MOVED
      ],
    },
    {
      type: "category",
      label: "Next-Gen Assistants",
      collapsed: false,
      items: [
        "concepts/flows", // MOVED
        "llms/unhappy-paths",
        "llms/llm-detector",
      ],
    },
    {
      type: "category",
      label: "Components",
      collapsed: false,
      items: [
        "llms/llm-intent",
        "llms/llm-command-generator", // DONE
        "llms/llm-nlg",
        "llms/llm-docsearch",
        "llms/llm-intentless",
        "llms/flow-policy",
      ],
    },
    "llms/llm-custom",
  ],
};
