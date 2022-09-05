// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const path = require('path');
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

const isDev = process.env.NODE_ENV === 'development';
const isStaging = process.env.NETLIFY && process.env.CONTEXT === 'staging';
const isPreview = process.env.NETLIFY && process.env.CONTEXT === 'deploy-preview';

const BASE_URL = '/docs/rasa/';
const SITE_URL = 'https://rasa.com';
// NOTE: this allows switching between local dev instances of rasa/rasa-enterprise
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;
/** @type {import('@docusaurus/types').Config} */



const versionLabels = {
  current: 'Main/Unreleased'
};

const config = {
  title: 'Rasa Open Source Documentation',
  tagline: 'An open source machine learning framework for automated text and voice-based conversations',
  url: SITE_URL,
  baseUrl: BASE_URL,
  favicon: '/img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  customFields: {
    // FIXME: this is a simplistic solution to https://github.com/RasaHQ/rasa/issues/7011
    // either (A): create a more sophisticated solution to link the precise branch and doc to be edited, according to branch settings
    // or (B): create a README document (or a section in the main README) which explains how to contribute docs fixes, and link all edit links to this
    rootEditUrl: 'https://github.com/rasahq/rasa/',
    productLogo: '/img/logo-rasa-oss.png',
    versionLabels,
    legacyVersions: [{
      label: 'Legacy 1.x',
      href: 'https://legacy-docs-v1.rasa.com',
      target: '_blank',
      rel: 'nofollow noopener noreferrer',
    }],
    redocPages: [
      {
        title: 'Rasa HTTP API',
        specUrl: '/spec/rasa.yml',
        slug: '/pages/http-api',
      },
      {
        title: 'Rasa Action Server API',
        specUrl: '/spec/action-server.yml',
        slug: '/pages/action-server-api',
      }
    ]
  },
  // themes:[
  //   'classic',
  //   path.resolve(__dirname, './src/themes/theme-custom'),
  //   // path.resolve(__dirname, './src/themes/old-tabula'),
  // ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      announcementBar: {
        id: 'pre_release_notice', // Any value that will identify this message.
        content: 'These docs are for version 3.x of Rasa Open Source. <a href="https://rasa.com/docs/rasa/2.x/">Docs for the 2.x series can be found here.</a>',
        backgroundColor: '#6200F5', // Defaults to `#fff`.
        textColor: '#fff', // Defaults to `#000`.
        // isCloseable: false, // Defaults to `true`.
      },
      // algolia: {
      //   // this is configured via DocSearch here:
      //   // https://github.com/algolia/docsearch-configs/blob/master/configs/rasa.json
      //   apiKey: '1f9e0efb89e98543f6613a60f847b176',
      //   indexName: 'rasa',
      //   inputSelector: '.search-bar',
      //   searchParameters: {
      //     'facetFilters': ["tags:rasa"]
      //   }
      // },
      navbar: {
        hideOnScroll: false,
        title: 'Rasa Open Source',
        items: [
          {
            label: 'Rasa Open Source',
            to: path.join('/', BASE_URL),
            position: 'left',
          },
          {
            target: '_self',
            label: 'Rasa Enterprise',
            position: 'left',
            href: `${SWAP_URL}/docs/rasa-enterprise/`,
          },
          {
            target: '_self',
            label: 'Rasa Action Server',
            position: 'left',
            href: 'https://rasa.com/docs/action-server',
          },
          {
            href: 'https://github.com/rasahq/rasa',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
            position: 'right',
          },
          {
            target: '_self',
            href: 'https://blog.rasa.com/',
            label: 'Blog',
            position: 'right',
          },
          {
            label: 'Community',
            position: 'right',
            items: [
              {
                target: '_self',
                href: 'https://rasa.com/community/join/',
                label: 'Community Hub',
              },
              {
                target: '_self',
                href: 'https://forum.rasa.com',
                label: 'Forum',
              },
              {
                target: '_self',
                href: 'https://rasa.com/community/contribute/',
                label: 'How to Contribute',
              },
              {
                target: '_self',
                href: 'https://rasa.com/showcase/',
                label: 'Community Showcase',
              },
            ],
          },
        ],
      },
      footer: {
        copyright: `Copyright © ${new Date().getFullYear()} Rasa Technologies GmbH`,
      },
      gtm: {
        containerID: 'GTM-MMHSZCS',
      },
    }),
};

module.exports = config;
