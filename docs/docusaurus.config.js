const path = require('path');
const { remarkProgramOutput } = require('./plugins/program_output');
const {
  rehypePlugins: themeRehypePlugins,
  remarkPlugins: themeRemarkPlugins,
} = require('@rasahq/docusaurus-theme-tabula');

const isDev = process.env.NODE_ENV === 'development';
const isStaging = process.env.NETLIFY && process.env.CONTEXT === 'staging';
const isPreview = process.env.NETLIFY && process.env.CONTEXT === 'deploy-preview';

const BASE_URL = '/docs/rasa/';
const SITE_URL = 'https://rasa.com';
// NOTE: this allows switching between local dev instances of rasa/rasa-enterprise
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;

let existingVersions = [];
try { existingVersions = require('./versions.json'); } catch (e) { console.info('no versions.json file found') }

const routeBasePath = '/';

const versionLabels = {
  current: 'Main/Unreleased'
};

module.exports = {
  customFields: {
    // FIXME: this is a simplistic solution to https://github.com/RasaHQ/rasa/issues/7011
    // either (A): create a more sophisticated solution to link the precise branch and doc to be edited, according to branch settings
    // or (B): create a README document (or a section in the main README) which explains how to contribute docs fixes, and link all edit links to this
    rootEditUrl: 'https://github.com/rasahq/rasa/',
    productLogo: '/img/blocks.png',
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
  title: 'Rasa & Rasa Pro Documentation',
  url: SITE_URL,
  baseUrl: BASE_URL,
  favicon: '/img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  themeConfig: {
    announcementBar: {
      id: 'rasa_sdk_change', // Any value that will identify this message.
      content: '<strong>Rasa SDK<strong> documentation has been moved to a <a href="https://rasa.com/docs/rasa/action-server/">section</a> of <strong>Rasa Open Source</strong>.',
      backgroundColor: '#6200F5', // Defaults to `#fff`.
      textColor: '#fff', // Defaults to `#000`.
      // isCloseable: false, // Defaults to `true`.
    },
    algolia: {
      appId: '94J0KRFPTZ',
      apiKey: '75ea5a8c8f4b16405c560a4ba786256b',
      indexName: 'rasa',
      inputSelector: '.search-bar',

    },
    navbar: {
      hideOnScroll: false,
      title: 'Rasa',
      items: [
        {
          label: 'Rasa',
          to: path.join('/', BASE_URL),
          position: 'left',
        },
        {
          target: '_self',
          label: 'Rasa X/Enterprise',
          position: 'left',
          href: `${SWAP_URL}/docs/rasa-enterprise/`,
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
  },
  themes: [
    '@docusaurus/theme-search-algolia',
    '@rasahq/docusaurus-theme-tabula',
    path.resolve(__dirname, './themes/theme-custom')
  ],
  plugins: [
    ['@docusaurus/plugin-content-docs/', {
      routeBasePath,
      sidebarPath: require.resolve('./sidebars.js'),
      editUrl: 'https://github.com/rasahq/rasa/edit/main/docs/',
      showLastUpdateTime: true,
      showLastUpdateAuthor: true,
      rehypePlugins: [
        ...themeRehypePlugins,
      ],
      remarkPlugins: [
        ...themeRemarkPlugins,
        remarkProgramOutput,
      ],
      lastVersion: existingVersions[0] || 'current', // aligns / to last versioned folder in production
      versions: {
        current: {
          label: versionLabels['current'],
          path: existingVersions.length < 1 ? '' : 'next',
        },
      },
    }],
    ['@docusaurus/plugin-content-pages', {}],
    [
      '@docusaurus/plugin-ideal-image',
      {
        sizes: [160, 226, 320, 452, 640, 906, 1280, 1810, 2560],
        quality: 70,
      },
    ],
    ['@docusaurus/plugin-sitemap',
      {
        cacheTime: 600 * 1000, // 600 sec - cache purge period
        changefreq: 'weekly',
        priority: 0.5,
      }],
    isDev && ['@docusaurus/plugin-debug', {}],
    [path.resolve(__dirname, './plugins/google-tagmanager'), {}],
  ].filter(Boolean),
};
