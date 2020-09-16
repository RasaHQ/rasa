const path = require('path');
const { remarkProgramOutput } = require('./plugins/program_output');
const {
  rehypePlugins: themeRehypePlugins,
  remarkPlugins: themeRemarkPlugins,
} = require('@rasahq/docusaurus-theme-tabula');

const isDev = process.env.NODE_ENV === 'development';
const isStaging = process.env.NETLIFY && process.env.CONTEXT === 'staging';
const isPreview = process.env.NETLIFY && process.env.CONTEXT === 'deploy-preview';

// FIXME: remove "next/" when releasing + remove the "next/" in
// https://github.com/RasaHQ/rasa-website/blob/master/netlify.toml
const BASE_URL = '/docs/rasa/next/';
const SITE_URL = 'https://rasa.com';
// NOTE: this allows switching between local dev instances of rasa/rasa-x
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;

let existingVersions = [];
try { existingVersions = require('./versions.json'); } catch (e) { console.info('no versions.json file found') }
const currentVersionPath = isDev || isPreview ? '/' : `${existingVersions[0]}/`;

const routeBasePath = '/';
const existingVersionRE = new RegExp(
  `${routeBasePath}/(${existingVersions.reduce((s, v, i) => `${s}${i > 0 ? '|' : ''}${v}`, '')}).?`,
);
const currentVersionRE = new RegExp(`(${routeBasePath})(.?)`);

const versionLabels = {
  current:
    isDev || isPreview
      ? `Next (${isPreview ? 'deploy preview' : 'dev'})`
      : existingVersions.length < 1
      ? 'Current'
      : 'Next',
};

module.exports = {
  onBrokenLinks: 'warn',
  customFields: {
    versionLabels,
    legacyVersions: [{
      label: 'Legacy 1.x',
      href: 'https://legacy-docs-v1.rasa.com',
      target: '_self',
    }]
  },
  title: 'Rasa Open Source Documentation',
  tagline: 'An open source machine learning framework for automated text and voice-based conversations',
  url: SITE_URL,
  baseUrl: BASE_URL,
  favicon: '/img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  themeConfig: {
    algolia: {
      apiKey: '09ef5d111ebd9002df72575d516693af',
      indexName: 'BH4D9OD16A',
      // searchParameters: {}, // Optional (if provided by Algolia)
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
    navbar: {
      hideOnScroll: false,
      title: 'Rasa Open Source',
      logo: {
        alt: 'Rasa Logo',
        src: `/img/rasa-logo.svg`,
        href: SITE_URL,
      },
      items: [
        {
          label: 'Rasa Open Source',
          to: path.join('/', BASE_URL),
          position: 'left',
        },
        {
          label: 'Rasa X',
          position: 'left',
          href: `${SWAP_URL}/docs/rasa-x/next/`,
          target: '_self',
        },
        {
          label: 'Rasa Action Server',
          position: 'left',
          href: 'https://rasa.com/docs/action-server',
        },
        {
          target: '_self',
          href: 'https://blog.rasa.com/',
          label: 'Blog',
          position: 'right',
        },
        {
          target: '_self',
          href: `${SITE_URL}/community/join/`,
          label: 'Community',
          position: 'right',
        },
      ],
    },
    // footer: {
    //   copyright: `Copyright © ${new Date().getFullYear()} Rasa Technologies GmbH`,
    // },
    gtm: {
      containerID: 'GTM-PK448GB',
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
      editUrl: 'https://github.com/rasahq/rasa/edit/master/docs/',
      showLastUpdateTime: true,
      showLastUpdateAuthor: true,
      rehypePlugins: [
        ...themeRehypePlugins,
      ],
      remarkPlugins: [
        ...themeRemarkPlugins,
        remarkProgramOutput,
      ],
      lastVersion: isDev || isPreview || existingVersions.length < 1 ? 'current' : undefined, // aligns / to last versioned folder in production
      // includeCurrentVersion: true, // default is true
      versions: {
        current: {
          label: versionLabels['current'],
          path: isDev || isPreview || existingVersions.length < 1 ? '' : 'next',
        },
      },
    }],
    ['@docusaurus/plugin-content-pages', {}],
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
