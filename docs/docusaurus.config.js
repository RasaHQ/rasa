const path = require('path');
const { remarkProgramOutput } = require('./plugins/program_output');
const {
  rehypePlugins: themeRehypePlugins,
  remarkPlugins: themeRemarkPlugins,
} = require('@rasahq/docusaurus-theme-tabula');


// FIXME: remove "next/" when releasing + remove the "next/" in
// http://github.com/RasaHQ/rasa-website/blob/master/netlify.toml
const BASE_URL = '/docs/rasa/next/';
const SITE_URL = 'https://rasa.com';

// NOTE: this allows switching between local dev instances of rasa/rasa-x
const isDev = process.env.NODE_ENV === 'development';
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;

/* VERSIONING: WIP */

const routeBasePath = '/';
let versions = [];
try { versions = require('./versions.json'); } catch (ex) { console.info('no versions.json file found; assuming dev mode.') }

const legacyVersion = {
  label: 'Legacy 1.x',
  href: 'https://legacy-docs-v1.rasa.com',
  target: '_self',
};

const allVersions = {
  label: 'Versions',
  to: '/', // "fake" link
  position: 'left',
  items: versions.length > 0 ? [
    {
      label: versions[0],
      to: '/',
      activeBaseRegex: versions[0],
    },
    ...versions.slice(1).map((version) => ({
      label: version,
      to: `${version}/`,
      activeBaseRegex: version,
    })),
    {
      label: 'Master/Unreleased',
      to: 'next/',
      activeBaseRegex: `next`,
    },
    legacyVersion,
  ]
: [
    {
      label: 'Master/Unreleased',
      to: '/',
      activeBaseRegex: `/`,
    },
    legacyVersion,
  ],
}

module.exports = {
  customFields: {
    // NOTE: all non-standard options should go in this object
  },
  title: 'Rasa Open Source Documentation',
  tagline: 'An open source machine learning framework for automated text and voice-based conversations',
  url: SITE_URL,
  baseUrl: BASE_URL,
  favicon: '/img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  themeConfig: {
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
          target: '_self',
          href: 'http://blog.rasa.com/',
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
    footer: {
      copyright: `Copyright © ${new Date().getFullYear()} Rasa Technologies GmbH`,
    },
    gtm: {
      containerID: 'GTM-PK448GB',
    },
  },
  themes: [
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
