const path = require('path');
const { remarkProgramOutput } = require('./plugins/program_output');
const { rehypePlugins, remarkPlugins } = require('@rasahq/docusaurus-theme-tabula');

const isDev = process.env.NODE_ENV === 'development';

// FIXME: when deploying this for real, change to '/docs/rasa/'
const BASE_URL = '/docs/rasa/next/';
const SITE_URL = 'https://rasa.com';

// this allows switching doc sites in development
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;

let versions = [];
try {
  versions = require('./versions.json');
} catch (ex) {
  // Nothing to do here, in dev mode, only
  // one version of the doc is available
}

const legacyVersion = {
  label: 'Legacy 1.x',
  href: 'https://legacy-docs-v1.rasa.com',
  target: '_self',
};

const allVersions = {
  label: 'Versions',
  to: '/', // "fake" link
  position: 'left',
  items:
    versions.length > 0
      ? [
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
  tagline: 'Cras justo odio, dapibus ac facilisis in, egestas eget quam.',
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
          href: `${SITE_URL}/docs/rasa-overview/`,
          label: 'Overview of Rasa',
          position: 'left',
          target: '_self',
        },
        {
          label: 'Rasa Open Source',
          to: path.join('/', BASE_URL), // for purpose of route match styling
          position: 'left',
          items: [
            {
              to: path.join('/', BASE_URL),
              label: 'Usage',
            },
            {
              to: path.join('/', BASE_URL, 'api'),
              label: 'API',
            },
          ],
        },
        {
          label: 'Rasa X',
          position: 'left',
          items: [
            {
              href: `${SWAP_URL}/docs/rasa-x/next/`,
              label: 'Usage',
              target: '_self',
            },
            {
              href: `${SWAP_URL}/docs/rasa-x/next/api`,
              label: 'API',
              target: '_self',
            },
          ],
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
    path.resolve(__dirname, './themes/theme-live-codeblock')
  ],
  plugins: [
    ['@docusaurus/plugin-content-docs/', {
      routeBasePath: '/',
      sidebarPath: require.resolve('./sidebars.js'),
      editUrl: 'https://github.com/rasahq/rasa/edit/master/docs/',
      showLastUpdateTime: true,
      showLastUpdateAuthor: true,
      rehypePlugins: [
        ...rehypePlugins,
      ],
      remarkPlugins: [
        ...remarkPlugins,
        remarkProgramOutput,
      ],
      /*
        VERSIONING
        TODO: figure out these options (new since alpha 62)
      */
      // excludeNextVersionDocs: false,
      // includeCurrentVersion: true,
      // disableVersioning: false,
      // lastVersion: undefined,
      // onlyIncludeVersions: undefined,
      versions: {},
    }],
    ['@docusaurus/plugin-content-pages', {}],
    ['@docusaurus/plugin-sitemap',
      {
        cacheTime: 600 * 1000, // 600 sec - cache purge period
        changefreq: 'weekly',
        priority: 0.5,
      }],
    /*
    TODO: configure this plugin:
    https://v2.docusaurus.io/docs/using-plugins#docusaurusplugin-ideal-image
     */
    ['@docusaurus/plugin-ideal-image', {}],
    /*
    TODO: configure this plugin:
    https://v2.docusaurus.io/docs/using-plugins#docusaurusplugin-client-redirects
     */
    // ['@docusaurus/plugin-client-redirects', {
    //   fromExtensions: ['html', 'htm']
    // }],
    isDev && ['@docusaurus/plugin-debug', {}],
    [path.resolve(__dirname, './plugins/google-tagmanager'), {}],
  ].filter(Boolean),
};
