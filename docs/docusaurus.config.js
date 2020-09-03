const path = require('path');
const { remarkProgramOutput } = require('./plugins/program_output');
const { rehypePlugins, remarkPlugins } = require('@rasahq/docusaurus-theme-tabula');

const isDev = process.env.NODE_ENV === 'development';

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

// FIXME: when deploying this for real, change to '/docs/rasa/'
const BASE_URL = '/docs/rasa/next/';
const SITE_URL = 'https://rasa.com';

// this allows switching doc sites in development
const SWAP_URL = isDev ? 'http://localhost:3001' : SITE_URL;

module.exports = {
  title: 'Rasa Open Source Documentation',
  // FIXME: tagline should be different from the title
  tagline: 'Rasa Open Source Documentation',
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
        {
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
        }      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Rasa Technologies GmbH`,
    },
    gtm: {
      containerID: 'GTM-PK448GB',
    },
  },
  themes: [
    '@rasahq/docusaurus-theme-tabula',
    // path.resolve(__dirname, './themes/theme-live-codeblock')
  ],
  plugins: [
    [
      '@docusaurus/plugin-content-docs/',
      {
        // https://v2.docusaurus.io/docs/next/docs-introduction/#docs-only-mode
        routeBasePath: '/',
        // FIXME: the following option is now deprecated
        homePageId: 'index',
        sidebarPath: require.resolve('./sidebars.js'),
        editUrl: 'https://github.com/rasahq/rasa/edit/master/docs/',
        rehypePlugins: [
          ...rehypePlugins,
        ],
        remarkPlugins: [
          ...remarkPlugins,
          remarkProgramOutput,
        ],
        /* TODO review all of these options ↓↓↓↓↓ */
        // path: 'docs',
        // routeBasePath: 'docs',
        // homePageId: undefined,
        // include: ['**/*.{md,mdx}'],
        // sidebarPath: 'sidebars.json',
        // docLayoutComponent: '@theme/DocPage',
        // docItemComponent: '@theme/DocItem',
        // showLastUpdateTime: false,
        // showLastUpdateAuthor: false,
        // admonitions: {},
        // excludeNextVersionDocs: false,
        // includeCurrentVersion: true,
        // disableVersioning: false,
        // lastVersion: undefined,
        // versions: {},
      },
    ],
    ['@docusaurus/plugin-content-pages', {}],
    ['@docusaurus/plugin-sitemap',
      {
        cacheTime: 600 * 1000, // 600 sec - cache purge period
        changefreq: 'weekly',
        priority: 0.5,
      }],
    [path.resolve(__dirname, './plugins/google-tagmanager'), {}],
    isDev && ['@docusaurus/plugin-debug', {}],
  ].filter(Boolean),
};
