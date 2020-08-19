const path = require('path');
const remarkSources = require('remark-sources');
const remarkCollapse = require('remark-collapse');
const { remarkProgramOutput } = require("./plugins/program_output");

let versions = [];
try {
  versions = require('./versions.json');
} catch (ex) {
  // Nothing to do here, in dev mode, only
  // one version of the doc is available
}

const legacyVersion = {
  label: 'Legacy 1.x',
  to: 'https://legacy-docs-v1.rasa.com',
};

  // FIXME: when deploying this for real, change to '/docs/rasa/'
const BASE_URL = '/docs/rasa/next/';
const SITE_URL = 'https://rasa.com';

module.exports = {
  title: 'Rasa Open Source Documentation',
  tagline: 'The tagline of my site',
  url: SITE_URL,
  baseUrl: BASE_URL,
  favicon: 'img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
    versions: {
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
      ] : [
        {
          label: 'Master/Unreleased',
          to: '/',
          activeBaseRegex: `/`,
        },
        legacyVersion,
      ],
    },
    navbar: {
      hideOnScroll: false,
      title: 'Rasa Docs',
      logo: {
        alt: 'Rasa',
        src: `${SITE_URL}/static/60e441f8eadef13bea0cc790c8cf188b/rasa-logo.svg`,
      },
      items: [
        {
          href: path.join(SITE_URL, '/docs/'),
          label: 'Overview',
          position: 'left',
        },
        {
          to: path.join('/', BASE_URL),
          label: 'Rasa Open Source',
          position: 'left',
        },
        {
          href: path.join(SITE_URL, '/docs/rasa-x/next/'),
          label: 'Rasa X',
          position: 'left',
        },
        {
          to: path.join('/', BASE_URL, 'api'),
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://blog.rasa.com/',
          label: 'Blog',
          position: 'right',
        },
        {
          href: path.join(SITE_URL, '/community/join/'),
          label: 'Community',
          position: 'right',
        },
      ],
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
    [path.resolve(__dirname, './themes/theme-classic'), {
      customCss: require.resolve('./src/stylesheets/custom.scss'),
    }],
    path.resolve(__dirname, './themes/theme-live-codeblock'),
  ],
  plugins: [
    ['@docusaurus/plugin-content-docs/', {
      // https://v2.docusaurus.io/docs/next/docs-introduction/#docs-only-mode
      routeBasePath: '/',
      // It is recommended to set document id as docs home page (`docs/` path).
      homePageId: 'index',
      sidebarPath: require.resolve('./sidebars.js'),
      editUrl: 'https://github.com/rasahq/rasa/edit/master/docs/',
      remarkPlugins: [
        [ remarkCollapse, { test: '' }],
        remarkSources,
        remarkProgramOutput
      ],
    }],
    ['@docusaurus/plugin-content-pages', {}],
    ['@docusaurus/plugin-sitemap', {
      cacheTime: 600 * 1000, // 600 sec - cache purge period
      changefreq: 'weekly',
      priority: 0.5,
    }],
    [path.resolve(__dirname, './plugins/google-tagmanager'), {}],
    [path.resolve(__dirname, './plugins/dart-sass'), {
      sassOptions: {
        fiber: require('fibers'),
        includePaths: [path.join(__dirname, 'node_modules')],
        additionalData: '$env: ' + process.env.NODE_ENV + ';',
        webpackImporter: false,
      }
    }],
  ],
};
