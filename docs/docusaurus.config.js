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

module.exports = {
  title: 'Rasa Open Source Documentation',
  tagline: 'The tagline of my site',
  url: 'https://your-docusaurus-test-site.com',
  // FIXME: when deploying this for real, change to '/docs/rasa/'
  baseUrl: '/docs/rasa/next/',
  favicon: 'img/favicon.ico',
  organizationName: 'RasaHQ',
  projectName: 'rasa',
  themeConfig: {
    navbar: {
      title: 'Rasa Open Source',
      logo: {
        alt: 'Rasa',
        src: 'https://rasa.com/static/60e441f8eadef13bea0cc790c8cf188b/rasa-logo.svg',
      },
      items: [
        {
          label: 'Docs',
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
          ] : [{
              label: 'Master/Unreleased',
              to: '/',
              activeBaseRegex: `/`,
            },],
        },
        {
          href: 'https://github.com/rasahq/rasa',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Rasa Technologies GmbH`,
    },
    // algolia: {
    //   apiKey: 'YOUR_API_KEY',
    //   indexName: 'YOUR_INDEX_NAME',
    //   searchParameters: {}, // Optional (if provided by Algolia)
    // },
  },
  themes: [
    ['@docusaurus/theme-classic', {
      customCss: require.resolve('./src/css/custom.css'),
    }],
    // FIXME: apply for and setup algolia docSearchv3 https://docsearch.algolia.com/apply/
    // ['@docusaurus/theme-search-algolia', {}],
    path.resolve(__dirname, './themes/theme-live-codeblock'),
  ],
  plugins: [
    ['@docusaurus/plugin-content-docs', {
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
    // ['@docusaurus/plugin-google-analytics', {}],
    // ['@docusaurus/plugin-google-gtag', {}],
    ['@docusaurus/plugin-sitemap', {
      cacheTime: 600 * 1000, // 600 sec - cache purge period
      changefreq: 'weekly',
      priority: 0.5,
    }],
  ],
};

/*










 */
