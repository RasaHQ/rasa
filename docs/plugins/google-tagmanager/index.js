const path = require('path');

module.exports = function (context, options) {
  const { siteConfig } = context;
  const { themeConfig } = siteConfig;
  const { gtm } = themeConfig || {};

  if (!gtm) {
    throw new Error(
      `You need to specify 'gtm' object in 'themeConfig' with 'containerID' field in it to use google-tagmanager`,
    );
  }

  const { containerID } = gtm;

  if (!containerID) {
    throw new Error(
      'You specified the `gtm` object in `themeConfig` but the `containerID` field was missing. ' +
        'Please ensure this is not a mistake.',
    );
  }

  return {
    name: 'google-tagmanager',

    getClientModules() {
      return [path.resolve(__dirname, './client')];
    },

    injectHtmlTags() {
      return {
        headTags: [
          {
            tagName: 'link',
            attributes: {
              rel: 'preconnect',
              href: 'https://www.google-analytics.com',
            },
          },
          {
            tagName: 'link',
            attributes: {
              rel: 'preconnect',
              href: 'https://www.googletagmanager.com',
            },
          },
          {
            tagName: 'link',
            attributes: {
              rel: 'stylesheet',
              href: 'https://assets.rasa.com/styles/klaro.css',
            },
          },
          {
            tagName: 'script',
            attributes: {
              'type': 'opt-in',
              'data-type': 'text/javascript',
              'data-name': 'analytics',
            },
            innerHTML: `
window.dataLayer = window.dataLayer || [{
  deployContext: (window.netlifyMeta && window.netlifyMeta.CONTEXT) || 'development',
  branchName: window.netlifyMeta && window.netlifyMeta.BRANCH,
}];
(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','${containerID}');
            `,
          },
          {
            tagName: 'script',
            attributes: {
              defer: true,
              src: 'https://assets.rasa.com/scripts/klaro_config.js'
            }
          },
          {
            tagName: 'script',
            attributes: {
              defer: true,
              src: 'https://assets.rasa.com/scripts/klaro.js'
            }
          },
        ],
        preBodyTags: [],
        postBodyTags: [],
      };
    },
  };
};
