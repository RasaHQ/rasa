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

  const isProd = process.env.NODE_ENV === 'production';

  return {
    name: 'google-tagmanager',

    getClientModules() {
      return isProd ? [path.resolve(__dirname, './client')] : [];
    },

    injectHtmlTags() {
      if (!isProd) {
        return {};
      }
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
            tagName: 'script',
            attributes: {
              async: true,
              src: `https://www.googletagmanager.com/gtm.js?id=${containerID}`,
            },
          },
          {
            tagName: 'script',
            innerHTML: `(function(w,d,l){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'}); })(window,document,'dataLayer');`,
          },
        ],
        preBodyTags: [
          {
            tagName: 'noscript',
            innerHTML: `<iframe src="https://www.googletagmanager.com/ns.html?id=${containerID}" height="0" width="0" style="display:none;visibility:hidden"></iframe>`,
          },
        ],
        postBodyTags: [],
      };
    },
  };
};
