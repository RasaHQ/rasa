const path = require("path");

// FIXME: this package is copied from
// https://github.com/facebook/docusaurus/tree/afe9ff91a4247316f0081c9b080655d575298416/packages/docusaurus-theme-live-codeblock/src
module.exports = function (context) {
  const {
    siteConfig: { url: siteUrl, baseUrl },
  } = context;

  return {
    name: "theme-custom",

    getThemePath() {
      return path.resolve(__dirname, "./theme");
    },

    // FIXME: this needs to be fixed in the theme, see https://github.com/RasaHQ/docusaurus-tabula/issues/11
    getClientModules() {
      return [require.resolve("./custom.css")];
    },

    injectHtmlTags() {
      return {
        headTags: [
          {
            tagName: "meta",
            attributes: {
              property: "og:image",
              content: `${siteUrl}${baseUrl}img/og-image.png`,
            },
          },
        ],
      };
    },
  };
};
