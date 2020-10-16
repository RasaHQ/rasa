const path = require('path');


// FIXME: this package is copied from
// https://github.com/facebook/docusaurus/tree/afe9ff91a4247316f0081c9b080655d575298416/packages/docusaurus-theme-live-codeblock/src
module.exports = function() {
  return {
    name: 'theme-live-codeblock',

    getThemePath() {
      return path.resolve(__dirname, './theme');
    },

    // FIXME: this allows to disable searchbox shortcuts. It's a quickfix and shouldn't be located here,
    //        but it's temporary and should be removed when we enable Algolia search
    getClientModules() {
      const modules = [
        require.resolve('./styles.css')
      ];

      return modules;
    },

    configureWebpack() {
      return {
        resolve: {
          alias: {
            // fork of Buble which removes Buble's large dependency and weighs in at a smaller size of ~51kB
            // https://github.com/FormidableLabs/react-live#what-bundle-size-can-i-expect
            buble: '@philpl/buble',
          },
        },
      };
    },
  };
};
