const path = require('path');

const { validateThemeConfig } = require('./validateThemeConfig');

module.exports = function (context, options) {
  const { customCss } = options || {};

  return {
    name: 'theme-tabula',

    getThemePath() {
      return path.resolve(__dirname, './theme');
    },

    getClientModules() {
      const modules = [path.resolve(__dirname, './prism-include-languages')];

      if (customCss) {
        modules.push(customCss);
      }

      return modules;
    },
  };
};

module.exports.validateThemeConfig = validateThemeConfig;
