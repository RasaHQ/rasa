const getIncludedSources = require('../plugins/included_source.js');

console.info('Computing included sources');
getIncludedSources({
  docsDir: './docs',
  include: ['**.mdx', '**.md'],
});
