const copyMarkdownFiles = require('./plugins/copy_md_files.js');



console.info('Copying markdown files');
copyMarkdownFiles({
    docsDir: './docs',
    files: {
        '../CHANGELOG.mdx': 'changelog.mdx',
    }
});
