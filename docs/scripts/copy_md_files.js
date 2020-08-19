const fs = require('fs');

const { copyFile } = fs.promises;

const defaultOptions = {
	files: {},
	docsDir: './docs',
};

/**
    This function is used to copy markdown files from a source
    outside the `docs/` folder to a destination inside the `docs/` folder.

    Options:
    - files:          a mapping of source: destination
    - docsDir:        the docs folder
*/
async function copyMarkdownFiles(options) {
	options = { ...defaultOptions, ...options };
	const { docsDir, files } = options;

	for (const [source, destination] of Object.entries(files)) {
		await copyFile(source, `${docsDir}/${destination}`);
	}
}

console.info('Copying markdown files');
copyMarkdownFiles({
	docsDir: './docs',
	files: {
		'../CHANGELOG.mdx': 'changelog.mdx',
	},
});
