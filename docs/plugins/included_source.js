/**
    This plugin gives us the ability to include source files
    in a code-block in the docs, by using the following
    syntax:

        ```python (docs/sources/path/to/file.py)
        ```

    To make it work, you need to prefix the source file by `docs/sources/`.

    It relies on `remark-source` and on a pre-build phase,
    before docusaurus is started (or built). It allows us to support separate
    versions of the docs (and of the program outputs).
*/
const fs = require('fs-extra');
const globby = require('globby');

const { readFile, outputFile } = fs;

const defaultOptions = {
    docsDir: './docs',
    sourceDir: './docs/sources',
    include: ['**.mdx', '**.md'],
    pathPrefix: '../',
};

/**
    This function is used copy the included sources
    requested in the docs. It parses all the docs files,
    finds the included sources and copy them under the `sourceDir`.

    Options:
    - docsDir:        the directory containing the docs files
    - sourceDir:      the directory that will contain the included sources
    - include:        list of patterns to look for doc files
    - pathPrefix:     a path prefix to use for reading the sources
*/
async function getIncludedSources(options) {

    options = { ...defaultOptions, ...options };
    const { docsDir, include, sourceDir, pathPrefix } = options;
    const cleanedSourceDir = sourceDir.replace('./', '');
    const includedSourceRe =`\`\`\`[a-z\-]+ \\(${cleanedSourceDir}/([^\\]\\s]+)\\)\n\`\`\``;

    // first, gather all the docs files
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    // second, read every file source
    let sourceFiles = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        const sourceFiles = [];
        let group, sourceFile, content;
        // third, find out if there is a source to be included
        // there can be multiple sources in the same file
        const re = new RegExp(includedSourceRe, 'gi');
        while ((group = re.exec(data)) !== null) {
            sourceFile = group[1];
            if (seen.has(sourceFile)) {
                continue;
            }
            seen.add(sourceFile);
            // fourth, read the source file
            content = await readFile(`${pathPrefix}${sourceFile}`);
            sourceFiles.push([sourceFile, content]);
        }
        return sourceFiles;
    }));
    sourceFiles = sourceFiles.flat().filter(pair => pair.length > 0);

    // finally, write all the source files in the `sourceDir`
    return await Promise.all(sourceFiles.map(async ([sourceFile, content]) => {
        return await outputFile(`${sourceDir}/${sourceFile}`, content);
    }));
};


module.exports = getIncludedSources;
