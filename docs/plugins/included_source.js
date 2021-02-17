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
const path = require('path');
const fs = require('fs-extra');
const globby = require('globby');

const { readFile, outputFile } = fs;

const defaultOptions = {
    docsDir: './docs',
    relativeSourceDir: 'sources',
    include: ['**.mdx', '**.md'],
    pathPrefix: '../',
};

const _getIncludedSourceRe = (sourceDir) => `\`\`\`([a-z\-]+) \\(${sourceDir}/([^\\]\\s]+)\\)\n\`\`\``;

/**
    This function is used copy the included sources
    requested in the docs. It parses all the docs files,
    finds the included sources and copy them under the `sourceDir`.

    Options:
    - docsDir:             the directory containing the docs files
    - relativeSourceDir:   the directory that will contain the included sources
    - include:             list of patterns to look for doc files
    - pathPrefix:          a path prefix to use for reading the sources
*/
async function getIncludedSources(options) {

    options = { ...defaultOptions, ...options };
    const { docsDir, include, relativeSourceDir, pathPrefix } = options;
    const cleanedSourceDir = path.join(docsDir.replace('./', ''), relativeSourceDir);
    const includedSourceRe = _getIncludedSourceRe(cleanedSourceDir);

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
            sourceFile = group[2];
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

    // finally, write all the source files in the source directory
    return await Promise.all(sourceFiles.map(async ([sourceFile, content]) => {
        return await outputFile(`${docsDir}/${relativeSourceDir}/${sourceFile}`, content);
    }));
};


/**
    This function is used to ensure that all the documentation files
    which rely on the `remark-source` plugin to load source files into
    the docs continue to work across documentation versions. There is no way for
    `remark-source` to know about the file structure that we have, that's
    why we need this extra function. The workflow is:
    1. Doc files are put in a version folder upon release of a new version;
    2. The files, previously on the `main` branch, used to point to source files
       contained in `docs/source/...`;
    3. Now that they are in a versioned folder, we need to them to point to source files
       contained in `versioned_docs/version-xxx/sources/...`. This is what this function
       does.

    Options:
    - docsDir:                the directory containing the versionned docs files
    - relativeSourceDir:      the directory that will contain the included sources
    - include:                list of patterns to look for doc files
*/
async function updateVersionedSources(options) {
    options = { ...defaultOptions, ...options };
    const { docsDir, include, relativeSourceDir } = options;
    const originalSourceDir = path.join(defaultOptions.docsDir.replace('./', ''), relativeSourceDir);
    const newSourceDir = path.join(docsDir.replace('./', ''), relativeSourceDir);
    const includedSourceRe = _getIncludedSourceRe(originalSourceDir);

    // first, gather all the docs files
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    // second, read every doc file and compute their updated content
    let newDocsFiles = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        // third, find out if there is a source to be included
        // there can be multiple sources in the same file
        const re = new RegExp(includedSourceRe, 'gi');
        const updatedData = data.toString().replace(re, `\`\`\`$1 (${newSourceDir}/$2)\n\`\`\``);
        return (updatedData != data) ? [`${docsDir}/${source}`, updatedData] : [];
    }));

    newDocsFiles = newDocsFiles.filter(pair => pair.length > 0);

    // finally, update all the docs files with the path to the versioned source
    return await Promise.all(newDocsFiles.map(async ([docsFile, updatedContent]) => {
        return await outputFile(docsFile, updatedContent);
    }));
}


module.exports = getIncludedSources;
module.exports.updateVersionedSources = updateVersionedSources;
