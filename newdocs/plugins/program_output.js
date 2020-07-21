/**
    This plugin gives us the ability to insert the output of
    a program in a code-block in the docs, by using the following
    syntax:

        ```text [rasa --help]
        ```
    It is inspired by `remark-source` and also relies on a pre-build phase,
    before docusaurus is started (or built). It allows us to support separate
    versions of the docs (and of the program outputs).
*/
const fs = require('fs');
const globby = require('globby');
const visitChildren = require('unist-util-visit-children');
const { promisify } = require('util');

const exec = promisify(require('child_process').exec);
const { readFile, writeFile } = fs.promises;


const PROGRAM_OUTPUT_RE = /```[a-z]+ \[([^\]]+)\]\n```/;

const defaultOptions = {
    docsDir: './docs',
    sourceDir: './docs/sources',
    include: ['**.mdx', '**.md'],
    commandPrefix: '',
};

/**
    This function is use to get output of programs
    requested in the docs. It parses all the docs files,
    generates outputs and save them as files.

    Options:
    - docsDir:        the directory containing the docs files
    - sourceDir:      the directory that will contain the program outputs
    - include:        list of patterns to look for doc files
    - commandPrefix:  a prefix to be prepended before each command
*/
async function getProgramOutputs(options) {

    options = { ...defaultOptions, ...options };
    const { docsDir, include, sourceDir, commandPrefix } = options;
    // first, gather all the docs files
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    // second, read every file source
    let commands = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        const commands = [];
        let group, command, stdout;
        // third, find out if there is a program output to be generated
        // there can be multiple outputs in the same file
        const re = new RegExp(PROGRAM_OUTPUT_RE, 'gi');
        while ((group = re.exec(data)) !== null) {
            command = group[1];
            if (seen.has(command)) {
                continue;
            }
            seen.add(command);
            // fourth, call the command to generate the output
            output = await exec(`${commandPrefix} ${command}`);
            commands.push([command, output.stdout]);
        }
        return commands;
    }));
    commands = commands.flat().filter(pair => pair.length > 0);

    // finally, write all the command outputs as files in the `sourceDir`
    return await Promise.all(commands.map(async ([command, output]) => {
        return await writeFile(`${sourceDir}/${commandToFilename(command)}`, output);
    }));
};


/**
    Custom remark plugin to replace the following blocks:

    ```text [rasa --help]
    ```

    with the actual output of the program (here `rasa --help`).
    It relies on the output of `getProgramOutputs()` above,
    and is inspired by `remark-sources` plugin.
*/
function remarkProgramOutput(options = {}) {
    options = { ...defaultOptions, ...options };
    return (root) => {
        visitChildren((node, index, parent) => {
            if (node && node.type === 'code') {
                const content = readCommandOutput(node.meta, options);
                if (content !== undefined) {
                    node.value = content;
                }
            }
        })(root);
    };
}


function readCommandOutput(meta, { sourceDir }) {
    if (!meta) {
        return undefined;
    }
    if (meta[0] !== '[' || meta[meta.length - 1] !== ']') {
        return undefined;
    }
    meta = meta.slice(1, -1);
    try {
        return fs.readFileSync(`${sourceDir}/${commandToFilename(meta)}`, { encoding: 'utf8' });
    } catch (e) {
        throw new Error(`Failed to read file: ${meta}`);
    }
}


function commandToFilename(command) {
    return command.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.txt';
}


module.exports = getProgramOutputs;
module.exports.remarkProgramOutput = remarkProgramOutput;
