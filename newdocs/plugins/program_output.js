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

async function getProgramOutputs(options) {

    options = { ...defaultOptions, ...options };
    const { docsDir, include, sourceDir, commandPrefix } = options;
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    let commands = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        const commands = [];
        let group, command, stdout;
        const re = new RegExp(PROGRAM_OUTPUT_RE, 'gi');
        while ((group = re.exec(data)) !== null) {
            command = group[1];
            if (seen.has(command)) {
                continue;
            }
            seen.add(command);
            output = await exec(`${commandPrefix} ${command}`);
            commands.push([command, output.stdout]);
        }
        return commands;
    }));
    commands = commands.flat().filter(pair => pair.length > 0);

    return await Promise.all(commands.map(async ([command, output]) => {
        return await writeFile(`${sourceDir}/${commandToFilename(command)}`, output);
    }));
};


function remarkProgramOutput(options = {}) {
    options = { ...defaultOptions, ...options };
    return (root) => {
        visitChildren((node, index, parent) => {
            // console.info("node.type", node.type);
            if (node && node.type === 'code') {
                const content = readCommandOutput(node.meta, options);
                console.info('CONTENT', content);
                if (content !== undefined) {
                    console.info('assigning value', content);
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
    console.info("META", meta, sourceDir, commandToFilename(meta));
    try {
        return fs.readFileSync(`${sourceDir}/${commandToFilename(meta)}`, { encoding: 'utf8' });
    } catch (e) {
        throw new Error(`Failed to read file: ${meta}`);
    }
}


function commandToFilename(command) {
    return command.replace(/[^a-z0-9\-]/gi, '_').toLowerCase() + '.txt';
}


module.exports = getProgramOutputs;
module.exports.remarkProgramOutput = remarkProgramOutput;
