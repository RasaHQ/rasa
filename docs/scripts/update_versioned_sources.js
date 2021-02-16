const fs = require("fs");
const includedSources = require('../plugins/included_source.js');


const version = process.argv[2];
if (!version) {;
    throw new Error("Missing version argument.");
}

const docsDir = `./versioned_docs/version-${version}`;
if (!fs.existsSync(docsDir)) {
    throw new Error(`Documentation for version ${version} doesn't exist.`);
}

console.info(`Updating sources in ${version} documentation`);
includedSources.updateVersionedSources({ docsDir });
