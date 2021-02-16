const includedSources = require('../plugins/included_source.js');


const version = process.argv[2];
if (!version) {;
    throw new Error("Missing version argument.");
}

console.info(`Updating sources in ${version} documentation`);
includedSources.updateVersionedSources({
  docsDir: `./versioned_docs/version-${version}`,
});
