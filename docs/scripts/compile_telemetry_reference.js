const fs = require('fs');
const config = require("../package.json").telemetryReference;
const eventsSchema = require("../docs/telemetry/events.json");

const { writeFile } = fs.promises;

const isFrontendEvent = (event) => event.indexOf(" ") === -1 && event.indexOf(":") !== -1;

const MDX_HEADER = `---
id: reference
sidebar_label: Telemetry Event Reference
title: Telemetry Event Reference
abstract: |
  Event descriptions of telemetry data we report in order to improve our products.
---
Telemetry events are only reported if telemetry is enabled. A detailed explanation 
on the reasoning behind collecting optional telemetry events can be found in our 
[telemetry documentation](./telemetry.mdx).
`;

const renderEventBadge = (event) => {
  const kind = isFrontendEvent(event) ? 'frontend' : 'backend';
  return `<span class="badge badge--primary">${kind}</span>`;
};

const renderTelemetryEventDetails = (type, properties) => {
  const renderEnum = (_enum) => _enum.length ? ` Can be one of: ${_enum.map(e => `\`"${e}"\``).join(', ')}.` : '';

  switch (type) {
    case 'object':
      return `Event properties:\n
${Object.entries(properties)
  .map(([property, { type, description, enum: _enum = [] }]) => `- \`${property}\` *(${type})*: ${description}${renderEnum(_enum)}`)
  .join('\n')
}
`;
    case 'null':
      return '';
    case undefined:
      return '';
    default:
      throw new Error(`Unexpected event type: "${type}".`);
  }
};

const renderTelemetryEventDescription = (event, schema) => {
  if (schema.description) {
    return schema.description;
  }
  throw new Error(`Missing description for event ${event}`);
};

const renderTelemetryEvent = (event, schema) => (`
### ${event}
${renderEventBadge(event)} ${renderTelemetryEventDescription(event, schema)}
${renderTelemetryEventDetails(schema.type, schema.properties)}
`);

const renderTelemetrySection = (section, events) => (`
## ${section}
${Object.entries(events).map(([event, schema]) => renderTelemetryEvent(event, schema)).join('\n')}
`);

/**
    This function is used to write the reference of our telemetry events
    inside the docs. It's run as part of the pre-build step.
    Options:
    - outputPath:     where to output the telemetry reference (mdx file)
*/
async function writeTelemetryReference({ outputPath }) {

  const sections = Object.fromEntries(eventsSchema.sections.map((section) => [section, {}]));
  Object.entries(eventsSchema.events).forEach(([event, schema]) => {
    const section = schema.section || eventsSchema.defaultSection;
    if (sections[section] === undefined) {
      throw new Error(`Unknown section ${section}, please add it to the "sections" array.`);
    }
    sections[section][event] = schema;
  });

  // const content = Object.entries(eventsSchema.events).map(([event, schema]) => renderTelemetryEvent(event, schema)).join('\n');
  const content = Object.entries(sections).map(([section, events]) => renderTelemetrySection(section, events)).join('\n');
  await writeFile(outputPath, `${MDX_HEADER}\n\n${content}`);
};


console.info('Compiling telemetry reference');
writeTelemetryReference(config);
