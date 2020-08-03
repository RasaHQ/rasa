import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';



const Redoc = (props) => (
  <BrowserOnly fallback={<div>Pre-rendering...</div>}>
    {() => {
        // we need to import this here instead of at the top-level
        // because it causes issues in production builds
        const RedocStandalone = require('redoc').RedocStandalone;
        return <RedocStandalone {...props} />;
    }}
  </BrowserOnly>
);

export default Redoc;
