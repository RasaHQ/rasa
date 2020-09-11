import React from 'react';
// import { RedocStandalone } from 'redoc';
import BrowserOnly from '@docusaurus/BrowserOnly';
// import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';

const Redoc = (props) => {
  // const specUrl = useBaseUrl('/spec/action-server.yml');

  return (
    <Layout>
      <BrowserOnly fallback={<div>Loading...</div>}>
        {/* {() => <RedocStandalone specUrl={specUrl} {...props} />} */}
      </BrowserOnly>
    </Layout>
  );
};

export default Redoc;
