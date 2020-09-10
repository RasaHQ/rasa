import React from 'react';
import usePromise from 'react-promise';
import BrowserOnly from '@docusaurus/BrowserOnly';

const Redoc = (props) => {
  const getRedocStandalone = React.useCallback(async () => {
    // using import() instead of require() keeps the package tree-shakeable.
    const redoc = await import('redoc');
    return redoc.RedocStandalone;
  }, []);

  return (
    <BrowserOnly fallback={<div>Loading...</div>}>
      {() => {
        // we need to import this here instead of at the top-level
        // because it causes issues in production builds
        const { value: RedocStandalone, loading } = usePromise(getRedocStandalone);
        return loading ? <div>Loading...</div> : <RedocStandalone {...props} />;
      }}
    </BrowserOnly>
  );
};

export default Redoc;
