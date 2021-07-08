import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import usePrismTheme from '@theme/hooks/usePrismTheme';
import Playground from '@theme/Playground';
import ReactLiveScope from '@theme/ReactLiveScope';
import CodeBlock from '@theme-init/CodeBlock';

const withLiveEditor = (Component) => {
  return (props) => {
    const {isClient} = useDocusaurusContext();
    const prismTheme = usePrismTheme();

    if (props.live) {
      return (
        <Playground
          key={isClient}
          scope={ReactLiveScope}
          theme={prismTheme}
          {...props}
        />
      );
    }

    return <Component {...props} />;
  };
};

export default withLiveEditor(CodeBlock);
