import * as React from 'react';
import {LiveProvider, LiveEditor, LiveError, LivePreview} from 'react-live';
import clsx from 'clsx';
import ThemeContext from '@theme/_contexts/ThemeContext';

import styles from './styles.module.css';


function Playground({children, theme, transformCode, noResult, name, ...props}) {
  const code = children.replace(/\n$/, '');
  const themeContext = React.useContext(ThemeContext);

  // only run this when mounting
  React.useEffect(() => {
    themeContext.onLiveCodeStart(name, code);
  }, []);

  return (
    <LiveProvider
      code={code}
      transformCode={transformCode || ((code) => `${code};`)}
      theme={theme}
      {...props}>
      <div
        className={clsx(
          styles.playgroundHeader,
          styles.playgroundEditorHeader,
        )}>
        Live Editor
      </div>
      <LiveEditor className={styles.playgroundEditor} onChange={value => themeContext.onLiveCodeChange(name, value)} />
      {noResult ? undefined : <div
        className={clsx(
          styles.playgroundHeader,
          styles.playgroundPreviewHeader,
        )}>
        Result
      </div>}
      {noResult ? undefined : <div className={styles.playgroundPreview}>
        <LivePreview />
        <LiveError />
      </div>}
    </LiveProvider>
  );
}

export default Playground;
