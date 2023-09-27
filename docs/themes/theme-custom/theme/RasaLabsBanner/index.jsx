import * as React from 'react';
import clsx from 'clsx';
import CodeBlock from '@theme/CodeBlock';

import styles from './styles.module.css';

function RasaLabsBanner({isLoading, ...props}) {
  return (
    <>
      <div className="mdx-box admonition admonition-tip alert alert--success">
        <div className="mdx-box admonition-heading">
          <h5>
            <span className="admonition-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="16" viewBox="0 0 12 16">
                <path fillRule="evenodd" d="M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z">
                </path>
              </svg>
            </span>Rasa Labs access {props.version && <>
                <span>- New in </span>
                <span className={clsx(styles.titleExtension)}>{props.version}</span>
              </>}
          </h5>
        </div>
      <div className="mdx-box admonition-content">
        <p>
          Rasa Labs features are <strong>experimental</strong>. We introduce experimental
          features to co-create with our customers. To find out more about how to participate
          in our Labs program visit our
          {' '}
          <a href="https://rasa.com/rasa-labs/" target="_blank" rel="noopener noreferrer">
            Rasa Labs page
          </a>.
          <br />
          <br />
          We are continuously improving Rasa Labs features based on customer feedback. To benefit from the latest
          bug fixes and feature improvements, please install the latest pre-release using:

          <CodeBlock language="bash">
            pip install 'rasa-plus&gt;3.6' --pre --upgrade
          </CodeBlock>
        </p>
      </div>
    </div>
  </>
  )
}

export default RasaLabsBanner;
