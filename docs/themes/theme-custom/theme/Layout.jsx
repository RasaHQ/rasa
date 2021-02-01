import React from 'react';
import Layout from '@rasahq/docusaurus-theme-tabula/theme/Layout';

import { ChatBot, ChatLoader } from './ChatRoom';

function OverrideLayout(props) {
  return (
    <>
      <Layout {...props} />
      <ChatLoader>
        <ChatBot />
      </ChatLoader>
    </>
  );
}
export default OverrideLayout;
