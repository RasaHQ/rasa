import React, { useEffect, useRef } from 'react';
import Head from '@docusaurus/Head';
import ScriptLoader from 'react-script-loader-hoc';

export const ChatLoader = ScriptLoader(
  'https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.js'
)(({ scriptsLoadedSuccessfully: loaded, children }) => {
  return loaded ? children : null;
});

export const ChatBot = () => {
  const chatRef = useRef(null);
  useEffect(() => {
    const poll = window.setInterval(() => {
      if (window.Chatroom) {
        window.clearInterval(poll);
        var chatroom = new window.Chatroom({
          host: 'http://localhost:3000',
          title: 'Chat with Mike',
          container: chatRef.current,
          welcomeMessage: 'Hi, I am Mike. How may I help you?',
          speechRecognition: 'en-US',
          voiceLang: 'en-US',
        });
        chatroom.openChat();
      }
    }, 250);
  }, [chatRef]);
  return (
    <>
      <Head>
        <link
          rel="stylesheet"
          href="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.css"
        />
      </Head>
      <div
        ref={chatRef}
        style={{
          position: 'fixed',
          bottom: 40,
          right: 40,
          width: 500,
        }}
      />
    </>
  );
};
