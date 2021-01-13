// @flow
import type { ChatMessage } from "./Chatroom";

import "unfetch/polyfill";
import "@babel/polyfill";
import React from "react";
import ReactDOM from "react-dom";

import Chatroom from "./Chatroom";
import { noop, sleep, uuidv4 } from "./utils";
import ConnectedChatroom from "./ConnectedChatroom";
import DebuggerView from "./DebuggerView";

const USERID_STORAGE_KEY = "simple-chatroom-cid";

type ChatroomOptions = {
  host: string,
  title?: string,
  welcomeMessage?: string,
  speechRecognition?: string,
  startMessage?: string,
  container: HTMLElement,
  waitingTimeout?: number,
  fetchOptions?: RequestOptions,
  rasaToken?: string,
  voiceLang?: string
};

window.Chatroom = function(options: ChatroomOptions) {
  let sessionUserId = window.sessionStorage.getItem(USERID_STORAGE_KEY);

  const isNewSession = sessionUserId == null;

  if (isNewSession) {
    sessionUserId = uuidv4();
    window.sessionStorage.setItem(USERID_STORAGE_KEY, sessionUserId);
  }

  this.ref = ReactDOM.render(
    <ConnectedChatroom
      userId={sessionUserId}
      host={options.host}
      title={options.title || "Chat"}
      speechRecognition={options.speechRecognition}
      welcomeMessage={options.welcomeMessage}
      waitingTimeout={options.waitingTimeout}
      fetchOptions={options.fetchOptions}
      voiceLang={options.voiceLang}
    />,
    options.container
  );

  this.openChat = () => {
    this.ref.setState({ isOpen: true });
  };

  if (isNewSession && options.startMessage != null) {
    this.ref.sendMessage(options.startMessage);
  }
};

type DemoChatroomOptions = {
  title: string,
  container: HTMLElement
};

window.DemoChatroom = function(options: DemoChatroomOptions) {
  this.demoIsPlaying = false;

  this.render = (
    messages: Array<ChatMessage>,
    showWaitingBubble: boolean = false
  ) => {
    this.ref = ReactDOM.render(
      <Chatroom
        messages={messages}
        waitingForBotResponse={showWaitingBubble}
        speechRecognition={null}
        voiceLang={null}
        isOpen={true}
        title={options.title || "Chat"}
        onButtonClick={noop}
        onToggleChat={noop}
        onSendMessage={noop}
      />,
      options.container
    );
  };

  const sleepEffect = (time: number) => ({ type: "SLEEP", time });

  // Works like redux-saga
  function* demoSaga(
    _messages: Array<ChatMessage>,
    delay: number = 1000,
    keyDelay: number = 100
  ) {
    if (this.demoIsPlaying) return;
    this.demoIsPlaying = true;

    if (_messages.length === 0) return;

    const messages = _messages.map((m, i) => ({
      message: m.message,
      username: m.username || "user",
      time: Date.now() + delay * i,
      uuid: uuidv4()
    }));

    for (let i = -1; i < messages.length; i++) {
      if (i < 0) {
        this.render([], messages[0].username === "bot");
      } else {
        const currentMessage = messages[i];
        const currentMessageContent = currentMessage.message;

        // Show waiting when next message is a bot message
        const showWaitingBubble =
          i + 1 < messages.length && messages[i + 1].username === "bot";

        // Show typing animation if current message is a user message
        if (
          currentMessage.username !== "bot" &&
          currentMessageContent.type === "text"
        ) {
          const messageText = currentMessageContent.text;
          this.ref.getInputRef().focus();
          for (let j = 0; j < messageText.length && this.demoIsPlaying; j++) {
            const currentMessageText = messageText.substring(0, j + 1);
            this.ref.getInputRef().value = currentMessageText;
            this.ref.getInputRef().scrollLeft = 100000;
            yield sleepEffect(keyDelay);
          }
          yield sleepEffect(delay);
          this.ref.getInputRef().value = "";
          this.ref.getInputRef().blur();
        }
        if (
          currentMessageContent.type === "button" &&
          currentMessageContent.buttons.some(b => b.selected)
        ) {
          this.render(
            messages.slice(0, i).concat({
              username: currentMessage.username,
              message: {
                type: "button",
                buttons: currentMessageContent.buttons.map(b => ({
                  title: b.title,
                  payload: b.payload
                }))
              }
            })
          );
          yield sleepEffect(delay);
        }
        this.render(messages.slice(0, i + 1), showWaitingBubble);
      }
      yield sleepEffect(delay);
    }

    this.demoIsPlaying = false;
  }

  this.demo = async (
    messages: Array<ChatMessage>,
    delay: number = 1000,
    keyDelay: number = 100
  ) => {
    const saga = demoSaga.call(this, messages, delay, keyDelay);
    let currentEffect = saga.next();
    while (!currentEffect.done && this.demoIsPlaying) {
      if (currentEffect.value.type === "SLEEP") {
        await sleep(currentEffect.value.time);
      }
      currentEffect = saga.next();
    }

    // Cleanup
    if (!currentEffect.done) {
      this.render([]);
      this.ref.getInputRef().value = "";
      this.ref.getInputRef().blur();
    }
  };

  this.clear = () => {
    this.demoIsPlaying = false;
    this.render([]);
  };

  this.render([]);
};

window.DebugChatroom = function(options: ChatroomOptions) {
  let sessionUserId = window.sessionStorage.getItem(USERID_STORAGE_KEY);

  const isNewSession = sessionUserId == null;

  if (isNewSession) {
    sessionUserId = uuidv4();
    window.sessionStorage.setItem(USERID_STORAGE_KEY, sessionUserId);
  }

  this.ref = ReactDOM.render(
    <DebuggerView
      rasaToken={options.rasaToken}
      userId={sessionUserId}
      host={options.host}
      title={options.title || "Chat"}
      speechRecognition={options.speechRecognition}
      welcomeMessage={options.welcomeMessage}
      waitingTimeout={options.waitingTimeout}
      fetchOptions={options.fetchOptions}
      voiceLang={options.voiceLang}
    />,
    options.container
  );

  const { startMessage } = options;
  if (isNewSession && startMessage != null) {
    this.ref.getChatroom().sendMessage(startMessage);
  }
};
