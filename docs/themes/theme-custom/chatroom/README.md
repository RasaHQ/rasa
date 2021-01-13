# React-based Chatroom Component for Rasa Stack

[![CircleCI](https://circleci.com/gh/scalableminds/chatroom.svg?style=svg)](https://circleci.com/gh/scalableminds/chatroom)

<a href="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/demo.html"><img src="https://npm-scalableminds.s3.amazonaws.com/%40scalableminds/chatroom/demo.gif" alt="Demo" width="409" height="645" /></a>

[Watch a demo of our Chatroom in action](https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/demo.html)

## Features

* React-based component
* Supports Text with Markdown formatting, Images, and Buttons
* Customizable with SASS variables
* Generates a unique session id and keeps it in `sessionStorage`
* Queues consecutive bot messages for better readability
* Speech input (only in Chrome for now)
* Text to Speech (only in Chrome for now)
* Demo mode included (ideal for scripted screencasts)
* Hosted on S3 for easy use
* Simple setup. Works with Rasa's [REST channel](https://rasa.com/docs/rasa/user-guide/connectors/your-own-website/#rest-channels)

## Usage
1. Embed the `chatroom.js` in the HTML of your website and configure it to connect to your Rasa bot. Either use the S3 hosted version or build it yourself. (see below)

```html
<head>
  <link rel="stylesheet" href="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.css" />
</head>
<body>
  <div class="chat-container"></div>

  <script src="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.js"/></script>
  <script type="text/javascript">
    var chatroom = new window.Chatroom({
      host: "http://localhost:5005",
      title: "Chat with Mike",
      container: document.querySelector(".chat-container"),
      welcomeMessage: "Hi, I am Mike. How may I help you?",
      speechRecognition: "en-US",
      voiceLang: "en-US"
    });
    chatroom.openChat();
  </script>
</body>
```


2. In your Rasa bot setup, make sure to include the Rasa [REST channel](https://rasa.com/docs/rasa/user-guide/connectors/your-own-website/#rest-channels) in your `credentials.yml` file:
```
rest:
  # pass
```

Depending on your setup you might need to start the Rasa CLI / Rasa server with the right CORS headers, e.g. `--cors "*"`.

Note, the version of the Chatroom's Javascript file is encoded in the URL. `chatroom@master` is always the latest version from the GitHub master branch. Use e.g. `chatroom@0.10.0` to load a specific release. [All Releases can be found here.](https://github.com/scalableminds/chatroom/releases)


| Chatroom Version  | Compatible Rasa Core Version |
|-------------------|------------------------------|
| 0.10.x            | 1.0                          |
| 0.9.x (Deprecated)| 0.11.4+, 0.13.7              |
| 0.8.x (Deprecated)| 0.11.4+                      |
| 0.7.8 (Deprecated)| 0.10.4+                      |

Note, versions prior to `0.10.x` used a custom Python channel to connect the chatroom frontend with a Rasa bot backend. Upgrading, from version `0.9.x` or below will require you to modify the `credentials.yml` and include the Rasa REST channel. (see installation instructions above)


## Development

### Install dependencies

```
yarn install
```

### Continuously build the Chatroom component

```
yarn watch
yarn serve
```

Open `http://localhost:8080/demo.html` in your browser.

## Build

```
yarn build
```

Distributable files will be created in folder `dist`.

## License

AGPL v3

Made by [scalable minds](https://scalableminds.com)
