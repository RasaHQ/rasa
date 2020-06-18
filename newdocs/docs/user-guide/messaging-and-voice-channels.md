# Messaging and Voice Channels

To make your assistant available on a messaging platform you need to provide credentials
in a `credentials.yml` file.
An example file is created when you run `rasa init`, so it’s easiest to edit that file
and add your credentials there. Here is an example with Facebook credentials:

```
facebook:
  verify: "rasa-bot"
  secret: "3e34709d01ea89032asdebfe5a74518"
  page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"
```

Learn how to make your assistant available on:


* Your Own Website


* Facebook Messenger


* Slack


* Telegram


* Twilio


* Microsoft Bot Framework


* Cisco Webex Teams


* RocketChat


* Mattermost


* Google Hangouts Chat


* Custom Connectors


## Testing Channels on Your Local Machine with Ngrok

You can use [ngrok](https://ngrok.com/) to create a connection to your local
computer that is publicly available on the internet.
You don’t need this when running Rasa on a server because, you can set up a domain
name to point to that server’s IP address, or use the IP address itself.

After installing ngrok, run:

Your webhook address will look like the following:


* `https://yyyyyy.ngrok.io/webhooks/<CHANNEL>/webhook`, e.g.


* `https://yyyyyy.ngrok.io/webhooks/facebook/webhook`

**WARNING**: With the free-tier of ngrok, you can run into limits on how many connections you can make per minute.
As of writing this, it is set to 40 connections / minute.
