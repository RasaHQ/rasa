import { DeepChat } from 'deep-chat-react'
import { MessageContent } from 'deep-chat/dist/types/messages'
import { Flex, FlexProps } from '@chakra-ui/react'
import { Command, Event } from '../types'

interface Props extends FlexProps {
  events: Event[]
}

export const Chat = ({ sx, events, ...props }: Props) => {
  const containerSx = {
    ...sx,
    p: 0,
    flexDirection: 'column',
  }

  const maxHeight = document.documentElement.scrollHeight - 64
  // 21 and 25 are the rem number we're using for the columns. We add 0.75rem for the padding
  // A potential improvement would be to add a onresize event for both width and height
  let remReference = 21.75
  if (document.documentElement.clientWidth > 1631) {
    remReference = 25.75
  }

  const columnWidth =
    remReference *
    parseFloat(getComputedStyle(document.documentElement).fontSize)

  // function that maps a command dict to a human string to display in the chat
  const mapCommand = (command: Command) => {
    let commandName = command.command
    if (commandName === 'start flow') {
      return `Start (${command.flow})`
    }
    if (commandName === 'set slot') {
      return `Set (${command.name} = ${command.value})`
    }
    return commandName
  }

  // collect user and bot messages
  const messages: MessageContent[] = events
    .filter(
      (event: Event) =>
        event.event === 'user' ||
        event.event === 'bot' ||
        event.event === 'session_ended',
    )
    // @ts-expect-error
    .flatMap((event: Event) => {
      if (event.event === 'user') {
        let commands =
          event.parse_data?.commands?.map(
            (command) => `<div>${mapCommand(command)}</div>`,
          ) || []
        return [
          {
            role: event.event,
            text: event.text || '',
          },
          {
            role: 'system',
            html: `<div>${commands.join('')}</div>`,
          },
        ]
      } else if (event.event === 'session_ended') {
        return [
          {
            role: 'system',
            html: `<div>
              session ended
              <div style="margin-top: 8px;">
                <button 
                  onclick="(() => {
                     window.restartConversation();
                     return false;
                   })()"
                  style="background-color: transparent; border: 1px solid #ccc; border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 14px;"
                >
                  Start a new conversation
                </button>
              </div>
            </div>`,
          },
        ]
      } else {
        return [
          {
            role: event.event,
            text: event.text || '',
          },
        ]
      }
    })

  return (
    <Flex sx={containerSx} {...props}>
      <DeepChat
        avatars={true}
        textInput={{
          disabled: true,
        }}
        inputAreaStyle={{ display: 'none' }}
        messageStyles={{
          html: {
            shared: { bubble: { backgroundColor: 'unset', padding: '0px' } },
          },
        }}
        style={{
          borderRadius: '10px',
          border: 'none',
          width: columnWidth,
          height: maxHeight,
        }}
        history={messages}
        demo={true}
      />
    </Flex>
  )
}
