import { Box, Heading, Text } from "@chakra-ui/react";

interface CommonEventInfoProps {
  event: {
    timestamp: string;
    id: string;
    conversationEventType: string;
  };
}

export const CommonEventInfo = ({ event }: CommonEventInfoProps) => {
  const formatted = new Date(event.timestamp).toLocaleString();

  return (
    <>
      <Box data-testid="event-occurred-at">
        <Heading textStyle="sm" mb="1">
          Occurred at
        </Heading>
        <Text textStyle="sm" mb="2">
          {formatted}
        </Text>
      </Box>

      <Box>
        <Heading textStyle="sm" mb="1">
          ID
        </Heading>
        <Text textStyle="sm" mb="2" wordBreak="break-all">
          {event.id}
        </Text>
      </Box>

      <Box data-testid="event-type">
        <Heading textStyle="sm" mb="1">
          Type
        </Heading>
        <Text textStyle="sm" mb="2">
          {event.conversationEventType}
        </Text>
      </Box>
    </>
  );
};
