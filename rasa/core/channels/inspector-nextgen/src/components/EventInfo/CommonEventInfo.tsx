import { Box, Text } from "@chakra-ui/react";

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
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          Occurred at
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem">
          {formatted}
        </Text>
      </Box>

      <Box>
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          ID
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem" wordBreak="break-all">
          {event.id}
        </Text>
      </Box>

      <Box data-testid="event-type">
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          Type
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem">
          {event.conversationEventType}
        </Text>
      </Box>
    </>
  );
};
