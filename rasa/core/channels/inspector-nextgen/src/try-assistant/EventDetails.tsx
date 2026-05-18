import { Box } from "@chakra-ui/react";
import { EventInfo } from "../components/EventInfo";
import type { Flow, UnionEventType } from "../types";

interface EventDetailsProps {
  event: UnionEventType;
  onClose: () => void;
  flows: Flow[];
}

export const EventDetails = ({ event, onClose, flows }: EventDetailsProps) => {
  return (
    <Box height="100%" borderRadius="lg">
      <EventInfo event={event} onClose={onClose} flows={flows} />
    </Box>
  );
};
