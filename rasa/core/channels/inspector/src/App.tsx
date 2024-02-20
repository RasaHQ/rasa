import {
  Grid,
  GridItem,
  useBreakpointValue,
  useColorModeValue,
  useToast,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { useInterval } from "usehooks-ts";
import axios from "axios";
import { useOurTheme } from "./theme";
import { Welcome } from "./components/Welcome";
import { DialogueStack } from "./components/DialogueStack";
import { DialougeInformation } from "./components/DialogueInformation";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { DiagramFlow } from "./components/DiagramFlow";
import { formatSlots } from "./helpers/formatters";
import { Slot, Stack, Event, Flow, SelectedStack } from "./types";
import { updatedActiveFrame } from "./helpers/utils";

const storyController = new AbortController();
const trackerController = new AbortController();
const pollingInterval = 1000;

export function App() {
  const toast = useToast();
  const { rasaSpace, rasaRadii } = useOurTheme();
  const [rasaChatSessionId, setRasaChatSessionId] = useState<string>("");
  const [flows, setFlows] = useState<Flow[]>([]);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [story, setStory] = useState<string>("");
  const [stack, setStack] = useState<Stack[]>([]);
  const [frame, setFrame] = useState<SelectedStack | undefined>(undefined);

  useEffect(() => {
    axios
      .get("/flows")
      .then((response) => setFlows(response.data))
      .catch((error) => {
        // don't show a new toast if it's already active
        if (toast.isActive("flows")) return;
        toast({
          id: "flows",
          title: "Flows could not be retrieved",
          description: error?.message || "An unknown error happened.",
          status: "error",
          duration: 4000,
          isClosable: true,
        });
      });
  }, [toast]);

  // `rasaChatSessionId` is set by @rasahq/rasa-chat on the window object 😞
  // Since we can't control when that happens, we need to poll for it.
  useInterval(
    () => setRasaChatSessionId(window?.rasaChatSessionId || ""),
    // null means: stop polling. We stop polling once we retrieve `rasaChatSessionId`
    rasaChatSessionId ? null : 1000
  );

  useInterval(
    () => {
      if (!rasaChatSessionId) return;
      storyController.abort();
      axios
        .get(`/conversations/${rasaChatSessionId}/story`)
        .then((response) => setStory(response.data))
        .catch((error) => {
          // don't show a new toast if it's already active
          if (toast.isActive("story-error")) return;
          toast({
            id: "story-error",
            title: "Stories could not be retrieved",
            description: error?.message || "An unknown error happened.",
            status: "error",
            duration: 4000,
            isClosable: true,
          });
        });
    },
    // null means: stop polling. We start polling once we retrieve `rasaChatSessionId`
    rasaChatSessionId ? pollingInterval : null
  );

  useInterval(
    () => {
      if (!rasaChatSessionId) return;
      trackerController.abort();
      axios
        .get(`/conversations/${rasaChatSessionId}/tracker?start_session=false`)
        .then((response) => {
          setSlots(formatSlots(response.data.slots));
          setEvents(response.data.events);
          setStack(response.data.stack);
          setFrame(updatedActiveFrame(frame, response.data.stack));
        })
        .catch((error) => {
          // don't show a new toast if it's already active
          if (toast.isActive("tracker-error")) return;
          toast({
            id: "tracker-error",
            title: "Tracker could not be retrieved",
            description: error?.message || "An unknown error happened.",
            status: "error",
            duration: 4000,
            isClosable: true,
          });
        });
    },
    // null means: stop polling. We start polling once we retrieve `rasaChatSessionId`
    rasaChatSessionId ? pollingInterval : null
  );

  const borderRadiusSx = {
    borderRadius: rasaRadii.normal,
  };
  const gridTemplateColumns = useBreakpointValue({
    base: "21rem minmax(20rem, auto) 21rem",
    "2xl": "25rem auto 25rem",
  });
  const gridSx = {
    gridTemplateColumns,
    gridTemplateRows: "1fr",
    gridColumnGap: rasaSpace[1],
    height: "100vh",
    padding: rasaSpace[2],
  };
  const boxSx = {
    ...borderRadiusSx,
    padding: rasaSpace[1],
    bg: useColorModeValue("neutral.50", "neutral.50"),
    overflow: "hidden",
  };
  const leftColumnSx = {
    height: "100%",
    overflow: "hidden",
    gridTemplateColumns: "1fr",
    gridTemplateRows: "max-content minmax(10rem, 17.5rem) minmax(10rem, auto)",
    gridRowGap: rasaSpace[1],
  };

  if (!rasaChatSessionId) return <LoadingSpinner />;

  return (
    <Grid sx={gridSx}>
      <GridItem overflow="hidden">
        <Grid sx={leftColumnSx}>
          <Welcome sx={boxSx} />
          <DialogueStack
            sx={boxSx}
            stack={stack}
            active={frame?.stack}
            onItemClick={setFrame}
          />
          <DialougeInformation
            sx={boxSx}
            rasaChatSessionId={rasaChatSessionId}
            slots={slots}
            events={events}
            story={story}
          />
        </Grid>
      </GridItem>
      <GridItem sx={boxSx}>
        <DiagramFlow stackFrame={frame?.stack} flows={flows} slots={slots} />
      </GridItem>
    </Grid>
  );
}
