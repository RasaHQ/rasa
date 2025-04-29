import {
  Grid,
  GridItem,
  useBreakpointValue,
  useColorModeValue,
  useToast,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import axios from "axios";
import { useOurTheme } from "./theme";
import { Welcome } from "./components/Welcome";
import { DialogueStack } from "./components/DialogueStack";
import { DialougeInformation } from "./components/DialogueInformation";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { DiagramFlow } from "./components/DiagramFlow";
import { RecruitmentPanel } from "./components/RecruitmentPanel";
import { formatSlots } from "./helpers/formatters";
import { Slot, Stack, Event, Flow, SelectedStack, Tracker } from "./types";
import {
  createHistoricalStack,
  flowStepTrail,
  updatedActiveFrame,
} from "./helpers/utils";
import queryString from "query-string";
import { Chat } from "./components/Chat";
import useWebSocket, { ReadyState } from "react-use-websocket";

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

  // State to control the visibility of the RecruitmentPanel
  const [showRecruitmentPanel, setShowRecruitmentPanel] = useState(true);

  // we only show the transcript if we are not on the socket io channel
  // on the socketio channel, we show the chat component instead
  const shouldShowTranscript = !window.location.href.includes("socketio");

  const inspectWS = window.location.href
    .replace("inspect.html", "tracker_stream")
    .replace("http", "ws");
  const { sendJsonMessage, lastJsonMessage, readyState } =
    useWebSocket<Tracker>(inspectWS, {
      share: false,
      shouldReconnect: () => true,
    });
    const token = new URLSearchParams(window.location.search).get("token");

  useEffect(() => {
    if (readyState === ReadyState.OPEN && rasaChatSessionId) {
      sendJsonMessage({
        action: "retrieve",
        sender_id: rasaChatSessionId,
      });
    }
  }, [readyState, sendJsonMessage, rasaChatSessionId]);

  useEffect(() => {
    // if we are on a "dynamic" input channel where the user can chat on
    // the inspector page, these chat-uis (rasa chat, vortex chat) don't
    // support loading an existing conversation. therefore, we don't set
    // the sender id in the url and a reload of the page will start a new
    // conversation.
    if(!shouldShowTranscript) return;

    const queryParameters = queryString.parse(window.location.search);
    const urlSender = queryParameters.sender as string;
    if (urlSender && urlSender !== rasaChatSessionId) {
      setRasaChatSessionId(urlSender);
    } else if (!urlSender && rasaChatSessionId) {
      // update the sender query parameter
      const url = new URL(window.location.href);
      url.searchParams.set("sender", rasaChatSessionId);
      window.history.pushState(null, "", url.toString());
    }
  }, [rasaChatSessionId, shouldShowTranscript]);

  useEffect(() => {
    axios
      .get("/flows", { params: { token } })
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

  function fetchStory() {
    axios
      .get(`/conversations/${rasaChatSessionId}/story`, { params: { token } })
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
  }

  // Run when a new WebSocket message is received (lastJsonMessage)
  useEffect(() => {
    // if the tracker update that we received is for the current chat session,
    // update the tracker.
    if (!lastJsonMessage) return;
    if (
      !rasaChatSessionId ||
      lastJsonMessage?.sender_id === rasaChatSessionId
    ) {
      setSlots(formatSlots(lastJsonMessage.slots));
      setEvents(lastJsonMessage.events);
      const updatedStack = createHistoricalStack(
        lastJsonMessage.stack,
        lastJsonMessage.events
      );
      setStack(updatedStack);
      setFrame(updatedActiveFrame(frame, updatedStack, lastJsonMessage.events));
      setRasaChatSessionId(lastJsonMessage.sender_id);
      fetchStory();
    } else if (
      rasaChatSessionId &&
      lastJsonMessage.sender_id !== rasaChatSessionId
    ) {
      // show a header link with the new sender the user can click a button to
      // switch to the new chat session. the alert should be dissmissable.
    }
  }, [lastJsonMessage, rasaChatSessionId]);

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
    gridTemplateRows: showRecruitmentPanel
      ? "max-content max-content minmax(10rem, auto)"
      : "max-content minmax(10rem, 17.5rem) minmax(10rem, auto)",
    gridRowGap: rasaSpace[1],
  };

  const onFrameSelected = (stack: Stack) => {
    setFrame({
      stack,
      activatedSteps: flowStepTrail(events)[stack.flow_id],
      isUserSelected: true,
    });
  };

  const handleCloseRecruitmentPanel = () => {
    setShowRecruitmentPanel(false);
  };

  if (!rasaChatSessionId && !window.location.href.includes("socketio")) return <LoadingSpinner />;

  return (
    <Grid sx={gridSx}>
      <GridItem overflow="hidden">
        <Grid sx={leftColumnSx}>
          <Welcome sx={boxSx} isRecruitmentVisible={showRecruitmentPanel} />
          {showRecruitmentPanel && (
            <RecruitmentPanel onClose={handleCloseRecruitmentPanel} />
          )}
          <DialogueStack
            sx={boxSx}
            stack={stack}
            active={frame?.stack}
            onItemClick={onFrameSelected}
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
        <DiagramFlow
          stackFrame={frame?.stack}
          stepTrail={frame?.activatedSteps || []}
          flows={flows}
          slots={slots}
        />
      </GridItem>
      {shouldShowTranscript && (
        <GridItem>
          <Chat events={events || []} />
        </GridItem>
      )}
    </Grid>
  );
}
