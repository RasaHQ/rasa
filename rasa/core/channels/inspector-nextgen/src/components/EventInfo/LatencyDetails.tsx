import { Accordion, Heading, HStack, Tag, Text } from "@chakra-ui/react";
import type { ReactNode } from "react";
import type { RasaExecutionTime, VoiceLatency } from "../../types/conversation";
import { isRasaExecutionTime, isVoiceLatency } from "../../utils/latency";

function LatencyPill({ ms }: { readonly ms: number }) {
  return (
    <Tag.Root variant="surface" colorPalette="orange" size="lg">
      <Tag.Label>
        ~{ms} ms
      </Tag.Label>
    </Tag.Root>
  );
}

function LatencyRow({
  label,
  valueMs,
}: {
  readonly label: string;
  readonly valueMs: number;
}) {
  return (
    <HStack justify="flex-start" py="2" gap="1">
      <Heading textStyle="sm">
        {label}
      </Heading>
      <Text textStyle="sm">
        {Math.round(valueMs)} ms
      </Text>
    </HStack >
  );
}

function VoiceLatencyBody({ latency }: { readonly latency: VoiceLatency }) {
  return (
    <>
      <LatencyRow
        label="Rasa:"
        valueMs={latency.rasa_processing_latency_ms}
      />
      <LatencyRow
        label="TTS First Byte:"
        valueMs={latency.tts_first_byte_latency_ms}
      />
      <LatencyRow label="ASR:" valueMs={latency.asr_latency_ms} />
      <LatencyRow
        label="TTS Complete:"
        valueMs={latency.tts_complete_latency_ms}
      />
    </>
  );
}

function RasaExecutionBody({ times }: { readonly times: RasaExecutionTime }) {
  return (
    <>
      <LatencyRow label="Command processor:" valueMs={times.command_processor} />
      <LatencyRow label="Prediction loop:" valueMs={times.prediction_loop} />
    </>
  );
}

export function BotLatencyAccordionItem({
  executionTimes,
  voiceLatency,
}: {
  readonly executionTimes?: RasaExecutionTime;
  readonly voiceLatency?: VoiceLatency;
}) {
  let titleMs: number;
  let body: ReactNode;

  if (isVoiceLatency(voiceLatency)) {
    titleMs = Math.round(
      voiceLatency.rasa_processing_latency_ms +
      voiceLatency.tts_first_byte_latency_ms,
    );
    body = <VoiceLatencyBody latency={voiceLatency} />;
  } else if (isRasaExecutionTime(executionTimes)) {
    titleMs = Math.round(
      executionTimes.command_processor + executionTimes.prediction_loop,
    );
    body = <RasaExecutionBody times={executionTimes} />;
  } else {
    return null;
  }

  return (
    <Accordion.Item
      value="latency"
      py="1"
    >
      <Accordion.ItemTrigger px="0" cursor="pointer">
        <HStack flex="1" justify="flex-start" align="center" gap="2">
          <Heading textStyle="sm">
            Latency per turn:
          </Heading>
          <LatencyPill ms={titleMs} />
        </HStack>
        <Accordion.ItemIndicator />
      </Accordion.ItemTrigger>
      <Accordion.ItemContent>
        <Accordion.ItemBody px="0">{body}</Accordion.ItemBody>
      </Accordion.ItemContent>
    </Accordion.Item>
  );
}
