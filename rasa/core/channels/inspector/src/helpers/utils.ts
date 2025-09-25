import {SelectedStack, Stack, Event, Agent, AGENT_EVENT_TYPES} from '../types'
import { immutableJSONPatch } from 'immutable-json-patch'

export const shouldShowTooltip = (text: string) => {
  if (!text) {
    return false
  }

  const textLength = text.length

  if (textLength > 10 && textLength < 89) {
    return true
  }

  return false
}

export const createHistoricalStack = (
  activeStack: Stack[],
  events: Event[],
): Stack[] => {
  let stackFrames = activeStack.map((frame) => ({
    ...frame,
    ended: false,
  }))
  // go through the events looking for flow_completed and append them to the stack
  let historicalStack: Stack[] = []
  let pastStackFrames: Stack[] = []
  for (const event of events) {
    if (event.event === 'restart') {
      historicalStack = []
      stackFrames = []
    } else if (event.event === 'stack') {
      let stackUpdate = JSON.parse(event.update || '')
      historicalStack = immutableJSONPatch(historicalStack, stackUpdate)
      for (const frame of historicalStack) {
        if (!frame.flow_id) {
          // this is not a stack frame from the flow handler
          continue
        }
        // if the frame is already in stackFrames, skip it
        if (stackFrames.find((f) => f.frame_id === frame.frame_id)) {
          continue
        }
        // if the frame is in pastStackFrames, update the step_id otherwise add it
        const pastFrame = pastStackFrames.find(
          (f) => f.frame_id === frame.frame_id,
        )
        if (pastFrame) {
          pastFrame.step_id = frame.step_id
          continue
        }
        pastStackFrames.push({ ...frame, ended: true })
      }
    }
  }
  // filter out pattern_collect_information frames
  pastStackFrames = pastStackFrames.filter(
    (frame) => frame.flow_id !== 'pattern_collect_information',
  )
  return [...pastStackFrames, ...stackFrames]
}

export const flowStepTrail = (events: Event[]): Record<string, string[]> => {
  let stack: Stack[] = []
  // mapping from flow id to the steps that were active in that flow
  let activeSteps: { [key: string]: string[] } = {}
  for (const event of events) {
    if (event.event === 'restart') {
      stack = []
      activeSteps = {}
    } else if (event.event === 'stack') {
      let stackUpdate = JSON.parse(event.update || '')
      stack = immutableJSONPatch(stack, stackUpdate)
      if (stack.length > 0) {
        let topFrame = stack[stack.length - 1]
        if (!topFrame.flow_id) {
          // this is not a stack frame from the flow handler
          continue
        }
        if (!activeSteps[topFrame.flow_id] || topFrame.step_id === 'START') {
          activeSteps[topFrame.flow_id] = []
        }
        if (!activeSteps[topFrame.flow_id].includes(topFrame.step_id)) {
          activeSteps[topFrame.flow_id].push(topFrame.step_id)
        }
      }
    }
  }
  return activeSteps
}

export const updatedActiveFrame = (
  previous: SelectedStack | undefined,
  updatedStack: Stack[],
  events: Event[],
) => {
  // try to find the currently active frame in the updated stack
  // if it isn't there anymore, we will show the first non-pattern frame
  // instead

  // reset previously active stack frame, if it was not user selected
  if (!previous?.isUserSelected) {
    previous = undefined
  }

  const activeFrame = updatedStack.find(
    (stackFrame) => stackFrame.frame_id === previous?.stack.frame_id,
  )
  if (!activeFrame || activeFrame.ended) {
    if (!updatedStack) {
      return undefined
    }
    // iterate over the stack. select the first frame where the name does not
    // contain "pattern" and that has not
    // ended yet. If there is no such frame, select the topmost frame that has
    // not ended yet. If there is no such frame, select the topmost frame.
    const updatedFrame =
      updatedStack
        .slice()
        .reverse()
        .find(
          (frame) => !frame.flow_id?.startsWith('pattern_') && !frame.ended,
        ) ||
      updatedStack
        .slice()
        .reverse()
        .find((frame) => !frame.ended) ||
      updatedStack[updatedStack.length - 1]

    if (updatedFrame !== undefined) {
      return {
        stack: updatedFrame,
        isUserSelected: false,
        activatedSteps: flowStepTrail(events)[updatedFrame.flow_id] || [],
      }
    } else {
      return undefined
    }
  } else {
    if (previous) {
      previous.activatedSteps =
        flowStepTrail(events)[previous.stack.flow_id] || []
      previous.stack = activeFrame
    }
    return previous
  }
}

export function updateAgentsStatus(agents: Agent[], events: Event[]): Agent[] {
  if (!agents || agents.length === 0 || !events || events.length === 0) {
    return agents || []
  }

  const agentEventTypes = new Set<Event['event']>(AGENT_EVENT_TYPES as unknown as Event['event'][] )

  type AgentEventLite = Agent & {
    agent_id: string
    event: Event['event']
    timestamp?: string | number | null
  }
  
  const latestByAgent = new Map<string, AgentEventLite>()

  for (let i = 0; i < events.length; i++) {
    const event = events[i] as unknown as AgentEventLite
    if (!event || !agentEventTypes.has(event.event)) continue

    const agentId = event.agent_id
    if (!agentId) continue

    const previousEvent = latestByAgent.get(agentId)

    const currentTimestamp = event.timestamp == null ? undefined : Number(event.timestamp)
    const previousTimestamp = previousEvent?.timestamp == null ? undefined : Number(previousEvent.timestamp)

    const isNewer =
      previousEvent == null ||
      (currentTimestamp != null && previousTimestamp != null && currentTimestamp > previousTimestamp) ||
      (currentTimestamp != null && previousTimestamp == null)
    // If both timestamps are missing, prefer later in the array (current)

    if (isNewer || (currentTimestamp == null && previousTimestamp == null)) {
      latestByAgent.set(agentId, event)
    }
  }

  return agents.map((agent) => {
    const latest = latestByAgent.get(agent.name)
    if (!latest) return agent

    let status = agent.status
    switch (latest.event) {
      case 'agent_started':
      case 'agent_resumed':
        status = 'running'
        break
      case 'agent_interrupted':
        status = 'interrupted'
        break
      case 'agent_cancelled':
        status = 'cancelled'
        break
      case 'agent_completed':
        status = latest.status || 'completed'
        break
      default:
        break
    }

    return { ...agent, status }
  })
}