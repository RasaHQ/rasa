import {rasaColors} from '../theme/base/colors'
import type {Event, Flow, Slot, Stack} from '../types'

export function formatSlots(slots: { [key: string]: unknown }) {
  if (!slots) {
    return []
  }

  return Object.entries(slots)
    .filter(
      (slotDuple) =>
        slotDuple[0] !== 'dialogue_stack' &&
        slotDuple[0] !== 'flow_hashes' &&
        slotDuple[1] != null,
    )
    .map((slotDuple) => ({name: slotDuple[0], value: slotDuple[1]}))
}

export const formatTestCases = (events: Event[], sessionId: string) => {
  let testCases = `test_cases:
  - test_case: ${sessionId}
    steps:`

  const steps = events
    .map((event) => {
      const escapedText = JSON.stringify(event.text)
      if (event.event === 'user') {
        return `    - user: ${escapedText}`
      } else if (event.event === 'bot') {
        return `${
          event.metadata?.utter_action
            ? `    - utter: ${event.metadata.utter_action}`
            : `    - bot: ${escapedText}`
        }`
      }
    })
    .filter(Boolean)

  if (steps.length === 0) {
    testCases = `${testCases} []`
  } else {
    testCases = `${testCases}
${steps.join('\n')}`
  }

  return testCases
}

function hashCode(str: string) {
  var hash = 0,
    i,
    chr
  if (str.length === 0) return hash
  for (i = 0; i < str.length; i++) {
    chr = str.charCodeAt(i)
    hash = (hash << 5) - hash + chr
    hash |= 0 // Convert to 32bit integer
  }
  return hash
}

function mermaidIdForTitle(title: string) {
  return `id${hashCode(title)}`
}

const encodeDoubleQuotes = (str: string) =>
  /**
   * Transforms all " characters of a string into html codes.
   */
  str.replace(/"/g, `#34;`)

export const formatFlow = (
  slots: Slot[],
  currentStack?: Stack,
  flow?: Flow,
  stepTrail?: string[],
) => {
  const activeStep = currentStack?.step_id

  if (!flow) {
    return ''
  }

  const mermaidText = [
    `flowchart TD
classDef collect stroke-width:1px
classDef action fill:#FBFCFD,stroke:#A0B8CF
classDef noop fill:#FBFCFD
classDef callstep fill:#FBFCFD
classDef link fill:#f43
classDef slot fill:#e8f3db,stroke:#c5e1a5
classDef endstep fill:#ccc,stroke:#444
classDef previous stroke:${rasaColors.rasaOrange[400]},stroke-width:1px
classDef active stroke:${rasaColors.rasaOrange[400]},stroke-width:3px,fill:${rasaColors.warning[50]}
classDef pulse animation:pulse 2s infinite
`,
  ]
  try {
    const text = renderStepSequence(
      flow.steps,
      slots,
      currentStack,
      activeStep,
      stepTrail,
    )
    mermaidText.push(text)
    mermaidText.push(colorDoubleEdges(mermaidText.join('')))
    return mermaidText.join('')
  } catch (e) {
    return `${mermaidText}\nA["Something went wrong!"]\nB["${e}"]`
  }
}

function truncate(str: string, limit = 35) {
  // if a string is too long, use an ellipsis
  if (str.length > limit) {
    return str.substring(0, limit) + '...'
  }
  return str
}

function colorDoubleEdges(mermaidText: string) {
  // go through the lines of mermaid text. keep a counter counting edges
  // ("-->"" or "==>"). if "==>" is found in a line, add the line number to
  // a list.
  const lines = mermaidText.split('\n')
  const coloredEdges = []
  let edgeCounter = 0
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('-->')) {
      edgeCounter++
    } else if (lines[i].includes('==>')) {
      coloredEdges.push(edgeCounter)
      edgeCounter++
    }
  }
  if (coloredEdges.length > 0) {
    return `linkStyle ${coloredEdges.join(',')} stroke:${
      rasaColors.rasaOrange[400]
    }, ;\n`
  } else {
    return ''
  }
}

export function parseFieldUsingStack(name: string, stack?: Stack): string {
  // name might be in the `{{context.field_in_stack}}` format so we're stripping everything except the field in the stack name
  const parsedField = name.split(/{{context\.|}}/)
  if (parsedField.length === 1) {
    return parsedField[0]
  }

  const stackField = parsedField[1]

  // @ts-expect-error `stack[stackField]` doesn't necessary exists this might return `undefined`
  const stackValue = stack ? stack[stackField] : undefined

  // name might also be in the `condition {{context.field_in_stack}} condition` format
  // so we want to keep that if there is any
  if (parsedField[2] && parsedField[2] !== '') {
    return `${parsedField[0]}${stackValue}${parsedField[2]}`
  }

  return `${parsedField[0]}${stackValue}`
}

function arrowTypeFor(stepId: string, nextId: string, stepTrail?: string[]) {
  return stepTrail?.includes(stepId) && stepTrail?.includes(nextId)
    ? '==>'
    : '-->'
}

function renderStepSequence(
  steps: Flow['steps'],
  slots: Slot[],
  currentStack?: Stack,
  activeStep?: string,
  stepTrail?: string[],
) {
  let hasUsedEndStep = false
  let mermaidTextFragment = ''
  steps.forEach((step) => {
    const stepId = parseFieldUsingStack(step.id, currentStack)
    const mermaidId = mermaidIdForTitle(stepId)

    if (step.collect) {
      const slot = slots.find((slot) => slot.name === step.collect)
      const slotValue =
        slot && typeof slot.value === 'string'
          ? `"${encodeDoubleQuotes(truncate(slot.value))}"`
          : '💬'
      mermaidTextFragment += `${mermaidId}["${encodeDoubleQuotes(
        truncate(parseFieldUsingStack(step.collect, currentStack)),
      )}\n${slotValue}"]:::collect\n`
    }

    if (step.action) {
      mermaidTextFragment += `${mermaidId}["${encodeDoubleQuotes(
        truncate(parseFieldUsingStack(step.action, currentStack)),
      )}"]:::action\n`
    }

    if (step.link) {
      mermaidTextFragment += `${mermaidId}["\uD83D\uDD17 ${parseFieldUsingStack(
        step.link,
        currentStack,
      )}"]:::link\n`
    }

    if (step.set_slots) {
      mermaidTextFragment += `${mermaidId}["✍️ ${encodeDoubleQuotes(
        stepId,
      )}"]:::slot\n`
    }

    if (step.call) {
      const isAgentWaitingOnInput = !!currentStack?.agent_id && currentStack?.state === 'waiting_for_input'
      mermaidTextFragment += `${mermaidId}["${encodeDoubleQuotes(
        stepId,
      )} 🤖"]:::callstep\n`

      if (isAgentWaitingOnInput) {
        mermaidTextFragment += `class ${mermaidId} pulse\n`
      }
    }

    if (step.noop) {
      mermaidTextFragment += `${mermaidId}["${parseFieldUsingStack(
        stepId,
        currentStack
      )}"]:::noop\n`
    }

    if (activeStep && stepId === activeStep) {
      mermaidTextFragment += `class ${mermaidId} active\n`
    } else if (stepTrail?.includes(stepId)) {
      mermaidTextFragment += `class ${mermaidId} previous\n`
    }

    // if next is an id, then it is a link
    if (step.next && typeof step.next === 'string') {
      const nextId = parseFieldUsingStack(step.next, currentStack)

      mermaidTextFragment += `${mermaidId} ${arrowTypeFor(
        stepId,
        nextId,
        stepTrail,
      )} ${mermaidIdForTitle(nextId)}\n`
      if (step.next == 'END') {
        hasUsedEndStep = true
      }
    }

    // if next is an array, then it is a list of conditions
    if (step.next && Array.isArray(step.next)) {
      step.next.forEach((condition) => {
        if (condition.then && typeof condition.then === 'string') {
          mermaidTextFragment += `${mermaidId} ${arrowTypeFor(
            stepId,
            condition.then,
            stepTrail,
          )}|"${encodeDoubleQuotes(
            parseFieldUsingStack(condition.if, currentStack),
          )}"| ${mermaidIdForTitle(condition.then)}\n`
          if (condition.then == 'END') {
            hasUsedEndStep = true
          }
        } else if (condition.then) {
          mermaidTextFragment += `${mermaidId} ${arrowTypeFor(
            stepId,
            condition.then[0].id,
            stepTrail,
          )}|"${encodeDoubleQuotes(
            parseFieldUsingStack(condition.if, currentStack),
          )}"| ${mermaidIdForTitle(condition.then[0].id)}\n`
          mermaidTextFragment += renderStepSequence(
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
            condition.then,
            slots,
            currentStack,
            activeStep,
            stepTrail,
          )
        }

        // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
        if (condition.else && typeof condition.else === 'string') {
          mermaidTextFragment += `${mermaidId} ${arrowTypeFor(
            stepId,
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
            condition.else,
            stepTrail,
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
          )}|else| ${mermaidIdForTitle(condition.else)}\n`
          // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
          if (condition.else == 'END') {
            hasUsedEndStep = true
          }
          // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
        } else if (condition.else) {
          mermaidTextFragment += `${mermaidId} ${arrowTypeFor(
            stepId,
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
            condition.else[0].id,
            stepTrail,
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
          )}|else| ${mermaidIdForTitle(condition.else[0].id)}\n`
          mermaidTextFragment += renderStepSequence(
            // @ts-expect-error Currently the param for renderStepSequence only accepts a Step, for further improvements we need to change the type to know that it can also be a then step
            condition.else,
            slots,
            currentStack,
            activeStep,
            stepTrail,
          )
        }
      })
    }
  })
  if (hasUsedEndStep) {
    mermaidTextFragment += `${mermaidIdForTitle('END')}["🏁 END"]:::endstep\n`
    if (activeStep && 'END' === activeStep) {
      mermaidTextFragment += `class ${mermaidIdForTitle('END')} active\n`
    }
  }
  return mermaidTextFragment
}
