import { updateAgentsStatus } from './utils'
import type { Agent, Event } from '../types'

// Helper to clone agents (immutability check)
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x))

describe('updateAgentsStatus', () => {
  test('returns empty array when agents undefined', () => {
    // @ts-expect-error testing runtime behavior with undefined
    expect(updateAgentsStatus(undefined, [])).toEqual([])
  })

  test('returns same agents when agents empty', () => {
    const agents: Agent[] = []
    const events: Event[] = [
      { event: 'agent_started', timestamp: '1' } as any,
    ]
    expect(updateAgentsStatus(agents, events)).toEqual([])
  })

  test('returns same agents when events empty', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'completed' },
    ]
    const events: Event[] = []
    expect(updateAgentsStatus(clone(agents), events)).toEqual(agents)
  })

  test('ignores non-agent events', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'completed' },
    ]
    const events: Event[] = [
      { event: 'user', text: 'hi', timestamp: '1' } as any,
      { event: 'bot', text: 'hey', timestamp: '2' } as any,
    ]
    expect(updateAgentsStatus(clone(agents), events)).toEqual(agents)
  })

  test('uses latest event by timestamp per agent', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'completed' },
      { name: 'beta', status: 'completed' },
    ]

    const events: Event[] = [
      { event: 'agent_started', agent_id: 'alpha', timestamp: 1 } as any,
      { event: 'agent_interrupted', agent_id: 'alpha', timestamp: 2 } as any,
      { event: 'agent_resumed', agent_id: 'alpha', timestamp: 3 } as any,
      { event: 'agent_cancelled', agent_id: 'beta', timestamp: 5 } as any,
    ]

    const result = updateAgentsStatus(clone(agents), events)
    expect(result).toEqual([
      { name: 'alpha', status: 'running' },
      { name: 'beta', status: 'cancelled' },
    ])
  })

  test('later event wins when timestamps missing', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'completed' },
    ]

    const events: Event[] = [
      { event: 'agent_started', agent_id: 'alpha', timestamp: null } as any,
      { event: 'agent_interrupted', agent_id: 'alpha', timestamp: null } as any,
    ]

    const result = updateAgentsStatus(clone(agents), events)
    expect(result).toEqual([
      { name: 'alpha', status: 'interrupted' },
    ])
  })

  test('agent_completed maps to provided status and falls back to completed', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'running' },
      { name: 'beta', status: 'running' },
    ]

    const events: Event[] = [
      { event: 'agent_completed', agent_id: 'alpha', status: 'completed', timestamp: 10 } as any,
      { event: 'agent_completed', agent_id: 'beta', timestamp: 10 } as any,
    ]

    const result = updateAgentsStatus(clone(agents), events)
    expect(result).toEqual([
      { name: 'alpha', status: 'completed' },
      { name: 'beta', status: 'completed' },
    ])
  })

  test('unmatched agents remain unchanged', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'running' },
      { name: 'gamma', status: 'interrupted' },
    ]

    const events: Event[] = [
      { event: 'agent_resumed', agent_id: 'alpha', timestamp: 1 } as any,
      { event: 'agent_cancelled', agent_id: 'beta', timestamp: 2 } as any, // beta not in agents list
    ]

    const result = updateAgentsStatus(clone(agents), events)
    expect(result).toEqual([
      { name: 'alpha', status: 'running' },
      { name: 'gamma', status: 'interrupted' },
    ])
  })

  test('handles mixed timestamp types (string/number)', () => {
    const agents: Agent[] = [
      { name: 'alpha', status: 'completed' },
    ]

    const events: Event[] = [
      { event: 'agent_started', agent_id: 'alpha', timestamp: '1' } as any,
      { event: 'agent_interrupted', agent_id: 'alpha', timestamp: 2 } as any,
    ]

    const result = updateAgentsStatus(clone(agents), events)
    expect(result).toEqual([
      { name: 'alpha', status: 'interrupted' },
    ])
  })
})
