import { Event } from '../types'
import {
  formatSlots,
  formatTestCases,
  formatFlow,
  parseFieldUsingStack,
} from './formatters'

describe('helpers', () => {
  describe('formatSlots', () => {
    test('should return an empty array if no slots are provided', () => {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      const result = formatSlots(undefined)
      expect(result).toEqual([])
    })

    test('should filter out slots with no value', () => {
      const slots = {
        slot1: null,
        slot2: null,
      }
      const result = formatSlots(slots)
      expect(result).toEqual([])
    })

    test('should filter out `dialogue_stack` slots', () => {
      const slots = {
        dialogue_stack: 'anything',
      }
      const result = formatSlots(slots)
      expect(result).toEqual([])
    })

    test('should filter out `flow_hashes` slots', () => {
      const slots = {
        flow_hashes: 'anything',
      }
      const result = formatSlots(slots)
      expect(result).toEqual([])
    })

    test('should format a list of slots', () => {
      const slots = {
        slot1: true,
        slot2: 'value',
      }
      const result = formatSlots(slots)
      expect(result).toEqual([
        {
          name: 'slot1',
          value: true,
        },
        {
          name: 'slot2',
          value: 'value',
        },
      ])
    })
  })

  describe('formatTestCases', () => {
    const sessionId = 'session-id'

    test('should format an empty list of events', () => {
      const events: Event[] = []
      const result = formatTestCases(events, sessionId)
      expect(result).toEqual(`test_cases:
  - test_case: ${sessionId}
    steps: []`)
    })

    test('should format a list of events', () => {
      const events: Event[] = [
        {
          event: 'user',
          text: 'book a restaurant',
          timestamp: '123',
        },
        {
          event: 'bot',
          metadata: {
            utter_action: 'utter_ask_book_restaurant_name_of_restaurant',
          },
          text: "What's the name of the restaurant you are interested in?",
          timestamp: '124',
        },
        {
          event: 'user',
          text: 'simsim',
          timestamp: '125',
        },
        {
          event: 'bot',
          text: 'How many people are we talking?',
          timestamp: '126',
        },
        {
          event: 'user',
          text: '100',
          timestamp: '127',
        },
        {
          event: 'bot',
          metadata: {
            utter_action: 'utter_ask_book_restaurant_date',
          },
          text: 'For which day do you want to book?',
          timestamp: '128',
        },
      ]
      const result = formatTestCases(events, sessionId)

      expect(result).toEqual(`test_cases:
  - test_case: ${sessionId}
    steps:
    - user: "book a restaurant"
    - utter: utter_ask_book_restaurant_name_of_restaurant
    - user: "simsim"
    - bot: "How many people are we talking?"
    - user: "100"
    - utter: utter_ask_book_restaurant_date`)
    })
  })

  describe('formatFlow', () => {
    test('should render a step sequence', () => {
      const slots = [
        {
          name: 'book_restaurant_offered_alternative_dates',
          value: [],
        },
        {
          name: 'book_restaurant_is_date_flexible',
          value: true,
        },
        {
          name: 'book_restaurant_name_of_restaurant',
          value: 'fonfon',
        },
        {
          name: 'book_restaurant_number_of_people',
          value: '30',
        },
        {
          name: 'book_restaurant_date',
          value: 'tomorrow',
        },
      ]
      const currentStack = {
        frame_id: 'EPQQUPJA',
        flow_id: 'book_restaurant',
        step_id: '3_collect_book_restaurant_time',
        frame_type: 'regular',
        type: 'flow',
      }
      const flow = {
        id: 'book_restaurant',
        name: 'book a restaurant',
        description: 'This flow books a restaurant',
        steps: [
          {
            next: '1_collect_book_restaurant_number_of_people',
            id: '0_collect_book_restaurant_name_of_restaurant',
            collect: 'book_restaurant_name_of_restaurant',
            utter: 'utter_ask_book_restaurant_name_of_restaurant',
            ask_before_filling: false,
            reset_after_flow_ends: true,
          },
          {
            next: '2_collect_book_restaurant_date',
            id: '1_collect_book_restaurant_number_of_people',
            collect: 'book_restaurant_number_of_people',
            utter: 'utter_ask_book_restaurant_number_of_people',
            ask_before_filling: false,
            reset_after_flow_ends: true,
          },
          {
            next: '3_collect_book_restaurant_time',
            id: '2_collect_book_restaurant_date',
            collect: 'book_restaurant_date',
            utter: 'utter_ask_book_restaurant_date',
            ask_before_filling: false,
            reset_after_flow_ends: true,
          },
          {
            next: '4_collect_book_restaurant_is_date_flexible',
            id: '3_collect_book_restaurant_time',
            collect: 'book_restaurant_time',
            utter: 'utter_ask_book_restaurant_time',
            ask_before_filling: false,
            reset_after_flow_ends: true,
          },
          {
            next: '5_check_restaurant_availability',
            id: '4_collect_book_restaurant_is_date_flexible',
            description:
              '(True/False) whether offering alternative dates make sense',
            collect: 'book_restaurant_is_date_flexible',
            utter: 'utter_ask_book_restaurant_is_date_flexible',
            ask_before_filling: false,
            reset_after_flow_ends: true,
          },
          {
            next: [
              {
                if: 'not is_restaurant_available',
                then: [
                  {
                    next: 'ask_alternative',
                    id: '6_utter_restaurant_not_available',
                    action: 'utter_restaurant_not_available',
                  },
                  {
                    next: 'ask_alternative',
                    id: 'ask_alternative',
                    description:
                      'do not fill this slot, instead fill the given parts of the alternative into their respective slots',
                    collect: 'book_restaurant_alternative_dummy',
                    utter: 'utter_ask_book_restaurant_alternative_dummy',
                    ask_before_filling: false,
                    reset_after_flow_ends: true,
                    rejections: [],
                  },
                ],
              },
              {
                else: 'available',
              },
            ],
            id: '5_check_restaurant_availability',
            action: 'check_restaurant_availability',
          },
          {
            next: '9_collect_book_restaurant_reservation_name',
            id: 'available',
            action: 'utter_restaurant_available',
          },
          {
            next: '10_collect_book_restaurant_confirmation',
            id: '9_collect_book_restaurant_reservation_name',
            collect: 'book_restaurant_reservation_name',
            utter: 'utter_ask_book_restaurant_reservation_name',
            ask_before_filling: true,
            reset_after_flow_ends: true,
          },
          {
            next: [
              {
                if: 'book_restaurant_confirmation',
                then: [
                  {
                    next: 'END',
                    id: '11_utter_confirm_restaurant_booking',
                    action: 'utter_confirm_restaurant_booking',
                  },
                ],
              },
              {
                else: [
                  {
                    next: 'END',
                    id: '12_utter_cancel_book_restaurant',
                    action: 'utter_cancel_book_restaurant',
                  },
                ],
              },
            ],
            id: '10_collect_book_restaurant_confirmation',
            collect: 'book_restaurant_confirmation',
            utter: 'utter_ask_book_restaurant_confirmation',
            ask_before_filling: true,
            reset_after_flow_ends: true,
          },
        ],
      }
      const activeStep = '3_collect_book_restaurant_time'

      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      const result = formatFlow(slots, currentStack, flow, activeStep)
      expect(result).toEqual(restaurantFlow)
    })
  })

  describe('parseFieldUsingStack', () => {
    test('should parse a filed using a stack', () => {
      const fieldValue = 'transfer_money_recipient'

      expect(
        parseFieldUsingStack('{{context.collect}}', {
          frame_id: 'frame_id',
          flow_id: 'flow_id',
          step_id: 'step_id',
          collect: fieldValue,
          ended: false,
          type: 'flow',
        }),
      ).toEqual(fieldValue)
    })

    test("should return the same field if it doesn't have `context`", () => {
      const fieldValue = 'transfer_money_recipient'

      expect(
        parseFieldUsingStack(fieldValue, {
          frame_id: 'frame_id',
          flow_id: 'flow_id',
          step_id: 'step_id',
          ended: false,
          type: 'flow',
        }),
      ).toEqual(fieldValue)
    })

    test('should return the value if it has conditions as postfix', () => {
      const fieldValue = 'transfer_money_recipient'

      expect(
        parseFieldUsingStack('{{context.collect}} is not null', {
          frame_id: 'frame_id',
          flow_id: 'flow_id',
          step_id: 'step_id',
          collect: fieldValue,
          ended: false,
          type: 'flow',
        }),
      ).toEqual(`${fieldValue} is not null`)
    })

    test('should return the value if it has conditions as prefix', () => {
      const fieldValue = 'transfer_money_recipient'

      expect(
        parseFieldUsingStack('not {{context.collect}}', {
          frame_id: 'frame_id',
          flow_id: 'flow_id',
          step_id: 'step_id',
          collect: fieldValue,
          ended: false,
          type: 'flow',
        }),
      ).toEqual(`not ${fieldValue}`)
    })
  })
})

const restaurantFlow = `flowchart TD
classDef collect stroke-width:1px
classDef action fill:#FBFCFD,stroke:#A0B8CF
classDef link fill:#f43
classDef slot fill:#e8f3db,stroke:#c5e1a5
classDef endstep fill:#ccc,stroke:#444
classDef active stroke:#F8941A,stroke-width:3px,fill:#FFF8E0
0_collect_book_restaurant_name_of_restaurant["book_restaurant_name_of_restaurant
'fonfon'"]:::collect
0_collect_book_restaurant_name_of_restaurant --> 1_collect_book_restaurant_number_of_people
1_collect_book_restaurant_number_of_people["book_restaurant_number_of_people
'30'"]:::collect
1_collect_book_restaurant_number_of_people --> 2_collect_book_restaurant_date
2_collect_book_restaurant_date["book_restaurant_date
'tomorrow'"]:::collect
2_collect_book_restaurant_date --> 3_collect_book_restaurant_time
3_collect_book_restaurant_time["book_restaurant_time
💬"]:::collect
class 3_collect_book_restaurant_time active
3_collect_book_restaurant_time --> 4_collect_book_restaurant_is_date_flexible
4_collect_book_restaurant_is_date_flexible["book_restaurant_is_date_flexible
'true'"]:::collect
4_collect_book_restaurant_is_date_flexible --> 5_check_restaurant_availability
5_check_restaurant_availability["check_restaurant_availability"]:::action
5_check_restaurant_availability -->|not is_restaurant_available| 6_utter_restaurant_not_available
6_utter_restaurant_not_available["utter_restaurant_not_available"]:::action
6_utter_restaurant_not_available --> ask_alternative
ask_alternative["book_restaurant_alternative_dummy
💬"]:::collect
ask_alternative --> ask_alternative
END["🏁 END"]:::endstep
5_check_restaurant_availability -->|else| available
available["utter_restaurant_available"]:::action
available --> 9_collect_book_restaurant_reservation_name
9_collect_book_restaurant_reservation_name["book_restaurant_reservation_name
💬"]:::collect
9_collect_book_restaurant_reservation_name --> 10_collect_book_restaurant_confirmation
10_collect_book_restaurant_confirmation["book_restaurant_confirmation
💬"]:::collect
10_collect_book_restaurant_confirmation -->|book_restaurant_confirmation| 11_utter_confirm_restaurant_booking
11_utter_confirm_restaurant_booking["utter_confirm_restaurant_booking"]:::action
11_utter_confirm_restaurant_booking --> END
END["🏁 END"]:::endstep
10_collect_book_restaurant_confirmation -->|else| 12_utter_cancel_book_restaurant
12_utter_cancel_book_restaurant["utter_cancel_book_restaurant"]:::action
12_utter_cancel_book_restaurant --> END
END["🏁 END"]:::endstep
END["🏁 END"]:::endstep
`
