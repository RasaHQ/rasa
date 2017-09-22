## greet
* _greet
    - action_greet
    
## goodbye
* _goodbye
    - action_goodbye

## inform_but_no_change_wanted
* _inform_addresschange
    - action_ask_for_address_change

## deny
* _deny
    - action_no_address_change

## inform_with_complete_address_present
* _inform_addresschange[GPE=house street state]
    - action_ask_for_address_change

## request_change_with_complete_address
* _request_addresschange[GPE=house street state]
    - action_handle_partial_address
    - action_ask_confirm_address

## affirm_1
* _affirm
    - action_handle_partial_address
    - action_ask_for_address_change

## affirm_2
* _affirm OR _thankyou
    - action_ask_confirm_address

## affirm_3
* _affirm OR _thankyou
    - action_confirm_address
