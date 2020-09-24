## Get started from website
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> uh-huh
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build

## Get started from website + not interested in step 2
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> not yet
    - utter_anything_else

## Direct "install Rasa"
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build

## Installation went well
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> that sounds fine
    - utter_run_rasa_init
    - utter_direct_to_step3

## Installation went well
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yes give me information
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> yea
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
<U> ok great
    - utter_ask_describe_problem
* technical_question OR enter_data OR out_of_scope
    - action_store_problem_description
    - slot{"problem_description": "I don't know how to customize the pipeline"}
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> uh-huh
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
<U> ok great
    - utter_ask_describe_problem
* technical_question OR enter_data OR out_of_scope
    - action_store_problem_description
    - slot{"problem_description": "I don't know how to customize the pipeline"}
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (cut to the chase)
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yep
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
* technical_question OR enter_data OR out_of_scope
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (cut to the chase)
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
* technical_question OR enter_data OR out_of_scope
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (but FAQ)
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> $0.00
    - utter_ask_if_problem
<U> alright
    - utter_ask_describe_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (but FAQ)
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yes pleae
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> the master of desaster
    - utter_ask_if_problem
<U> that sounds fine
    - utter_ask_describe_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (but FAQ + cut to the chase)
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> i want that
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> no. u r idiot
    - utter_ask_if_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing (but FAQ + cut to the chase)
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
* faq
    - utter_faq
    - action_set_faq_slot
    - action_store_problem_description
    - utter_installation_instructions
    - utter_direct_to_forum_for_help
    - utter_run_rasa_init
    - utter_direct_to_step3

## Just don't want to continue
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> no i won't
    - utter_ask_if_problem
<U> not yet
    - utter_anything_else

## Just don't want to continue
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yes that's great
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> not yet
    - utter_ask_if_problem
<U> not yet
    - utter_anything_else
