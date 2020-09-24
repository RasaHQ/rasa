## Get started from website
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> yes, i'd love to
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build

## Get started from website + not interested in step 2
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> i don't want to say
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
<U> yes ...
    - utter_run_rasa_init
    - utter_direct_to_step3

## Installation went well
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> ofcoure i do
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> fine
    - utter_run_rasa_init
    - utter_direct_to_step3

## Problems installing
* install_rasa
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> i dont want to accept :p lol
    - utter_ask_if_problem
<U> ok i accept
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
<U> great lets do that
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> i don't want to give it to you
    - utter_ask_if_problem
<U> amazing!
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
<U> sure!
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> i'm not sure
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
<U> no thank you
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
<U> na
    - utter_ask_if_problem
<U> yres
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
<U> yep that's nice
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> na
    - utter_ask_if_problem
<U> of course
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
<U> not bad
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> definitely not
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
<U> no, thankyou
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
<U> never mind
    - utter_ask_if_problem
<U> no, thank you
    - utter_anything_else

## Just don't want to continue
* get_started_step2
    - action_greet_user
    - slot{"step": "2"}
<U> oui
    - utter_installation_command
    - utter_having_trouble_installing
    - utter_ask_ready_to_build
<U> no i can't
    - utter_ask_if_problem
<U> i don't want to say
    - utter_anything_else
