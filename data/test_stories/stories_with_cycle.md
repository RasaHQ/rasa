## Greeting
* greet
- utter_greet
> get_name

## Get Name - None
> get_name
* default{"name":null}
> process_name

## Get Name - Value
> get_name
* default{"name":"Josh"}
> process_name

## Have name
> process_name{"name":"Josh"}
- utter_goodbye
- action_restart

## Dont have name
> process_name{"name":null}
- utter_default
> get_name
