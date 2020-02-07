## intent:affirm
- yes
- yep
- yeah
- indeed
- that's right
- ok
- great
- right, thank you
- correct
- great choice
- sounds really good
- sure
- yup
- right
- definitely
- agreed

## intent:goodbye
- bye
- goodbye
- good bye
- stop
- end
- farewell
- Bye bye
- have a good one
- cheers
- pip pip
- peace out
- cya
- till later
- gbye

## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir
- yo
- what up
- greetings
- ahoy
- hi hi
- sup


## intent:restaurant_search
- i'm looking for a place to eat
- I want to grab lunch
- I am searching for a dinner spot
- i'm looking for a place in the [north](location) of town
- show me [chinese](cuisine) restaurants
- show me [chines](cuisine:chinese) restaurants in the [north](location)
- show me a [mexican](cuisine) place in the [centre](location)
- i am looking for an [indian](cuisine) spot called olaolaolaolaolaola
- search for restaurants
- anywhere in the [west](location)
- anywhere near [18328](location)
- I am looking for [asian fusion](cuisine) food
- I am looking a restaurant in [29432](location)
- I am looking for [mexican indian fusion](cuisine)
- [central](location) [indian](cuisine) restaurant

## synonym:chinese
+ Chines
* Chinese

## synonym:vegetarian
- vegg
- veggie

## regex:zipcode
- [0-9]{5}

## regex:greet
- hey[^\s]*

## intent:out_of_scope/non_english
- Wie fange ich mit Rasa an?
- hilf mir beim start
- tschüssikowski
- ¿Qué pasa?
- ça va ?
- como te llamas
- wer bist Du?
- como inicio en rasa
- come stai?
- como estas
- de donde eres
- de que lugar eres?
- epdi iruka
- eres humana
- kalhmera sara ti kaneis
- kannst du auch deutsch?
- kannst du dies auch auf deutsch?
- oui je besoine de l'aide
- que puedes hacer?
- tu parles francais?
- tudo bom

## intent:out_of_scope/other
- I am asking you an out of scope question
- 4 + 2 = ?
- After registration I see that I have an available balance of 0.00000000. What does this balance represent?
- Are you ready?
- But you're an english site :(
- Can I ask you questions first?
- Can I die
- Can I get a hamburger?
- Can YouTube talk?
- Can you call me back ?
- Can you give me your datacenter's password
- Can you give me your datacenter's password?
- Can you make sandwiches?
- Can you please send me an uber
- Do I have to accept?
- Do you know me
- Find nearest pizzahut
- Have we met before?
- HomeBase is advertised as a community. Is there a way to interact with other members of the community?
- How long does it take to set up a Rasa bot?
- I already told you! I'm a shitmuncher
- I am User
- I am an opioid addic
- I am an opioid addict
- I am hungry

