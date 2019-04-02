## greet
* greet
    - utter_greet

## happy
* thankyou
    - utter_youarewelcome

## goodbye
* goodbye
    - utter_goodbye

## venue_search
* search_venues
    - action_search_venues
    - slot{"venues": [{"name": "Big Arena", "reviews": 4.5}]}

## concert_search
* search_concerts
    - action_search_concerts
    - slot{"concerts": [{"artist": "Foo Fighters", "reviews": 4.5}]}

## compare_reviews_venues
* search_venues
    - action_search_venues
    - slot{"venues": [{"name": "Big Arena", "reviews": 4.5}]}
* compare_reviews
    - action_show_venue_reviews

## compare_reviews_concerts
* search_concerts
    - action_search_concerts
    - slot{"concerts": [{"artist": "Foo Fighters", "reviews": 4.5}]}
* compare_reviews
    - action_show_concert_reviews
## Generated Story 6575220025073800191
* greet
    - utter_greet

