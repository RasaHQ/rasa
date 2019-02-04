from rasa_core.event_brokers.kafka_producer import KafkaProducer  # nopep8
from rasa_core.event_brokers.pika_producer import PikaProducer  # nopep8

event_broker_classes = [
    PikaProducer, KafkaProducer
]  # type: List[InputChannel]

BUILTIN_EVENT_BROKERS = {
    c.name(): c
    for c in event_broker_classes}  # type: Dict[Text, InputChannel]
