from dataclasses import dataclass


@dataclass
class ASREvent:
    @classmethod
    def name(cls) -> str:
        return cls.__name__


@dataclass
class NewTranscript(ASREvent):
    text: str


@dataclass
class UserIsSpeaking(ASREvent):
    text: str


@dataclass
class UserSilence(ASREvent):
    pass
