import pytest

from rasa.core.channels.voice_ready.jambonz_protocol import (
    CallStatusChanged,
    NewSessionMessage,
)


@pytest.fixture
def session_new_message():
    return {
        "type": "session:new",
        "msgid": "heDUewwTZT51kXwauek3Zk",
        "call_sid": "f31285ee-0bd4-4f1e-907d-f239bcbea5c1",
        "data": {
            "sip": {
                "headers": {
                    "via": "SIP/2.0/UDP 54.236.168.131;rport=5060;branch=z9hG4bKHg6B8e5638mXH;received=172.20.10.254",  # noqa E501
                    "max-forwards": "70",
                    "from": "<sip:+491604697810@54.236.168.131: 5060>;tag=cv3mj1eyZ7U4e",  # noqa E501
                    "to": "<sip:+493041733972@shailendra.sip.jambonz.cloud>",
                    "call-id": "7e6e04fa-f451-123d-9aab-0e9252b57157",
                    "cseq": "88995033 INVITE",
                    "contact": "<sip: 172.20.10.254: 5060>",
                    "user-agent": "Twilio Gateway",
                    "allow": "INVITE, ACK, BYE, CANCEL, OPTIONS, REFER, NOTIFY",
                    "content-type": "application/sdp",
                    "content-length": "289",
                    "X-Account-Sid": "55312bec-9135-4615-9f72-667e3719a911",
                    "X-CID": "30cea8a53174afe5bb05348945e4a52a@0.0.0.0",
                    "X-Forwarded-For": "35.156.191.129",
                    "X-Originating-Carrier": "Twilio-n7xbPvgiNLmPuFASP2Z2AX",
                    "X-Voip-Carrier-Sid": "7f01a2ba-aadc-47bf-b93e-27912a43e740",
                    "X-Application-Sid": "95d89ed8-a0cd-4f1a-881c-bdcbc855971a",
                    "P-Early-Media": "supported",
                    "Diversion": "<sip:+493041733972@twilio.com>;reason=unconditional",
                    "X-Twilio-AccountSid": "ACbc2d4fd426ce33de19d54bdcd6e41186",
                    "X-Twilio-CallSid": "CA378d270bea3f4b60856f5b80a83ee483",
                    "p-asserted-identity": "<sip:+491604697810@3U.net;user=phone>",
                },
                "raw": "INVITE sip:+493041733972@shailendra.sip.jambonz.cloud SIP/2.0\r\nVia: SIP/2.0/UDP 54.236.168.131;rport=5060;branch=z9hG4bKHg6B8e5638mXH;received=172.20.10.254\r\nMax-Forwards: 70\r\nFrom: <sip:+491604697810@54.236.168.131: 5060>;tag=cv3mj1eyZ7U4e\r\nTo: <sip:+493041733972@shailendra.sip.jambonz.cloud>\r\nCall-ID: 7e6e04fa-f451-123d-9aab-0e9252b57157\r\nCSeq: 88995033 INVITE\r\nContact: <sip: 172.20.10.254: 5060>\r\nUser-Agent: Twilio Gateway\r\nAllow: INVITE, ACK, BYE, CANCEL, OPTIONS, REFER, NOTIFY\r\nContent-Type: application/sdp\r\nContent-Length: 289\r\nX-Account-Sid: 55312bec-9135-4615-9f72-667e3719a911\r\nX-CID: 30cea8a53174afe5bb05348945e4a52a@0.0.0.0\r\nX-Forwarded-For: 35.156.191.129\r\nX-Originating-Carrier: Twilio-n7xbPvgiNLmPuFASP2Z2AX\r\nX-Voip-Carrier-Sid: 7f01a2ba-aadc-47bf-b93e-27912a43e740\r\nX-Application-Sid: 95d89ed8-a0cd-4f1a-881c-bdcbc855971a\r\nP-Early-Media: supported\r\nDiversion: <sip:+493041733972@twilio.com>;reason=unconditional\r\nX-Twilio-AccountSid: ACbc2d4fd426ce33de19d54bdcd6e41186\r\nX-Twilio-CallSid: CA378d270bea3f4b60856f5b80a83ee483\r\nP-Asserted-Identity: <sip:+491604697810@3U.net;user=phone>\r\n\r\nv=0\r\no=root 738528152 738528152 IN IP4 172.20.10.254\r\ns=Twilio Media Gateway\r\nc=IN IP4 172.20.10.254\r\nt=0 0\r\nm=audio 47412 RTP/AVP 0 8 101\r\na=maxptime: 20\r\na=rtpmap: 0 PCMU/8000\r\na=rtpmap: 8 PCMA/8000\r\na=rtpmap: 101 telephone-event/8000\r\na=fmtp: 101 0-16\r\na=sendrecv\r\na=rtcp: 47413\r\na=ptime: 20\r\n",  # noqa E501
                "body": "v=0\r\no=root 738528152 738528152 IN IP4 172.20.10.254\r\ns=Twilio Media Gateway\r\nc=IN IP4 172.20.10.254\r\nt=0 0\r\nm=audio 47412 RTP/AVP 0 8 101\r\na=maxptime: 20\r\na=rtpmap: 0 PCMU/8000\r\na=rtpmap: 8 PCMA/8000\r\na=rtpmap: 101 telephone-event/8000\r\na=fmtp: 101 0-16\r\na=sendrecv\r\na=rtcp: 47413\r\na=ptime: 20\r\n",  # noqa E501
                "method": "INVITE",
                "version": "2.0",
                "uri": "sip:+493041733972@shailendra.sip.jambonz.cloud",
                "payload": [
                    {
                        "type": "application/sdp",
                        "content": "v=0\r\no=root 738528152 738528152 IN IP4 172.20.10.254\r\ns=Twilio Media Gateway\r\nc=IN IP4 172.20.10.254\r\nt=0 0\r\nm=audio 47412 RTP/AVP 0 8 101\r\na=maxptime: 20\r\na=rtpmap: 0 PCMU/8000\r\na=rtpmap: 8 PCMA/8000\r\na=rtpmap: 101 telephone-event/8000\r\na=fmtp: 101 0-16\r\na=sendrecv\r\na=rtcp: 47413\r\na=ptime: 20\r\n",  # noqa E501
                    }
                ],
            },
            "direction": "inbound",
            "trace_id": "617072655b976fa722a26230ec86fd2b",
            "caller_name": "",
            "call_sid": "f31285ee-0bd4-4f1e-907d-f239bcbea5c1",
            "account_sid": "55312bec-9135-4615-9f72-667e3719a911",
            "application_sid": "95d89ed8-a0cd-4f1a-881c-bdcbc855971a",
            "from": "+491604697810",
            "to": "+493041733972",
            "call_id": "7e6e04fa-f451-123d-9aab-0e9252b57157",
            "sip_status": 100,
            "sip_reason": "Trying",
            "call_status": "trying",
            "originating_sip_ip": "35.156.191.129",
            "originating_sip_trunk_name": "Twilio-n7xbPvgiNLmPuFASP2Z2AX",
            "local_sip_address": "172.20.10.61: 5060",
            "public_ip": "34.207.132.111",
            "service_provider_sid": "6072f617-6151-4c15-8eb9-74a4967ab701",
            "defaults": {
                "synthesizer": {
                    "vendor": "deepgram",
                    "language": "en-US",
                    "voice": "aura-asteria-en",
                },
                "recognizer": {"vendor": "deepgram", "language": "en-US"},
            },
        },
        "b3": "617072655b976fa722a26230ec86fd2b-ba9124cce9093a40-1",
    }


@pytest.fixture
def call_status_message():
    return {
        "type": "call:status",
        "msgid": "f1mbETGkVDXJK4jhCBcmpY",
        "call_sid": "74dd6db2-7cb3-45c1-a575-d0935c55a219",
        "data": {
            "call_sid": "74dd6db2-7cb3-45c1-a575-d0935c55a219",
            "direction": "inbound",
            "from": "+491604697810",
            "to": "+493041733972",
            "call_id": "b5876a77-f453-123d-9aab-0e9252b57157",
            "sip_status": 200,
            "sip_reason": "OK",
            "call_status": "completed",
            "account_sid": "55312bec-9135-4615-9f72-667e3719a911",
            "trace_id": "e4f06a9d9c7bc293682a6c21a16e60aa",
            "application_sid": "95d89ed8-a0cd-4f1a-881c-bdcbc855971a",
            "fs_sip_address": "172.20.10.61:5060",
            "originating_sip_ip": "35.156.191.129",
            "originating_sip_trunk_name": "Twilio-n7xbPvgiNLmPuFASP2Z2AX",
            "call_termination_by": "caller",
            "duration": 3,
            "api_base_url": "http://jambonz.cloud/v1",
            "fs_public_ip": "34.207.132.111",
        },
        "b3": "e4f06a9d9c7bc293682a6c21a16e60aa-9b14e64931712301-1",
    }


def test_new_session_message(session_new_message):
    new_session_message = NewSessionMessage.from_message(session_new_message)

    assert new_session_message.call_sid == "f31285ee-0bd4-4f1e-907d-f239bcbea5c1"
    assert new_session_message.message_id == "heDUewwTZT51kXwauek3Zk"
    assert (
        new_session_message.call_params.call_id
        == "f31285ee-0bd4-4f1e-907d-f239bcbea5c1"
    )
    assert new_session_message.call_params.user_phone == "+491604697810"
    assert new_session_message.call_params.bot_phone == "+493041733972"


def test_call_status_changed(call_status_message):
    call_status = CallStatusChanged.from_message(call_status_message)

    assert call_status.call_sid == "74dd6db2-7cb3-45c1-a575-d0935c55a219"
    assert call_status.status == "completed"
