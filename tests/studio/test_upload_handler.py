import argparse
import base64
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from rasa.shared.exceptions import RasaException

import rasa.studio.upload
from rasa.studio.config import StudioConfig


def base64_calm_domain_yaml(calm_domain_yaml):
    return base64.b64encode(calm_domain_yaml.encode("utf-8")).decode("utf-8")


def base64_calm_flows_yaml(calm_flows_yaml):
    return base64.b64encode(calm_flows_yaml.encode("utf-8")).decode("utf-8")


@pytest.mark.parametrize(
    "args, endpoint, expected",
    [
        (
            argparse.Namespace(
                assistant_name=["test"],
                domain="data/upload/domain.yml",
                data=["data/upload/data/nlu.yml"],
                entities=["name"],
                intents=["greet", "inform"],
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation ImportFromEncodedYaml"
                    "($input: ImportFromEncodedYamlInput!)"
                    "{\n  importFromEncodedYaml(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": (
                            "dmVyc2lvbjogJzMuMScKaW50ZW50czoKLSBncmVldAotIGluZm9ybQplbn"
                            "RpdGllczoKLSBuYW1lOgogICAgcm9sZXM6CiAgICAtIGZpcnN0X25hbWUK"
                            "ICAgIC0gbGFzdF9uYW1lCi0gYWdlCg=="
                        ),
                        "nlu": (
                            "dmVyc2lvbjogIjMuMSIKbmx1OgotIGludGVudDogZ3JlZXQKICBleGFtcGxlc"
                            "zogfAogICAgLSBoZXkKICAgIC0gaGVsbG8KICAgIC0gaGkKICAgIC0gaGVsbG8"
                            "gdGhlcmUKICAgIC0gZ29vZCBtb3JuaW5nCiAgICAtIGdvb2QgZXZlbmluZwogI"
                            "CAgLSBtb2luCiAgICAtIGhleSB0aGVyZQogICAgLSBsZXQncyBnbwogICAgLSB"
                            "oZXkgZHVkZQogICAgLSBnb29kbW9ybmluZwogICAgLSBnb29kZXZlbmluZwogI"
                            "CAgLSBnb29kIGFmdGVybm9vbgotIGludGVudDogaW5mb3JtCiAgZXhhbXBsZXM"
                            "6IHwKICAgIC0gbXkgbmFtZSBpcyBbVXJvc117ImVudGl0eSI6ICJuYW1lIiwgI"
                            "nJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEknbSBbSm9obl17ImVudGl0eSI"
                            "6ICJuYW1lIiwgInJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEhpLCBteSBma"
                            "XJzdCBuYW1lIGlzIFtMdWlzXXsiZW50aXR5IjogIm5hbWUiLCAicm9sZSI6ICJ"
                            "maXJzdF9uYW1lIn0KICAgIC0gTWlsaWNhCiAgICAtIEthcmluCiAgICAtIFN0Z"
                            "XZlbgogICAgLSBJJ20gWzE4XShhZ2UpCiAgICAtIEkgYW0gWzMyXShhZ2UpIHl"
                            "lYXJzIG9sZAogICAgLSA5Cg=="
                        ),
                    }
                },
            },
        ),
        (
            argparse.Namespace(
                assistant_name=["test"],
                calm=True,
                domain="data/upload/calm/domain/domain.yml",
                data=["data/upload/calm/"],
                config="data/upload/calm/config.yml",
                flows="data/upload/flows.yml",
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation UploadModernAssistant"
                    "($input: UploadModernAssistantInput!)"
                    "{\n  uploadModernAssistant(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": (
                            "dmVyc2lvbjogJzMuMScKYWN0aW9uczoKLSBhY3Rpb25fYWRkX2NvbnRhY3QKLSBhY3Rpb25fY2hl"
                            "Y2tfYmFsYW5jZQotIGFjdGlvbl9leGVjdXRlX3RyYW5zZmVyCi0gYWN0aW9uX3JlbW92ZV9jb250"
                            "YWN0Ci0gYWN0aW9uX3NlYXJjaF9ob3RlbAotIGFjdGlvbl90cmFuc2FjdGlvbl9zZWFyY2gKLSBh"
                            "Y3Rpb25fY2hlY2tfdHJhbnNmZXJfZnVuZHMKcmVzcG9uc2VzOgogIHV0dGVyX2FkZF9jb250YWN0"
                            "X2NhbmNlbGxlZDoKICAtIHRleHQ6IE9rYXksIEkgYW0gY2FuY2VsbGluZyB0aGlzIGFkZGluZyBv"
                            "ZiBhIGNvbnRhY3QuCiAgdXR0ZXJfYWRkX2NvbnRhY3RfZXJyb3I6CiAgLSB0ZXh0OiBTb21ldGhp"
                            "bmcgd2VudCB3cm9uZywgcGxlYXNlIHRyeSBhZ2Fpbi4KICB1dHRlcl9jYV9pbmNvbWVfaW5zdWZm"
                            "aWNpZW50OgogIC0gdGV4dDogVW5mb3J0dW5hdGVseSwgd2UgY2Fubm90IGluY3JlYXNlIHlvdXIg"
                            "dHJhbnNmZXIgbGltaXRzIHVuZGVyIHRoZXNlIGNpcmN1bXN0YW5jZXMuCiAgdXR0ZXJfY2FudF9h"
                            "ZHZpY2Vfb25faGVhbHRoOgogIC0gdGV4dDogSSdtIHNvcnJ5LCBJIGNhbid0IGdpdmUgeW91IGFk"
                            "dmljZSBvbiB5b3VyIGhlYWx0aC4KICB1dHRlcl9jb250YWN0X2FkZGVkOgogIC0gdGV4dDogQ29u"
                            "dGFjdCBhZGRlZCBzdWNjZXNzZnVsbHkuCiAgdXR0ZXJfY29udGFjdF9hbHJlYWR5X2V4aXN0czoK"
                            "ICAtIHRleHQ6IFRoZXJlJ3MgYWxyZWFkeSBhIGNvbnRhY3Qgd2l0aCB0aGF0IGhhbmRsZSBpbiB5"
                            "b3VyIGxpc3QuCiAgdXR0ZXJfY29udGFjdF9ub3RfaW5fbGlzdDoKICAtIHRleHQ6IFRoYXQgY29u"
                            "dGFjdCBpcyBub3QgaW4geW91ciBsaXN0LgogIHV0dGVyX2N1cnJlbnRfYmFsYW5jZToKICAtIHRl"
                            "eHQ6IFlvdSBzdGlsbCBoYXZlIHtjdXJyZW50X2JhbGFuY2V9IGluIHlvdXIgYWNjb3VudC4KICB1"
                            "dHRlcl9ob3RlbF9pbmZvcm1fcmF0aW5nOgogIC0gdGV4dDogVGhlIHtob3RlbF9uYW1lfSBoYXMg"
                            "YW4gYXZlcmFnZSByYXRpbmcgb2Yge2hvdGVsX2F2ZXJhZ2VfcmF0aW5nfQogIHV0dGVyX3JlbW92"
                            "ZV9jb250YWN0X2NhbmNlbGxlZDoKICAtIHRleHQ6IE9rYXksIEkgYW0gY2FuY2VsbGluZyB0aGlz"
                            "IHJlbW92YWwgb2YgYSBjb250YWN0LgogIHV0dGVyX3JlbW92ZV9jb250YWN0X2Vycm9yOgogIC0g"
                            "dGV4dDogU29tZXRoaW5nIHdlbnQgd3JvbmcsIHBsZWFzZSB0cnkgYWdhaW4uCiAgdXR0ZXJfcmVt"
                            "b3ZlX2NvbnRhY3Rfc3VjY2VzczoKICAtIHRleHQ6IFJlbW92ZWQge3JlbW92ZV9jb250YWN0X2hh"
                            "bmRsZX0oe3JlbW92ZV9jb250YWN0X25hbWV9KSBmcm9tIHlvdXIgY29udGFjdHMuCiAgdXR0ZXJf"
                            "dHJhbnNhY3Rpb25zOgogIC0gdGV4dDogJ1lvdXIgY3VycmVudCB0cmFuc2FjdGlvbnMgYXJlOiAg"
                            "e3RyYW5zYWN0aW9uc19saXN0fScKICB1dHRlcl90cmFuc2Zlcl9jYW5jZWxsZWQ6CiAgLSB0ZXh0"
                            "OiBUcmFuc2ZlciBjYW5jZWxsZWQuCiAgdXR0ZXJfdHJhbnNmZXJfY29tcGxldGU6CiAgLSB0ZXh0"
                            "OiBTdWNjZXNzZnVsbHkgdHJhbnNmZXJyZWQge3RyYW5zZmVyX21vbmV5X2Ftb3VudH0gdG8ge3Ry"
                            "YW5zZmVyX21vbmV5X3JlY2lwaWVudH0uCiAgdXR0ZXJfdHJhbnNmZXJfZmFpbGVkOgogIC0gdGV4"
                            "dDogc29tZXRoaW5nIHdlbnQgd3JvbmcgdHJhbnNmZXJyaW5nIHRoZSBtb25leS4KICB1dHRlcl92"
                            "ZXJpZnlfYWNjb3VudF9jYW5jZWxsZWQ6CiAgLSB0ZXh0OiBDYW5jZWxsaW5nIGFjY291bnQgdmVy"
                            "aWZpY2F0aW9uLi4uCiAgICBtZXRhZGF0YToKICAgICAgcmVwaHJhc2U6IHRydWUKICB1dHRlcl92"
                            "ZXJpZnlfYWNjb3VudF9zdWNjZXNzOgogIC0gdGV4dDogWW91ciBhY2NvdW50IHdhcyBzdWNjZXNz"
                            "ZnVsbHkgdmVyaWZpZWQKICB1dHRlcl9hc2tfYWRkX2NvbnRhY3RfY29uZmlybWF0aW9uOgogIC0g"
                            "dGV4dDogRG8geW91IHdhbnQgdG8gYWRkIHthZGRfY29udGFjdF9uYW1lfSh7YWRkX2NvbnRhY3Rf"
                            "aGFuZGxlfSkgdG8geW91ciBjb250YWN0cz8KICB1dHRlcl9hc2tfYWRkX2NvbnRhY3RfaGFuZGxl"
                            "OgogIC0gdGV4dDogV2hhdCdzIHRoZSBoYW5kbGUgb2YgdGhlIHVzZXIgeW91IHdhbnQgdG8gYWRk"
                            "PwogIHV0dGVyX2Fza19hZGRfY29udGFjdF9uYW1lOgogIC0gdGV4dDogV2hhdCdzIHRoZSBuYW1l"
                            "IG9mIHRoZSB1c2VyIHlvdSB3YW50IHRvIGFkZD8KICAgIG1ldGFkYXRhOgogICAgICByZXBocmFz"
                            "ZTogdHJ1ZQogIHV0dGVyX2Fza19iYXNlZF9pbl9jYWxpZm9ybmlhOgogIC0gdGV4dDogQXJlIHlv"
                            "dSBiYXNlZCBpbiBDYWxpZm9ybmlhPwogICAgYnV0dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAg"
                            "ICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5vCiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJf"
                            "YXNrX3JlbW92ZV9jb250YWN0X2NvbmZpcm1hdGlvbjoKICAtIHRleHQ6IFNob3VsZCBJIHJlbW92"
                            "ZSB7cmVtb3ZlX2NvbnRhY3RfaGFuZGxlfSBmcm9tIHlvdXIgY29udGFjdCBsaXN0PwogICAgYnV0"
                            "dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAgICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5v"
                            "CiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJfYXNrX3JlbW92ZV9jb250YWN0X2hhbmRsZToKICAt"
                            "IHRleHQ6IFdoYXQncyB0aGUgaGFuZGxlIG9mIHRoZSB1c2VyIHlvdSB3YW50IHRvIHJlbW92ZT8K"
                            "ICB1dHRlcl9hc2tfdHJhbnNmZXJfbW9uZXlfYW1vdW50OgogIC0gdGV4dDogSG93IG11Y2ggbW9u"
                            "ZXkgZG8geW91IHdhbnQgdG8gdHJhbnNmZXI/CiAgdXR0ZXJfYXNrX3RyYW5zZmVyX21vbmV5X2Zp"
                            "bmFsX2NvbmZpcm1hdGlvbjoKICAtIHRleHQ6IFdvdWxkIHlvdSBsaWtlIHRvIHRyYW5zZmVyIHt0"
                            "cmFuc2Zlcl9tb25leV9hbW91bnR9IHRvIHt0cmFuc2Zlcl9tb25leV9yZWNpcGllbnR9PwogICAg"
                            "YnV0dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAgICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6"
                            "IE5vCiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJfYXNrX3RyYW5zZmVyX21vbmV5X3JlY2lwaWVu"
                            "dDoKICAtIHRleHQ6IFdobyBkbyB5b3Ugd2FudCB0byB0cmFuc2ZlciBtb25leSB0bz8KICB1dHRl"
                            "cl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uOgogIC0gdGV4dDogWW91ciBlbWFpbCBh"
                            "ZGRyZXNzIGlzIHt2ZXJpZnlfYWNjb3VudF9lbWFpbH0gYW5kIHlvdSBhcmUgbm90IGJhc2VkIGlu"
                            "IENhbGlmb3JuaWEsIGNvcnJlY3Q/CiAgICBidXR0b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAg"
                            "IHBheWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAgICAgcGF5bG9hZDogTm8KICB1dHRlcl9h"
                            "c2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3JuaWE6CiAgLSB0ZXh0OiBZb3Vy"
                            "IGVtYWlsIGFkZHJlc3MgaXMge3ZlcmlmeV9hY2NvdW50X2VtYWlsfSBhbmQgeW91IGFyZSBiYXNl"
                            "ZCBpbiBDYWxpZm9ybmlhIHdpdGggYSB5ZWFybHkgaW5jb21lIGV4Y2VlZGluZyAxMDAsMDAwJCwg"
                            "Y29ycmVjdD8KICAgIGJ1dHRvbnM6CiAgICAtIHRpdGxlOiBZZXMKICAgICAgcGF5bG9hZDogWWVz"
                            "CiAgICAtIHRpdGxlOiBObwogICAgICBwYXlsb2FkOiBObwogIHV0dGVyX2Fza192ZXJpZnlfYWNj"
                            "b3VudF9lbWFpbDoKICAtIHRleHQ6IFdoYXQncyB5b3VyIGVtYWlsIGFkZHJlc3M/CiAgdXR0ZXJf"
                            "YXNrX3ZlcmlmeV9hY2NvdW50X3N1ZmZpY2llbnRfY2FsaWZvcm5pYV9pbmNvbWU6CiAgLSB0ZXh0"
                            "OiBEb2VzIHlvdXIgeWVhcmx5IGluY29tZSBleGNlZWQgMTAwLDAwMCBVU0Q/CiAgICBidXR0b25z"
                            "OgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBheWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAg"
                            "ICAgcGF5bG9hZDogTm8Kc2xvdHM6CiAgYWRkX2NvbnRhY3RfY29uZmlybWF0aW9uOgogICAgdHlw"
                            "ZTogYm9vbAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIGFkZF9jb250YWN0X2hh"
                            "bmRsZToKICAgIHR5cGU6IHRleHQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICBh"
                            "ZGRfY29udGFjdF9uYW1lOgogICAgdHlwZTogdGV4dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6"
                            "IGN1c3RvbQogIGJhc2VkX2luX2NhbGlmb3JuaWE6CiAgICB0eXBlOiBib29sCiAgICBtYXBwaW5n"
                            "czoKICAgIC0gdHlwZTogY3VzdG9tCiAgY3VycmVudF9iYWxhbmNlOgogICAgdHlwZTogZmxvYXQK"
                            "ICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICBob3RlbF9hdmVyYWdlX3JhdGluZzoK"
                            "ICAgIHR5cGU6IGZsb2F0CiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgaG90ZWxf"
                            "bmFtZToKICAgIHR5cGU6IHRleHQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICBy"
                            "ZW1vdmVfY29udGFjdF9jb25maXJtYXRpb246CiAgICB0eXBlOiBib29sCiAgICBtYXBwaW5nczoK"
                            "ICAgIC0gdHlwZTogY3VzdG9tCiAgcmVtb3ZlX2NvbnRhY3RfaGFuZGxlOgogICAgdHlwZTogdGV4"
                            "dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIHJlbW92ZV9jb250YWN0X25hbWU6"
                            "CiAgICB0eXBlOiB0ZXh0CiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgcmV0dXJu"
                            "X3ZhbHVlOgogICAgdHlwZTogdGV4dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQog"
                            "IHRyYW5zYWN0aW9uc19saXN0OgogICAgdHlwZTogdGV4dAogICAgbWFwcGluZ3M6CiAgICAtIHR5"
                            "cGU6IGN1c3RvbQogIHRyYW5zZmVyX21vbmV5X2Ftb3VudDoKICAgIHR5cGU6IGZsb2F0CiAgICBt"
                            "YXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgdHJhbnNmZXJfbW9uZXlfZmluYWxfY29uZmly"
                            "bWF0aW9uOgogICAgdHlwZTogYm9vbAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQog"
                            "IHRyYW5zZmVyX21vbmV5X3JlY2lwaWVudDoKICAgIHR5cGU6IHRleHQKICAgIG1hcHBpbmdzOgog"
                            "ICAgLSB0eXBlOiBjdXN0b20KICB0cmFuc2Zlcl9tb25leV90cmFuc2Zlcl9zdWNjZXNzZnVsOgog"
                            "ICAgdHlwZTogYm9vbAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIHNldF9zbG90"
                            "c190ZXN0X3RleHQ6CiAgICB0eXBlOiB0ZXh0CiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3Vz"
                            "dG9tCiAgc2V0X3Nsb3RzX3Rlc3RfY2F0ZWdvcmljYWw6CiAgICB0eXBlOiBjYXRlZ29yaWNhbAog"
                            "ICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogICAgdmFsdWVzOgogICAgLSB2YWx1ZV8x"
                            "CiAgICAtIHZhbHVlXzIKICB0cmFuc2Zlcl9tb25leV9oYXNfc3VmZmljaWVudF9mdW5kczoKICAg"
                            "IHR5cGU6IGJvb2wKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICB2ZXJpZnlfYWNj"
                            "b3VudF9jb25maXJtYXRpb246CiAgICB0eXBlOiBib29sCiAgICBtYXBwaW5nczoKICAgIC0gdHlw"
                            "ZTogY3VzdG9tCiAgdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3JuaWE6CiAgICB0"
                            "eXBlOiBib29sCiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgdmVyaWZ5X2FjY291"
                            "bnRfZW1haWw6CiAgICB0eXBlOiB0ZXh0CiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9t"
                            "CiAgdmVyaWZ5X2FjY291bnRfc3VmZmljaWVudF9jYWxpZm9ybmlhX2luY29tZToKICAgIHR5cGU6"
                            "IGJvb2wKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KaW50ZW50czoKLSBoZWFsdGhf"
                            "YWR2aWNlCnNlc3Npb25fY29uZmlnOgogIHNlc3Npb25fZXhwaXJhdGlvbl90aW1lOiA2MAogIGNh"
                            "cnJ5X292ZXJfc2xvdHNfdG9fbmV3X3Nlc3Npb246IHRydWUK"
                        ),
                        "flows": (
                            "Zmxvd3M6CiAgaGVhbHRoX2FkdmljZToKICAgIHN0ZXBzOgogICAgLSBpZDogMF91dHRlcl9jYW50"
                            "X2FkdmljZV9vbl9oZWFsdGgKICAgICAgbmV4dDogRU5ECiAgICAgIGFjdGlvbjogdXR0ZXJfY2Fu"
                            "dF9hZHZpY2Vfb25faGVhbHRoCiAgICBuYW1lOiBoZWFsdGggYWR2aWNlCiAgICBkZXNjcmlwdGlv"
                            "bjogdXNlciBhc2tzIGZvciBoZWFsdGggYWR2aWNlCiAgICBubHVfdHJpZ2dlcjoKICAgIC0gaW50"
                            "ZW50OgogICAgICAgIG5hbWU6IGhlYWx0aF9hZHZpY2UKICAgICAgICBjb25maWRlbmNlX3RocmVz"
                            "aG9sZDogMC44CiAgYWRkX2NvbnRhY3Q6CiAgICBzdGVwczoKICAgIC0gaWQ6IDBfY29sbGVjdF9h"
                            "ZGRfY29udGFjdF9oYW5kbGUKICAgICAgbmV4dDogMV9jb2xsZWN0X2FkZF9jb250YWN0X25hbWUK"
                            "ICAgICAgZGVzY3JpcHRpb246IGEgdXNlciBoYW5kbGUgc3RhcnRpbmcgd2l0aCBACiAgICAgIGNv"
                            "bGxlY3Q6IGFkZF9jb250YWN0X2hhbmRsZQogICAgICB1dHRlcjogdXR0ZXJfYXNrX2FkZF9jb250"
                            "YWN0X2hhbmRsZQogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IGZhbHNlCiAgICAgIHJlc2V0X2Fm"
                            "dGVyX2Zsb3dfZW5kczogdHJ1ZQogICAgICByZWplY3Rpb25zOiBbXQogICAgLSBpZDogMV9jb2xs"
                            "ZWN0X2FkZF9jb250YWN0X25hbWUKICAgICAgbmV4dDogMl9jb2xsZWN0X2FkZF9jb250YWN0X2Nv"
                            "bmZpcm1hdGlvbgogICAgICBkZXNjcmlwdGlvbjogYSBuYW1lIG9mIGEgcGVyc29uCiAgICAgIGNv"
                            "bGxlY3Q6IGFkZF9jb250YWN0X25hbWUKICAgICAgdXR0ZXI6IHV0dGVyX2Fza19hZGRfY29udGFj"
                            "dF9uYW1lCiAgICAgIGFza19iZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVzZXRfYWZ0ZXJf"
                            "Zmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAyX2NvbGxlY3Rf"
                            "YWRkX2NvbnRhY3RfY29uZmlybWF0aW9uCiAgICAgIG5leHQ6CiAgICAgIC0gaWY6IG5vdCBzbG90"
                            "cy5hZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgICB0aGVuOgogICAgICAgIC0gaWQ6IDNf"
                            "dXR0ZXJfYWRkX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAgICAgICBuZXh0OiBFTkQKICAgICAgICAg"
                            "IGFjdGlvbjogdXR0ZXJfYWRkX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAgIC0gZWxzZTogYWN0aW9u"
                            "X2FkZF9jb250YWN0CiAgICAgIGRlc2NyaXB0aW9uOiBhIGNvbmZpcm1hdGlvbiB0byBhZGQgY29u"
                            "dGFjdAogICAgICBjb2xsZWN0OiBhZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgdXR0ZXI6"
                            "IHV0dGVyX2Fza19hZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgYXNrX2JlZm9yZV9maWxs"
                            "aW5nOiBmYWxzZQogICAgICByZXNldF9hZnRlcl9mbG93X2VuZHM6IHRydWUKICAgICAgcmVqZWN0"
                            "aW9uczogW10KICAgIC0gaWQ6IGFjdGlvbl9hZGRfY29udGFjdAogICAgICBuZXh0OgogICAgICAt"
                            "IGlmOiBzbG90cy5yZXR1cm5fdmFsdWUgaXMgJ2FscmVhZHlfZXhpc3RzJwogICAgICAgIHRoZW46"
                            "CiAgICAgICAgLSBpZDogNV91dHRlcl9jb250YWN0X2FscmVhZHlfZXhpc3RzCiAgICAgICAgICBu"
                            "ZXh0OiBFTkQKICAgICAgICAgIGFjdGlvbjogdXR0ZXJfY29udGFjdF9hbHJlYWR5X2V4aXN0cwog"
                            "ICAgICAtIGlmOiBzbG90cy5yZXR1cm5fdmFsdWUgaXMgJ3N1Y2Nlc3MnCiAgICAgICAgdGhlbjoK"
                            "ICAgICAgICAtIGlkOiA2X3V0dGVyX2NvbnRhY3RfYWRkZWQKICAgICAgICAgIG5leHQ6IEVORAog"
                            "ICAgICAgICAgYWN0aW9uOiB1dHRlcl9jb250YWN0X2FkZGVkCiAgICAgIC0gZWxzZToKICAgICAg"
                            "ICAtIGlkOiA3X3V0dGVyX2FkZF9jb250YWN0X2Vycm9yCiAgICAgICAgICBuZXh0OiBFTkQKICAg"
                            "ICAgICAgIGFjdGlvbjogdXR0ZXJfYWRkX2NvbnRhY3RfZXJyb3IKICAgICAgYWN0aW9uOiBhY3Rp"
                            "b25fYWRkX2NvbnRhY3QKICAgIG5hbWU6IGFkZF9jb250YWN0CiAgICBkZXNjcmlwdGlvbjogYWRk"
                            "IGEgY29udGFjdCB0byB5b3VyIGNvbnRhY3QgbGlzdAogIGNoZWNrX2JhbGFuY2U6CiAgICBzdGVw"
                            "czoKICAgIC0gaWQ6IDBfYWN0aW9uX2NoZWNrX2JhbGFuY2UKICAgICAgbmV4dDogMV91dHRlcl9j"
                            "dXJyZW50X2JhbGFuY2UKICAgICAgYWN0aW9uOiBhY3Rpb25fY2hlY2tfYmFsYW5jZQogICAgLSBp"
                            "ZDogMV91dHRlcl9jdXJyZW50X2JhbGFuY2UKICAgICAgbmV4dDogRU5ECiAgICAgIGFjdGlvbjog"
                            "dXR0ZXJfY3VycmVudF9iYWxhbmNlCiAgICBuYW1lOiBjaGVja19iYWxhbmNlCiAgICBkZXNjcmlw"
                            "dGlvbjogY2hlY2sgdGhlIHVzZXIncyBhY2NvdW50IGJhbGFuY2UuCiAgaG90ZWxfc2VhcmNoOgog"
                            "ICAgc3RlcHM6CiAgICAtIGlkOiAwX2FjdGlvbl9zZWFyY2hfaG90ZWwKICAgICAgbmV4dDogMV91"
                            "dHRlcl9ob3RlbF9pbmZvcm1fcmF0aW5nCiAgICAgIGFjdGlvbjogYWN0aW9uX3NlYXJjaF9ob3Rl"
                            "bAogICAgLSBpZDogMV91dHRlcl9ob3RlbF9pbmZvcm1fcmF0aW5nCiAgICAgIG5leHQ6IEVORAog"
                            "ICAgICBhY3Rpb246IHV0dGVyX2hvdGVsX2luZm9ybV9yYXRpbmcKICAgIG5hbWU6IGhvdGVsX3Nl"
                            "YXJjaAogICAgZGVzY3JpcHRpb246IHNlYXJjaCBmb3IgaG90ZWxzCiAgcmVtb3ZlX2NvbnRhY3Q6"
                            "CiAgICBzdGVwczoKICAgIC0gaWQ6IDBfY29sbGVjdF9yZW1vdmVfY29udGFjdF9oYW5kbGUKICAg"
                            "ICAgbmV4dDogMV9jb2xsZWN0X3JlbW92ZV9jb250YWN0X2NvbmZpcm1hdGlvbgogICAgICBkZXNj"
                            "cmlwdGlvbjogYSBjb250YWN0IGhhbmRsZSBzdGFydGluZyB3aXRoIEAKICAgICAgY29sbGVjdDog"
                            "cmVtb3ZlX2NvbnRhY3RfaGFuZGxlCiAgICAgIHV0dGVyOiB1dHRlcl9hc2tfcmVtb3ZlX2NvbnRh"
                            "Y3RfaGFuZGxlCiAgICAgIGFza19iZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVzZXRfYWZ0"
                            "ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAxX2NvbGxl"
                            "Y3RfcmVtb3ZlX2NvbnRhY3RfY29uZmlybWF0aW9uCiAgICAgIG5leHQ6CiAgICAgIC0gaWY6IG5v"
                            "dCBzbG90cy5yZW1vdmVfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgICB0aGVuOgogICAgICAg"
                            "IC0gaWQ6IDJfdXR0ZXJfcmVtb3ZlX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAgICAgICBuZXh0OiBF"
                            "TkQKICAgICAgICAgIGFjdGlvbjogdXR0ZXJfcmVtb3ZlX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAg"
                            "IC0gZWxzZTogYWN0aW9uX3JlbW92ZV9jb250YWN0CiAgICAgIGNvbGxlY3Q6IHJlbW92ZV9jb250"
                            "YWN0X2NvbmZpcm1hdGlvbgogICAgICB1dHRlcjogdXR0ZXJfYXNrX3JlbW92ZV9jb250YWN0X2Nv"
                            "bmZpcm1hdGlvbgogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgcmVzZXRfYWZ0"
                            "ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiBhY3Rpb25f"
                            "cmVtb3ZlX2NvbnRhY3QKICAgICAgbmV4dDoKICAgICAgLSBpZjogc2xvdHMucmV0dXJuX3ZhbHVl"
                            "IGlzICdub3RfZm91bmQnCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlkOiA0X3V0dGVyX2NvbnRh"
                            "Y3Rfbm90X2luX2xpc3QKICAgICAgICAgIG5leHQ6IEVORAogICAgICAgICAgYWN0aW9uOiB1dHRl"
                            "cl9jb250YWN0X25vdF9pbl9saXN0CiAgICAgIC0gaWY6IHNsb3RzLnJldHVybl92YWx1ZSBpcyAn"
                            "c3VjY2VzcycKICAgICAgICB0aGVuOgogICAgICAgIC0gaWQ6IDVfdXR0ZXJfcmVtb3ZlX2NvbnRh"
                            "Y3Rfc3VjY2VzcwogICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICBhY3Rpb246IHV0dGVyX3Jl"
                            "bW92ZV9jb250YWN0X3N1Y2Nlc3MKICAgICAgLSBlbHNlOgogICAgICAgIC0gaWQ6IDZfdXR0ZXJf"
                            "cmVtb3ZlX2NvbnRhY3RfZXJyb3IKICAgICAgICAgIG5leHQ6IEVORAogICAgICAgICAgYWN0aW9u"
                            "OiB1dHRlcl9yZW1vdmVfY29udGFjdF9lcnJvcgogICAgICBhY3Rpb246IGFjdGlvbl9yZW1vdmVf"
                            "Y29udGFjdAogICAgbmFtZTogcmVtb3ZlX2NvbnRhY3QKICAgIGRlc2NyaXB0aW9uOiByZW1vdmUg"
                            "YSBjb250YWN0IGZyb20geW91ciBjb250YWN0IGxpc3QKICB0cmFuc2FjdGlvbl9zZWFyY2g6CiAg"
                            "ICBzdGVwczoKICAgIC0gaWQ6IDBfYWN0aW9uX3RyYW5zYWN0aW9uX3NlYXJjaAogICAgICBuZXh0"
                            "OiAxX3V0dGVyX3RyYW5zYWN0aW9ucwogICAgICBhY3Rpb246IGFjdGlvbl90cmFuc2FjdGlvbl9z"
                            "ZWFyY2gKICAgIC0gaWQ6IDFfdXR0ZXJfdHJhbnNhY3Rpb25zCiAgICAgIG5leHQ6IEVORAogICAg"
                            "ICBhY3Rpb246IHV0dGVyX3RyYW5zYWN0aW9ucwogICAgbmFtZTogdHJhbnNhY3Rpb25fc2VhcmNo"
                            "CiAgICBkZXNjcmlwdGlvbjogbGlzdHMgdGhlIGxhc3QgdHJhbnNhY3Rpb25zIG9mIHRoZSB1c2Vy"
                            "IGFjY291bnQKICB0cmFuc2Zlcl9tb25leToKICAgIHN0ZXBzOgogICAgLSBpZDogMF9jb2xsZWN0"
                            "X3RyYW5zZmVyX21vbmV5X3JlY2lwaWVudAogICAgICBuZXh0OiAxX2NvbGxlY3RfdHJhbnNmZXJf"
                            "bW9uZXlfYW1vdW50CiAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVzZXIgZm9yIHRoZSByZWNpcGll"
                            "bnQncyBuYW1lLgogICAgICBjb2xsZWN0OiB0cmFuc2Zlcl9tb25leV9yZWNpcGllbnQKICAgICAg"
                            "dXR0ZXI6IHV0dGVyX2Fza190cmFuc2Zlcl9tb25leV9yZWNpcGllbnQKICAgICAgYXNrX2JlZm9y"
                            "ZV9maWxsaW5nOiBmYWxzZQogICAgICByZXNldF9hZnRlcl9mbG93X2VuZHM6IHRydWUKICAgICAg"
                            "cmVqZWN0aW9uczogW10KICAgIC0gaWQ6IDFfY29sbGVjdF90cmFuc2Zlcl9tb25leV9hbW91bnQK"
                            "ICAgICAgbmV4dDogMl9hY3Rpb25fY2hlY2tfdHJhbnNmZXJfZnVuZHMKICAgICAgZGVzY3JpcHRp"
                            "b246IEFza3MgdXNlciBmb3IgdGhlIGFtb3VudCB0byB0cmFuc2Zlci4KICAgICAgY29sbGVjdDog"
                            "dHJhbnNmZXJfbW9uZXlfYW1vdW50CiAgICAgIHV0dGVyOiB1dHRlcl9hc2tfdHJhbnNmZXJfbW9u"
                            "ZXlfYW1vdW50CiAgICAgIGFza19iZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVzZXRfYWZ0"
                            "ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAyX2FjdGlv"
                            "bl9jaGVja190cmFuc2Zlcl9mdW5kcwogICAgICBuZXh0OgogICAgICAtIGlmOiBub3Qgc2xvdHMu"
                            "dHJhbnNmZXJfbW9uZXlfaGFzX3N1ZmZpY2llbnRfZnVuZHMKICAgICAgICB0aGVuOgogICAgICAg"
                            "IC0gaWQ6IDNfdXR0ZXJfdHJhbnNmZXJfbW9uZXlfaW5zdWZmaWNpZW50X2Z1bmRzCiAgICAgICAg"
                            "ICBuZXh0OiA0X3NldF9zbG90cwogICAgICAgICAgYWN0aW9uOiB1dHRlcl90cmFuc2Zlcl9tb25l"
                            "eV9pbnN1ZmZpY2llbnRfZnVuZHMKICAgICAgICAtIGlkOiA0X3NldF9zbG90cwogICAgICAgICAg"
                            "bmV4dDogRU5ECiAgICAgICAgICBzZXRfc2xvdHM6CiAgICAgICAgICAtIHRyYW5zZmVyX21vbmV5"
                            "X2Ftb3VudDogbnVsbAogICAgICAgICAgLSB0cmFuc2Zlcl9tb25leV9oYXNfc3VmZmljaWVudF9m"
                            "dW5kczogbnVsbAogICAgICAgICAgLSBzZXRfc2xvdHNfdGVzdF90ZXh0OiBUaGlzIGlzIGEgdGVz"
                            "dCEKICAgICAgICAgIC0gc2V0X3Nsb3RzX3Rlc3RfY2F0ZWdvcmljYWw6IHZhbHVlXzEKICAgICAg"
                            "LSBlbHNlOiBjb2xsZWN0X3RyYW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbgogICAgICBh"
                            "Y3Rpb246IGFjdGlvbl9jaGVja190cmFuc2Zlcl9mdW5kcwogICAgLSBpZDogY29sbGVjdF90cmFu"
                            "c2Zlcl9tb25leV9maW5hbF9jb25maXJtYXRpb24KICAgICAgbmV4dDoKICAgICAgLSBpZjogbm90"
                            "IHNsb3RzLnRyYW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbgogICAgICAgIHRoZW46CiAg"
                            "ICAgICAgLSBpZDogNl91dHRlcl90cmFuc2Zlcl9jYW5jZWxsZWQKICAgICAgICAgIG5leHQ6IEVO"
                            "RAogICAgICAgICAgYWN0aW9uOiB1dHRlcl90cmFuc2Zlcl9jYW5jZWxsZWQKICAgICAgLSBlbHNl"
                            "OiBhY3Rpb25fZXhlY3V0ZV90cmFuc2ZlcgogICAgICBkZXNjcmlwdGlvbjogQXNrcyB1c2VyIGZv"
                            "ciBmaW5hbCBjb25maXJtYXRpb24gdG8gdHJhbnNmZXIgbW9uZXkuCiAgICAgIGNvbGxlY3Q6IHRy"
                            "YW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbgogICAgICB1dHRlcjogdXR0ZXJfYXNrX3Ry"
                            "YW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbgogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6"
                            "IHRydWUKICAgICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6"
                            "IFtdCiAgICAtIGlkOiBhY3Rpb25fZXhlY3V0ZV90cmFuc2ZlcgogICAgICBuZXh0OgogICAgICAt"
                            "IGlmOiBzbG90cy50cmFuc2Zlcl9tb25leV90cmFuc2Zlcl9zdWNjZXNzZnVsCiAgICAgICAgdGhl"
                            "bjoKICAgICAgICAtIGlkOiA4X3V0dGVyX3RyYW5zZmVyX2NvbXBsZXRlCiAgICAgICAgICBuZXh0"
                            "OiBFTkQKICAgICAgICAgIGFjdGlvbjogdXR0ZXJfdHJhbnNmZXJfY29tcGxldGUKICAgICAgLSBl"
                            "bHNlOgogICAgICAgIC0gaWQ6IDlfdXR0ZXJfdHJhbnNmZXJfZmFpbGVkCiAgICAgICAgICBuZXh0"
                            "OiBFTkQKICAgICAgICAgIGFjdGlvbjogdXR0ZXJfdHJhbnNmZXJfZmFpbGVkCiAgICAgIGFjdGlv"
                            "bjogYWN0aW9uX2V4ZWN1dGVfdHJhbnNmZXIKICAgIG5hbWU6IHRyYW5zZmVyX21vbmV5CiAgICBk"
                            "ZXNjcmlwdGlvbjogVGhpcyBmbG93IGxldCdzIHVzZXJzIHNlbmQgbW9uZXkgdG8gZnJpZW5kcyBh"
                            "bmQgZmFtaWx5LgogIHZlcmlmeV9hY2NvdW50OgogICAgc3RlcHM6CiAgICAtIGlkOiAwX2NvbGxl"
                            "Y3RfdmVyaWZ5X2FjY291bnRfZW1haWwKICAgICAgbmV4dDogMV9jb2xsZWN0X2Jhc2VkX2luX2Nh"
                            "bGlmb3JuaWEKICAgICAgZGVzY3JpcHRpb246IEFza3MgdXNlciBmb3IgdGhlaXIgZW1haWwgYWRk"
                            "cmVzcy4KICAgICAgY29sbGVjdDogdmVyaWZ5X2FjY291bnRfZW1haWwKICAgICAgdXR0ZXI6IHV0"
                            "dGVyX2Fza192ZXJpZnlfYWNjb3VudF9lbWFpbAogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRy"
                            "dWUKICAgICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtd"
                            "CiAgICAtIGlkOiAxX2NvbGxlY3RfYmFzZWRfaW5fY2FsaWZvcm5pYQogICAgICBuZXh0OgogICAg"
                            "ICAtIGlmOiBzbG90cy5iYXNlZF9pbl9jYWxpZm9ybmlhCiAgICAgICAgdGhlbjoKICAgICAgICAt"
                            "IGlkOiAyX2NvbGxlY3RfdmVyaWZ5X2FjY291bnRfc3VmZmljaWVudF9jYWxpZm9ybmlhX2luY29t"
                            "ZQogICAgICAgICAgbmV4dDoKICAgICAgICAgIC0gaWY6IG5vdCBzbG90cy52ZXJpZnlfYWNjb3Vu"
                            "dF9zdWZmaWNpZW50X2NhbGlmb3JuaWFfaW5jb21lCiAgICAgICAgICAgIHRoZW46CiAgICAgICAg"
                            "ICAgIC0gaWQ6IDNfdXR0ZXJfY2FfaW5jb21lX2luc3VmZmljaWVudAogICAgICAgICAgICAgIG5l"
                            "eHQ6IEVORAogICAgICAgICAgICAgIGFjdGlvbjogdXR0ZXJfY2FfaW5jb21lX2luc3VmZmljaWVu"
                            "dAogICAgICAgICAgLSBlbHNlOiBjb2xsZWN0X3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbl9j"
                            "YWxpZm9ybmlhCiAgICAgICAgICBkZXNjcmlwdGlvbjogQXNrcyB1c2VyIGlmIHRoZXkgaGF2ZSBz"
                            "dWZmaWNpZW50IGluY29tZSBpbiBDYWxpZm9ybmlhLgogICAgICAgICAgY29sbGVjdDogdmVyaWZ5"
                            "X2FjY291bnRfc3VmZmljaWVudF9jYWxpZm9ybmlhX2luY29tZQogICAgICAgICAgdXR0ZXI6IHV0"
                            "dGVyX2Fza192ZXJpZnlfYWNjb3VudF9zdWZmaWNpZW50X2NhbGlmb3JuaWFfaW5jb21lCiAgICAg"
                            "ICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgICAgIHJlc2V0X2FmdGVyX2Zsb3df"
                            "ZW5kczogdHJ1ZQogICAgICAgICAgcmVqZWN0aW9uczogW10KICAgICAgICAtIGlkOiBjb2xsZWN0"
                            "X3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbl9jYWxpZm9ybmlhCiAgICAgICAgICBuZXh0Ogog"
                            "ICAgICAgICAgLSBpZjogc2xvdHMudmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3Ju"
                            "aWEKICAgICAgICAgICAgdGhlbjoKICAgICAgICAgICAgLSBpZDogNV91dHRlcl92ZXJpZnlfYWNj"
                            "b3VudF9zdWNjZXNzCiAgICAgICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICAgICAgYWN0aW9u"
                            "OiB1dHRlcl92ZXJpZnlfYWNjb3VudF9zdWNjZXNzCiAgICAgICAgICAtIGVsc2U6CiAgICAgICAg"
                            "ICAgIC0gaWQ6IDZfdXR0ZXJfdmVyaWZ5X2FjY291bnRfY2FuY2VsbGVkCiAgICAgICAgICAgICAg"
                            "bmV4dDogRU5ECiAgICAgICAgICAgICAgYWN0aW9uOiB1dHRlcl92ZXJpZnlfYWNjb3VudF9jYW5j"
                            "ZWxsZWQKICAgICAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVzZXIgZm9yIGZpbmFsIGNvbmZpcm1h"
                            "dGlvbiB0byB2ZXJpZnkgdGhlaXIgYWNjb3VudCBpbiBDYWxpZm9ybmlhLgogICAgICAgICAgY29s"
                            "bGVjdDogdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3JuaWEKICAgICAgICAgIHV0"
                            "dGVyOiB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3JuaWEKICAg"
                            "ICAgICAgIGFza19iZWZvcmVfZmlsbGluZzogdHJ1ZQogICAgICAgICAgcmVzZXRfYWZ0ZXJfZmxv"
                            "d19lbmRzOiB0cnVlCiAgICAgICAgICByZWplY3Rpb25zOiBbXQogICAgICAtIGVsc2U6IGNvbGxl"
                            "Y3RfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uCiAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVz"
                            "ZXIgaWYgdGhleSBhcmUgYmFzZWQgaW4gQ2FsaWZvcm5pYS4KICAgICAgY29sbGVjdDogYmFzZWRf"
                            "aW5fY2FsaWZvcm5pYQogICAgICB1dHRlcjogdXR0ZXJfYXNrX2Jhc2VkX2luX2NhbGlmb3JuaWEK"
                            "ICAgICAgYXNrX2JlZm9yZV9maWxsaW5nOiB0cnVlCiAgICAgIHJlc2V0X2FmdGVyX2Zsb3dfZW5k"
                            "czogdHJ1ZQogICAgICByZWplY3Rpb25zOiBbXQogICAgLSBpZDogY29sbGVjdF92ZXJpZnlfYWNj"
                            "b3VudF9jb25maXJtYXRpb24KICAgICAgbmV4dDoKICAgICAgLSBpZjogc2xvdHMudmVyaWZ5X2Fj"
                            "Y291bnRfY29uZmlybWF0aW9uCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlkOiA4X3V0dGVyX3Zl"
                            "cmlmeV9hY2NvdW50X3N1Y2Nlc3MKICAgICAgICAgIG5leHQ6IEVORAogICAgICAgICAgYWN0aW9u"
                            "OiB1dHRlcl92ZXJpZnlfYWNjb3VudF9zdWNjZXNzCiAgICAgIC0gZWxzZToKICAgICAgICAtIGlk"
                            "OiA5X3V0dGVyX3ZlcmlmeV9hY2NvdW50X2NhbmNlbGxlZAogICAgICAgICAgbmV4dDogRU5ECiAg"
                            "ICAgICAgICBhY3Rpb246IHV0dGVyX3ZlcmlmeV9hY2NvdW50X2NhbmNlbGxlZAogICAgICBkZXNj"
                            "cmlwdGlvbjogQXNrcyB1c2VyIGZvciBmaW5hbCBjb25maXJtYXRpb24gdG8gdmVyaWZ5IHRoZWly"
                            "IGFjY291bnQuCiAgICAgIGNvbGxlY3Q6IHZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbgogICAg"
                            "ICB1dHRlcjogdXR0ZXJfYXNrX3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbgogICAgICBhc2tf"
                            "YmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAg"
                            "ICAgIHJlamVjdGlvbnM6IFtdCiAgICBuYW1lOiB2ZXJpZnlfYWNjb3VudAogICAgZGVzY3JpcHRp"
                            "b246IFZlcmlmeSBhbiBhY2NvdW50IGZvciBoaWdoZXIgdHJhbnNmZXIgbGltaXRzCg=="
                        ),
                        "nlu": (
                            "dmVyc2lvbjogIjMuMSIKbmx1OgotIGludGVudDogaGVhbHRoX2FkdmljZQogIGV4YW1wbGVzOiB8"
                            "CiAgICAtIEkgbmVlZCBzb21lIG1lZGljYWwgYWR2aWNlLgogICAgLSBDYW4geW91IGhlbHAgbWUg"
                            "d2l0aCBzb21lIGhlYWx0aCBpc3N1ZXM/CiAgICAtIEkgbmVlZCBtZWRpY2FsIHN1cHBvcnQuCiAg"
                            "ICAtIEknbSBleHBlcmllbmNpbmcgc29tZSBzeW1wdG9tcyBhbmQgSSBuZWVkIGd1aWRhbmNlIG9u"
                            "IHdoYXQgdG8gZG8uCiAgICAtIENhbiB5b3UgcHJvdmlkZSBtZSB3aXRoIGhlYWx0aCByZWNvbW1l"
                            "bmRhdGlvbnM/CiAgICAtIEknbSBzdHJ1Z2dsaW5nIHdpdGggc29tZSBoZWFsdGggY29uY2VybnMu"
                            "IENhbiB5b3Ugb2ZmZXIgYWR2aWNlPwogICAgLSBDYW4geW91IHN1Z2dlc3Qgd2F5cyB0byBpbXBy"
                            "b3ZlIG15IG92ZXJhbGwgd2VsbC1iZWluZz8KICAgIC0gSSdtIGxvb2tpbmcgZm9yIHRpcHMgb24g"
                            "bWFuYWdpbmcgc3RyZXNzIGFuZCBhbnhpZXR5LiBBbnkgYWR2aWNlPwogICAgLSBJIGhhdmUgYSBz"
                            "cGVjaWZpYyBoZWFsdGggcXVlc3Rpb24uIENhbiB5b3Ugb2ZmZXIgbWUgc29tZSBpbnNpZ2h0cz8K"
                            "ICAgIC0gSSBuZWVkIHN1Z2dlc3Rpb25zIG9uIG1haW50YWluaW5nIGEgaGVhbHRoeSBkaWV0IGFu"
                            "ZCBleGVyY2lzZSByb3V0aW5lLgogICAgLSBJcyB0aGVyZSBhbnlvbmUga25vd2xlZGdlYWJsZSBh"
                            "Ym91dCBuYXR1cmFsIHJlbWVkaWVzIHdobyBjYW4gZ2l2ZSBtZSBhZHZpY2U/CiAgICAtIENhbiB5"
                            "b3UgcHJvdmlkZSBtZSB3aXRoIGluZm9ybWF0aW9uIG9uIHByZXZlbnRpbmcgY29tbW9uIGlsbG5l"
                            "c3Nlcz8KICAgIC0gSSdtIGludGVyZXN0ZWQgaW4gbGVhcm5pbmcgYWJvdXQgYWx0ZXJuYXRpdmUg"
                            "dGhlcmFwaWVzLiBDYW4geW91IHNoYXJlIHlvdXIgZXhwZXJ0aXNlPwogICAgLSBDYW4geW91IHJl"
                            "Y29tbWVuZCBhIGdvb2QgZG9jdG9yPyBJJ20gbm90IGZlZWxpbmcgd2VsbC4KcmVzcG9uc2VzOgog"
                            "IHV0dGVyX3RyYW5zZmVyX21vbmV5X2luc3VmZmljaWVudF9mdW5kczoKICAtIHRleHQ6IFlvdSBk"
                            "b24ndCBoYXZlIHNvIG11Y2ggbW9uZXkgb24geW91ciBhY2NvdW50IQogIHV0dGVyX3RyYW5zZmVy"
                            "X2ZhaWxlZDoKICAtIHRleHQ6IHNvbWV0aGluZyB3ZW50IHdyb25nIHRyYW5zZmVycmluZyB0aGUg"
                            "bW9uZXkuCiAgdXR0ZXJfb3V0X29mX3Njb3BlOgogIC0gdGV4dDogU29ycnksIEknbSBub3Qgc3Vy"
                            "ZSBob3cgdG8gcmVzcG9uZCB0byB0aGF0LiBUeXBlICJoZWxwIiBmb3IgYXNzaXN0YW5jZS4KICB1"
                            "dHRlcl9hc2tfdHJhbnNmZXJfbW9uZXlfYW1vdW50X29mX21vbmV5OgogIC0gdGV4dDogSG93IG11"
                            "Y2ggbW9uZXkgZG8geW91IHdhbnQgdG8gdHJhbnNmZXI/CiAgdXR0ZXJfYXNrX3RyYW5zZmVyX21v"
                            "bmV5X3JlY2lwaWVudDoKICAtIHRleHQ6IFdobyBkbyB5b3Ugd2FudCB0byB0cmFuc2ZlciBtb25l"
                            "eSB0bz8KICB1dHRlcl90cmFuc2Zlcl9jb21wbGV0ZToKICAtIHRleHQ6IFN1Y2Nlc3NmdWxseSB0"
                            "cmFuc2ZlcnJlZCB7dHJhbnNmZXJfbW9uZXlfYW1vdW50fSB0byB7dHJhbnNmZXJfbW9uZXlfcmVj"
                            "aXBpZW50fS4KICB1dHRlcl90cmFuc2Zlcl9jYW5jZWxsZWQ6CiAgLSB0ZXh0OiBUcmFuc2ZlciBj"
                            "YW5jZWxsZWQuCiAgdXR0ZXJfYXNrX3RyYW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbjoK"
                            "ICAtIHRleHQ6IFdvdWxkIHlvdSBsaWtlIHRvIHRyYW5zZmVyIHt0cmFuc2Zlcl9tb25leV9hbW91"
                            "bnR9IHRvIHt0cmFuc2Zlcl9tb25leV9yZWNpcGllbnR9PwogICAgYnV0dG9uczoKICAgIC0gdGl0"
                            "bGU6IFllcwogICAgICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5vCiAgICAgIHBheWxvYWQ6"
                            "IE5vCiAgdXR0ZXJfYWRkX2NvbnRhY3RfY2FuY2VsbGVkOgogIC0gdGV4dDogT2theSwgSSBhbSBj"
                            "YW5jZWxsaW5nIHRoaXMgYWRkaW5nIG9mIGEgY29udGFjdC4KICB1dHRlcl9hZGRfY29udGFjdF9l"
                            "cnJvcjoKICAtIHRleHQ6IFNvbWV0aGluZyB3ZW50IHdyb25nLCBwbGVhc2UgdHJ5IGFnYWluLgog"
                            "IHV0dGVyX2NhX2luY29tZV9pbnN1ZmZpY2llbnQ6CiAgLSB0ZXh0OiBVbmZvcnR1bmF0ZWx5LCB3"
                            "ZSBjYW5ub3QgaW5jcmVhc2UgeW91ciB0cmFuc2ZlciBsaW1pdHMgdW5kZXIgdGhlc2UgY2lyY3Vt"
                            "c3RhbmNlcy4KICB1dHRlcl9jYW50X2FkdmljZV9vbl9oZWFsdGg6CiAgLSB0ZXh0OiBJJ20gc29y"
                            "cnksIEkgY2FuJ3QgZ2l2ZSB5b3UgYWR2aWNlIG9uIHlvdXIgaGVhbHRoLgogIHV0dGVyX2NvbnRh"
                            "Y3RfYWRkZWQ6CiAgLSB0ZXh0OiBDb250YWN0IGFkZGVkIHN1Y2Nlc3NmdWxseS4KICB1dHRlcl9j"
                            "b250YWN0X2FscmVhZHlfZXhpc3RzOgogIC0gdGV4dDogVGhlcmUncyBhbHJlYWR5IGEgY29udGFj"
                            "dCB3aXRoIHRoYXQgaGFuZGxlIGluIHlvdXIgbGlzdC4KICB1dHRlcl9jb250YWN0X25vdF9pbl9s"
                            "aXN0OgogIC0gdGV4dDogVGhhdCBjb250YWN0IGlzIG5vdCBpbiB5b3VyIGxpc3QuCiAgdXR0ZXJf"
                            "Y3VycmVudF9iYWxhbmNlOgogIC0gdGV4dDogWW91IHN0aWxsIGhhdmUge2N1cnJlbnRfYmFsYW5j"
                            "ZX0gaW4geW91ciBhY2NvdW50LgogIHV0dGVyX2hvdGVsX2luZm9ybV9yYXRpbmc6CiAgLSB0ZXh0"
                            "OiBUaGUge2hvdGVsX25hbWV9IGhhcyBhbiBhdmVyYWdlIHJhdGluZyBvZiB7aG90ZWxfYXZlcmFn"
                            "ZV9yYXRpbmd9CiAgdXR0ZXJfcmVtb3ZlX2NvbnRhY3RfY2FuY2VsbGVkOgogIC0gdGV4dDogT2th"
                            "eSwgSSBhbSBjYW5jZWxsaW5nIHRoaXMgcmVtb3ZhbCBvZiBhIGNvbnRhY3QuCiAgdXR0ZXJfcmVt"
                            "b3ZlX2NvbnRhY3RfZXJyb3I6CiAgLSB0ZXh0OiBTb21ldGhpbmcgd2VudCB3cm9uZywgcGxlYXNl"
                            "IHRyeSBhZ2Fpbi4KICB1dHRlcl9yZW1vdmVfY29udGFjdF9zdWNjZXNzOgogIC0gdGV4dDogUmVt"
                            "b3ZlZCB7cmVtb3ZlX2NvbnRhY3RfaGFuZGxlfSh7cmVtb3ZlX2NvbnRhY3RfbmFtZX0pIGZyb20g"
                            "eW91ciBjb250YWN0cy4KICB1dHRlcl90cmFuc2FjdGlvbnM6CiAgLSB0ZXh0OiAnWW91ciBjdXJy"
                            "ZW50IHRyYW5zYWN0aW9ucyBhcmU6ICB7dHJhbnNhY3Rpb25zX2xpc3R9JwogIHV0dGVyX3Zlcmlm"
                            "eV9hY2NvdW50X2NhbmNlbGxlZDoKICAtIHRleHQ6IENhbmNlbGxpbmcgYWNjb3VudCB2ZXJpZmlj"
                            "YXRpb24uLi4KICAgIG1ldGFkYXRhOgogICAgICByZXBocmFzZTogdHJ1ZQogIHV0dGVyX3Zlcmlm"
                            "eV9hY2NvdW50X3N1Y2Nlc3M6CiAgLSB0ZXh0OiBZb3VyIGFjY291bnQgd2FzIHN1Y2Nlc3NmdWxs"
                            "eSB2ZXJpZmllZAogIHV0dGVyX2Fza19hZGRfY29udGFjdF9jb25maXJtYXRpb246CiAgLSB0ZXh0"
                            "OiBEbyB5b3Ugd2FudCB0byBhZGQge2FkZF9jb250YWN0X25hbWV9KHthZGRfY29udGFjdF9oYW5k"
                            "bGV9KSB0byB5b3VyIGNvbnRhY3RzPwogIHV0dGVyX2Fza19hZGRfY29udGFjdF9oYW5kbGU6CiAg"
                            "LSB0ZXh0OiBXaGF0J3MgdGhlIGhhbmRsZSBvZiB0aGUgdXNlciB5b3Ugd2FudCB0byBhZGQ/CiAg"
                            "dXR0ZXJfYXNrX2FkZF9jb250YWN0X25hbWU6CiAgLSB0ZXh0OiBXaGF0J3MgdGhlIG5hbWUgb2Yg"
                            "dGhlIHVzZXIgeW91IHdhbnQgdG8gYWRkPwogICAgbWV0YWRhdGE6CiAgICAgIHJlcGhyYXNlOiB0"
                            "cnVlCiAgdXR0ZXJfYXNrX2Jhc2VkX2luX2NhbGlmb3JuaWE6CiAgLSB0ZXh0OiBBcmUgeW91IGJh"
                            "c2VkIGluIENhbGlmb3JuaWE/CiAgICBidXR0b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBh"
                            "eWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAgICAgcGF5bG9hZDogTm8KICB1dHRlcl9hc2tf"
                            "cmVtb3ZlX2NvbnRhY3RfY29uZmlybWF0aW9uOgogIC0gdGV4dDogU2hvdWxkIEkgcmVtb3ZlIHty"
                            "ZW1vdmVfY29udGFjdF9oYW5kbGV9IGZyb20geW91ciBjb250YWN0IGxpc3Q/CiAgICBidXR0b25z"
                            "OgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBheWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAg"
                            "ICAgcGF5bG9hZDogTm8KICB1dHRlcl9hc2tfcmVtb3ZlX2NvbnRhY3RfaGFuZGxlOgogIC0gdGV4"
                            "dDogV2hhdCdzIHRoZSBoYW5kbGUgb2YgdGhlIHVzZXIgeW91IHdhbnQgdG8gcmVtb3ZlPwogIHV0"
                            "dGVyX2Fza190cmFuc2Zlcl9tb25leV9hbW91bnQ6CiAgLSB0ZXh0OiBIb3cgbXVjaCBtb25leSBk"
                            "byB5b3Ugd2FudCB0byB0cmFuc2Zlcj8KICB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmly"
                            "bWF0aW9uOgogIC0gdGV4dDogWW91ciBlbWFpbCBhZGRyZXNzIGlzIHt2ZXJpZnlfYWNjb3VudF9l"
                            "bWFpbH0gYW5kIHlvdSBhcmUgbm90IGJhc2VkIGluIENhbGlmb3JuaWEsIGNvcnJlY3Q/CiAgICBi"
                            "dXR0b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBheWxvYWQ6IFllcwogICAgLSB0aXRsZTog"
                            "Tm8KICAgICAgcGF5bG9hZDogTm8KICB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0"
                            "aW9uX2NhbGlmb3JuaWE6CiAgLSB0ZXh0OiBZb3VyIGVtYWlsIGFkZHJlc3MgaXMge3ZlcmlmeV9h"
                            "Y2NvdW50X2VtYWlsfSBhbmQgeW91IGFyZSBiYXNlZCBpbiBDYWxpZm9ybmlhIHdpdGggYSB5ZWFy"
                            "bHkgaW5jb21lIGV4Y2VlZGluZyAxMDAsMDAwJCwgY29ycmVjdD8KICAgIGJ1dHRvbnM6CiAgICAt"
                            "IHRpdGxlOiBZZXMKICAgICAgcGF5bG9hZDogWWVzCiAgICAtIHRpdGxlOiBObwogICAgICBwYXls"
                            "b2FkOiBObwogIHV0dGVyX2Fza192ZXJpZnlfYWNjb3VudF9lbWFpbDoKICAtIHRleHQ6IFdoYXQn"
                            "cyB5b3VyIGVtYWlsIGFkZHJlc3M/CiAgdXR0ZXJfYXNrX3ZlcmlmeV9hY2NvdW50X3N1ZmZpY2ll"
                            "bnRfY2FsaWZvcm5pYV9pbmNvbWU6CiAgLSB0ZXh0OiBEb2VzIHlvdXIgeWVhcmx5IGluY29tZSBl"
                            "eGNlZWQgMTAwLDAwMCBVU0Q/CiAgICBidXR0b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBh"
                            "eWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAgICAgcGF5bG9hZDogTm8K"
                        ),
                        "config": (
                            "cmVjaXBlOiBkZWZhdWx0LnYxCmxhbmd1YWdlOiBlbgpwaXBlbGluZToKLSBuYW1lOiBMTE1Db21t"
                            "YW5kR2VuZXJhdG9yCiAgbGxtOgogICAgbW9kZWxfbmFtZTogZ3B0LTQKcG9saWNpZXM6Ci0gbmFt"
                            "ZTogcmFzYS5jb3JlLnBvbGljaWVzLmZsb3dfcG9saWN5LkZsb3dQb2xpY3kK"
                        ),
                    }
                },
            },
        ),
    ],
)
def test_handle_upload(
    monkeypatch: MonkeyPatch,
    args: argparse.Namespace,
    endpoint: str,
    expected: Dict[str, Any],
) -> None:
    mock = MagicMock()
    mock_token = MagicMock()
    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(rasa.studio.upload, "requests", mock)
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", mock_token)
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    rasa.studio.upload.handle_upload(args)

    assert mock.post.called
    assert mock.post.call_args[0][0] == endpoint
    assert mock.post.call_args[1]["json"] == expected


@pytest.mark.parametrize(
    "is_calm_bot, mock_fn_name",
    [
        (True, "upload_calm_assistant"),
        (False, "upload_nlu_assistant"),
    ],
)
def test_handle_upload_no_domain_path_specified(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    is_calm_bot: bool,
    mock_fn_name: str,
) -> None:
    """Test the handle_upload function when no domain path is specified in the CLI."""
    # setup test
    assistant_name = "test"
    endpoint = "http://studio.amazonaws.com/api/graphql"
    args = argparse.Namespace(
        assistant_name=[assistant_name],
        # this is the default value when running the cmd without specifying -d flag
        domain="domain.yml",
        calm=is_calm_bot,
    )

    domain_dir = tmp_path / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_path = domain_dir / "domain.yml"
    domain_path.write_text("test domain")

    domain_paths = [str(domain_dir), str(tmp_path / "domain.yml")]
    # we need to monkeypatch the DEFAULT_DOMAIN_PATHS to be able to use temporary paths
    monkeypatch.setattr(rasa.studio.upload, "DEFAULT_DOMAIN_PATHS", domain_paths)

    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    mock = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, mock_fn_name, mock)

    rasa.studio.upload.handle_upload(args)

    expected_args = argparse.Namespace(
        assistant_name=[assistant_name],
        calm=is_calm_bot,
        domain=str(domain_dir),
    )

    mock.assert_called_once_with(expected_args, assistant_name, endpoint)


@pytest.mark.parametrize(
    "assistant_name, nlu_examples_yaml, domain_yaml",
    [
        (
            "test",
            dedent(
                """\
                version: '3.1'
                intents:
                - greet
                - inform
                entities:
                - name:
                    roles:
                    - first_name
                    - last_name
                - age"""
            ),
            dedent(
                """\
                version: "3.1"
                nlu:
                - intent: greet
                examples: |
                    - hey
                    - hello
                    - hi
                    - hello there
                    - good morning
                    - good evening
                    - hey there
                    - let's go
                    - hey dude
                    - good afternoon
                - intent: inform
                examples: |
                    - I'm [John]{"entity": "name", "role": "first_name"}
                    - My first name is [Luis]{"entity": "name", "role": "first_name"}
                    - Karin
                    - Steven
                    - I'm [18](age)
                    - I am [32](age) years old"""
            ),
        )
    ],
)
def test_build_request(
    assistant_name: str, nlu_examples_yaml: str, domain_yaml: str
) -> None:
    domain_base64 = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")

    nlu_examples_base64 = base64.b64encode(nlu_examples_yaml.encode("utf-8")).decode(
        "utf-8"
    )

    graphQL_req = rasa.studio.upload.build_request(
        assistant_name, nlu_examples_yaml, domain_yaml
    )

    assert graphQL_req["variables"]["input"]["domain"] == domain_base64
    assert graphQL_req["variables"]["input"]["nlu"] == nlu_examples_base64
    assert graphQL_req["variables"]["input"]["assistantName"] == assistant_name


@pytest.mark.parametrize("assistant_name", ["test"])
def test_build_import_request(
    assistant_name: str, calm_domain_yaml, calm_flows_yaml, calm_nlu_yaml
) -> None:
    """Test the build_import_request function.

    :param assistant_name: The name of the assistant
    :return: None
    """
    base64_domain = base64.b64encode(calm_domain_yaml.encode("utf-8")).decode("utf-8")
    base64_flows = base64.b64encode(calm_flows_yaml.encode("utf-8")).decode("utf-8")
    base64_config = base64.b64encode("".encode("utf-8")).decode("utf-8")
    base64_nlu = base64.b64encode(calm_nlu_yaml.encode("utf-8")).decode("utf-8")

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name, calm_flows_yaml, calm_domain_yaml, base64_config, calm_nlu_yaml
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["nlu"] == base64_nlu


@pytest.mark.parametrize("assistant_name", ["test"])
def test_build_import_request_no_nlu(
    assistant_name: str, calm_domain_yaml, calm_flows_yaml
) -> None:
    """Test the build_import_request function when there is no NLU content to upload.

    :return: None
    """
    assistant_name = "test"
    base64_domain = base64.b64encode("domain".encode("utf-8")).decode("utf-8")
    base64_flows = base64.b64encode("flows".encode("utf-8")).decode("utf-8")
    base64_config = base64.b64encode("".encode("utf-8")).decode("utf-8")

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name, "flows", "domain", base64_config
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name


@pytest.mark.parametrize(
    "graphQL_req, endpoint, return_value, expected_response, expected_status",
    [
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"importFromEncodedYaml": ""}},
                "status_code": 200,
            },
            "Upload successful!",
            True,
        ),
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"importFromEncodedYaml": ""}},
                "status_code": 405,
            },
            "Upload failed with status code 405",
            False,
        ),
        (
            {
                "query": """\
                    mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)\
                        {\n  importFromEncodedYaml(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": "dmFyY2FzZSxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0WJhc2U=",
                        "nlu": "hcyBhIGxvYWRpbmcgY29udGVudCBvZiB0aGUgZGF0YWJhc2U=",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"importFromEncodedYaml": None}},
                "status_code": 500,
            },
            "Upload failed with status code 500",
            False,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": base64_calm_domain_yaml,
                        "flows": base64_calm_flows_yaml,
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"uploadModernAssistant": ""}},
                "status_code": 200,
            },
            "Upload successful!",
            True,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": base64_calm_domain_yaml,
                        "flows": base64_calm_flows_yaml,
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"uploadModernAssistant": ""}},
                "status_code": 405,
            },
            "Upload failed with status code 405",
            False,
        ),
        (
            {
                "query": """\
          mutation UploadModernAssistant($input: UploadModernAssistantInput!)\
              {\n  uploadModernAssistant(input: $input)\n}""",
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": base64_calm_domain_yaml,
                        "flows": base64_calm_flows_yaml,
                        "config": "",
                    }
                },
            },
            "http://studio.test/api/graphql/",
            {
                "json": {"data": {"uploadModernAssistant": None}},
                "status_code": 500,
            },
            "Upload failed with status code 500",
            False,
        ),
    ],
)
def test_make_request(
    monkeypatch: MonkeyPatch,
    graphQL_req: Dict[str, Any],
    return_value: Dict[str, Any],
    expected_response: str,
    expected_status: bool,
    endpoint: str,
    calm_domain_yaml: str,
    calm_flows_yaml: str,
) -> None:
    return_mock = MagicMock()
    return_mock.status_code = return_value["status_code"]
    return_mock.json.return_value = return_value["json"]
    mock = MagicMock(return_value=return_mock)
    mock_token = MagicMock()
    monkeypatch.setattr(rasa.studio.upload.requests, "post", mock)
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", mock_token)

    ret_val, status = rasa.studio.upload.make_request(endpoint, graphQL_req)

    assert mock.called
    assert mock.call_args[0][0] == endpoint
    assert mock.call_args[1]["json"] == graphQL_req

    assert mock_token.called

    assert ret_val == expected_response
    assert status == expected_status


@pytest.mark.parametrize(
    "domain_from_files, intents, entities, expected_domain",
    [
        (
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                    "goodbye",
                    "deny",
                ],
                "entities": [
                    {"name": {"roles": ["first_name", "last_name"]}},
                    "age",
                    "destination",
                    "origin",
                ],
            },
            ["greet", "inform"],
            ["name"],
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                ],
                "entities": [{"name": {"roles": ["first_name", "last_name"]}}],
            },
        ),
    ],
)
def test_filter_domain(
    domain_from_files: Dict[str, Any],
    intents: List[str],
    entities: List[Union[str, Dict[Any, Any]]],
    expected_domain: Dict[str, Any],
) -> None:

    filtered_domain = rasa.studio.upload._filter_domain(
        domain_from_files=domain_from_files, intents=intents, entities=entities
    )
    assert filtered_domain == expected_domain


@pytest.mark.parametrize(
    "intents, entities, found_intents, found_entities",
    [
        (
            ["greet", "inform"],
            ["name"],
            ["greet", "goodbye", "deny"],
            ["name", "destination", "origin"],
        ),
    ],
)
def test_check_for_missing_primitives(
    intents: List[str],
    entities: List[str],
    found_intents: List[str],
    found_entities: List[str],
) -> None:

    with pytest.raises(RasaException) as excinfo:
        rasa.studio.upload._check_for_missing_primitives(
            intents, entities, found_intents, found_entities
        )
        assert "The following intents were not found in the domain: inform" in str(
            excinfo.value
        )
        assert "The following entities were not found in the domain: age" in str(
            excinfo.value
        )


@pytest.mark.parametrize(
    "response, expected",
    [
        ({"errors": None}, False),
        ({"errors": []}, False),
        ({"errors": ["error"]}, True),
        ({"errors": ["error", "error2"]}, True),
    ],
)
def test_response_has_errors(response: Dict, expected: bool) -> None:
    assert rasa.studio.upload._response_has_errors(response) == expected


@pytest.mark.parametrize(
    "args, intents_from_files, entities_from_files, "
    "expected_intents, expected_entities",
    [
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents=None,
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities=None,
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
    ],
)
def test_get_selected_entities_and_intents(
    args: argparse.Namespace,
    intents_from_files: Set[Text],
    entities_from_files: List[Text],
    expected_intents: List[Text],
    expected_entities: List[Text],
) -> None:
    entities, intents = rasa.studio.upload._get_selected_entities_and_intents(
        args=args,
        intents_from_files=intents_from_files,
        entities_from_files=entities_from_files,
    )

    assert intents.sort() == expected_intents.sort()
    assert entities.sort() == expected_entities.sort()
