import argparse
import base64
import os
from textwrap import dedent
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from rasa.shared.exceptions import RasaException

import rasa.studio.upload
from rasa.studio.config import StudioConfig

# Globals for the calm files
calm_domain_yaml = open(
    os.path.dirname(__file__) + "/../../data/upload/calm/domain.yml", "r"
).read()

calm_flows_yaml = open(
    os.path.dirname(__file__) + "/../../data/upload/calm/flows.yml", "r"
).read()

base64_calm_domain_yaml = base64.b64encode(calm_domain_yaml.encode("utf-8")).decode(
    "utf-8"
)

base64_calm_flows_yaml = base64.b64encode(calm_flows_yaml.encode("utf-8")).decode(
    "utf-8"
)


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
                domain="data/upload/calm/domain.yml",
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
                            "YWN0Ci0gYWN0aW9uX3NlYXJjaF9ob3RlbAotIGFjdGlvbl90cmFuc2FjdGlvbl9zZWFyY2gKcmVz"
                            "cG9uc2VzOgogIHV0dGVyX2FkZF9jb250YWN0X2NhbmNlbGxlZDoKICAtIHRleHQ6IE9rYXksIEkg"
                            "YW0gY2FuY2VsbGluZyB0aGlzIGFkZGluZyBvZiBhIGNvbnRhY3QuCiAgdXR0ZXJfYWRkX2NvbnRh"
                            "Y3RfZXJyb3I6CiAgLSB0ZXh0OiBTb21ldGhpbmcgd2VudCB3cm9uZywgcGxlYXNlIHRyeSBhZ2Fp"
                            "bi4KICB1dHRlcl9jYV9pbmNvbWVfaW5zdWZmaWNpZW50OgogIC0gdGV4dDogVW5mb3J0dW5hdGVs"
                            "eSwgd2UgY2Fubm90IGluY3JlYXNlIHlvdXIgdHJhbnNmZXIgbGltaXRzIHVuZGVyIHRoZXNlIGNp"
                            "cmN1bXN0YW5jZXMuCiAgdXR0ZXJfY2FudF9hZHZpY2Vfb25faGVhbHRoOgogIC0gdGV4dDogSSdt"
                            "IHNvcnJ5LCBJIGNhbid0IGdpdmUgeW91IGFkdmljZSBvbiB5b3VyIGhlYWx0aC4KICB1dHRlcl9j"
                            "b250YWN0X2FkZGVkOgogIC0gdGV4dDogQ29udGFjdCBhZGRlZCBzdWNjZXNzZnVsbHkuCiAgdXR0"
                            "ZXJfY29udGFjdF9hbHJlYWR5X2V4aXN0czoKICAtIHRleHQ6IFRoZXJlJ3MgYWxyZWFkeSBhIGNv"
                            "bnRhY3Qgd2l0aCB0aGF0IGhhbmRsZSBpbiB5b3VyIGxpc3QuCiAgdXR0ZXJfY29udGFjdF9ub3Rf"
                            "aW5fbGlzdDoKICAtIHRleHQ6IFRoYXQgY29udGFjdCBpcyBub3QgaW4geW91ciBsaXN0LgogIHV0"
                            "dGVyX2N1cnJlbnRfYmFsYW5jZToKICAtIHRleHQ6IFlvdSBzdGlsbCBoYXZlIHtjdXJyZW50X2Jh"
                            "bGFuY2V9IGluIHlvdXIgYWNjb3VudC4KICB1dHRlcl9ob3RlbF9pbmZvcm1fcmF0aW5nOgogIC0g"
                            "dGV4dDogVGhlIHtob3RlbF9uYW1lfSBoYXMgYW4gYXZlcmFnZSByYXRpbmcgb2Yge2hvdGVsX2F2"
                            "ZXJhZ2VfcmF0aW5nfQogIHV0dGVyX3JlbW92ZV9jb250YWN0X2NhbmNlbGxlZDoKICAtIHRleHQ6"
                            "IE9rYXksIEkgYW0gY2FuY2VsbGluZyB0aGlzIHJlbW92YWwgb2YgYSBjb250YWN0LgogIHV0dGVy"
                            "X3JlbW92ZV9jb250YWN0X2Vycm9yOgogIC0gdGV4dDogU29tZXRoaW5nIHdlbnQgd3JvbmcsIHBs"
                            "ZWFzZSB0cnkgYWdhaW4uCiAgdXR0ZXJfcmVtb3ZlX2NvbnRhY3Rfc3VjY2VzczoKICAtIHRleHQ6"
                            "IFJlbW92ZWQge3JlbW92ZV9jb250YWN0X2hhbmRsZX0oe3JlbW92ZV9jb250YWN0X25hbWV9KSBm"
                            "cm9tIHlvdXIgY29udGFjdHMuCiAgdXR0ZXJfdHJhbnNhY3Rpb25zOgogIC0gdGV4dDogJ1lvdXIg"
                            "Y3VycmVudCB0cmFuc2FjdGlvbnMgYXJlOiAge3RyYW5zYWN0aW9uc19saXN0fScKICB1dHRlcl90"
                            "cmFuc2Zlcl9jYW5jZWxsZWQ6CiAgLSB0ZXh0OiBUcmFuc2ZlciBjYW5jZWxsZWQuCiAgdXR0ZXJf"
                            "dHJhbnNmZXJfY29tcGxldGU6CiAgLSB0ZXh0OiBTdWNjZXNzZnVsbHkgdHJhbnNmZXJyZWQge3Ry"
                            "YW5zZmVyX21vbmV5X2Ftb3VudH0gdG8ge3RyYW5zZmVyX21vbmV5X3JlY2lwaWVudH0uCiAgdXR0"
                            "ZXJfdHJhbnNmZXJfZmFpbGVkOgogIC0gdGV4dDogc29tZXRoaW5nIHdlbnQgd3JvbmcgdHJhbnNm"
                            "ZXJyaW5nIHRoZSBtb25leS4KICB1dHRlcl92ZXJpZnlfYWNjb3VudF9jYW5jZWxsZWQ6CiAgLSB0"
                            "ZXh0OiBDYW5jZWxsaW5nIGFjY291bnQgdmVyaWZpY2F0aW9uLi4uCiAgICBtZXRhZGF0YToKICAg"
                            "ICAgcmVwaHJhc2U6IHRydWUKICB1dHRlcl92ZXJpZnlfYWNjb3VudF9zdWNjZXNzOgogIC0gdGV4"
                            "dDogWW91ciBhY2NvdW50IHdhcyBzdWNjZXNzZnVsbHkgdmVyaWZpZWQKICB1dHRlcl9hc2tfYWRk"
                            "X2NvbnRhY3RfY29uZmlybWF0aW9uOgogIC0gdGV4dDogRG8geW91IHdhbnQgdG8gYWRkIHthZGRf"
                            "Y29udGFjdF9uYW1lfSh7YWRkX2NvbnRhY3RfaGFuZGxlfSkgdG8geW91ciBjb250YWN0cz8KICB1"
                            "dHRlcl9hc2tfYWRkX2NvbnRhY3RfaGFuZGxlOgogIC0gdGV4dDogV2hhdCdzIHRoZSBoYW5kbGUg"
                            "b2YgdGhlIHVzZXIgeW91IHdhbnQgdG8gYWRkPwogIHV0dGVyX2Fza19hZGRfY29udGFjdF9uYW1l"
                            "OgogIC0gdGV4dDogV2hhdCdzIHRoZSBuYW1lIG9mIHRoZSB1c2VyIHlvdSB3YW50IHRvIGFkZD8K"
                            "ICAgIG1ldGFkYXRhOgogICAgICByZXBocmFzZTogdHJ1ZQogIHV0dGVyX2Fza19iYXNlZF9pbl9j"
                            "YWxpZm9ybmlhOgogIC0gdGV4dDogQXJlIHlvdSBiYXNlZCBpbiBDYWxpZm9ybmlhPwogICAgYnV0"
                            "dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAgICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5v"
                            "CiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJfYXNrX3JlbW92ZV9jb250YWN0X2NvbmZpcm1hdGlv"
                            "bjoKICAtIHRleHQ6IFNob3VsZCBJIHJlbW92ZSB7cmVtb3ZlX2NvbnRhY3RfaGFuZGxlfSBmcm9t"
                            "IHlvdXIgY29udGFjdCBsaXN0PwogICAgYnV0dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAgICBw"
                            "YXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5vCiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJfYXNr"
                            "X3JlbW92ZV9jb250YWN0X2hhbmRsZToKICAtIHRleHQ6IFdoYXQncyB0aGUgaGFuZGxlIG9mIHRo"
                            "ZSB1c2VyIHlvdSB3YW50IHRvIHJlbW92ZT8KICB1dHRlcl9hc2tfdHJhbnNmZXJfbW9uZXlfYW1v"
                            "dW50OgogIC0gdGV4dDogSG93IG11Y2ggbW9uZXkgZG8geW91IHdhbnQgdG8gdHJhbnNmZXI/CiAg"
                            "dXR0ZXJfYXNrX3RyYW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1hdGlvbjoKICAtIHRleHQ6IFdv"
                            "dWxkIHlvdSBsaWtlIHRvIHRyYW5zZmVyIHt0cmFuc2Zlcl9tb25leV9hbW91bnR9IHRvIHt0cmFu"
                            "c2Zlcl9tb25leV9yZWNpcGllbnR9PwogICAgYnV0dG9uczoKICAgIC0gdGl0bGU6IFllcwogICAg"
                            "ICBwYXlsb2FkOiBZZXMKICAgIC0gdGl0bGU6IE5vCiAgICAgIHBheWxvYWQ6IE5vCiAgdXR0ZXJf"
                            "YXNrX3RyYW5zZmVyX21vbmV5X3JlY2lwaWVudDoKICAtIHRleHQ6IFdobyBkbyB5b3Ugd2FudCB0"
                            "byB0cmFuc2ZlciBtb25leSB0bz8KICB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0"
                            "aW9uOgogIC0gdGV4dDogWW91ciBlbWFpbCBhZGRyZXNzIGlzIHt2ZXJpZnlfYWNjb3VudF9lbWFp"
                            "bH0gYW5kIHlvdSBhcmUgbm90IGJhc2VkIGluIENhbGlmb3JuaWEsIGNvcnJlY3Q/CiAgICBidXR0"
                            "b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBheWxvYWQ6IFllcwogICAgLSB0aXRsZTogTm8K"
                            "ICAgICAgcGF5bG9hZDogTm8KICB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9u"
                            "X2NhbGlmb3JuaWE6CiAgLSB0ZXh0OiBZb3VyIGVtYWlsIGFkZHJlc3MgaXMge3ZlcmlmeV9hY2Nv"
                            "dW50X2VtYWlsfSBhbmQgeW91IGFyZSBiYXNlZCBpbiBDYWxpZm9ybmlhIHdpdGggYSB5ZWFybHkg"
                            "aW5jb21lIGV4Y2VlZGluZyAxMDAsMDAwJCwgY29ycmVjdD8KICAgIGJ1dHRvbnM6CiAgICAtIHRp"
                            "dGxlOiBZZXMKICAgICAgcGF5bG9hZDogWWVzCiAgICAtIHRpdGxlOiBObwogICAgICBwYXlsb2Fk"
                            "OiBObwogIHV0dGVyX2Fza192ZXJpZnlfYWNjb3VudF9lbWFpbDoKICAtIHRleHQ6IFdoYXQncyB5"
                            "b3VyIGVtYWlsIGFkZHJlc3M/CiAgdXR0ZXJfYXNrX3ZlcmlmeV9hY2NvdW50X3N1ZmZpY2llbnRf"
                            "Y2FsaWZvcm5pYV9pbmNvbWU6CiAgLSB0ZXh0OiBEb2VzIHlvdXIgeWVhcmx5IGluY29tZSBleGNl"
                            "ZWQgMTAwLDAwMCBVU0Q/CiAgICBidXR0b25zOgogICAgLSB0aXRsZTogWWVzCiAgICAgIHBheWxv"
                            "YWQ6IFllcwogICAgLSB0aXRsZTogTm8KICAgICAgcGF5bG9hZDogTm8Kc2xvdHM6CiAgYWRkX2Nv"
                            "bnRhY3RfY29uZmlybWF0aW9uOgogICAgdHlwZTogYm9vbAogICAgbWFwcGluZ3M6CiAgICAtIHR5"
                            "cGU6IGN1c3RvbQogIGFkZF9jb250YWN0X2hhbmRsZToKICAgIHR5cGU6IHRleHQKICAgIG1hcHBp"
                            "bmdzOgogICAgLSB0eXBlOiBjdXN0b20KICBhZGRfY29udGFjdF9uYW1lOgogICAgdHlwZTogdGV4"
                            "dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIGJhc2VkX2luX2NhbGlmb3JuaWE6"
                            "CiAgICB0eXBlOiBib29sCiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgY3VycmVu"
                            "dF9iYWxhbmNlOgogICAgdHlwZTogZmxvYXQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0"
                            "b20KICBob3RlbF9hdmVyYWdlX3JhdGluZzoKICAgIHR5cGU6IGZsb2F0CiAgICBtYXBwaW5nczoK"
                            "ICAgIC0gdHlwZTogY3VzdG9tCiAgaG90ZWxfbmFtZToKICAgIHR5cGU6IHRleHQKICAgIG1hcHBp"
                            "bmdzOgogICAgLSB0eXBlOiBjdXN0b20KICByZW1vdmVfY29udGFjdF9jb25maXJtYXRpb246CiAg"
                            "ICB0eXBlOiBib29sCiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAgcmVtb3ZlX2Nv"
                            "bnRhY3RfaGFuZGxlOgogICAgdHlwZTogdGV4dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1"
                            "c3RvbQogIHJlbW92ZV9jb250YWN0X25hbWU6CiAgICB0eXBlOiB0ZXh0CiAgICBtYXBwaW5nczoK"
                            "ICAgIC0gdHlwZTogY3VzdG9tCiAgcmV0dXJuX3ZhbHVlOgogICAgdHlwZTogdGV4dAogICAgbWFw"
                            "cGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIHRyYW5zYWN0aW9uc19saXN0OgogICAgdHlwZTog"
                            "dGV4dAogICAgbWFwcGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIHRyYW5zZmVyX21vbmV5X2Ft"
                            "b3VudDoKICAgIHR5cGU6IGZsb2F0CiAgICBtYXBwaW5nczoKICAgIC0gdHlwZTogY3VzdG9tCiAg"
                            "dHJhbnNmZXJfbW9uZXlfZmluYWxfY29uZmlybWF0aW9uOgogICAgdHlwZTogYm9vbAogICAgbWFw"
                            "cGluZ3M6CiAgICAtIHR5cGU6IGN1c3RvbQogIHRyYW5zZmVyX21vbmV5X3JlY2lwaWVudDoKICAg"
                            "IHR5cGU6IHRleHQKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICB0cmFuc2Zlcl9t"
                            "b25leV90cmFuc2Zlcl9zdWNjZXNzZnVsOgogICAgdHlwZTogYm9vbAogICAgbWFwcGluZ3M6CiAg"
                            "ICAtIHR5cGU6IGN1c3RvbQogIHZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbjoKICAgIHR5cGU6"
                            "IGJvb2wKICAgIG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICB2ZXJpZnlfYWNjb3VudF9j"
                            "b25maXJtYXRpb25fY2FsaWZvcm5pYToKICAgIHR5cGU6IGJvb2wKICAgIG1hcHBpbmdzOgogICAg"
                            "LSB0eXBlOiBjdXN0b20KICB2ZXJpZnlfYWNjb3VudF9lbWFpbDoKICAgIHR5cGU6IHRleHQKICAg"
                            "IG1hcHBpbmdzOgogICAgLSB0eXBlOiBjdXN0b20KICB2ZXJpZnlfYWNjb3VudF9zdWZmaWNpZW50"
                            "X2NhbGlmb3JuaWFfaW5jb21lOgogICAgdHlwZTogYm9vbAogICAgbWFwcGluZ3M6CiAgICAtIHR5"
                            "cGU6IGN1c3RvbQpzZXNzaW9uX2NvbmZpZzoKICBzZXNzaW9uX2V4cGlyYXRpb25fdGltZTogNjAK"
                            "ICBjYXJyeV9vdmVyX3Nsb3RzX3RvX25ld19zZXNzaW9uOiB0cnVlCg=="
                        ),
                        "flows": (
                            "Zmxvd3M6CiAgaGVhbHRoX2FkdmljZToKICAgIHN0ZXBzOgogICAgLSBpZDogMF91dHRlcl9jYW50"
                            "X2FkdmljZV9vbl9oZWFsdGgKICAgICAgbmV4dDogRU5ECiAgICAgIGFjdGlvbjogdXR0ZXJfY2Fu"
                            "dF9hZHZpY2Vfb25faGVhbHRoCiAgICBuYW1lOiBoZWFsdGhfYWR2aWNlCiAgICBkZXNjcmlwdGlv"
                            "bjogdXNlciBhc2tzIGZvciBoZWFsdGggYWR2aWNlCiAgYWRkX2NvbnRhY3Q6CiAgICBzdGVwczoK"
                            "ICAgIC0gaWQ6IDBfY29sbGVjdF9hZGRfY29udGFjdF9oYW5kbGUKICAgICAgbmV4dDogMV9jb2xs"
                            "ZWN0X2FkZF9jb250YWN0X25hbWUKICAgICAgZGVzY3JpcHRpb246IGEgdXNlciBoYW5kbGUgc3Rh"
                            "cnRpbmcgd2l0aCBACiAgICAgIGNvbGxlY3Q6IGFkZF9jb250YWN0X2hhbmRsZQogICAgICB1dHRl"
                            "cjogdXR0ZXJfYXNrX2FkZF9jb250YWN0X2hhbmRsZQogICAgICBjb2xsZWN0X2FjdGlvbjogYWN0"
                            "aW9uX2Fza19hZGRfY29udGFjdF9oYW5kbGUKICAgICAgYXNrX2JlZm9yZV9maWxsaW5nOiBmYWxz"
                            "ZQogICAgICByZXNldF9hZnRlcl9mbG93X2VuZHM6IHRydWUKICAgICAgcmVqZWN0aW9uczogW10K"
                            "ICAgIC0gaWQ6IDFfY29sbGVjdF9hZGRfY29udGFjdF9uYW1lCiAgICAgIG5leHQ6IDJfY29sbGVj"
                            "dF9hZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgZGVzY3JpcHRpb246IGEgbmFtZSBvZiBh"
                            "IHBlcnNvbgogICAgICBjb2xsZWN0OiBhZGRfY29udGFjdF9uYW1lCiAgICAgIHV0dGVyOiB1dHRl"
                            "cl9hc2tfYWRkX2NvbnRhY3RfbmFtZQogICAgICBjb2xsZWN0X2FjdGlvbjogYWN0aW9uX2Fza19h"
                            "ZGRfY29udGFjdF9uYW1lCiAgICAgIGFza19iZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVz"
                            "ZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAy"
                            "X2NvbGxlY3RfYWRkX2NvbnRhY3RfY29uZmlybWF0aW9uCiAgICAgIG5leHQ6CiAgICAgIC0gaWY6"
                            "IG5vdCBzbG90cy5hZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgICB0aGVuOgogICAgICAg"
                            "IC0gaWQ6IDNfdXR0ZXJfYWRkX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAgICAgICBuZXh0OiBFTkQK"
                            "ICAgICAgICAgIGFjdGlvbjogdXR0ZXJfYWRkX2NvbnRhY3RfY2FuY2VsbGVkCiAgICAgIC0gZWxz"
                            "ZTogYWN0aW9uX2FkZF9jb250YWN0CiAgICAgIGRlc2NyaXB0aW9uOiBhIGNvbmZpcm1hdGlvbiB0"
                            "byBhZGQgY29udGFjdAogICAgICBjb2xsZWN0OiBhZGRfY29udGFjdF9jb25maXJtYXRpb24KICAg"
                            "ICAgdXR0ZXI6IHV0dGVyX2Fza19hZGRfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgY29sbGVj"
                            "dF9hY3Rpb246IGFjdGlvbl9hc2tfYWRkX2NvbnRhY3RfY29uZmlybWF0aW9uCiAgICAgIGFza19i"
                            "ZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAg"
                            "ICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiBhY3Rpb25fYWRkX2NvbnRhY3QKICAgICAgbmV4"
                            "dDoKICAgICAgLSBpZjogc2xvdHMucmV0dXJuX3ZhbHVlIGlzICdhbHJlYWR5X2V4aXN0cycKICAg"
                            "ICAgICB0aGVuOgogICAgICAgIC0gaWQ6IDVfdXR0ZXJfY29udGFjdF9hbHJlYWR5X2V4aXN0cwog"
                            "ICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICBhY3Rpb246IHV0dGVyX2NvbnRhY3RfYWxyZWFk"
                            "eV9leGlzdHMKICAgICAgLSBpZjogc2xvdHMucmV0dXJuX3ZhbHVlIGlzICdzdWNjZXNzJwogICAg"
                            "ICAgIHRoZW46CiAgICAgICAgLSBpZDogNl91dHRlcl9jb250YWN0X2FkZGVkCiAgICAgICAgICBu"
                            "ZXh0OiBFTkQKICAgICAgICAgIGFjdGlvbjogdXR0ZXJfY29udGFjdF9hZGRlZAogICAgICAtIGVs"
                            "c2U6CiAgICAgICAgLSBpZDogN191dHRlcl9hZGRfY29udGFjdF9lcnJvcgogICAgICAgICAgbmV4"
                            "dDogRU5ECiAgICAgICAgICBhY3Rpb246IHV0dGVyX2FkZF9jb250YWN0X2Vycm9yCiAgICAgIGFj"
                            "dGlvbjogYWN0aW9uX2FkZF9jb250YWN0CiAgICBuYW1lOiBhZGRfY29udGFjdAogICAgZGVzY3Jp"
                            "cHRpb246IGFkZCBhIGNvbnRhY3QgdG8geW91ciBjb250YWN0IGxpc3QKICBjaGVja19iYWxhbmNl"
                            "OgogICAgc3RlcHM6CiAgICAtIGlkOiAwX2FjdGlvbl9jaGVja19iYWxhbmNlCiAgICAgIG5leHQ6"
                            "IDFfdXR0ZXJfY3VycmVudF9iYWxhbmNlCiAgICAgIGFjdGlvbjogYWN0aW9uX2NoZWNrX2JhbGFu"
                            "Y2UKICAgIC0gaWQ6IDFfdXR0ZXJfY3VycmVudF9iYWxhbmNlCiAgICAgIG5leHQ6IEVORAogICAg"
                            "ICBhY3Rpb246IHV0dGVyX2N1cnJlbnRfYmFsYW5jZQogICAgbmFtZTogY2hlY2tfYmFsYW5jZQog"
                            "ICAgZGVzY3JpcHRpb246IGNoZWNrIHRoZSB1c2VyJ3MgYWNjb3VudCBiYWxhbmNlLgogIGhvdGVs"
                            "X3NlYXJjaDoKICAgIHN0ZXBzOgogICAgLSBpZDogMF9hY3Rpb25fc2VhcmNoX2hvdGVsCiAgICAg"
                            "IG5leHQ6IDFfdXR0ZXJfaG90ZWxfaW5mb3JtX3JhdGluZwogICAgICBhY3Rpb246IGFjdGlvbl9z"
                            "ZWFyY2hfaG90ZWwKICAgIC0gaWQ6IDFfdXR0ZXJfaG90ZWxfaW5mb3JtX3JhdGluZwogICAgICBu"
                            "ZXh0OiBFTkQKICAgICAgYWN0aW9uOiB1dHRlcl9ob3RlbF9pbmZvcm1fcmF0aW5nCiAgICBuYW1l"
                            "OiBob3RlbF9zZWFyY2gKICAgIGRlc2NyaXB0aW9uOiBzZWFyY2ggZm9yIGhvdGVscwogIHJlbW92"
                            "ZV9jb250YWN0OgogICAgc3RlcHM6CiAgICAtIGlkOiAwX2NvbGxlY3RfcmVtb3ZlX2NvbnRhY3Rf"
                            "aGFuZGxlCiAgICAgIG5leHQ6IDFfY29sbGVjdF9yZW1vdmVfY29udGFjdF9jb25maXJtYXRpb24K"
                            "ICAgICAgZGVzY3JpcHRpb246IGEgY29udGFjdCBoYW5kbGUgc3RhcnRpbmcgd2l0aCBACiAgICAg"
                            "IGNvbGxlY3Q6IHJlbW92ZV9jb250YWN0X2hhbmRsZQogICAgICB1dHRlcjogdXR0ZXJfYXNrX3Jl"
                            "bW92ZV9jb250YWN0X2hhbmRsZQogICAgICBjb2xsZWN0X2FjdGlvbjogYWN0aW9uX2Fza19yZW1v"
                            "dmVfY29udGFjdF9oYW5kbGUKICAgICAgYXNrX2JlZm9yZV9maWxsaW5nOiBmYWxzZQogICAgICBy"
                            "ZXNldF9hZnRlcl9mbG93X2VuZHM6IHRydWUKICAgICAgcmVqZWN0aW9uczogW10KICAgIC0gaWQ6"
                            "IDFfY29sbGVjdF9yZW1vdmVfY29udGFjdF9jb25maXJtYXRpb24KICAgICAgbmV4dDoKICAgICAg"
                            "LSBpZjogbm90IHNsb3RzLnJlbW92ZV9jb250YWN0X2NvbmZpcm1hdGlvbgogICAgICAgIHRoZW46"
                            "CiAgICAgICAgLSBpZDogMl91dHRlcl9yZW1vdmVfY29udGFjdF9jYW5jZWxsZWQKICAgICAgICAg"
                            "IG5leHQ6IEVORAogICAgICAgICAgYWN0aW9uOiB1dHRlcl9yZW1vdmVfY29udGFjdF9jYW5jZWxs"
                            "ZWQKICAgICAgLSBlbHNlOiBhY3Rpb25fcmVtb3ZlX2NvbnRhY3QKICAgICAgY29sbGVjdDogcmVt"
                            "b3ZlX2NvbnRhY3RfY29uZmlybWF0aW9uCiAgICAgIHV0dGVyOiB1dHRlcl9hc2tfcmVtb3ZlX2Nv"
                            "bnRhY3RfY29uZmlybWF0aW9uCiAgICAgIGNvbGxlY3RfYWN0aW9uOiBhY3Rpb25fYXNrX3JlbW92"
                            "ZV9jb250YWN0X2NvbmZpcm1hdGlvbgogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAg"
                            "ICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAt"
                            "IGlkOiBhY3Rpb25fcmVtb3ZlX2NvbnRhY3QKICAgICAgbmV4dDoKICAgICAgLSBpZjogc2xvdHMu"
                            "cmV0dXJuX3ZhbHVlIGlzICdub3RfZm91bmQnCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlkOiA0"
                            "X3V0dGVyX2NvbnRhY3Rfbm90X2luX2xpc3QKICAgICAgICAgIG5leHQ6IEVORAogICAgICAgICAg"
                            "YWN0aW9uOiB1dHRlcl9jb250YWN0X25vdF9pbl9saXN0CiAgICAgIC0gaWY6IHNsb3RzLnJldHVy"
                            "bl92YWx1ZSBpcyAnc3VjY2VzcycKICAgICAgICB0aGVuOgogICAgICAgIC0gaWQ6IDVfdXR0ZXJf"
                            "cmVtb3ZlX2NvbnRhY3Rfc3VjY2VzcwogICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICBhY3Rp"
                            "b246IHV0dGVyX3JlbW92ZV9jb250YWN0X3N1Y2Nlc3MKICAgICAgLSBlbHNlOgogICAgICAgIC0g"
                            "aWQ6IDZfdXR0ZXJfcmVtb3ZlX2NvbnRhY3RfZXJyb3IKICAgICAgICAgIG5leHQ6IEVORAogICAg"
                            "ICAgICAgYWN0aW9uOiB1dHRlcl9yZW1vdmVfY29udGFjdF9lcnJvcgogICAgICBhY3Rpb246IGFj"
                            "dGlvbl9yZW1vdmVfY29udGFjdAogICAgbmFtZTogcmVtb3ZlX2NvbnRhY3QKICAgIGRlc2NyaXB0"
                            "aW9uOiByZW1vdmUgYSBjb250YWN0IGZyb20geW91ciBjb250YWN0IGxpc3QKICB0cmFuc2FjdGlv"
                            "bl9zZWFyY2g6CiAgICBzdGVwczoKICAgIC0gaWQ6IDBfYWN0aW9uX3RyYW5zYWN0aW9uX3NlYXJj"
                            "aAogICAgICBuZXh0OiAxX3V0dGVyX3RyYW5zYWN0aW9ucwogICAgICBhY3Rpb246IGFjdGlvbl90"
                            "cmFuc2FjdGlvbl9zZWFyY2gKICAgIC0gaWQ6IDFfdXR0ZXJfdHJhbnNhY3Rpb25zCiAgICAgIG5l"
                            "eHQ6IEVORAogICAgICBhY3Rpb246IHV0dGVyX3RyYW5zYWN0aW9ucwogICAgbmFtZTogdHJhbnNh"
                            "Y3Rpb25fc2VhcmNoCiAgICBkZXNjcmlwdGlvbjogbGlzdHMgdGhlIGxhc3QgdHJhbnNhY3Rpb25z"
                            "IG9mIHRoZSB1c2VyIGFjY291bnQKICB0cmFuc2Zlcl9tb25leToKICAgIHN0ZXBzOgogICAgLSBp"
                            "ZDogMF9jb2xsZWN0X3RyYW5zZmVyX21vbmV5X3JlY2lwaWVudAogICAgICBuZXh0OiAxX2NvbGxl"
                            "Y3RfdHJhbnNmZXJfbW9uZXlfYW1vdW50CiAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVzZXIgZm9y"
                            "IHRoZSByZWNpcGllbnQncyBuYW1lLgogICAgICBjb2xsZWN0OiB0cmFuc2Zlcl9tb25leV9yZWNp"
                            "cGllbnQKICAgICAgdXR0ZXI6IHV0dGVyX2Fza190cmFuc2Zlcl9tb25leV9yZWNpcGllbnQKICAg"
                            "ICAgY29sbGVjdF9hY3Rpb246IGFjdGlvbl9hc2tfdHJhbnNmZXJfbW9uZXlfcmVjaXBpZW50CiAg"
                            "ICAgIGFza19iZWZvcmVfZmlsbGluZzogZmFsc2UKICAgICAgcmVzZXRfYWZ0ZXJfZmxvd19lbmRz"
                            "OiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAxX2NvbGxlY3RfdHJhbnNmZXJf"
                            "bW9uZXlfYW1vdW50CiAgICAgIG5leHQ6IDJfY29sbGVjdF90cmFuc2Zlcl9tb25leV9maW5hbF9j"
                            "b25maXJtYXRpb24KICAgICAgZGVzY3JpcHRpb246IEFza3MgdXNlciBmb3IgdGhlIGFtb3VudCB0"
                            "byB0cmFuc2Zlci4KICAgICAgY29sbGVjdDogdHJhbnNmZXJfbW9uZXlfYW1vdW50CiAgICAgIHV0"
                            "dGVyOiB1dHRlcl9hc2tfdHJhbnNmZXJfbW9uZXlfYW1vdW50CiAgICAgIGNvbGxlY3RfYWN0aW9u"
                            "OiBhY3Rpb25fYXNrX3RyYW5zZmVyX21vbmV5X2Ftb3VudAogICAgICBhc2tfYmVmb3JlX2ZpbGxp"
                            "bmc6IGZhbHNlCiAgICAgIHJlc2V0X2FmdGVyX2Zsb3dfZW5kczogdHJ1ZQogICAgICByZWplY3Rp"
                            "b25zOiBbXQogICAgLSBpZDogMl9jb2xsZWN0X3RyYW5zZmVyX21vbmV5X2ZpbmFsX2NvbmZpcm1h"
                            "dGlvbgogICAgICBuZXh0OgogICAgICAtIGlmOiBub3Qgc2xvdHMudHJhbnNmZXJfbW9uZXlfZmlu"
                            "YWxfY29uZmlybWF0aW9uCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlkOiAzX3V0dGVyX3RyYW5z"
                            "ZmVyX2NhbmNlbGxlZAogICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICBhY3Rpb246IHV0dGVy"
                            "X3RyYW5zZmVyX2NhbmNlbGxlZAogICAgICAtIGVsc2U6IGFjdGlvbl9leGVjdXRlX3RyYW5zZmVy"
                            "CiAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVzZXIgZm9yIGZpbmFsIGNvbmZpcm1hdGlvbiB0byB0"
                            "cmFuc2ZlciBtb25leS4KICAgICAgY29sbGVjdDogdHJhbnNmZXJfbW9uZXlfZmluYWxfY29uZmly"
                            "bWF0aW9uCiAgICAgIHV0dGVyOiB1dHRlcl9hc2tfdHJhbnNmZXJfbW9uZXlfZmluYWxfY29uZmly"
                            "bWF0aW9uCiAgICAgIGNvbGxlY3RfYWN0aW9uOiBhY3Rpb25fYXNrX3RyYW5zZmVyX21vbmV5X2Zp"
                            "bmFsX2NvbmZpcm1hdGlvbgogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgcmVz"
                            "ZXRfYWZ0ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiBh"
                            "Y3Rpb25fZXhlY3V0ZV90cmFuc2ZlcgogICAgICBuZXh0OgogICAgICAtIGlmOiBzbG90cy50cmFu"
                            "c2Zlcl9tb25leV90cmFuc2Zlcl9zdWNjZXNzZnVsCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlk"
                            "OiA1X3V0dGVyX3RyYW5zZmVyX2NvbXBsZXRlCiAgICAgICAgICBuZXh0OiBFTkQKICAgICAgICAg"
                            "IGFjdGlvbjogdXR0ZXJfdHJhbnNmZXJfY29tcGxldGUKICAgICAgLSBlbHNlOgogICAgICAgIC0g"
                            "aWQ6IDZfdXR0ZXJfdHJhbnNmZXJfZmFpbGVkCiAgICAgICAgICBuZXh0OiBFTkQKICAgICAgICAg"
                            "IGFjdGlvbjogdXR0ZXJfdHJhbnNmZXJfZmFpbGVkCiAgICAgIGFjdGlvbjogYWN0aW9uX2V4ZWN1"
                            "dGVfdHJhbnNmZXIKICAgIG5hbWU6IHRyYW5zZmVyX21vbmV5CiAgICBkZXNjcmlwdGlvbjogVGhp"
                            "cyBmbG93IGxldCdzIHVzZXJzIHNlbmQgbW9uZXkgdG8gZnJpZW5kcyBhbmQgZmFtaWx5LgogIHZl"
                            "cmlmeV9hY2NvdW50OgogICAgc3RlcHM6CiAgICAtIGlkOiAwX2NvbGxlY3RfdmVyaWZ5X2FjY291"
                            "bnRfZW1haWwKICAgICAgbmV4dDogMV9jb2xsZWN0X2Jhc2VkX2luX2NhbGlmb3JuaWEKICAgICAg"
                            "ZGVzY3JpcHRpb246IEFza3MgdXNlciBmb3IgdGhlaXIgZW1haWwgYWRkcmVzcy4KICAgICAgY29s"
                            "bGVjdDogdmVyaWZ5X2FjY291bnRfZW1haWwKICAgICAgdXR0ZXI6IHV0dGVyX2Fza192ZXJpZnlf"
                            "YWNjb3VudF9lbWFpbAogICAgICBjb2xsZWN0X2FjdGlvbjogYWN0aW9uX2Fza192ZXJpZnlfYWNj"
                            "b3VudF9lbWFpbAogICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgcmVzZXRfYWZ0"
                            "ZXJfZmxvd19lbmRzOiB0cnVlCiAgICAgIHJlamVjdGlvbnM6IFtdCiAgICAtIGlkOiAxX2NvbGxl"
                            "Y3RfYmFzZWRfaW5fY2FsaWZvcm5pYQogICAgICBuZXh0OgogICAgICAtIGlmOiBzbG90cy5iYXNl"
                            "ZF9pbl9jYWxpZm9ybmlhCiAgICAgICAgdGhlbjoKICAgICAgICAtIGlkOiAyX2NvbGxlY3RfdmVy"
                            "aWZ5X2FjY291bnRfc3VmZmljaWVudF9jYWxpZm9ybmlhX2luY29tZQogICAgICAgICAgbmV4dDoK"
                            "ICAgICAgICAgIC0gaWY6IG5vdCBzbG90cy52ZXJpZnlfYWNjb3VudF9zdWZmaWNpZW50X2NhbGlm"
                            "b3JuaWFfaW5jb21lCiAgICAgICAgICAgIHRoZW46CiAgICAgICAgICAgIC0gaWQ6IDNfdXR0ZXJf"
                            "Y2FfaW5jb21lX2luc3VmZmljaWVudAogICAgICAgICAgICAgIG5leHQ6IEVORAogICAgICAgICAg"
                            "ICAgIGFjdGlvbjogdXR0ZXJfY2FfaW5jb21lX2luc3VmZmljaWVudAogICAgICAgICAgLSBlbHNl"
                            "OiBjb2xsZWN0X3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbl9jYWxpZm9ybmlhCiAgICAgICAg"
                            "ICBkZXNjcmlwdGlvbjogQXNrcyB1c2VyIGlmIHRoZXkgaGF2ZSBzdWZmaWNpZW50IGluY29tZSBp"
                            "biBDYWxpZm9ybmlhLgogICAgICAgICAgY29sbGVjdDogdmVyaWZ5X2FjY291bnRfc3VmZmljaWVu"
                            "dF9jYWxpZm9ybmlhX2luY29tZQogICAgICAgICAgdXR0ZXI6IHV0dGVyX2Fza192ZXJpZnlfYWNj"
                            "b3VudF9zdWZmaWNpZW50X2NhbGlmb3JuaWFfaW5jb21lCiAgICAgICAgICBjb2xsZWN0X2FjdGlv"
                            "bjogYWN0aW9uX2Fza192ZXJpZnlfYWNjb3VudF9zdWZmaWNpZW50X2NhbGlmb3JuaWFfaW5jb21l"
                            "CiAgICAgICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUKICAgICAgICAgIHJlc2V0X2FmdGVy"
                            "X2Zsb3dfZW5kczogdHJ1ZQogICAgICAgICAgcmVqZWN0aW9uczogW10KICAgICAgICAtIGlkOiBj"
                            "b2xsZWN0X3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbl9jYWxpZm9ybmlhCiAgICAgICAgICBu"
                            "ZXh0OgogICAgICAgICAgLSBpZjogc2xvdHMudmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2Nh"
                            "bGlmb3JuaWEKICAgICAgICAgICAgdGhlbjoKICAgICAgICAgICAgLSBpZDogNV91dHRlcl92ZXJp"
                            "ZnlfYWNjb3VudF9zdWNjZXNzCiAgICAgICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICAgICAg"
                            "YWN0aW9uOiB1dHRlcl92ZXJpZnlfYWNjb3VudF9zdWNjZXNzCiAgICAgICAgICAtIGVsc2U6CiAg"
                            "ICAgICAgICAgIC0gaWQ6IDZfdXR0ZXJfdmVyaWZ5X2FjY291bnRfY2FuY2VsbGVkCiAgICAgICAg"
                            "ICAgICAgbmV4dDogRU5ECiAgICAgICAgICAgICAgYWN0aW9uOiB1dHRlcl92ZXJpZnlfYWNjb3Vu"
                            "dF9jYW5jZWxsZWQKICAgICAgICAgIGRlc2NyaXB0aW9uOiBBc2tzIHVzZXIgZm9yIGZpbmFsIGNv"
                            "bmZpcm1hdGlvbiB0byB2ZXJpZnkgdGhlaXIgYWNjb3VudCBpbiBDYWxpZm9ybmlhLgogICAgICAg"
                            "ICAgY29sbGVjdDogdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3JuaWEKICAgICAg"
                            "ICAgIHV0dGVyOiB1dHRlcl9hc2tfdmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uX2NhbGlmb3Ju"
                            "aWEKICAgICAgICAgIGNvbGxlY3RfYWN0aW9uOiBhY3Rpb25fYXNrX3ZlcmlmeV9hY2NvdW50X2Nv"
                            "bmZpcm1hdGlvbl9jYWxpZm9ybmlhCiAgICAgICAgICBhc2tfYmVmb3JlX2ZpbGxpbmc6IHRydWUK"
                            "ICAgICAgICAgIHJlc2V0X2FmdGVyX2Zsb3dfZW5kczogdHJ1ZQogICAgICAgICAgcmVqZWN0aW9u"
                            "czogW10KICAgICAgLSBlbHNlOiBjb2xsZWN0X3ZlcmlmeV9hY2NvdW50X2NvbmZpcm1hdGlvbgog"
                            "ICAgICBkZXNjcmlwdGlvbjogQXNrcyB1c2VyIGlmIHRoZXkgYXJlIGJhc2VkIGluIENhbGlmb3Ju"
                            "aWEuCiAgICAgIGNvbGxlY3Q6IGJhc2VkX2luX2NhbGlmb3JuaWEKICAgICAgdXR0ZXI6IHV0dGVy"
                            "X2Fza19iYXNlZF9pbl9jYWxpZm9ybmlhCiAgICAgIGNvbGxlY3RfYWN0aW9uOiBhY3Rpb25fYXNr"
                            "X2Jhc2VkX2luX2NhbGlmb3JuaWEKICAgICAgYXNrX2JlZm9yZV9maWxsaW5nOiB0cnVlCiAgICAg"
                            "IHJlc2V0X2FmdGVyX2Zsb3dfZW5kczogdHJ1ZQogICAgICByZWplY3Rpb25zOiBbXQogICAgLSBp"
                            "ZDogY29sbGVjdF92ZXJpZnlfYWNjb3VudF9jb25maXJtYXRpb24KICAgICAgbmV4dDoKICAgICAg"
                            "LSBpZjogc2xvdHMudmVyaWZ5X2FjY291bnRfY29uZmlybWF0aW9uCiAgICAgICAgdGhlbjoKICAg"
                            "ICAgICAtIGlkOiA4X3V0dGVyX3ZlcmlmeV9hY2NvdW50X3N1Y2Nlc3MKICAgICAgICAgIG5leHQ6"
                            "IEVORAogICAgICAgICAgYWN0aW9uOiB1dHRlcl92ZXJpZnlfYWNjb3VudF9zdWNjZXNzCiAgICAg"
                            "IC0gZWxzZToKICAgICAgICAtIGlkOiA5X3V0dGVyX3ZlcmlmeV9hY2NvdW50X2NhbmNlbGxlZAog"
                            "ICAgICAgICAgbmV4dDogRU5ECiAgICAgICAgICBhY3Rpb246IHV0dGVyX3ZlcmlmeV9hY2NvdW50"
                            "X2NhbmNlbGxlZAogICAgICBkZXNjcmlwdGlvbjogQXNrcyB1c2VyIGZvciBmaW5hbCBjb25maXJt"
                            "YXRpb24gdG8gdmVyaWZ5IHRoZWlyIGFjY291bnQuCiAgICAgIGNvbGxlY3Q6IHZlcmlmeV9hY2Nv"
                            "dW50X2NvbmZpcm1hdGlvbgogICAgICB1dHRlcjogdXR0ZXJfYXNrX3ZlcmlmeV9hY2NvdW50X2Nv"
                            "bmZpcm1hdGlvbgogICAgICBjb2xsZWN0X2FjdGlvbjogYWN0aW9uX2Fza192ZXJpZnlfYWNjb3Vu"
                            "dF9jb25maXJtYXRpb24KICAgICAgYXNrX2JlZm9yZV9maWxsaW5nOiB0cnVlCiAgICAgIHJlc2V0"
                            "X2FmdGVyX2Zsb3dfZW5kczogdHJ1ZQogICAgICByZWplY3Rpb25zOiBbXQogICAgbmFtZTogdmVy"
                            "aWZ5X2FjY291bnQKICAgIGRlc2NyaXB0aW9uOiBWZXJpZnkgYW4gYWNjb3VudCBmb3IgaGlnaGVy"
                            "IHRyYW5zZmVyIGxpbWl0cwo="
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


@pytest.mark.parametrize("assistant_name", [("test")])
def test_build_import_request(assistant_name: str) -> None:
    """Test the build_import_request function.

    :param assistant_name: The name of the assistant
    :return: None
    """
    base64_domain = base64.b64encode(calm_domain_yaml.encode("utf-8")).decode("utf-8")
    base64_flows = base64.b64encode(calm_flows_yaml.encode("utf-8")).decode("utf-8")
    base64_config = base64.b64encode("".encode("utf-8")).decode("utf-8")

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name, calm_flows_yaml, calm_domain_yaml, base64_config
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
                        "nlu": base64_calm_flows_yaml,
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
