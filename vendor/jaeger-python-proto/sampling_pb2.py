# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sampling.proto
"""Generated protocol buffer code."""

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="sampling.proto",
    package="jaeger.api_v2",
    syntax="proto3",
    serialized_options=b"\n\027io.jaegertracing.api_v2Z\006api_v2\310\342\036\001\320\342\036\001\340\342\036\001",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0esampling.proto\x12\rjaeger.api_v2\x1a\x14gogoproto/gogo.proto\x1a\x1cgoogle/api/annotations.proto"5\n\x1dProbabilisticSamplingStrategy\x12\x14\n\x0csamplingRate\x18\x01 \x01(\x01":\n\x1cRateLimitingSamplingStrategy\x12\x1a\n\x12maxTracesPerSecond\x18\x01 \x01(\x05"{\n\x19OperationSamplingStrategy\x12\x11\n\toperation\x18\x01 \x01(\t\x12K\n\x15probabilisticSampling\x18\x02 \x01(\x0b\x32,.jaeger.api_v2.ProbabilisticSamplingStrategy"\xe2\x01\n\x1ePerOperationSamplingStrategies\x12"\n\x1a\x64\x65\x66\x61ultSamplingProbability\x18\x01 \x01(\x01\x12(\n defaultLowerBoundTracesPerSecond\x18\x02 \x01(\x01\x12H\n\x16perOperationStrategies\x18\x03 \x03(\x0b\x32(.jaeger.api_v2.OperationSamplingStrategy\x12(\n defaultUpperBoundTracesPerSecond\x18\x04 \x01(\x01"\xb7\x02\n\x18SamplingStrategyResponse\x12\x39\n\x0cstrategyType\x18\x01 \x01(\x0e\x32#.jaeger.api_v2.SamplingStrategyType\x12K\n\x15probabilisticSampling\x18\x02 \x01(\x0b\x32,.jaeger.api_v2.ProbabilisticSamplingStrategy\x12I\n\x14rateLimitingSampling\x18\x03 \x01(\x0b\x32+.jaeger.api_v2.RateLimitingSamplingStrategy\x12H\n\x11operationSampling\x18\x04 \x01(\x0b\x32-.jaeger.api_v2.PerOperationSamplingStrategies"1\n\x1aSamplingStrategyParameters\x12\x13\n\x0bserviceName\x18\x01 \x01(\t*<\n\x14SamplingStrategyType\x12\x11\n\rPROBABILISTIC\x10\x00\x12\x11\n\rRATE_LIMITING\x10\x01\x32\xa2\x01\n\x0fSamplingManager\x12\x8e\x01\n\x13GetSamplingStrategy\x12).jaeger.api_v2.SamplingStrategyParameters\x1a\'.jaeger.api_v2.SamplingStrategyResponse"#\x82\xd3\xe4\x93\x02\x1d"\x18/api/v2/samplingStrategy:\x01*B-\n\x17io.jaegertracing.api_v2Z\x06\x61pi_v2\xc8\xe2\x1e\x01\xd0\xe2\x1e\x01\xe0\xe2\x1e\x01\x62\x06proto3',
    dependencies=[
        gogoproto_dot_gogo__pb2.DESCRIPTOR,
        google_dot_api_dot_annotations__pb2.DESCRIPTOR,
    ],
)

_SAMPLINGSTRATEGYTYPE = _descriptor.EnumDescriptor(
    name="SamplingStrategyType",
    full_name="jaeger.api_v2.SamplingStrategyType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="PROBABILISTIC",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="RATE_LIMITING",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=919,
    serialized_end=979,
)
_sym_db.RegisterEnumDescriptor(_SAMPLINGSTRATEGYTYPE)

SamplingStrategyType = enum_type_wrapper.EnumTypeWrapper(_SAMPLINGSTRATEGYTYPE)
PROBABILISTIC = 0
RATE_LIMITING = 1


_PROBABILISTICSAMPLINGSTRATEGY = _descriptor.Descriptor(
    name="ProbabilisticSamplingStrategy",
    full_name="jaeger.api_v2.ProbabilisticSamplingStrategy",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="samplingRate",
            full_name="jaeger.api_v2.ProbabilisticSamplingStrategy.samplingRate",
            index=0,
            number=1,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=85,
    serialized_end=138,
)


_RATELIMITINGSAMPLINGSTRATEGY = _descriptor.Descriptor(
    name="RateLimitingSamplingStrategy",
    full_name="jaeger.api_v2.RateLimitingSamplingStrategy",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="maxTracesPerSecond",
            full_name="jaeger.api_v2.RateLimitingSamplingStrategy.maxTracesPerSecond",
            index=0,
            number=1,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=140,
    serialized_end=198,
)


_OPERATIONSAMPLINGSTRATEGY = _descriptor.Descriptor(
    name="OperationSamplingStrategy",
    full_name="jaeger.api_v2.OperationSamplingStrategy",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="operation",
            full_name="jaeger.api_v2.OperationSamplingStrategy.operation",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="probabilisticSampling",
            full_name="jaeger.api_v2.OperationSamplingStrategy.probabilisticSampling",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=200,
    serialized_end=323,
)


_PEROPERATIONSAMPLINGSTRATEGIES = _descriptor.Descriptor(
    name="PerOperationSamplingStrategies",
    full_name="jaeger.api_v2.PerOperationSamplingStrategies",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="defaultSamplingProbability",
            full_name="jaeger.api_v2.PerOperationSamplingStrategies.defaultSamplingProbability",
            index=0,
            number=1,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="defaultLowerBoundTracesPerSecond",
            full_name="jaeger.api_v2.PerOperationSamplingStrategies.defaultLowerBoundTracesPerSecond",
            index=1,
            number=2,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="perOperationStrategies",
            full_name="jaeger.api_v2.PerOperationSamplingStrategies.perOperationStrategies",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="defaultUpperBoundTracesPerSecond",
            full_name="jaeger.api_v2.PerOperationSamplingStrategies.defaultUpperBoundTracesPerSecond",
            index=3,
            number=4,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=326,
    serialized_end=552,
)


_SAMPLINGSTRATEGYRESPONSE = _descriptor.Descriptor(
    name="SamplingStrategyResponse",
    full_name="jaeger.api_v2.SamplingStrategyResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="strategyType",
            full_name="jaeger.api_v2.SamplingStrategyResponse.strategyType",
            index=0,
            number=1,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="probabilisticSampling",
            full_name="jaeger.api_v2.SamplingStrategyResponse.probabilisticSampling",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="rateLimitingSampling",
            full_name="jaeger.api_v2.SamplingStrategyResponse.rateLimitingSampling",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="operationSampling",
            full_name="jaeger.api_v2.SamplingStrategyResponse.operationSampling",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=555,
    serialized_end=866,
)


_SAMPLINGSTRATEGYPARAMETERS = _descriptor.Descriptor(
    name="SamplingStrategyParameters",
    full_name="jaeger.api_v2.SamplingStrategyParameters",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="serviceName",
            full_name="jaeger.api_v2.SamplingStrategyParameters.serviceName",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=868,
    serialized_end=917,
)

_OPERATIONSAMPLINGSTRATEGY.fields_by_name[
    "probabilisticSampling"
].message_type = _PROBABILISTICSAMPLINGSTRATEGY
_PEROPERATIONSAMPLINGSTRATEGIES.fields_by_name[
    "perOperationStrategies"
].message_type = _OPERATIONSAMPLINGSTRATEGY
_SAMPLINGSTRATEGYRESPONSE.fields_by_name[
    "strategyType"
].enum_type = _SAMPLINGSTRATEGYTYPE
_SAMPLINGSTRATEGYRESPONSE.fields_by_name[
    "probabilisticSampling"
].message_type = _PROBABILISTICSAMPLINGSTRATEGY
_SAMPLINGSTRATEGYRESPONSE.fields_by_name[
    "rateLimitingSampling"
].message_type = _RATELIMITINGSAMPLINGSTRATEGY
_SAMPLINGSTRATEGYRESPONSE.fields_by_name[
    "operationSampling"
].message_type = _PEROPERATIONSAMPLINGSTRATEGIES
DESCRIPTOR.message_types_by_name["ProbabilisticSamplingStrategy"] = (
    _PROBABILISTICSAMPLINGSTRATEGY
)
DESCRIPTOR.message_types_by_name["RateLimitingSamplingStrategy"] = (
    _RATELIMITINGSAMPLINGSTRATEGY
)
DESCRIPTOR.message_types_by_name["OperationSamplingStrategy"] = (
    _OPERATIONSAMPLINGSTRATEGY
)
DESCRIPTOR.message_types_by_name["PerOperationSamplingStrategies"] = (
    _PEROPERATIONSAMPLINGSTRATEGIES
)
DESCRIPTOR.message_types_by_name["SamplingStrategyResponse"] = _SAMPLINGSTRATEGYRESPONSE
DESCRIPTOR.message_types_by_name["SamplingStrategyParameters"] = (
    _SAMPLINGSTRATEGYPARAMETERS
)
DESCRIPTOR.enum_types_by_name["SamplingStrategyType"] = _SAMPLINGSTRATEGYTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProbabilisticSamplingStrategy = _reflection.GeneratedProtocolMessageType(
    "ProbabilisticSamplingStrategy",
    (_message.Message,),
    {
        "DESCRIPTOR": _PROBABILISTICSAMPLINGSTRATEGY,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.ProbabilisticSamplingStrategy)
    },
)
_sym_db.RegisterMessage(ProbabilisticSamplingStrategy)

RateLimitingSamplingStrategy = _reflection.GeneratedProtocolMessageType(
    "RateLimitingSamplingStrategy",
    (_message.Message,),
    {
        "DESCRIPTOR": _RATELIMITINGSAMPLINGSTRATEGY,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.RateLimitingSamplingStrategy)
    },
)
_sym_db.RegisterMessage(RateLimitingSamplingStrategy)

OperationSamplingStrategy = _reflection.GeneratedProtocolMessageType(
    "OperationSamplingStrategy",
    (_message.Message,),
    {
        "DESCRIPTOR": _OPERATIONSAMPLINGSTRATEGY,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.OperationSamplingStrategy)
    },
)
_sym_db.RegisterMessage(OperationSamplingStrategy)

PerOperationSamplingStrategies = _reflection.GeneratedProtocolMessageType(
    "PerOperationSamplingStrategies",
    (_message.Message,),
    {
        "DESCRIPTOR": _PEROPERATIONSAMPLINGSTRATEGIES,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.PerOperationSamplingStrategies)
    },
)
_sym_db.RegisterMessage(PerOperationSamplingStrategies)

SamplingStrategyResponse = _reflection.GeneratedProtocolMessageType(
    "SamplingStrategyResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _SAMPLINGSTRATEGYRESPONSE,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.SamplingStrategyResponse)
    },
)
_sym_db.RegisterMessage(SamplingStrategyResponse)

SamplingStrategyParameters = _reflection.GeneratedProtocolMessageType(
    "SamplingStrategyParameters",
    (_message.Message,),
    {
        "DESCRIPTOR": _SAMPLINGSTRATEGYPARAMETERS,
        "__module__": "sampling_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.SamplingStrategyParameters)
    },
)
_sym_db.RegisterMessage(SamplingStrategyParameters)


DESCRIPTOR._options = None

_SAMPLINGMANAGER = _descriptor.ServiceDescriptor(
    name="SamplingManager",
    full_name="jaeger.api_v2.SamplingManager",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=982,
    serialized_end=1144,
    methods=[
        _descriptor.MethodDescriptor(
            name="GetSamplingStrategy",
            full_name="jaeger.api_v2.SamplingManager.GetSamplingStrategy",
            index=0,
            containing_service=None,
            input_type=_SAMPLINGSTRATEGYPARAMETERS,
            output_type=_SAMPLINGSTRATEGYRESPONSE,
            serialized_options=b'\202\323\344\223\002\035"\030/api/v2/samplingStrategy:\001*',
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_SAMPLINGMANAGER)

DESCRIPTOR.services_by_name["SamplingManager"] = _SAMPLINGMANAGER

# @@protoc_insertion_point(module_scope)
