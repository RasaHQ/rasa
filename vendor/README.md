This folder contains some dependencies.

# jaeger-python-proto

This is the generated gRPC code for the Jaeger query API.

Unfortunately, this does not seem to get published as an artifact in any way.

To regenerate these files, follow these steps (taken from [here](https://stackoverflow.com/questions/59577629/compiling-jaeger-grpc-proto-files-with-python)).

```
git clone --recurse-submodules https://github.com/jaegertracing/jaeger-idl
cd jaeger-idl/
make proto
```
And then use the files generated in the folder `proto-gen-python`.

Note that this code is only needed for running the integration tests and does not need to be part
of any published Rasa Pro artifacts.
