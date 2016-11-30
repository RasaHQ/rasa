class Interpreter(object):
    def parse(self, text):
        raise NotImplementedError()


config_keys = [
  "backend",
  "config",
  "data",
  "emulate",
  "language",
  "mitie_file",
  "path",
  "port",
  "server_model_dir",
  "token",
  "write"
]
