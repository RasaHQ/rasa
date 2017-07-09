import json, os.path, hashlib

class TrainerUtils():
    def __init__(self, logger):
      self.config = False
      self.logger = logger
      
      self.load_config()
      self.check_pending_data_file()

    def load_config(self):
      config_file_path = os.path.dirname(__file__) + '/../../trainer_conf.json'

      if os.path.exists(config_file_path) :
        with open(config_file_path) as config_file:
          config = json.load(config_file)
      else :
        self.logger.info("Cannot find trainer_conf.json in " + config_file_path + ", thus the trainer util won't be used")

      self.config = config

    def check_pending_data_file(self):
      if not self.config :
        return
      
      pending_file_path =  os.path.dirname(__file__) + '/../../' + self.config['pending_file']
      if not os.path.exists(pending_file_path) :
        self.logger.info("Pending string file doesn't exists")
        with open(pending_file_path,"w") as pending_file:
          pending_file.write("{}")
          self.logger.info("Pending string file created")

    def process_response(self, response):
      if not self.config :
        return

      if not response['intent'] or response['intent']['confidence'] < self.config['threshold']:
        string_hash = hashlib.md5(response['text']).hexdigest()

        pending_file_path =  os.path.dirname(__file__) + '/../../' + self.config['pending_file']
        with open(pending_file_path,"r+") as pending_file:
          try :
            pending_file_data = json.load(pending_file)
          except ValueError:
            pending_file_data = {}
          
          if string_hash not in pending_file_data :
            pending_file_data.setdefault(string_hash, response['text'])
            pending_file.seek(0)
            pending_file.write(json.dumps(pending_file_data))
            pending_file.flush()
            self.logger.info("Pending string added")

    