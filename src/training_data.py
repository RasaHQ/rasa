import json, warnings, re
from mitie import tokenize

class TrainingData(object):
    def __init__(self,filename):
        self.intent_examples = []
        self.entity_examples = []
        self.filename = filename
        self.filedata = json.loads(open(filename,'rb').read())
        self.fformat = self.guess_format(self.filedata)
        
        if (self.fformat == 'luis'):
            self.load_luis_data(self.filedata)
        elif (self.fformat == 'wit'):
            self.load_wit_data(self.filedata)
        elif (self.fformat == 'parsa'):
            self.load_data(self.filedata)
        else:
            raise ValueError("unknown training file format : {0}".format(self.fformat))
    
    def as_json(self,**kwargs):
        return json.dumps( {
          "parsa_data" : {
            "intent_examples" : self.intent_examples,
            "entity_examples" : self.entity_examples
          }
        }, **kwargs)
        
            
    def guess_format(self,filedata):
        fformat = 'unk'
        if (filedata.has_key("data") and type(filedata.get("data")) is list ):
            fformat = 'wit'
        elif (filedata.has_key("luis_schema_version")):
            fformat = 'luis'
        return fformat
        
    def load_wit_data(self,data):
        for s in data["data"]:
            entities = s.get("entities")
            if (entities is None): continue
            text = s.get("text")
            intents = [e["value"] for e in entities if e["entity"]=='intent']
            intent = intents[0] if intents else 'None'
            
            entities = [e for e in entities if (e.has_key("start") and e.has_key("end"))]
            for e in entities:
                e["value"] = e["value"][1:-1]

            self.intent_examples.append({"text":text,"intent":intent})            
            self.entity_examples.append({"text":text,"intent":intent,"entities":entities})
            
    def load_luis_data(self,data):
        warnings.warn(
        """LUIS data may not always be correctly imported because entity locations are specified by tokens. 
        If you use a tokenizer which behaves differently from LUIS's your entities might not be correct""")
        for s in data["utterances"]:
            text = unicode(s.get("text"))
            tokens = [t.decode('utf-8') for t in tokenize(text)]
            intent = s.get("intent")
            entities = []
            for e in s.get("entities") or []:
                i, ii = e["startPos"], e["endPos"]+1
                #print(u"full text:  {0}".format(text))
                val = u"\s*".join([unicode(s) for s in tokens[i:ii+1]])
                #print(u"entity val : {0}".format(val))
                expr = re.compile(val)
                m = expr.search(text)
                start, end = m.start(), m.end()
                #print(u"match : {0}".format(m.group()))
                #print(text[start:end])
                entities.append({"entity":e["entity"],"value":val,"start":start,"end":end})

            self.intent_examples.append({"text":text,"intent":intent})            
            self.entity_examples.append({"text":text,"intent":intent,"entities":entities})
                            
                
            
        
    def load_data(self,data):
        raise NotImplementedError()
        
        
