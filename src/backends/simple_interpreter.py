class HelloGoodbyeInterpreter(object):
    def __init__(self):
        self.name="hello_goodbye"
        self.his = ["hello","hi","hey"]
        self.byes = ["bye","goodbye"]

    def parse(self,text):
        text, intent = text, "None"

        is_present = lambda x: x in text.lower()
        if (True in map(is_present,self.his)):
            intent="hello"
        elif (True in map(is_present,self.byes)):
            intent="goodbye"

        return {'text':text,'intent':intent,'entities':[]}
