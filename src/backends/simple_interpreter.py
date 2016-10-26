class HelloGoodbyeInterpreter():
    def __init__(self):
        self.name="hello_goodbye"
        self.his = ["hello","hi","hey"]
        self.byes = ["bye","goodbye"]

    def parse(self,text):
        text, intent = text.lower(), "None"

        is_present = lambda x: x in text
        if (True in map(is_present,self.his)):
            intent="hello"
        elif (True in map(is_present,self.byes)):
            intent="goodbye"

        return {'intent':intent,'entities':[]}
