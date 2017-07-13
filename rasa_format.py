import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def rasa_format_intent(in_path, out_path, intent):
	"""
	Convert info from a nlu_benchmark training files into rasa's format.
	"""
	with open(in_path) as data_file:    
	    data = json.load(data_file)[intent]

	common_examples = []
	for msg in data:
		chunks = []
		for section in msg["data"]:
			chunks.append(section["text"])
		message = u''.join(chunks)
		rasa = {
				"text": message,
				"intent": intent,
				"entities": []
				}
		#print("rasa_entry: {}".format(rasa))
		common_examples.append(rasa)

	with open(out_path, "w") as outfile:
	    json.dump({"rasa_nlu_data": {"common_examples":common_examples}}, outfile, indent=4)


rasa_format_intent('/home/sarenne/nlu_benchmark/SearchScreeningEvent/train_SearchScreeningEvent.json',
		'/home/sarenne/nlu_benchmark/rasa_data/train_SearchScreeningEvent_rasa.json', 
		'SearchScreeningEvent')


p_list = ['/home/sarenne/nlu_benchmark/rasa_data/train_AddToPlaylist_rasa.json', 
			'/home/sarenne/nlu_benchmark/rasa_data/train_GetWeather_rasa.json', 
			'/home/sarenne/nlu_benchmark/rasa_data/train_BookRestaurant_rasa.json', 
			'/home/sarenne/nlu_benchmark/rasa_data/train_SearchScreeningEvent_rasa.json']
o_path = '/home/sarenne/nlu_benchmark/rasa_data/4intents'
s = .8

def combined_intents(path_list, out_path, split, validation=False):
	"""
	Create a test and train set (sizes specified by `split`) from json files in the `path_list`.  If validation is set to True, 
	return a validation set constructed using the same split value from the training set (ie. split = 0.8 validation is (1-0.8)% 
	of training, which is 0.8% of total)
	"""
	combined_intents_train = []
	combined_intents_test = []
	combined_intents_valid = []
	for path in path_list:
		with open(path) as data_file:
			intent_dict = json.load(data_file)['rasa_nlu_data']['common_examples']
			combined_intents_train.extend(intent_dict[:int(len(intent_dict)*s)])
			combined_intents_test.extend(intent_dict[int(len(intent_dict)*s):])
			print("intent_dict * s: {}, training set: {}, test set: {}".format(int(len(intent_dict)*(s)), len(combined_intents_train), len(combined_intents_test)))
			if validation:
				combined_intents_v_train.extend(combined_intents_train[:int(len(combined_intents_train * s))])
				combined_intents_v_valid.extend(combined_intents_train[int(len(combined_intents_train)*s):])

			#combined_intents.extend(i)
	    	#print("data file: {}\n-----".format(i))
	if validation:
		with open(out_path + '_train.json', "w") as outfile:
	   		json.dump({"rasa_nlu_data": {"common_examples":combined_intents_v_train}}, outfile, indent=2)
		with open(out_path + '_validation.json', "w") as outfile:
			json.dump({"rasa_nlu_data": {"common_examples":combined_intents_v_valid}}, outfile, indent=2)
		with open(out_path + '_test.json', "w") as outfile:
			json.dump({"rasa_nlu_data": {"common_examples":combined_intents_test}}, outfile, indent=2)
	else:
		with open(out_path + '_train.json', "w") as outfile:
			json.dump({"rasa_nlu_data": {"common_examples":combined_intents_train}}, outfile, indent=2)
		with open(out_path + '_test.json', "w") as outfile:
			json.dump({"rasa_nlu_data": {"common_examples":combined_intents_test}}, outfile, indent=2)

combined_intents(p_list, o_path, s)




