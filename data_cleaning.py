"""Created by Daniel-Iosif Trubacs for the AI society on 6 December 2022. The aim of this module is to clean the
CortAI dataset. The CortAI contains classified clauses separated by rows into a txt file. The cleaned data is saved
into the json format: {'text_data', 'label'}. More information about the labelled data can be found here: """

import re
import json

# reading the training dataset
# the type of data to be read (train or validation)
data_type = 'validation'
with open('cortAI/' + data_type + '.txt', 'r', encoding="utf8") as file:
    contents = file.readlines()

# the labels used to classify the text
labels = ['unc', 'ltd2', 'ter3', 'ch2', 'ter2', 'use2', 'a2', 'j3', 'ltd3', 'ltd1', 'law2', 'j1', 'cr3', 'cr2', 'law1',
          'a3', 'j2', 'a4']

# the number of data points in the saved dataset
n_data = 0

# dictionary object in which the data will be saved
dict_save = None

# start from i+1 in the contents array because the first elements is just the header of the table ('content,label')
for i in range(len(contents) - 1):
    # going through every label
    # check if the data was corrupted and there are more than 2 labels
    n_labels = 0
    # the data that will be saved
    saved_data = []

    # the label that will be saved
    saved_label = []

    # going through every possible label
    for label in labels:
        if label in contents[i + 1]:
            n_labels += 1
            saved_label = label

    # save the data only if the number of labels found is 1
    if n_labels == 1:
        n_data += 1
        # processing the data to keep only the necessary text
        # removing the label from the original text
        saved_data = re.sub(saved_label, ' ', contents[i + 1])

        # removing the coma mark from the end
        saved_data = saved_data[0:len(saved_data) - 3]

        # if there any quotation marks at the beginning and end remove them
        if saved_data[0] == '"':
            saved_data = saved_data[1:len(saved_data) - 1]

        # save the data in a dictionary object
        if dict_save is None:
            dict_save = {'text_data': [saved_data], 'labels': [saved_label]}

        else:
            dict_save['text_data'].append(saved_data)
            dict_save['labels'].append(saved_label)

print("There are", n_data, "data points in the cleaned dataset.")

# save the data in json format
with open(data_type + '_data.json', 'w') as json_file:
    json.dump(dict_save, json_file)
    print("The data has been saved in", data_type+'_data.json')
