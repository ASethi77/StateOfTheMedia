import json
from datetime import date, timedelta
from pprint import pprint

class LabelLoader(object):

    def __init__(self):
         # default target based on our server and data configuration
         self.target_path = "/opt/nlp_shared/data/presidential_approval_ratings"
         self.target_file = "approval_ratings.json"
         self.labels = {}

    # loads the raw data from the current target file
    def load_json(self):
        self.labels = {} # clear out old data

        with open(self.target_path + "/" + self.target_file) as data_file:
            data = json.load(data_file)
     	
        for president in data:
             for term in president['terms']:
                 for rating in term['ratings']:
                     start = rating['pollingStart'].split("-")
                     end = rating['pollingEnd'].split("-")
                     approve = rating['approve']
                     disapprove = rating['disapprove']
                     neutral = 100 - approve - disapprove

                     start_date = date(int(start[0]), int(start[1]), int(start[2]))
                     end_date = date(int(end[0]), int(end[1]), int(end[2]))
                     for i in range((end_date - start_date).days + 1): # do inclusive day range on both sides
                          curr_date = start_date + timedelta(days=i)
                          if curr_date not in self.labels:
                              self.labels[curr_date] = (float(approve), float(disapprove), float(neutral), 1)
                          else:
                              prev_approval = self.labels[curr_date]
                              num_entries = prev_approval[3]
                              new_approve = (prev_approval[0] * num_entries + approve) / (num_entries + 1)
                              new_disapprove = (prev_approval[1] * num_entries + disapprove) / (num_entries + 1)
                              new_neutral = (prev_approval[2] * num_entries + neutral) / (num_entries + 1)
                              self.labels[curr_date] = (new_approve, new_disapprove, new_neutral, num_entries + 1)

    # write the label vector out to file
    def write_to_file(self, file_name):
         pass

    # read in the label vector from the given file 
    def load_from_file(self, file_name):
         pass

    # gets the currently loaded label vector
    # NOTE: this is a dict with keys that are datetime date objects,
    #       and values of tuples representing (approval, disapproval, neutral, # of polls)
    #       as (double, double, double, int)
    def get_labels(self):
         return self.labels

# test
if __name__ == "__main__":
    label_loader = LabelLoader()
    label_loader.load_json()
    print(label_loader.get_labels())

