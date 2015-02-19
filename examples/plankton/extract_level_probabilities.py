#!/usr/bin/env python
import level_map

SUBMIT_FILE=root="submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv"
MAPPING_FILENAMES = ["../../data/plankton/leveld_levelc_mapping.csv",
                     "../../data/plankton/levelc_levelb_mapping.csv",
                     "../../data/plankton/levelb_levela_mapping.csv"]


def parse_submission_ind(submit_file):
    submit_data = open(submit_file,'r')
    first_line = True
    headers = []
    probabilities = {}
    count = 0
    for line in submit_data:
        line = line.rstrip()
        if first_line:
            headers = line.split(',')
            first_line = False
            continue

        count += 1
        fields = line.split(',')
        assert len(fields) == len(headers)
        image_name = fields[0]
        probabilities[image_name] = {}

        for i in range(1,len(fields)):
            probabilities[image_name][headers[i]] = float(fields[i])
            
    return probabilities

def parse_submission(submit_file):
    submit_data = open(submit_file,'r')
    first_line = True
    headers = []
    sum_probabilities = []
    count = 0
    for line in submit_data:
        line = line.rstrip()
        if first_line:
            headers = line.split(',')
            for _ in headers:
                sum_probabilities.append(0.0)
            first_line = False
            continue

        count += 1
        fields = line.split(',')
        assert len(fields) == len(headers)
        for i in range(1,len(fields)):
            sum_probabilities[i] += float(fields[i])

    probabilities = {}
    for i in range(1,len(headers)):
        probabilities[headers[i]] = sum_probabilities[i] / count

    return probabilities

def invert_map(map):
    inverted_map = {}
    for key in map:
        value = map[key][1]
        if value in inverted_map:
            value_list = inverted_map[value]
            if key not in value_list:
                value_list.append(key)
        else:
            inverted_map[value] = [key]
    return inverted_map

def derive_label_probabilities(probabilities,level_map):
    derived_probs = {}
    for key in level_map:
        derived_probs[key] = {}
        level_probs = derived_probs[key]
        sum_probs = 0.0
        for sublevel in level_map[key]:
            prob = probabilities[sublevel]
            level_probs[sublevel] = prob
            sum_probs += prob
        for sublevel in level_probs:
            level_probs[sublevel] /= sum_probs

    return derived_probs

def derive_label_probabilities_ind(probabilities,level_map):
    derived_probs = {}
    for image_name in probabilities:
        derived_probs[image_name] = \
            derive_label_probabilities(probabilities[image_name],level_map)
    return derived_probs


def get_derived_probabilities(prob_file,mapping_files):
    probabilities = parse_submission(prob_file)
    levmap = level_map.get_level_map(mapping_files)
    inverted_levmap = invert_map(levmap)
    return derive_label_probabilities(probabilities,inverted_levmap)

def get_derived_probabilities_ind(prob_file,mapping_files):
    probabilities = parse_submission_ind(prob_file)
    levmap = level_map.get_level_map(mapping_files)
    inverted_levmap = invert_map(levmap)
    return derive_label_probabilities_ind(probabilities,inverted_levmap)

def print_steps():
    probabilities = parse_submission(SUBMIT_FILE)
    print len(probabilities),probabilities
    print "--"
    levmap = level_map.get_level_map(MAPPING_FILENAMES)
    print len(levmap),levmap
    print "--"
    inverted_levmap = invert_map(levmap)
    print len(inverted_levmap), inverted_levmap
    print "--"
    derived_probs = derive_label_probabilities(probabilities,inverted_levmap)
    print len(derived_probs),derived_probs


def print_steps_ind():
    probabilities = parse_submission_ind(SUBMIT_FILE)
    # print len(probabilities),probabilities
    print "--"
    levmap = level_map.get_level_map(MAPPING_FILENAMES)
    print len(levmap),levmap
    print "--"
    inverted_levmap = invert_map(levmap)
    print len(inverted_levmap), inverted_levmap
    print "--"
    derived_probs = derive_label_probabilities_ind(probabilities,inverted_levmap)
    print len(derived_probs)
    count = 0
    for derived_prob_ind in derived_probs:
        print len(derived_prob_ind),derived_prob_ind, derived_probs[derived_prob_ind]
        count += 1
        if count > 5:
            break


def main():
    derived_probs = get_derived_probabilities(SUBMIT_FILE,MAPPING_FILENAMES)
    print len(derived_probs),derived_probs

if __name__ == "__main__":
    # main()
    print_steps_ind()
