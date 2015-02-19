#!/usr/bin/env python

def get_level_map(mapping_filenames):

    num_levels = len(mapping_filenames)+1

    # open all the files, read into arrays
    raw_maps = []
    for mapping_filename in mapping_filenames:
        mapping_file = open(mapping_filename,'r');

        raw_maps.append({})
        raw_map = raw_maps[len(raw_maps)-1]
        num_lines = 0
        for line in mapping_file:
            [lower_label,upper_label] = line.strip().split(",")
            raw_map[lower_label]=upper_label
            num_lines += 1

        assert num_lines == len(raw_map)

        mapping_file.close()

    # check the mappings
    for level_idx in range(num_levels-1):
        this_raw_map = raw_maps[level_idx]

        print "Level %d num_keys=%d num_unique_values=%d" % \
            (level_idx,len(this_raw_map.keys()),len(set(this_raw_map.values())))
        
        if level_idx < num_levels-2:
            next_raw_map = raw_maps[level_idx+1]
            print "NextLevel %d num_keys=%d num_unique_values=%d" % \
                 (level_idx+1,len(next_raw_map.keys()),len(set(next_raw_map.values())))

    level_map = {}
    if len(mapping_filenames) == 0:
        return level_map

    first_level = raw_maps[0]
    for label in first_level:
        mapped_label = label
        for raw_map in raw_maps:
            mapped_label = raw_map[mapped_label] 
        level_map[label] = (-1,mapped_label)

    label_to_index = {}
    for idx,label in enumerate(set(level_map.values())):
        label_to_index[label[1]] = idx
    for encoded_key in level_map:
        mlabel = level_map[encoded_key][1]
        level_map[encoded_key] = (label_to_index[mlabel],mlabel)

    return level_map
