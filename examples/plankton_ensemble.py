import csv


#submissions = ["plankton/submit_16_sorted.csv","plankton/submit_24_sorted.csv","plankton/submit_48_sorted.csv"]
#weights = [0.21952193,0.31418002,0.46629805]

# submissions = ["plankton/submit_48_notest_v4.csv","plankton/submit_48_notest_v2.csv","plankton/submit_48_notest.csv","plankton/submit_48_notest_v3.csv"]
# weights = [1.0/len(submissions)]*len(submissions)

#ensemble 6
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/sorted/submit_48_notest_v2.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/sorted/submit_48_notest_v3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv"]
# weights = [0.125,0.125,0.125,0.125,0.25,0.25]

### ensemble 7
# submissions = ["plankton/submissions/sorted/submit_48_notest_v2.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv"]
# weights = [1.0/len(submissions)]*len(submissions)

### ensemble 8 (bagging of model_48_5-3-3-3)
# submissions = ["plankton/submissions/submit_48_5-3-3-3_iter20000_split0.csv",
#                "plankton/submissions/submit_48_5-3-3-3_iter20000_split1.csv",
#                "plankton/submissions/submit_48_5-3-3-3_iter20000_split2.csv",
#                "plankton/submissions/submit_48_5-3-3-3_iter20000_split3.csv",
#                "plankton/submissions/submit_48_5-3-3-3_iter20000_split4.csv"]
# weights = [1.0/len(submissions)]*len(submissions)

### ensemble 9
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/sorted/submit_48_notest_v3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv"]
# weights = [0.1,0.2,0.1,0.1,0.25,0.25]

### ensemble 10
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/sorted/submit_48_notest_v3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv"]
# weights = [0.125,0.125,0.125,0.125,0.25,0.25]

### ensemble 11 (bagging of model_48_6-5-4-3)
# submissions = ["plankton/submissions/submit_48_6-5-4-3_iter20000_split0.csv",
#                "plankton/submissions/submit_48_6-5-4-3_iter20000_split1.csv",
#                "plankton/submissions/submit_48_6-5-4-3_iter20000_split2.csv",
#                "plankton/submissions/submit_48_6-5-4-3_iter20000_split3.csv",
#                "plankton/submissions/submit_48_6-5-4-3_iter20000_split4.csv"]
# weights = [1.0/len(submissions)]*len(submissions)

### ensemble 12
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/submit_ensemble_11.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv",
#                "plankton/submissions/submit_online_6-5-3-3_48insteadof52.csv"]
# weights = [1.0,2.0,1.0,2.0,3.0,3.0,4.0]
# output = "plankton/submissions/submit_ensemble_12.csv"

### ensemble 13
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/submit_ensemble_11.csv",
#                "plankton/submissions/submit_48_3x3-3x3-3-3_iter20000.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv",
#                "plankton/submissions/submit_online_6-5-3-3_48insteadof52.csv"]
# weights = [1.0,2.0,1.0,2.0,3.0,3.0,3.0,4.0]
# output = "plankton/submissions/submit_ensemble_13.csv"

### ensemble 14
# submissions = ["plankton/submissions/sorted/submit_48_notest_v4.csv",
#                "plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/sorted/submit_48_notest.csv",
#                "plankton/submissions/submit_ensemble_11.csv",
#                "plankton/submissions/submit_48_3x3-3x3-3-3_iter20000_oversampled.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3.csv",
#                "plankton/submissions/sorted/submit_online_6-5-3-3_more_iters.csv",
#                "plankton/submissions/submit_online_6-5-3-3_48insteadof52.csv"]
# weights = [1.0,2.0,1.0,2.0,3.0,3.0,3.0,4.0]
# output = "plankton/submissions/submit_ensemble_14.csv"

### ensemble 15
# submissions = ["plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/submit_ensemble_11.csv",
#                "plankton/submissions/submit_48_3x3-3x3-3-3_iter20000_oversampled.csv",
#                "plankton/submissions/submit_online_6-5-3-3_48insteadof52.csv"]
# weights = [1.0,1.0,2.0,2.0]
# output = "plankton/submissions/submit_ensemble_15.csv"

### ensemble 16
# submissions = ["plankton/submissions/submit_ensemble_8.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv",
#                "plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv"]
# weights = [1.0,1.0,2.0]
# output = "plankton/submissions/submit_ensemble_16.csv"

### ensemble 17
# submissions = ["plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv",
#                "plankton/submissions/submit_inet7_shai.csv"]
# weights = [0.4,0.6]
# output = "plankton/submissions/submit_ensemble_17.csv"


### ensemble 18
# submissions = ["plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv",
#                "plankton/submissions/submit_inet7_shai.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv"]
# weights = [0.3,0.6,0.1]
# output = "plankton/submissions/submit_ensemble_18.csv"


### ensemble 19 - hierarchy ensemble
# submissions = ["plankton/submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levelc_iter20000_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levelb_iter20000_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levela_iter20000_oversampled.csv"]
# weights = [0.8,0.1,0.07,0.03]
# output = "plankton/submissions/submit_ensemble_19.csv"


### ensemble 20 - hierarchy ensemble with better inference
# submissions = ["plankton/submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levelc_iter20000_ind_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levelb_iter20000_ind_oversampled.csv",
#                "plankton/submissions/submit_48_3x3-6-5-3_levela_iter20000_ind_oversampled.csv"]
# weights = [0.4,0.3,0.2,0.1]
# output = "plankton/submissions/submit_ensemble_20.csv"

### ensemble 21 - similar to 18, using the (better) ensemble from 20
# submissions = ["plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv",
#                "plankton/submissions/submit_inet7_shai.csv",
#                "plankton/submissions/submit_ensemble_20.csv"]
# weights = [2,3,1]
# output = "plankton/submissions/submit_ensemble_21.csv"

### ensemble 22 - similar to 21, different weights
# submissions = ["plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv",
#                "plankton/submissions/submit_inet7_shai.csv",
#                "plankton/submissions/submit_ensemble_20.csv"]
# weights = [2,4,1]
# output = "plankton/submissions/submit_ensemble_22.csv"

### ensemble 23 - adding in inet9
submissions = ["plankton/submissions/submit_inet9_shai.csv",
               "plankton/submissions/submit_imp-online_6-5-3-3_oversampled.csv",
               "plankton/submissions/submit_inet7_shai.csv",
               "plankton/submissions/submit_ensemble_20.csv"]
weights = [6,1,2,1]
output = "plankton/submissions/submit_ensemble_23.csv"

print "(Unnormalized) weights sum to", sum(weights)
normalized_weights = [float(w)/sum(weights) for w in weights]

print "Output file:",output

def main():

    if len(submissions) == 0:
        return

    average_probs = {} # map of weighted averages of probabilities
    first_submission = True
    header = []
    for submission_index in range(len(submissions)):
        submission = submissions[submission_index]
        weight = normalized_weights[submission_index]
        print submission,weight
        with open(submission,'r') as submission_csv:
            reader = csv.reader(submission_csv)
            first = True
            for row in reader:
                # skip header
                if first:
                    header = row
                    first = False
                    continue

                identifier=row[0]
                if first_submission:
                    average_probs[identifier] = {}
                    for col_index in range(1,len(row)):
                        average_probs[identifier][header[col_index]] = 0.0
                else:
                    if row[0] not in average_probs:
                        print "ERROR: identifier in submission %s invalid: %s" % \
                            (submission,identifier)
                        return
                        
                for col_index in range(1,len(row)):
                    average_probs[identifier][header[col_index]] += weight*float(row[col_index])

        first_submission = False

    print "Writing to",output
    with open(output,'w') as output_csv:
        writer=csv.writer(output_csv)
        writer.writerow(header)

        for identifier in average_probs:
            output_row = [identifier]

            for column_title in header:
                if column_title == "image":
                    continue

                output_row.append("%.16e" % average_probs[identifier][column_title])

            writer.writerow(output_row)
    print "Done!"

if __name__ == "__main__":
    main()




