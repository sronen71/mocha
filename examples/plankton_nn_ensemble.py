import csv
import numpy as np
import caffe

#SUBMISSIONS = ["val9.csv",
#               "val10.csv"]
SUBMISSIONS = ["plankton/submissions/submit_inet9_shai.csv",
               "plankton/submissions/submit_inet10_shai.csv"]

PRETRAINED_MODEL = "plankton/snapshots/ensemble/inet_train_iter_60000.caffemodel"
MODEL_DEF = "plankton/models/ensemble/inet_deploy_32_121.prototxt"
OUTPUT = "plankton/submissions/submit_nn_ensemble_4.csv"


# returns tuple of header (column titles) and averages_probs structure
# average probs is dictionary of <image ID> -> <submission file index> -> <column index> -> <prob>
def read_submissions(submissions):

    if len(submissions) == 0:
        return [], [], [], []
        
    num_submissions = len(submissions)
    
    # open the first file 
    header = []
    labels = []
    identifiers = []
    num_records = 0
    num_classes = 0
    with open(submissions[0],'r') as submission_csv:
        reader = csv.reader(submission_csv)
        first = True
        for row in reader:
            # skip header
            if first:
                header = row
                first = False
                continue
            num_records += 1
            if len(row) == len(header)+1:
                # csv has labels in final column                
                labels.append(int(row[len(row)-1]))
            else:
                assert len(row) == len(header)
            identifiers.append(row[0])
            # first column is image identifier
            num_classes = len(header)-1
    
    probs = np.zeros((num_records,num_classes*num_submissions))
    for submission_idx in range(len(submissions)):
        submission = submissions[submission_idx]
        print submission
        with open(submission,'r') as submission_csv:
            reader = csv.reader(submission_csv)
            first = True
            record_num = 0
            for row in reader:
                # skip header
                if first:
                    header = row
                    first = False
                    continue

                # skip the first column
                for col_idx in range(1,num_classes+1):
                    probs[record_num,num_classes*submission_idx+col_idx-1] = float(row[col_idx])
                record_num += 1
                        
#    average_probs = {} # map of weighted averages of probabilities
#    first_submission = True
#    header = []
#    labels = {}
#    for submission_index in range(len(submissions)):
#        submission = submissions[submission_index]
#        print submission
#        with open(submission,'r') as submission_csv:
#            reader = csv.reader(submission_csv)
#            first = True
#            record_num = 0
#            for row in reader:
#                # skip header
#                if first:
#                    header = row
#                    first = False
#                    continue
#
#                identifier=row[0]
#                if first_submission:
#                    average_probs[identifier] = {}
#                    if len(row) == len(header)+1:
#                        # csv has labels in final column
#                        #labels.append(int(row[len(row)-1]))   
#                        labels[record_num] = int(row[len(row)-1])
#                        #print identifier,int(row[len(row)-1])  
#                    else:
#                        assert len(row) == len(header)                
#                else:
#                    if row[0] not in average_probs:
#                        print "ERROR: identifier in submission %s invalid: %s" % \
#                            (submission,identifier)
#                        return
#                average_probs[identifier][submission_index] = {}
#
#                for col_index in range(1,len(header)):
#                    average_probs[identifier][submission_index][col_index] = float(row[col_index])
#                record_num += 1

#        first_submission = False
#    print probs[0:2,:]
    return header, probs, identifiers, labels

def write_output(output_file,header,identifiers,final_probs):
    print "Writing to",output_file
    with open(output_file,'w') as output_csv:
        writer=csv.writer(output_csv)
        writer.writerow(header)

        for i in range(final_probs.shape[0]):
            output_row = [identifiers[i]]
            sum = 0.0
            for j in range(final_probs.shape[1]):
                output_row.append("%.16e" % final_probs[i,j])
                sum += final_probs[i,j]
            writer.writerow(output_row)
            

def transform_probs(submission_probs):
    first_prob = submission_probs[submission_probs.keys()[0]]
#    print submission_probs.keys()[0],first_prob
    num_submissions = len(first_prob)
    num_columns = len(first_prob[first_prob.keys()[0]])
    num_records = len(submission_probs)
#    probs = []
    probs = np.zeros((num_records,num_columns*num_submissions))
    # average probs is dictionary of <image ID> -> <submission file index> -> <column index> -> <prob>
    for record_idx in range(len(submission_probs.keys())):
        record_id = submission_probs.keys()[record_idx]
        submission_map = submission_probs[record_id]
#        probs.append([0]*num_columns*num_submissions)
        for submission_idx, prob_map in submission_map.iteritems():
            for col_idx, prob in prob_map.iteritems():
                probs[record_idx,num_columns*submission_idx+col_idx-1] = prob
    
#    print probs[0,0:121]
#    print probs[0,121:]
    return probs    

def generate_output_validation(header, probs, labels, trained_model_file, trained_model):
    print "Try to create net..."
    net = caffe.Classifier(trained_model_file, trained_model,
        # mean=np.load("plankton/mean"+DATA_SUFFIX+SUFFIX+".npy")
        )
    print "Created net"
    net.set_phase_test()
    net.set_mode_gpu()
    # transform individual probabilities into horizontal stack
#    probs = transform_probs(submission_probs)    
    probs.astype(np.float32)
    probs = probs.reshape((probs.shape[0],1,1,probs.shape[1]))
    preds = net.predict(probs,oversample=False) 
#    print preds[0:3,:]
    print preds.shape
    
    if len(labels) > 0:
        final_probs=np.zeros(preds.shape[0])
        for i,label in enumerate(labels):
    #        print i, label
            final_probs[i]=preds[i,label]
        score=-np.mean(np.log(final_probs))
        print "log loss score:", score

    return preds

def main():
    print "Output file:",OUTPUT

    # get submissions from submit files
    header, submission_probs, identifiers, labels = read_submissions(SUBMISSIONS)
    
    # run submissions through nn ensemble
    final_probs = generate_output_validation(header,submission_probs,labels,MODEL_DEF,PRETRAINED_MODEL)
    
    # write output
    write_output(OUTPUT,header,identifiers,final_probs)
    print "Done!"

if __name__ == "__main__":
    main()




