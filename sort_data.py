import json
import numpy as np

# This simply sorts the raw generated dataset to include only a percentile to remove outliers.
#     E.g. interquartile range is the percentile [25, 75]. I use [5,95] to include 90% of recorded values.
#     This is because far outliers could hurt performance of neural network training.
#     And outliers are common given the unpredictable nature of the QAOA optimisation.

def main():
    #Pre-defined fixed parameters.
    no_features = 36
    no_outputs = 6

    #Load training raw data
    with open("dataset_training_raw.json", "r") as f:
        training_dataset = json.load(f)
    
    x_inputs, y_outputs, xmins, xminmaxrange, ymins, yminmaxrange = sort_dataset(training_dataset, no_features, no_outputs, normed=True)
    
    #Save arrays into csv
    np.savetxt("datasets/training_x_inputs.csv", x_inputs, delimiter=",")
    np.savetxt("datasets/training_y_outputs.csv", y_outputs, delimiter=",")
    np.savetxt("datasets/xmins.csv", xmins, delimiter=",")
    np.savetxt("datasets/xminmaxrange.csv", xminmaxrange, delimiter=",")
    np.savetxt("datasets/ymins.csv", ymins, delimiter=",")
    np.savetxt("datasets/yminmaxrange.csv", yminmaxrange, delimiter=",")
    print("training input shape: ", x_inputs.shape)
    print("training output shape: ", y_outputs.shape)
    print("x normalising shape: ", xmins.shape)
    print("y normalising shape: ", ymins.shape)
    print("x normalising range: ", xminmaxrange)
    print("y normalising range: ", yminmaxrange)


    #Load validation raw data
    with open("dataset_validation_raw.json", "r") as f:
        validation_dataset = json.load(f)
    
    #Use normalisation values of training dataset to normalise validation dataset
    x_validation, y_validation = sort_dataset(validation_dataset, no_features, no_outputs, normed=False)
    normalise_array_axis1(x_validation, use_provided = True, arraymins = xmins, arrayrange = xminmaxrange)
    normalise_array_axis1(y_validation, use_provided = True, arraymins = ymins, arrayrange = yminmaxrange)
    np.savetxt("datasets/validation_x_inputs.csv", x_validation, delimiter=",")
    np.savetxt("datasets/validation_y_outputs.csv", y_validation, delimiter=",")
    print("validation input shape: ", x_validation.shape)
    print("validation output shape: ", y_validation.shape)
    print("\nSuccessfully saved all arrays into .csv files within dataset folder.\n")

def sort_dataset(all_data, no_features, no_outputs, normed=True):
    for data in all_data.values():
        no_data = len(data)
        break
    print("\nThere are {} data in this raw dataset".format(4*no_data))
    sample_inputs = np.zeros( (no_data, 4, no_features) )
    sample_outputs = np.zeros( (no_data, 4, no_outputs) )
    for s in range(no_data):
        for k, instance in enumerate(["0.3 weighted", "0.7 weighted", "0.3 nonweighted","0.7 nonweighted"]):
            sample = all_data[instance][str(s)]
            input_J = sample[0]
            angle = np.array( sample[1] )
            angle[:3] = np.mod( angle[:3], np.pi/2 )
            angle[3:] *= -1
            sample_inputs[s, k, :] = input_J
            sample_outputs[s, k, :] = angle

    sample_percentiles = np.percentile(sample_outputs, [5, 95], axis=0)
    sample_mins, sample_maxs = sample_percentiles[0,:,:], sample_percentiles[1,:,:]
    allmin = np.min( sample_mins, axis = 0 )
    allmax = np.max( sample_maxs, axis = 0 )

    samples = []
    for k in range(4):
        sample_min = sample_mins[k, :]
        sample_max = sample_maxs[k, :]
        for s in range(no_data):
            sample_output = sample_outputs[s, k, :]
            if np.all( sample_output >= sample_min ) and np.all( sample_output <= sample_max ):
                samples += [ [list(sample_inputs[s,k,:]), list(sample_output)] ]

    #Vectorize input and outputs
    no_samples = len(samples)
    print("There are {} samples when choosing within the [5,95] percentile of each output feature".format(no_samples))
    x_inputs = np.zeros( ( no_features, no_samples ) ) #size (n,m) where n = no_features, m = no_samples
    y_outputs = np.zeros( ( no_outputs, no_samples ) ) #same for y
    for t, sample in enumerate(samples):
        x_inputs[:, t] = sample[0]
        y_outputs[:, t] = sample[1]
    
    if not normed:
        return x_inputs, y_outputs
    else:
        # Min-Max normalize input and output for each feature:
        # First get min, max of each feature
        xmins, xminmaxrange = normalise_array_axis1(x_inputs)
        ymins, yminmaxrange = normalise_array_axis1(y_outputs)
        return x_inputs, y_outputs, xmins, xminmaxrange, ymins, yminmaxrange

#Min Max normalise array in-place by columns so that is in range [0,1]
#Additionally return normalising values: min_value, max_value - minvalue
#arraymins, arrayrange should have ndim=1, with size = array.shape[0] if supplied
#If not supplied, it will calculate the min and max values along column of array
#if use_provided = False, it will ignore provided values.
def normalise_array_axis1(array, use_provided = False, arraymins = None, arrayrange = None):
    if not use_provided:
        arraymins, arraymaxs = np.min(array, axis=1), np.max(array, axis=1)
        arrayrange = arraymaxs - arraymins
    no_samples = array.shape[1]
    arraymins_stacked = np.broadcast_to(arraymins, (no_samples,)+arraymins.shape).T
    arrayrange_stacked = np.broadcast_to(arrayrange, (no_samples,)+arrayrange.shape).T
    array -= arraymins_stacked
    array /= arrayrange_stacked
    return arraymins, arrayrange

if __name__ == "__main__":
    main()
