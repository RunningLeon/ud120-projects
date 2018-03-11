#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    ###predictions ,ages, net_worths type are numpy.ndarray, need to add [i][0]
    for i in range(len(predictions)):
        p =  predictions[i][0]
        a = ages[i][0]
        n = net_worths[i][0]
        err = round(abs(p-n)/n, 4 )
        cleaned_data.append((a, n, err))
    # sort list using lambda small to large
    cleaned_data = sorted(cleaned_data, key=lambda d:d[2])
    #remove 10% outliers points
    cleaned_data = cleaned_data[:80]
    return cleaned_data
