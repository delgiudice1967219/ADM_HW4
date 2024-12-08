def sign_agreement(a, b):
    """
    Takes in input two lists and computes the frequency of ordered components agreements
    """
    # Initialize a counter for the number of agreements
    count = 0
    a_list = list(a)
    b_list = list(b)
    # Iterate through both lists while checking the elements at the same positions
    for i in range(len(a_list)):
        if a_list[i] == b_list[i]:
            count += 1

    return count/len(a)


def jaccard_sim(a, b):
    """
    Takes in input two lists and computes the jaccard similarity between them
    """
    set1 = set(a)
    set2 = set(b)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection/union
