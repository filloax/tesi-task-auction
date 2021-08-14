import numpy as np

# Più valori restituiti nel caso di pareggio
# opzionalmente da gestire con criterio a parte, ma è
# comunque abbastanza raro
# Equivalente di argmax
def find_max_index(lst):
    if type(lst) is not list:
        lst = list(lst)

    max_val = max(lst)
    return [idx for idx in range(len(lst)) if lst[idx] == max_val]

def bids_to_string(bids):
    if bids is list:
        bids = np.array(bids)

    return str(np.round(bids, 3))
