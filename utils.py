from decimal import Decimal
import numpy as np

# Più valori restituiti nel caso di pareggio
# opzionalmente da gestire con criterio a parte, ma è
# comunque abbastanza raro
# Equivalente di argmax
def find_max_index(lst):
    if type(lst) is not list:
        lst = list(lst)

    return lst.index(max(lst))

def bid_to_str(bid):
    if isinstance(bid, Decimal):
        return decimal_to_str(bid)
    else:
        return str(round(bid, 3))

def decimal_to_str(dec):
    parts1 = str(dec).split("E")
    partsDot = parts1[0].split(".")

    if len(partsDot) > 1:
        if len(parts1) > 1:
            return "{}.{}E{}".format(partsDot[0], partsDot[1][:2], parts1[1])
        else:
            return "{}.{}".format(partsDot[0], partsDot[1][:2])
    else:
        return partsDot[0]

def bids_to_string(bids):
    if type(bids) is list:
        bids = np.array(bids)

    if len(bids) > 0: 
        if isinstance(bids[0], Decimal):
            return str(np.array(list(map(decimal_to_str, bids)))).replace("'", "")
        elif isinstance(bids[0], (list, np.ndarray)) and isinstance(bids[0][0], Decimal):
            return str(np.array(list(map(lambda agent_bids: list(map(decimal_to_str, agent_bids)), bids)))).replace("'", "")

    return np.array2string(bids, precision = 3, max_line_width = 999)

def sol_to_string(agents = None, sol = None):
    if sol is None:
        sol = [agent.assigned_tasks for agent in agents]
    return np.array2string(np.array(sol)).replace("0.", "_").replace("1.", "1")