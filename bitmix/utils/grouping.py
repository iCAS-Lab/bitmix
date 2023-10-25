import re
def group_by_regex(alist, aregex):
    #TODO implement grouping by regex
    raise NotImplementedError

def group_by_substrings(alist, substrs):
    groups = [[] for _ in substrs] + [[]]
    
    for item in alist:
        found = False
        for i, substr_group in enumerate(substrs):
            if any(substr in item for substr in substr_group):
                groups[i].append(item)
                found = True
                break
        if not found:
            groups[-1].append(item)
    
    return groups
