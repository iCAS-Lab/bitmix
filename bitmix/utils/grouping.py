import re
def group_by_regex(alist, aregex):
    raise NotImplementedError

def group_by_substrings(alist, substrs):
    groups = []
    for substr_group in substrs:
        groups.append([])
    groups.append([])

    for item in alist:
        found=False
        for i, substr_group in enumerate(substrs):
            for substr in substr_group:
                if substr in item:
                    groups[i].append(item)
                    found=True
                    break
                if found:
                    break
            if found:
                break
        if not found:
            groups[len(groups)-1].append(item)
    return groups
