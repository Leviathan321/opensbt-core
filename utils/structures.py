
def update_list_unique(dest,src):
    for el in src:
        if el not in dest:
            dest.append(el)