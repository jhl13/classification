# encoding = utf-8
def replace_all(s, old, new, reg = False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s
    
def remove_all(s, sub):
    return replace_all(s, sub, '')
    
def split(s, splitter, reg = False):
    if not reg:
        return s.split(splitter)
    import re
    return re.split(splitter, s) 
    