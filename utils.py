
def load_data(path):
    with open(path,"r") as f:
        return eval(f.read())
