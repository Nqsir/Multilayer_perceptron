import pickle


# save object
def save_object(obj, name):
    f = open(name, 'wb')
    pickle.dump(obj, f)


# load object
def load_object(name):
    f = open(name, 'rb')
    return pickle.load(f)
