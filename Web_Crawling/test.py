import pickle

with open('brand_list_1.pkl', 'rb') as f:
    brand_list_1 = pickle.load(f)

with open('brand_list_2.pkl', 'rb') as f:
    brand_list_2 = pickle.load(f)

all_brand_list = brand_list_1+brand_list_2

with open('all_brand_list.pkl', 'wb') as f:
    pickle.dump(all_brand_list, f)