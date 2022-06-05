import pandas as pd

brand_dict = pd.read_excel("brand_dict_raw.xlsx")[['品牌','組合']]

brand_dict['組合'] = brand_dict['組合'].map(lambda x: x.strip("[]").split(","))
brand_dict['組合'] = brand_dict['組合'].map(lambda x: [i.strip().strip("'").strip('"').strip("'") for i in x])

new_brand_dict = {}
error_index = []
for i in brand_dict.index:
    for x in brand_dict['組合'][i]:
        try:
            new_brand_dict[x] = brand_dict['品牌'][i]
        except:
            error_index.append(i)

print(error_index)
df = pd.DataFrame(new_brand_dict.items(), columns=['original', 'organized'])
df.to_csv("brand_dict_new.csv", encoding="utf-8")
