import pandas as pd

brand_dict = pd.read_excel("Brand_Dict/brand_dict_raw.xlsx")[['品牌','組合']]

brand_dict['組合'] = brand_dict['組合'].map(lambda x: x.strip("[]").split(","))
brand_dict['組合'] = brand_dict['組合'].map(lambda x: [i.strip().strip("'").strip('"').strip("'") for i in x])

new_brand_dict = {}
error_index = []
count = 0
for i in brand_dict.index:
    for x in brand_dict['組合'][i]:
        try:
            try:
                new_brand_dict[x]+=""
                error_index.append(i)
            except:
                pass
            new_brand_dict[x] = brand_dict['品牌'][i]
        except:
            error_index.append(i)

error_index = list(set(error_index))
for i in error_index:
    print(brand_dict['品牌'][i])
df = pd.DataFrame(new_brand_dict.items(), columns=['original', 'organized'])
df.to_csv("Brand_Dict/brand_dict_new.csv", encoding="utf-8")
