# 資料清整說明
--------
### 品類
>資料路徑：Category -> category_data -> category.csv
>
>資料說明：已比照業師給予的資料欄位及名稱做出清整
>
>清整項目：
>
>>欄位['name']大寫轉小寫、全形轉半形、翻譯拉丁文、去除單位、去除亂數、去除表情符號、去除空格、去除全英文無法辨識之名稱
>
>>欄位['product']大寫轉小寫、刪去多標籤（預計優化方式：多標籤轉其他、相似名稱統整）
### 品牌
>資料路徑：Brand -> brand_data -> brand.csv
>
>資料說明：已比照業師給予的資料欄位及名稱做出清整
>
>清整項目：
>
>>欄位['name']大寫轉小寫、全形轉半形、翻譯拉丁文、去除單位、去除亂數、去除表情符號、去除空格、去除全英文無法辨識之名稱
>
>>欄位['brand']大寫轉小寫、全形轉半形、翻譯拉丁文、只刪去英文與中文之間空格
