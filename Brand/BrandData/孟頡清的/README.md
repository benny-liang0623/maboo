# 資料清整說明
--------

## 品牌

### 資料組成
brand_final + 原資料30筆以下  

### 文字處理
regex取所有中英文數字，英文轉小寫並去除所有空格  

### 品牌彙總
見Brand_Dict  

### 重新採樣
上採樣至25，下採樣至80，random_state都設為42，上採樣時replace=True  

