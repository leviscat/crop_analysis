# 精准农业 Python 数据分析指导书

> 这份文档包含每个项目的步骤、用到的库、API说明和专有名词解释。

---

## 目录
1. [作物推荐系统 + Streamlit](#1-作物推荐系统--streamlit)
2. [数据清洗](#2-数据清洗)
3. [产量预测回归 + 特征重要性](#3-产量预测回归--特征重要性)
4. [时间序列分析 + ARIMA预测](#4-时间序列分析--arima预测)
5. [地理空间分析](#5-地理空间分析)
6. [专有名词总表](#6-专有名词总表)

---

## 1. 作物推荐系统 + Streamlit

### 目标
根据土壤环境数据（氮磷钾、温度、湿度、pH、降雨量）预测最适合种植的作物。

### 数据集
- 文件：`Crop_recommendation.csv`
- 2200行，22种作物，每种100条，数据均衡干净

### 主要步骤

#### 步骤1：读取数据
```python
import pandas as pd
df = pd.read_csv("Crop_recommendation.csv")
df.head()        # 查看前5行
df.shape         # 查看行列数
df['label'].unique()  # 查看所有作物种类
```
**API说明：**
- `pd.read_csv()`：读取CSV文件，返回DataFrame
- `df.head()`：显示前5行，快速了解数据结构
- `df.shape`：返回(行数, 列数)
- `df['列名'].unique()`：返回该列所有不重复的值

#### 步骤2：可视化数据分布
```python
import matplotlib.pyplot as plt
df['label'].value_counts().plot(kind='bar', figsize=(12, 4))
plt.show()
```
**API说明：**
- `df['列名'].value_counts()`：统计每个值出现的次数，返回从多到少排序的Series
- `.plot(kind='bar')`：画柱状图，kind还可以是'line'折线图、'pie'饼图、'box'箱线图

#### 步骤3：训练随机森林分类器
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('label', axis=1)   # 特征：7个环境变量
y = df['label']                 # 标签：作物名称

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
**API说明：**
- `df.drop('label', axis=1)`：删除label列，axis=1表示按列删，axis=0表示按行删
- `train_test_split(X, y, test_size=0.2)`：把数据按8:2分成训练集和测试集，random_state=42保证每次分法一样
- `RandomForestClassifier(n_estimators=100)`：建立100棵决策树的随机森林
- `model.fit(X_train, y_train)`：用训练数据训练模型
- `model.predict(X_test)`：用测试数据预测结果
- `accuracy_score(y_test, y_pred)`：计算预测准确率

#### 步骤4：混淆矩阵
```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
```
**API说明：**
- `confusion_matrix()`：生成混淆矩阵，显示每个类别预测对了多少、错成什么了
- 对角线是预测正确的，非对角线是预测错误的

#### 步骤5：对比XGBoost
```python
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train_encoded)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb))
```
**API说明：**
- `LabelEncoder()`：把文字标签转成数字，XGBoost只接受数字标签
- `le.fit_transform(y_train)`：学习标签映射并转换训练集
- `le.transform(y_test)`：用同样的映射转换测试集（不能重新fit）

#### 步骤6：保存模型
```python
import joblib
joblib.dump(model, 'crop_model.pkl')   # 保存
model = joblib.load('crop_model.pkl')  # 加载
```
**API说明：**
- `joblib.dump()`：把Python对象序列化保存到文件
- `joblib.load()`：从文件加载回Python对象
- `.pkl`是pickle格式，Python常用的对象存储格式

#### 步骤7：Streamlit交互界面
```python
import streamlit as st
st.title("标题")
st.write("文字")
N = st.slider("Nitrogen", 0, 140, 50)   # 标签、最小值、最大值、默认值
st.success("预测结果")
```
**API说明：**
- `st.slider()`：创建滑动条，用户拖动时变量实时更新
- `st.success()`：绿色成功提示框
- Streamlit的运行逻辑：用户每次操作，整个脚本从头到尾重新运行一遍

**运行命令：**
```bash
streamlit run app.py
```

### 专有名词
- **分类（Classification）**：预测一个类别，输出是离散的标签
- **随机森林**：很多棵决策树投票，少数服从多数，比单棵树更稳定
- **训练集/测试集**：训练集用来学习，测试集用来评估，测试集不能参与训练
- **准确率（Accuracy）**：预测正确的比例
- **混淆矩阵**：展示每个类别预测对了多少、错成什么的矩阵

---

## 2. 数据清洗

### 目标
处理真实农业数据集中的异常值，让数据更适合模型训练。

### 数据集
- 文件：`yield_df.csv`
- 28242行，全球101个国家农业产量数据

### 主要步骤

#### 步骤1：检查数据质量
```python
print(df.isnull().sum())      # 检查每列缺失值数量
print(df.duplicated().sum())  # 检查重复行数量
print(df.describe())          # 基本统计：均值、标准差、最大最小值
```
**API说明：**
- `df.isnull()`：返回每个值是否为空的布尔DataFrame
- `.sum()`：统计True的数量，即缺失值数量
- `df.duplicated()`：返回每行是否重复的布尔Series
- `df.describe()`：对数值列计算count、mean、std、min、25%、50%、75%、max

#### 步骤2：删除多余列
```python
df = df.drop(columns=['Unnamed: 0'])
```
**说明：** CSV文件有时会多一列行号索引，对分析无意义，直接删掉。

#### 步骤3：箱线图可视化异常值
```python
df['列名'].plot(kind='box')
plt.show()
```
**箱线图结构：**
- 箱子：中间50%的数据范围（Q1到Q3）
- 中间线：中位数
- 箱子外的圆圈：异常值（离群点）

#### 步骤4：IQR方法去除异常值
```python
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)   # 下四分位数
    Q3 = df[column].quantile(0.75)   # 上四分位数
    IQR = Q3 - Q1                     # 四分位距
    lower = Q1 - 1.5 * IQR           # 下边界
    upper = Q3 + 1.5 * IQR           # 上边界
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_clean = remove_outliers(df, 'pesticides_tonnes')
df_clean = remove_outliers(df_clean, 'hg/ha_yield')
```
**API说明：**
- `df['列名'].quantile(0.25)`：计算该列的25%分位数
- `&`：布尔AND，两个条件同时满足
- IQR判断标准：低于Q1-1.5×IQR或高于Q3+1.5×IQR的视为异常值

#### 步骤5：保存清洗后的数据
```python
df_clean.to_csv('yield_clean.csv', index=False)
```
**原则：** 原始数据永远不动，清洗后的数据存新文件。

### 专有名词
- **缺失值（Missing Value）**：数据中空白或NaN的位置
- **异常值（Outlier）**：与大多数数据差距极大的值，可能是录入错误或极端情况
- **IQR（四分位距）**：Q3-Q1，中间50%数据的范围
- **右偏分布**：大多数数据集中在左边，少数极大值把均值拉向右边

---

## 3. 产量预测回归 + 特征重要性

### 目标
预测具体的农业产量数值，并分析哪些因素对产量影响最大。

### 主要步骤

#### 步骤1：处理文字类特征
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])
], remainder='passthrough')
```
**API说明：**
- `OneHotEncoder`：把文字类别转成0/1列，比如Area_China=1, Area_India=0
- 与LabelEncoder的区别：LabelEncoder给每个类别一个数字（China=1, India=2），暗示大小关系；OneHotEncoder每个类别独立一列，没有大小关系
- `ColumnTransformer`：对不同列应用不同的处理方式
- `remainder='passthrough'`：没有指定处理方式的列保持原样

#### 步骤2：Pipeline串联预处理和模型
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model_yield = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])
model_yield.fit(X_train, y_train)
```
**API说明：**
- `Pipeline`：把多个步骤串联，fit时自动按顺序执行，避免数据泄露
- `RandomForestRegressor`：回归版随机森林，输出连续数值而不是类别

#### 步骤3：评估回归模型
```python
from sklearn.metrics import mean_absolute_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```
**API说明：**
- `MAE（平均绝对误差）`：预测值与真实值的平均偏差，单位与目标变量相同
- `R²（决定系数）`：0到1之间，越接近1说明模型解释了越多的数据变化，0.98表示解释了98%

#### 步骤4：特征重要性
```python
feature_names = (
    model_yield.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(['Area', 'Item']).tolist()
    + ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
)
importances = model_yield.named_steps['regressor'].feature_importances_
```
**API说明：**
- `feature_importances_`：随机森林内置属性，每个特征对模型的贡献权重，所有值加起来=1
- `get_feature_names_out()`：获取OneHotEncoder展开后的列名

### 专有名词
- **回归（Regression）**：预测连续数值，输出是具体数字
- **分类vs回归**：分类输出"种什么"，回归输出"产多少"
- **OneHotEncoding**：把类别变量转成多列0/1，避免引入虚假的大小关系
- **Pipeline**：把数据预处理和模型训练串联成一个流程
- **MAE**：平均绝对误差，直觉上容易理解
- **R²**：决定系数，衡量模型整体解释能力

---

## 4. 时间序列分析 + ARIMA预测

### 目标
分析农业产量随时间的变化趋势，并预测未来几年的产量。

### 主要步骤

#### 步骤1：按时间聚合数据
```python
yearly = df.groupby('Year')['hg/ha_yield'].mean().reset_index()
```
**API说明：**
- `df.groupby('Year')`：按年份分组
- `['hg/ha_yield'].mean()`：对每组计算均值
- `.reset_index()`：把分组的索引变回普通列

#### 步骤2：多国趋势对比
```python
for country in countries:
    data = df[df['Area'] == country].groupby('Year')['hg/ha_yield'].mean()
    plt.plot(data.index, data.values, marker='o', label=country)
plt.legend()
plt.show()
```
**说明：** 用循环对每个国家画一条线，`.index`是年份，`.values`是产量数值。

#### 步骤3：unstack多维数据透视
```python
nz_crops = df[df['Area'] == 'New Zealand'].groupby(['Year', 'Item'])['hg/ha_yield'].mean().unstack()
```
**API说明：**
- 先按Year和Item两个维度分组
- `.unstack()`：把Item从行索引变成列，每种作物一列，行是年份

#### 步骤4：ARIMA时间序列预测
```python
from statsmodels.tsa.arima.model import ARIMA

train = nz_potato[nz_potato.index <= 2010]
test = nz_potato[nz_potato.index > 2010]

model = ARIMA(train, order=(2, 1, 1))
result = model.fit()
forecast = result.forecast(steps=3)
```
**API说明：**
- `ARIMA(data, order=(p, d, q))`：
  - p：自回归阶数，用几年前的数据预测
  - d：差分次数，1表示对数据做一次差分去除趋势
  - q：移动平均阶数，用几步的历史误差修正
- `model.fit()`：拟合模型
- `result.forecast(steps=3)`：预测未来3个时间步

### 专有名词
- **时间序列**：按时间顺序排列的数据，前后有依赖关系
- **趋势（Trend）**：数据长期增长或下降的方向
- **平稳性**：均值和方差不随时间变化，ARIMA要求数据平稳
- **差分**：用相邻两个时间点的差值代替原始值，去除趋势使数据平稳
- **ARIMA**：Auto-Regressive Integrated Moving Average，专门处理时间序列的预测模型
- **AR（自回归）**：用自己过去的值预测当前值
- **MA（移动平均）**：用过去的预测误差修正当前预测
- **MAE误差率**：预测值与实际值的偏差除以实际值，越小越好

---

## 5. 地理空间分析

### 目标
处理带地理坐标的数据，在地图上可视化，并做空间查询、缓冲区分析和面积计算。

### 核心库：geopandas
geopandas是pandas的地理空间扩展，多了一列geometry存储地理形状。

### 主要步骤

#### 步骤1：加载地图数据
```python
import geopandas as gpd
import requests, io

url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
response = requests.get(url)
world = gpd.read_file(io.BytesIO(response.content))
```
**说明：** 用requests先把文件下载到内存，再传给geopandas，绕过GDAL直接读URL的网络限制。

#### 步骤2：过滤和绘图
```python
nz = world[world['NAME'] == 'New Zealand']
nz.plot(figsize=(8, 10), color='lightgreen', edgecolor='black')
plt.show()
```
**说明：** GeoDataFrame的过滤和pandas一样，`.plot()`会自动识别geometry列画地图。

#### 步骤3：Choropleth地图（颜色深浅表示数值）
```python
world_yield = world.merge(yield_geo, on='NAME', how='left')
world_yield.plot(column='avg_yield',
                 legend=True,
                 missing_kwds={'color': 'lightgrey'},
                 cmap='YlOrRd')
```
**API说明：**
- `world.merge()`：跟pandas的merge一样，按国家名称把产量数据合并到地图数据
- `column='avg_yield'`：用这列的数值决定颜色深浅
- `cmap='YlOrRd'`：颜色映射，YlOrRd是黄色到红色渐变，还有Blues、Greens等
- `missing_kwds={'color': 'lightgrey'}`：没有数据的区域显示灰色

#### 步骤4：创建点数据
```python
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip(farm_points['longitude'], farm_points['latitude'])]
farms_gdf = gpd.GeoDataFrame(farm_points, geometry=geometry, crs='EPSG:4326')
```
**API说明：**
- `Point(经度, 纬度)`：创建一个地理点对象
- `zip(longitude, latitude)`：把经度和纬度列配对
- `crs='EPSG:4326'`：声明使用的坐标参考系

#### 步骤5：空间查询（sjoin）
```python
joined = gpd.sjoin(farms_gdf, world[['NAME', 'CONTINENT', 'geometry']],
                   how='left', predicate='within')
```
**API说明：**
- `gpd.sjoin()`：空间连接，按地理关系合并两个GeoDataFrame
- `predicate='within'`：判断条件为"点在多边形内部"
- 其他predicate：`intersects`（相交）、`contains`（包含）
- 与pandas merge的区别：merge按列值合并，sjoin按地理位置关系合并

#### 步骤6：缓冲区分析
```python
farms_meter = farms_gdf.to_crs('EPSG:3857')
farms_meter['buffer'] = farms_meter.geometry.buffer(50000)  # 50km
farms_buffer = farms_meter.set_geometry('buffer')
```
**API说明：**
- `.to_crs('EPSG:3857')`：转换坐标系，必须先转成米制才能用米做缓冲区
- `.buffer(50000)`：以每个几何对象为中心，创建半径50000米的缓冲区
- `.set_geometry('buffer')`：把活跃geometry列切换到buffer列

#### 步骤7：面积计算
```python
nz_nztm = nz.to_crs('EPSG:2193')   # 新西兰官方坐标系
area_m2 = nz_nztm.geometry.area.values[0]
area_km2 = area_m2 / 1_000_000
```
**API说明：**
- `.to_crs('EPSG:2193')`：转为新西兰官方坐标系，面积计算更准确
- `.geometry.area`：计算每个几何形状的面积，单位由坐标系决定

### 专有名词
- **GeoDataFrame**：带geometry列的DataFrame，可以存储地理形状
- **geometry**：地理形状列，可以是Point（点）、LineString（线）、Polygon（面）、MultiPolygon（多个面）
- **GeoJSON**：存储地理数据的JSON格式，广泛用于地图数据交换
- **Shapefile**：ESRI公司的地理数据格式，由.shp、.dbf、.shx等多个文件组成
- **坐标参考系（CRS）**：定义如何把球面上的位置用数字表示
  - EPSG:4326：经纬度坐标系，最通用，适合显示位置
  - EPSG:3857：Web墨卡托投影，Google Maps使用，适合网页地图
  - EPSG:2193：新西兰官方坐标系，在新西兰范围内面积距离最准确
- **投影变形**：把球面压成平面时产生的形状失真，高纬度地区更严重
- **Choropleth地图**：用颜色深浅表示数值大小的专题地图
- **空间查询（Spatial Join）**：按地理位置关系合并数据，而不是按列值
- **缓冲区（Buffer）**：以某个地理对象为中心，向外扩展指定距离形成的区域

---

## 6. 专有名词总表

| 名词 | 解释 |
|---|---|
| DataFrame | pandas的表格数据结构，有行有列 |
| GeoDataFrame | 带geometry列的DataFrame |
| 特征（Feature） | 用来做预测的输入变量，即X |
| 标签（Label） | 要预测的目标变量，即y |
| 训练集 | 用来训练模型的数据，通常80% |
| 测试集 | 用来评估模型的数据，通常20%，不参与训练 |
| 过拟合 | 模型把训练数据背下来了，对新数据预测很差 |
| 分类 | 预测类别，输出离散标签 |
| 回归 | 预测数值，输出连续数字 |
| 准确率 | 分类正确的比例 |
| MAE | 平均绝对误差，回归模型的评估指标 |
| R² | 决定系数，越接近1越好 |
| 随机森林 | 很多棵决策树投票的集成模型 |
| XGBoost | 梯度提升树，擅长处理不均衡数据 |
| ARIMA | 时间序列预测模型，靠历史规律预测未来 |
| OneHotEncoding | 把类别变量转成多列0/1 |
| LabelEncoding | 把类别变量转成数字 |
| Pipeline | 把多个处理步骤串联成一个流程 |
| IQR | 四分位距，Q3-Q1，用于识别异常值 |
| 异常值 | 与大多数数据差距极大的值 |
| 坐标系（CRS） | 定义如何用数字表示地球上的位置 |
| 空间连接（sjoin） | 按地理位置关系合并两个地理数据集 |
| 缓冲区（Buffer） | 以某个地理对象为中心向外扩展的区域 |
| Choropleth地图 | 用颜色深浅表示数值大小的专题地图 |
| NDVI | 归一化植被指数，用遥感数据衡量植被覆盖状况 |
