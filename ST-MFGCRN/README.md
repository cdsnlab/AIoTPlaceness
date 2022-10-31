# Supplementary Material

Paper Title: Multiple Regional Feature-aware Spatio-temporal Public Transit Prediction

**Table of Contents**
1. Details of Regional Feature
2. Supplementary Results
3. Full Dataset
4. Implementation Code

# 1. Details of Regional Feature

1. Transit feature: The higher degree of transit supply accompanies higher transit demand. The neighborhoods with dense transit stations give better accessibility to transit service and so it makes travelers use transit more. In addition, the possibility of going to diverse areas (i.e., the number of bus routes) attracts users to use transit.
    * (BUS INFO) This feature includes the number of bus stops in the cells and routes passing through the cells.
2. Land use feature: Land use type is an important feature characterizing urban areas. The spatial distribution of land use generate travel demand for certain activities and determines travel pattern spatially and temporally. However, land use can not capture neighborhood functions in detail because it is a relatively broad concept of the legal purpose of the lands. To overcome the limitation of land use type, we additionally included three features that express the neighborhood in more detail.
    * (LU: Land Use Types) the area of each type in individual cells and use type was categorized as residential, semi-residential use, commercial and office use, neighborhood commercial use, industrial use, and green use.
    * (BD: Building-based Area of POIs) the area of the major point of interest(POI) in each cell. This includes six POIs that are i) culture and museums, ii) department stores and grocery stores, iii) hospitals, iv) religious facilities, v) government offices and research facilities, vi)train stations and bus terminals, and vii) universities and colleges.
    * (ENT) the number of business by industry types within each cell. Its types are i) primary industry (agriculture, fishing, mining, and extraction of minerals), ii) secondary industry (manufacturing industry), and iii) trinary industry (service industry)
    * (POI) the number of retail stores and living facilities of 29 categories provided by Local administration licensing data https://www.localdata.go.kr/.
3. Demographic feature: This is a basic factor determining the degree of travel demand, which is related to the number of potential users within the catchment area of transit stations.
    * (POP) the number of residents by age in each cell aggregated in (15-25, 25-45, 45-65, 65-85).
    * (EMP) the number of employments by industry types in each cell. We use it concatenated with ENTEMP in the experiment.


# 2. Supplementary Results

## 2.1. Feature Influence (Section 6.2)    
| K | Proximity | LU | POI (LOCAL) | BUS | POP | ENT_EMP | BD |   MAE  |  RMSE  |  NMAE  |
|:-:|:---------:|:--:|:-----------:|:---:|:---:|:-------:|:--:|:------:|:------:|:------:|
| 0 |     O     |  X |      X      |  X  |  X  |    X    |  X | 5.5293 | 8.5130 | 0.1605 |
| 0 |     x     |  X |      X      |  X  |  X  |    X    |  X | 5.6883 | 8.8704 | 0.1651 |
| 1 |     O     |  O |      X      |  X  |  X  |    X    |  X | 5.5012 | 8.4625 | 0.1597 |
| 1 |     O     |  X |      O      |  X  |  X  |    X    |  X | 5.5043 | 8.4675 | 0.1598 |
| 1 |     O     |  X |      X      |  O  |  X  |    X    |  X | 5.5070 | 8.4747 | 0.1598 |
| 1 |     O     |  X |      X      |  X  |  O  |    X    |  X | 5.5216 | 8.5119 | 0.1603 |
| 1 |     O     |  X |      X      |  X  |  X  |    O    |  X | 5.5004 | 8.4638 | 0.1596 |
| 1 |     O     |  X |      X      |  X  |  X  |    X    |  O | 5.5203 | 8.5031 | 0.1602 |
| 1 |     X     |  O |      X      |  X  |  X  |    X    |  X | 5.5194 | 8.5000 | 0.1602 |
| 1 |     X     |  X |      O      |  X  |  X  |    X    |  X | 5.5348 | 8.5409 | 0.1606 |
| 1 |     X     |  X |      X      |  O  |  X  |    X    |  X | 5.5232 | 8.5141 | 0.1603 |
| 1 |     X     |  X |      X      |  X  |  O  |    X    |  X | 5.5204 | 8.5063 | 0.1602 |
| 1 |     X     |  X |      X      |  X  |  X  |    O    |  X | 5.5190 | 8.5100 | 0.1602 |
| 1 |     X     |  X |      X      |  X  |  X  |    X    |  O | 5.5293 | 8.5290 | 0.1605 |
| 2 |     O     |  O |      O      |  X  |  X  |    X    |  X | 5.5027 | 8.4671 | 0.1597 |
| 2 |     O     |  O |      X      |  O  |  X  |    X    |  X | 5.5035 | 8.4644 | 0.1597 |
| 2 |     O     |  O |      X      |  X  |  O  |    X    |  X | 5.5047 | 8.4741 | 0.1598 |
| 2 |     O     |  O |      X      |  X  |  X  |    O    |  X | 5.5121 | 8.4862 | 0.1600 |
| 2 |     O     |  O |      X      |  X  |  X  |    X    |  O | 5.5068 | 8.4800 | 0.1598 |
| 2 |     O     |  X |      O      |  O  |  X  |    X    |  X | 5.5063 | 8.4758 | 0.1598 |
| 2 |     O     |  X |      O      |  X  |  O  |    X    |  X | 5.5193 | 8.5018 | 0.1602 |
| 2 |     O     |  X |      O      |  X  |  X  |    O    |  X | 5.5097 | 8.4869 | 0.1599 |
| 2 |     O     |  X |      O      |  X  |  X  |    X    |  O | 5.5014 | 8.4680 | 0.1597 |
| 2 |     O     |  X |      X      |  O  |  O  |    X    |  X | 5.5135 | 8.4909 | 0.1600 |
| 2 |     O     |  X |      X      |  O  |  X  |    O    |  X | 5.5030 | 8.4656 | 0.1597 |
| 2 |     O     |  X |      X      |  O  |  X  |    X    |  O | 5.5155 | 8.4984 | 0.1601 |
| 2 |     O     |  X |      X      |  X  |  O  |    O    |  X | 5.5049 | 8.4713 | 0.1598 |
| 2 |     O     |  X |      X      |  X  |  O  |    X    |  O | 5.5058 | 8.4749 | 0.1598 |
| 2 |     O     |  X |      X      |  X  |  X  |    O    |  O | 5.5013 | 8.4702 | 0.1597 |
| 2 |     X     |  O |      O      |  X  |  X  |    X    |  X | 5.5261 | 8.5097 | 0.1604 |
| 2 |     X     |  O |      X      |  O  |  X  |    X    |  X | 5.5369 | 8.5375 | 0.1607 |
| 2 |     X     |  O |      X      |  X  |  O  |    X    |  X | 5.5133 | 8.4906 | 0.1600 |
| 2 |     X     |  O |      X      |  X  |  X  |    O    |  X | 5.5223 | 8.5060 | 0.1603 |
| 2 |     X     |  O |      X      |  X  |  X  |    X    |  O | 5.5253 | 8.5116 | 0.1604 |
| 2 |     X     |  X |      O      |  O  |  X  |    X    |  X | 5.5405 | 8.5375 | 0.1608 |
| 2 |     X     |  X |      O      |  X  |  O  |    X    |  X | 5.5262 | 8.5090 | 0.1604 |
| 2 |     X     |  X |      O      |  X  |  X  |    O    |  X | 5.5177 | 8.5018 | 0.1601 |
| 2 |     X     |  X |      O      |  X  |  X  |    X    |  O | 5.5143 | 8.4858 | 0.1600 |
| 2 |     X     |  X |      X      |  O  |  O  |    X    |  X | 5.5239 | 8.5127 | 0.1603 |
| 2 |     X     |  X |      X      |  O  |  X  |    O    |  X | 5.5240 | 8.5122 | 0.1603 |
| 2 |     X     |  X |      X      |  O  |  X  |    X    |  O | 5.5082 | 8.4793 | 0.1599 |
| 2 |     X     |  X |      X      |  X  |  O  |    O    |  X | 5.5192 | 8.5056 | 0.1602 |
| 2 |     X     |  X |      X      |  X  |  O  |    X    |  O | 5.5247 | 8.5162 | 0.1603 |
| 2 |     X     |  X |      X      |  X  |  X  |    O    |  O | 5.5214 | 8.5059 | 0.1602 |
| 3 |     O     |  O |      O      |  O  |  X  |    X    |  X | 5.5016 | 8.4604 | 0.1597 |
| 3 |     O     |  O |      O      |  X  |  O  |    X    |  X | 5.5198 | 8.5047 | 0.1602 |
| 3 |     O     |  O |      O      |  X  |  X  |    O    |  X | 5.5090 | 8.4812 | 0.1599 |
| 3 |     O     |  O |      O      |  X  |  X  |    X    |  O | 5.5099 | 8.4822 | 0.1599 |
| 3 |     O     |  O |      X      |  O  |  O  |    X    |  X | 5.5114 | 8.4764 | 0.1600 |
| 3 |     O     |  O |      X      |  O  |  X  |    O    |  X | 5.5108 | 8.4779 | 0.1599 |
| 3 |     O     |  O |      X      |  O  |  X  |    X    |  O | 5.5051 | 8.4708 | 0.1598 |
| 3 |     O     |  O |      X      |  X  |  O  |    O    |  X | 5.5153 | 8.4955 | 0.1601 |
| 3 |     O     |  O |      X      |  X  |  O  |    X    |  O | 5.5096 | 8.4843 | 0.1599 |
| 3 |     O     |  O |      X      |  X  |  X  |    O    |  O | 5.5117 | 8.4887 | 0.1600 |
| 3 |     O     |  X |      O      |  O  |  O  |    X    |  X | 5.5166 | 8.4992 | 0.1601 |
| 3 |     O     |  X |      O      |  O  |  X  |    O    |  X | 5.5037 | 8.4669 | 0.1597 |
| 3 |     O     |  X |      O      |  O  |  X  |    X    |  O | 5.5133 | 8.4910 | 0.1600 |
| 3 |     O     |  X |      O      |  X  |  O  |    O    |  X | 5.5061 | 8.4757 | 0.1598 |
| 3 |     O     |  X |      O      |  X  |  O  |    X    |  O | 5.5077 | 8.4781 | 0.1599 |
| 3 |     O     |  X |      O      |  X  |  X  |    O    |  O | 5.5036 | 8.4657 | 0.1597 |
| 3 |     O     |  X |      X      |  O  |  O  |    O    |  X | 5.5047 | 8.4669 | 0.1598 |
| 3 |     O     |  X |      X      |  O  |  O  |    X    |  O | 5.5044 | 8.4734 | 0.1598 |
| 3 |     O     |  X |      X      |  O  |  X  |    O    |  O | 5.5090 | 8.4824 | 0.1599 |
| 3 |     O     |  X |      X      |  X  |  O  |    O    |  O | 5.5028 | 8.4675 | 0.1597 |
| 4 |     O     |  O |      O      |  O  |  O  |    X    |  X | 5.5048 | 8.4696 | 0.1598 |
| 4 |     O     |  O |      O      |  O  |  X  |    O    |  X | 5.5124 | 8.4822 | 0.1600 |
| 4 |     O     |  O |      O      |  O  |  X  |    X    |  O | 5.5022 | 8.4585 | 0.1597 |
| 4 |     O     |  O |      O      |  X  |  O  |    O    |  X | 5.5115 | 8.4800 | 0.1600 |
| 4 |     O     |  O |      O      |  X  |  O  |    X    |  O | 5.4997 | 8.4590 | 0.1596 |
| 4 |     O     |  O |      O      |  X  |  X  |    O    |  O | 5.4978 | 8.4587 | 0.1596 |
| 4 |     O     |  O |      X      |  O  |  O  |    O    |  X | 5.5037 | 8.4668 | 0.1597 |
| 4 |     O     |  O |      X      |  O  |  O  |    X    |  O | 5.5094 | 8.4808 | 0.1599 |
| 4 |     O     |  O |      X      |  O  |  X  |    O    |  O | 5.5008 | 8.4608 | 0.1597 |
| 4 |     O     |  O |      X      |  X  |  O  |    O    |  O | 5.5093 | 8.4803 | 0.1599 |
| 4 |     O     |  X |      O      |  O  |  O  |    O    |  X | 5.5034 | 8.4681 | 0.1597 |
| 4 |     O     |  X |      O      |  O  |  O  |    X    |  O | 5.5109 | 8.4856 | 0.1599 |
| 4 |     O     |  X |      O      |  O  |  X  |    O    |  O | 5.5108 | 8.4829 | 0.1599 |
| 4 |     O     |  X |      O      |  X  |  O  |    O    |  O | 5.5093 | 8.4785 | 0.1599 |
| 4 |     O     |  X |      X      |  O  |  O  |    O    |  O | 5.5050 | 8.4693 | 0.1598 |
| 5 |     O     |  O |      O      |  O  |  O  |    O    |  X | 5.5196 | 8.4963 | 0.1602 |
| 5 |     O     |  O |      O      |  O  |  O  |    X    |  O | 5.5118 | 8.4816 | 0.1600 |
| 5 |     O     |  O |      O      |  O  |  X  |    O    |  O | 5.5028 | 8.4619 | 0.1597 |
| 5 |     O     |  O |      O      |  X  |  O  |    O    |  O | 5.5055 | 8.4670 | 0.1598 |
| 5 |     O     |  O |      X      |  O  |  O  |    O    |  O | 5.5117 | 8.4784 | 0.1600 |
| 5 |     O     |  X |      O      |  O  |  O  |    O    |  O | 5.4987 | 8.4635 | 0.1596 |
| 6 |     O     |  O |      O      |  O  |  O  |    O    |  O | 5.5099 | 8.4765 | 0.1599 |


## 2.2. Analysis of feature influence on the model (Section 6.2, Fig. 5)

### 2.2.1. LU
| LU_TY (kr)   | LU_TY (en)                  | Top10     | Average   | Poor      |
|--------------|-----------------------------|-----------|-----------|-----------|
| 자연녹지지역 | green use                   | 1815422.2 | 1923115.6 | 2081094.9 |
| 주거지역     | residential                 |  781337.8 |  635583.6 |  563454.0 |
| 준주거지역   | semi-residential use        |  125579.4 |   31008.4 |    8887.6 |
| 상업지역     | commercial and office use   |   78643.9 |   82142.2 |  102619.0 |
| 공업지역     | industrial use              |   64766.7 |   86134.0 |   47072.4 |
| 근린상업지역 | neighborhood commercial use |   10847.0 |    3313.9 |    3920.3 |

### 2.2.1. POI (LOCAL)

| POI (LOCAL-kr)             | POI (LOCAL-en)                             | Top10 | Average | Poor  |
|----------------------------|--------------------------------------------|-------|---------|-------|
| 일반음식점                 | general restaurant                         | 877.3 |   445.9 | 409.5 |
| 통신판매업                 | mail order business                        | 357.7 |   193.0 | 188.2 |
| 담배소매업                 | tobacco retail business                    | 214.5 |   119.9 | 115.0 |
| 즉석판매제조가공업         | instant food store                         | 292.2 |   113.8 | 116.7 |
| 미용업                     | beauty business                            | 192.0 |    94.2 |  89.3 |
| 휴게음식점                 | convenient restaurant                      | 229.5 |    92.3 |  80.2 |
| 건강기능식품일반판매업     | health functional food general sales       | 164.9 |    77.0 |  79.2 |
| 식품자동판매기업           | food vending company                       | 131.6 |    65.9 |  66.4 |
| 축산판매업                 | livestock sales                            |  53.4 |    40.8 |  39.4 |
| 의료기기판매(임대)업       | medical device sales (rental) business     |  85.2 |    39.8 |  44.6 |
| 방문판매업                 | door-to-door sales                         |  63.0 |    28.0 |  33.8 |
| 의원                       | hospital                                   |  78.9 |    26.8 |  26.5 |
| 노래연습장업               | karaoke                                    |  42.9 |    19.7 |  17.7 |
| 세탁업                     | laundry                                    |  26.1 |    19.1 |  19.0 |
| 집단급식소                 | group food service                         |  28.4 |    16.4 |  15.0 |
| 식품제조가공업             | food manufacturing and processing industry |  19.8 |    15.9 |  13.4 |
| 인터넷컴퓨터게임시설제공업 | internet computer game facility provider   |  26.6 |    15.9 |  16.8 |
| 출판사                     | publisher                                  |  49.9 |    15.1 |  10.9 |
| 식품소분업                 | food subdivision                           |  27.4 |    14.4 |  12.4 |
| 건물위생관리업             | building sanitation management business    |  26.4 |    14.4 |  15.7 |
| 안전상비의약품 판매업소    | safety emergency medicine sales place      |  25.8 |    14.5 |  13.5 |
| 이용업                     | hair shop                                  |  28.8 |    14.1 |  14.9 |
| 당구장업                   | billiards business                         |  33.0 |    13.7 |  12.0 |
| 약국                       | pharmacy                                   |  34.7 |    13.4 |  12.9 |
| 숙박업                     | lodging                                    |  36.4 |    13.3 |  17.0 |
| 제과점영업                 | bakery business                            |  27.8 |    12.7 |  11.4 |
| 유료직업소개소             | paid job placement office                  |  22.4 |    11.7 |  13.0 |
| 일반게임제공업             | General game manufacturing.                |  31.0 |    12.2 |  14.5 |
| 옥외광고업                 | outdoor advertising                        |  14.0 |    11.3 |  13.8 |

## BUS

| BUS_INFO | Top10 | Average | Poor |
|----------|-------|---------|------|
| NBUSSTOP |  28.8 |    20.1 | 17.9 |
| ROUTE_N  |  18.6 |    12.8 | 11.6 |


## ENT EMP

| ENT_EMP | ENT_EMP                              | Top10   | Average | Poor   |
|---------|--------------------------------------|---------|---------|--------|
| ENT1    | Primary Enterprise                   |     0.0 |     0.0 |    0.0 |
| ENT2    | Manufacturing (Secondary) Enterprise |    72.7 |    78.4 |   63.0 |
| ENT3    | Service (Trinary) Enterprise         |  1785.2 |   727.0 |  613.7 |
| EMP1    | Primary Employees                    |    19.0 |     2.9 |    5.0 |
| EMP2    | Manufacturing (Secondary) Employees  |  1129.4 |   831.0 |  564.7 |
| EMP3    | Service (Trinary) Employees          | 12011.4 |  4428.2 | 4136.6 |


## BD AREA
| BD_AREA                | Top10   | Average | Poor    |
|------------------------|---------|---------|---------|
| Culture and Museum     |   312.2 | 11900.6 | 28133.3 |
| Department and Market  |  5806.4 |  3677.7 |  8087.3 |
| Hospital               |  3423.9 |   678.9 |     0.0 |
| Religion               | 29827.6 | 30076.1 | 37015.2 |
| Research and Public    |  7236.4 | 42390.2 | 76571.9 |
| Terminal               | 19970.2 |  2444.8 |    82.1 |
| University and College | 74120.7 | 40469.1 | 50294.7 |

## POP
| POP   | Top10  | Average | Poor   |
|-------|--------|---------|--------|
| 15-25 | 2992.3 |  1836.8 | 1837.7 |
| 25-45 | 5309.1 |  3814.7 | 3729.3 |
| 45-65 | 6826.9 |  4312.2 | 4278.5 |
| 65-85 | 1964.9 |  1446.5 | 1650.1 |


## 2.3. Module ablation study (Section 6.3)

| Model Setting |      |        |  Overall MAE |             |        |             |        |        |         |        |
|:-------------:|:----:|:------:|:------------:|:-----------:|:------:|:-----------:|:------:|:------:|:-------:|:------:|
|      SEN      | MGCN | FUSION | None (N_K=0) | All (N_K=6) |  LU_TY | POI (LOCAL) |   BUS  |   POP  | ENT_EMP |   BD   |
|       X       |  Cat |   Add  |    5.6024    |    5.7581   | 5.5854 |    5.5799   | 5.5954 | 5.5889 |  5.5960 | 5.5953 |
|       O       |  Cat |   Add  |    5.6050    |    5.7982   | 5.5944 |    5.5865   | 5.5821 | 5.5791 |  5.5862 | 5.5868 |
|       X       | Mean |   Add  |    5.5914    |    5.5781   | 5.5724 |    5.5650   | 5.5752 | 5.5703 |  5.5720 | 5.5751 |
|       O       | Mean |   Add  |    5.5910    |    5.5658   | 5.5625 |    5.5711   | 5.5748 | 5.5670 |  5.5676 | 5.5685 |
|       X       |  Cat | Weight |    5.5446    |    5.6796   | 5.5337 |    5.5354   | 5.5268 | 5.5391 |  5.5379 | 5.5389 |
|       O       |  Cat | Weight |    5.5412    |    5.6524   | 5.5276 |    5.5128   | 5.5245 | 5.5230 |  5.5503 | 5.5373 |
|       X       | Mean | Weight |    5.5227    |    5.5376   | 5.5179 |    5.5126   | 5.5173 | 5.5206 |  5.5028 | 5.5136 |
|       O       | Mean | Weight |    5.5293    |    5.5099   | 5.5027 |    5.5061   | 5.5093 | 5.5239 |  5.5039 | 5.5225 |



# 3. Full Dataset

We share sample data in the Microsoft CMT submission system, however, since its maximal level is only 10MB. Unfortunately, we have not found any better file sharing service with anonymization. We share our full dataset through anonymous link:

Daejeon: https://anonymfile.com/6jVd/process-qt-cpt-190301-190501-190601-190701.pkl

TaxiBJ: https://anonymfile.com/JRZg/process-qt-cpt-bj-150201-150603-150617-150701.pkl

(original dataset in: https://github.com/panzheyi/ST-MetaNet)

# 4. Implementation Code

We open our implementation code which runs on tensorflow 1.15.1, cuda 10.1 version. We cleaned and organized the code we used for our experiment. We checked it produces consistent performance, however, in case we modified some module during the cleaning process, we leave the original code as well in "./old-backup".



