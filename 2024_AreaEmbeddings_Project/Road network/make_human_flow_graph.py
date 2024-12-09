import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import networkx as nx
import pickle

# 424개 행정구역 mapping 함수
def dong_name_dict():
    dong_mapping = pd.read_excel('../Dataset/dong_mapping.xlsx')
    dong_mapping.columns = dong_mapping.iloc[0]
    dong_mapping['H_SDNG_CD'] = dong_mapping['H_SDNG_CD'].astype(str)
    dong_mapping = dong_mapping.iloc[1:].reset_index(drop=True)
    dong_mapping.loc[dong_mapping['H_SDNG_CD']=='1121068', 'H_DNG_NM'] = '신사동(관악)'
    dong_mapping.loc[dong_mapping['H_SDNG_CD']=='1123051', 'H_DNG_NM'] = '신사동(강남)'
    #dong_mapping  ->  (H_SDNG_CD: 통계청코드,	H_DNG_CD: 행안부코드)

    code2name_dict = dong_mapping.set_index('H_DNG_CD')['H_DNG_NM'].to_dict()
    
    return code2name_dict

def make_graph(path, code2name_dict):
    # 데이터 로드
    c_hf = pd.read_csv(path)
    cols = list(c_hf.columns)
    
    # 인덱스 순서에 맞는 동이름 저장
    for i in range(len(cols)):
        cols[i] = code2name_dict[cols[i]]

    # 동 이름으로 변경    
    c_hf.columns = cols
    
    # 3분위수 계산
    third_quartile = np.quantile(c_hf.values[np.triu_indices_from(c_hf.values, k=1)], 0.75)   # 3Q(간선 생성 임계치)
    print(f"3분위수 : {third_quartile}")
    
    # NetworkX 그래프 생성
    G = nx.Graph()

    # 노드 추가
    for col in cols:
        G.add_node(col)

    # 엣지 추가
    array = c_hf.to_numpy()
    indices = np.triu_indices_from(array, k=1)  # 대각선 제외
    for i, j in zip(*indices):
        weight = array[i, j]
        if weight >= third_quartile:
            G.add_edge(cols[i], cols[j], weight=weight)

    # 그래프 저장
    # Save graph using pickle
    output_file = os.path.join('../Dataset/Human_flow_Graph', f"{path[-10:-4]}.gpickle")
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph for {path[-10:-4]} saved in {'../Dataset/Human_flow_Graph'} as Pickle. Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    
def main():
    basic_path = '../Pablo/results/cosineSimilarity_'
    start_year = 2017
    start_month = 1
    end_year = 2022
    end_month = 7
    
    # 경로들 확인
    path_list = [
        basic_path+f"{year}{month:02d}.csv"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        if not (year == end_year and month > end_month)
    ]
    
    code2name_dict = dong_name_dict()
    
    # 그래프 저장
    for path in path_list:
        print(path)
        make_graph(path, code2name_dict)
    
    
if __name__ == "__main__":
    main()
    