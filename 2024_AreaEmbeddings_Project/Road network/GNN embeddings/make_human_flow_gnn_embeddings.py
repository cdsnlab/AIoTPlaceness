import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import random

# 각 열별 정규화
def standardize_dataframe(df, exclude_columns=[]):
    scaler = StandardScaler()
    standardized_df = df.copy()
    for column in standardized_df.columns:
        if column not in exclude_columns:
            standardized_df[column] = scaler.fit_transform(standardized_df[[column]])
    return standardized_df

# 초기 노드 임베딩 가져오기
def get_basic_embeddings(path, dong_name):
    emb = pd.read_csv(path)
    #emb = standardize_dataframe(emb, dong_name)   # 정규화
    features = torch.tensor(emb.iloc[:, 1:].values, dtype=torch.float32)   # 임베딩만 추출
    return emb, features

# 그래프 불러오기
def get_graph(path, emb, dong_name):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    edges = G.edges(data=True)   # 엣지 추출
    node_to_idx = {name: idx for idx, name in enumerate(emb[dong_name])}   # 동이름 -> 인덱스 변환
    edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edges], dtype=torch.long).T   # edge list 재생산
    edge_weight = torch.tensor([edge[2]['weight'] for edge in edges], dtype=torch.float32)
    
    return edge_index, edge_weight

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
# GAT 모델 정의
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# Negative Sampling 구현
def negative_sampling(edge_index, num_nodes, num_neg_samples):
    """Negative samples 생성"""
    edge_set = set(tuple(edge) for edge in edge_index.T.tolist())
    neg_samples = []
    while len(neg_samples) < num_neg_samples:
        i, j = np.random.randint(0, num_nodes, size=2)
        if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
            neg_samples.append((i, j))
    return torch.tensor(neg_samples, dtype=torch.long).T

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 사용하는 경우
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 모델 학습 
def train_model(model, data, optimizer, num_epochs=200, patience=20, seed=42):

    best_loss = float('inf')
    counter = 0
    best_z = None  # `best_loss` 시점의 임베딩 저장

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        if isinstance(model, GAT):  # GAT 모델에는 edge_weight 사용하지 않음
            z = model(data.x, data.edge_index)
        else:  # GCN 모델에는 edge_weight 전달
            z = model(data.x, data.edge_index, data.edge_weight)

        # Positive edges
        pos_edges = data.edge_index.T
        pos_scores = (z[pos_edges[:, 0]] * z[pos_edges[:, 1]]).sum(dim=1)

        # Negative edges
        neg_edges = negative_sampling(data.edge_index, data.num_nodes, pos_edges.size(0) // 2)
        neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

        # Loss 계산
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        # Loss 출력
        tqdm.write(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Early Stopping 로직
        if loss.item() < best_loss:
            best_loss = loss.item()  # 최적 Loss 갱신
            best_z = z.detach().cpu()  # `best_loss` 시점의 임베딩 저장
            counter = 0  # 카운터 초기화
        else:
            counter += 1

        if counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    return best_z  # 최적 Loss 기준의 임베딩 반환

# 임베딩 저장
def get_output(z, emb, model_name="model"):
    # GPU 텐서를 NumPy 배열로 변환
    z_numpy = z.detach().cpu().numpy()

    # 새로운 데이터프레임 생성: ADM_NM + 학습된 임베딩
    embedding_columns = [f"{model_name}_embed{i}" for i in range(z_numpy.shape[1])]  # 임베딩 컬럼 이름 생성
    emb_updated = pd.DataFrame(emb[['기준일ID', '행정동코드']], columns=['기준일ID', '행정동코드'])  # ADM_NM 열만 복사
    emb_updated[embedding_columns] = z_numpy  # 임베딩 추가
    
    return emb_updated

def main():
    # 시드 고정
    set_seed(42)
    
    # 전체 기본 임베딩 불러오기
    hf_emb = pd.read_csv('../../Dataset/Raw_Embeddings/HumanFlow_Embeddings_kor.csv')
    group = hf_emb.groupby(['기준일ID'])
    
    # 그래프 기본 경로 설정
    basic_path = '../../Dataset/Human_flow_Graph/'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_total_embeddings = []
    gat_total_embeddings = []
    
    # 각 월별 GNN 학습
    for key, df in group:
        date = str(key[0])
        # 초기 임베딩 불러오기
        emb = df
        features = torch.tensor(emb.iloc[:, 2:].values, dtype=torch.float32)

        G_path = f'{basic_path}{date}.gpickle'   # 그래프 경로
        edge_index, edge_weight = get_graph(G_path, emb, '행정동코드')
        print(f'{date}데이터 불러오기 성공!')
        
        # 학습용 torch data 생성
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight)
        print('데이터 변환!')
        print(data)
        # Check for NaN in features
        #print(torch.isnan(data.x).any())  # True면 데이터에 NaN 존재
        #print(torch.isnan(data.edge_weight).any())  # True면 데이터에 NaN 존재
    
        # 학습 설정
        data = data.to(device)

        model_gcn = GCN(data.num_node_features, 64, 35).to(device)
        model_gat = GAT(data.num_node_features, 64, 35, heads=4).to(device)  # 각 노드에서 4개의 attention heads 사용

        # 최적화 방법 설정
        optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=5e-4)
        optimizer_gat = torch.optim.Adam(model_gat.parameters(), lr=0.01, weight_decay=5e-4)
        
        # 임베딩 학습
        z_gcn = train_model(model_gcn, data, optimizer_gcn, num_epochs=200, patience=20)
        print('==========================================================================')
        z_gat = train_model(model_gat, data, optimizer_gat, num_epochs=200, patience=20)
        
        # 학습된 임베딩 저장
        gcn_path = "../../Dataset/GCN_Embeddings/GCN_Human_Flow_Embeddings.csv"
        gat_path = "../../Dataset/GCN_Embeddings/GAT_Human_Flow_Embeddings.csv"
        
        gcn_df = get_output(z_gcn, emb, 'GCN')
        gat_df = get_output(z_gat, emb, 'GAT')
        
        # 결과를 리스트에 추가
        gcn_total_embeddings.append(gcn_df)
        gat_total_embeddings.append(gat_df)
        
    # 모든 결과를 하나의 데이터프레임으로 결합
    gcn_total_df = pd.concat(gcn_total_embeddings, ignore_index=True)
    gat_total_df = pd.concat(gat_total_embeddings, ignore_index=True)

    # 최종 데이터프레임 저장 (선택 사항)
    gcn_total_df.to_csv(gcn_path, index=False)
    gat_total_df.to_csv(gat_path, index=False)
    
if __name__ == "__main__":
    main()