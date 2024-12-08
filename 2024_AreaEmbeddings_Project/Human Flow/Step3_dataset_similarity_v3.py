import pandas as pd
import numpy as np
from numpy.linalg import norm


# Step 4. Compute similarity statistic
file_pathRead = "Dataset/Raw_Embeddings/HumanFlow_Embeddings_code.csv"
df = pd.read_csv(file_pathRead, index_col=False)
dongCodes = df["행정동코드"].unique()

#Remove columns that have low correlation with Revenue, Number of Reservations. Threshold is 0.28
columnsDrop = ['남자0세부터9세생활인구수', '남자10세부터14세생활인구수', '남자15세부터19세생활인구수', '남자40세부터44세생활인구수',
       '남자45세부터49세생활인구수', '남자50세부터54세생활인구수', '남자55세부터59세생활인구수', '남자60세부터64세생활인구수', 
       '남자65세부터69세생활인구수', '남자70세이상생활인구수', '여자0세부터9세생활인구수', '여자10세부터14세생활인구수',
       '여자40세부터44세생활인구수', '여자45세부터49세생활인구수', '여자50세부터54세생활인구수',
       '여자55세부터59세생활인구수', '여자60세부터64세생활인구수', '여자65세부터69세생활인구수', '여자70세이상생활인구수',
       '중국인체류인구수_LONG_FOREIGNER']
df.drop(columns=columnsDrop, inplace=True)

#Keep separate files for each month
for date in df["기준일ID"].unique():
    current_df = df[df["기준일ID"]==date]
    #Loop through the Dongs, computing the cosine similarity for all pairs
    df_CosineSimilarity = pd.DataFrame(0, index = dongCodes, columns = dongCodes) #index = np.arange(len(dongCodes))
    j = k = 0
    for dong in dongCodes:
        k = 0
        df_externalIterator = current_df.loc[current_df["행정동코드"] == int(dong)]
        array_externalIterator = df_externalIterator.to_numpy()
        for nextdong in dongCodes:
            df_internalIterator = current_df.loc[current_df["행정동코드"] == int(nextdong)]
            array_internalIterator = df_internalIterator.to_numpy()
            #Compute cosine similarity
            cosine = int(np.dot(array_externalIterator, array_internalIterator.T))/(norm(array_externalIterator)*norm(array_internalIterator))
            #Adjust resolution of cosine similarity
            #cosine = (cosine*1e13+1)%1
            #Update value in dataframe
            df_CosineSimilarity.iloc[j, k] = cosine
            print("j = ", j, "k = ", k, "cosine = ", cosine, "dong = ", dong)
            print(current_df.head(1))
            k+=1
        j+=1

    #Save the resulting dataframe
    #file_name = "Pablo/results" + "/" + "cosineSimilarity" + "_" + str(date) + ".csv"
    file_name = "Pablo/results" + "/" + "cosineSimilarity" + "_DroppedFeatures_" + str(date) + ".csv"
    df_CosineSimilarity.to_csv(file_name, index=False)