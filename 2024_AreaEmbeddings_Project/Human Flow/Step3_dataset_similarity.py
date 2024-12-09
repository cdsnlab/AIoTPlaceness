import pandas as pd
import numpy as np
from numpy.linalg import norm


# Step 4. Compute similarity statistic
date_start = 201701
date_end = 202207

#Loop through each monthly data
while date_start <= date_end:
    file_pathRead = "Pablo/results/monthlyAverages_" + str(date_start) + ".csv"
    df = pd.read_csv(file_pathRead, index_col=False)

    #Loop through the Dongs, computing the cosine similarity for all pairs
    dongCodes = df["행정동코드"].tolist()
    df_CosineSimilarity = pd.DataFrame(0, index = dongCodes, columns = dongCodes) #index = np.arange(len(dongCodes))
    j = k = 0
    for dong in dongCodes:
        k = 0
        dongCode = int(dong)
        df_externalIterator = df.loc[df["행정동코드"] == dongCode]
        array_externalIterator = df_externalIterator.to_numpy()
        #Need to find out why 중국인체류인구수,중국외외국인체류인구수 columns are NaN. For now set to 0. Find out later.
        array_externalIterator[np.isnan(array_externalIterator)] = 0
        for nextdong in dongCodes:
            df_internalIterator = df.loc[df["행정동코드"] == dongCode]
            array_internalIterator = df_internalIterator.to_numpy()
            #Need to find out why 중국인체류인구수,중국외외국인체류인구수 columns are NaN. For now set to 0. Find out later.
            array_internalIterator[np.isnan(array_internalIterator)] = 0
            #Compute cosine similarity
            cosine = int(np.dot(array_externalIterator, array_internalIterator.T))/(norm(array_externalIterator)*norm(array_internalIterator))
            #Adjust resolution of cosine similarity
            #cosine = (cosine*1e13+1)%1
            #Update value in dataframe
            df_CosineSimilarity.iloc[j, k] = cosine
            k+=1
        j+=1

    #Save the resulting dataframe
    file_name = "Pablo/results" + "/" + "cosineSimilarity" + "_" + str(date_start) + ".csv"
    df_CosineSimilarity.to_csv(file_name, index=False)

    #Update iterator to evaluate next month
    if date_start % 100 < 12:
        date_start += 1
    else:
        date_start += 89
