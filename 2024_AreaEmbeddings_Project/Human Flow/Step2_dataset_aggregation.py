import pandas as pd


# Step 3. Aggregate results
date_start = 201701
date_end = 202207

#Loop through each monthly data
df_aggregated = pd.DataFrame()   
while date_start <= date_end:
    file_pathRead = "Pablo/results/monthlyAverages_" + str(date_start) + ".csv"
    df = pd.read_csv(file_pathRead, index_col=False)
    
    #Aggregate the dataframes
    df_aggregated = pd.concat([df_aggregated, df], ignore_index=True)

    #Update iterator to evaluate next month
    if date_start % 100 < 12:                   #if last two digits of date are lower than 12, increment month in 1
        date_start += 1
    else:                                       #if last two digits are 12 (December) update value of date to January next year
        date_start += 89

#Save the resulting dataframe
file_pathWrite = "Dataset/PabloTask_AggregatedDataframe.csv"
df_aggregated.to_csv(file_pathWrite, index=False)