import pandas as pd 
import warnings


# Step 1. Build the "Dong Code x Dong name" conversion table
#file_path = "Pablo/Dong_conversion_codes.xlsx"
#dataframe_Code2Dong = pd.read_excel(file_path, usecols=["행정동코드","행정동"])

#Filtering, remove duplicates
#dataframe_Code2Dong.drop_duplicates(inplace = True, subset='행정동', keep="last")
#dataframe_Code2Dong.dropna(inplace = True)


# Step 2. Leverage datasets to append monthly data for each Dong, vectorizing features
datasetType1, datasetType2, datasetType3 = "LOCAL_PEOPLE_DONG", "LONG_FOREIGNER_DONG", "TEMP_FOREIGNER_DONG"

date_start = 201701                                                 #the datasets do not have data from 2014 to 2016, airbnb dataset does
date_end = 202207                                                   #airbnb dataset ends with data from 2022.07
#Loop through each monthly data
while date_start <= date_end:
    file_path1 = "Pablo/datasets" + "/" + datasetType1 + "/" + datasetType1 + "_" + str(date_start) + ".csv"
    file_path2 = "Pablo/datasets" + "/" + datasetType2 + "/" + datasetType2 + "_" + str(date_start) + ".csv"
    file_path3 = "Pablo/datasets" + "/" + datasetType3 + "/" + datasetType3 + "_" + str(date_start) + ".csv"
    df1 = pd.read_csv(file_path1,index_col=False)
    df2 = pd.read_csv(file_path2,index_col=False)
    df3 = pd.read_csv(file_path3,index_col=False)

    #Loop through the Dongs, averaging the values of the daily entries for each Dong
    df_monthlyAverages = pd.DataFrame()                                     #This is poorly optimized, every loop iteration the dataframe increases in size?
    for dong in df1["행정동코드"].unique():
        dongCode = int(dong)
        
        #Read data for the given Dong, filtering the data as needed
        df1_filtered = df1.loc[df1["행정동코드"] == dongCode]                #Evaluate LOCAL_PEOPLE_DONG   data per each Dong
        df1_filtered.drop(columns=["시간대구분"], inplace = True)                            #Remove from LOCAL_PEOPLE_DONG   column Polled Time
        #df1_filtered.iloc[:, 3:df1_filtered.shape[1]] = df1_filtered.iloc[:, 3:df1_filtered.shape[1]].div(df1_filtered['총생활인구수'].values,axis=0)
        
        df2_filtered = df2.loc[df2["행정동코드"] == dongCode]               #Evaluate LONG_FOREIGNER_DONG data per each Dong
        df2_filtered.drop(columns=["기준일ID", "시간대구분", "행정동코드"], inplace = True)   #Remove from LONG_FOREIGNER_DONG column Date, Dong Code and Polled Time
        df2_filtered.columns += '_LONG_FOREIGNER'
        #df2_filtered.iloc[:, 1:df2_filtered.shape[1]] = df2_filtered.iloc[:, 1:df2_filtered.shape[1]].div(df2_filtered['총생활인구수_LONG_FOREIGNER'].values,axis=0)
        
        df3_filtered = df3.loc[df3["행정동코드"] == dongCode]                #Evaluate TEMP_FOREIGNER_DONG data per each Dong
        df3_filtered.drop(columns=["기준일ID", "시간대구분", "행정동코드"], inplace = True)   #Remove from TEMP_FOREIGNER_DONG column Date, Dong Code and Polled Time
        df3_filtered.columns += '_TEMP_FOREIGNER'
        #df3_filtered.iloc[:, 1:df3_filtered.shape[1]] = df3_filtered.iloc[:, 1:df3_filtered.shape[1]].div(df3_filtered['총생활인구수_TEMP_FOREIGNER'].values,axis=0)
        
        #If there is no data for the given Dong, set values to null, keeping Dong value
        if df1_filtered.empty:
            zeros = pd.DataFrame([[0]*df1_filtered.shape[1]], columns=df1_filtered.columns)
            df1_filtered = pd.concat([df1_filtered, zeros], ignore_index=True)
            df1_filtered["행정동코드"] = dongCode
        if df2_filtered.empty:
            zeros = pd.DataFrame([[0]*df2_filtered.shape[1]], columns=df2_filtered.columns)
            df2_filtered = pd.concat([df2_filtered, zeros], ignore_index=True)
        if df3_filtered.empty:
            zeros = pd.DataFrame([[0]*df3_filtered.shape[1]], columns=df3_filtered.columns)
            df3_filtered = pd.concat([df3_filtered, zeros], ignore_index=True)
        
        #Do monthly average for the given Dong
        with warnings.catch_warnings():                                             
            warnings.simplefilter(action='ignore', category=FutureWarning)      #Suppress "FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated"
            # Warning-causing lines of code here
            monthlyAverage_df1 = pd.DataFrame(df1_filtered.mean()).T
            monthlyAverage_df1["기준일ID"] = date_start
            monthlyAverage_df2 = pd.DataFrame(df2_filtered.mean()).T
            monthlyAverage_df3 = pd.DataFrame(df3_filtered.mean()).T

        #Aggregate the dataframes    
        row_filtered = pd.concat([monthlyAverage_df1, monthlyAverage_df2, monthlyAverage_df3], axis=1)
        df_monthlyAverages = pd.concat([df_monthlyAverages, row_filtered], ignore_index=True)

    #Save the resulting dataframe
    file_name = "Pablo/results" + "/" + "monthlyAverages" + "_" + str(date_start) + ".csv"
    df_monthlyAverages.to_csv(file_name, index=False)

    #Update iterator to evaluate next month
    if date_start % 100 < 12:                   #if last two digits of date are lower than 12, increment month in 1
        date_start += 1
    else:                                       #if last two digits are 12 (December) update value of date to January next year
        date_start += 89

# Next steps:
# 3. Aggregate results
# 4. Compute similarity statistic
