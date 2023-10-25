# Gukmin_recommendation_system

# User matrix  
```markdown
train_prep(matrix, n)
find_ones(matrix)
get_matrix(matrix, indexes)
bce_loss(matrix, target_matrix)
extract_unique_words(data, column_name)
map_dict(input_set, input_list)
translate(df,column, translation_dict) # translate is a mapper (can also be worked with clusters) 
translations
code_translations
```

# Resume  
```markdown
resume = pd.read_csv('/content/resume.csv')
resume = one_hot_encode_column(resume,'text_keyword')
resume = one_hot_encode_date(resume,'updated_date')
resume = one_hot_encode_numeric_range(resume,'degree',1)
resume = one_hot_encode_numeric_range(resume,'graduate_date',2)
resume = one_hot_encode_numeric_range(resume,'hope_salary',100)
resume = one_hot_encode_numeric_range(resume,'last_salary',100)
resume = one_hot_encode_numeric_range(resume,'career_month',12)
resume = one_hot_encode_column(resume,'career_job_code')
resume_a = pd.concat([resume.iloc[:, 0], resume.iloc[:, 6:]], axis=1)

# ======================= certificate ======================== # 
certificate = pd.read_csv('/content/resume_certificate.csv')
certificate = certificate.dropna(subset=['certificate_contents'])
grouped = certificate.groupby('resume_seq')['certificate_contents'].apply(';'.join).reset_index()
grouped_a = one_hot_encode_column(grouped,'certificate_contents')
merged_df = pd.merge(resume_a, grouped_a, on='resume_seq', how='left')

# ======================= education ======================== #
education = pd.read_csv('/content/resume_education.csv')
education = education[['resume_seq','univ_major_type','univ_score','univ_location']]
education_a = one_hot_encode_numeric_range(education,'univ_major_type',1)
education_a = one_hot_encode_numeric_range(education_a,'univ_location',1)
education_a = one_hot_encode_numeric_range(education_a,'univ_score',1)
merged_df2 = pd.merge(merged_df, education_a, on='resume_seq', how='left')

# ======================= language ======================== #
language = pd.read_csv('/content/resume_language.csv')
# Drop duplicates based on 'resume_seq' and keep the first occurrence
language = language.drop_duplicates(subset='resume_seq', keep='first')
language_a = one_hot_encode_numeric_range(language,'language',1)
language_a= one_hot_encode_numeric_range(language_a,'exam_name',1)
language_a = one_hot_encode_numeric_range(language_a,'score',50)
merged_df3 = pd.merge(merged_df2, language_a, on='resume_seq', how='left')
resume_final = merged_df3.fillna(0)
```

# User based  
```markdown
user_item_matrix = train.groupby(['resume_seq', 'recruitment_seq']).size().unstack(fill_value=0)
user_item_matrix[user_item_matrix > 1] = 1
train_matrix = user_item_matrix.apply(lambda row: pick_n_ones(row, 2), axis=1)

train_r = train_matrix.values
target_r = user_item_matrix.values
train_r = pd.merge(train_matrix,final_df,on = 'resume_seq',how = 'left')
train_r = train_r.iloc[:,1:]
target_r = pd.merge(user_item_matrix, final_df, on = 'resume_seq',how = 'left')
target_r = target_r.iloc[:,1:]
train_r = train_r.values
target_r = target_r.values
```


