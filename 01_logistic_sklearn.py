from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import json
import progressbar
import warnings
warnings.simplefilter('ignore')

# 파일 열기
# with open('Online_sorted_addY.json', encoding='utf-8', mode='r') as f:
#     array = json.load(f)
# df = pd.DataFrame.from_dict(array)
df = pd.read_csv('Online_Y_Z.csv', encoding = 'utf_8')
# print(df['biz_unit'])



# 데이터 최적화 및 결측값 처리
# df_array = df.transpose()
# df_array = df_array.fillna(0)
df = df.fillna(0)

# 로지스틱 회귀분석
model = sm.Logit.from_formula("Y ~ tot_sess_hr_v + tot_pag_view_ct + dvc_ctg_nm + trfc_src + biz_unit", df)
result = model.fit()

print("[1차 로지스틱 회귀분석]")
print(result.summary())
print(np.exp(result.params))

# 교차검증 데이터 사전처리
col_fix = ['Y','tot_sess_hr_v','tot_pag_view_ct']
dummy_mpc = pd.get_dummies(df['dvc_ctg_nm'], prefix='dvc_ctg_nm') #이 방법으로 나누기
dummy_src = pd.get_dummies(df['trfc_src'], prefix='trfc_src')
dummy_unit = pd.get_dummies(df['biz_unit'], prefix='biz_unit')

data = df[col_fix].join(dummy_mpc)
data = data.join(dummy_src)
data = data.join(dummy_unit)

log_leg = LogisticRegression()

x = data[['tot_sess_hr_v', 'tot_pag_view_ct', 'dvc_ctg_nm_0',
       'dvc_ctg_nm_PC', 'dvc_ctg_nm_mobile_app', 'dvc_ctg_nm_mobile_web',
       'trfc_src_DIRECT', 'trfc_src_PORTAL_1', 'trfc_src_PORTAL_2',
       'trfc_src_PORTAL_3', 'trfc_src_PUSH', 'trfc_src_WEBSITE',
       'trfc_src_unknown', 'biz_unit_A01', 'biz_unit_A02', 'biz_unit_A03']]
y = data['Y']

# 사전처리 데이터로 2차 로지스틱 회귀분석
model2 = sm.Logit.from_formula("Y ~ tot_sess_hr_v + tot_pag_view_ct + dvc_ctg_nm_PC + dvc_ctg_nm_mobile_app + dvc_ctg_nm_mobile_web + trfc_src_PORTAL_1 +trfc_src_PORTAL_2 + trfc_src_PORTAL_3 + trfc_src_PUSH + trfc_src_WEBSITE + trfc_src_unknown + biz_unit_A02 + biz_unit_A03", data )
result2 = model2.fit()

print("\n [2차 로지스틱 회귀분석]")
print(result2.summary())
print(np.exp(result2.params))

#학습 및 테스트 세트 분할
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=100)
log_leg.fit(x_train, y_train)
result2 = log_leg.score(x_test, y_test)
print("\n정확도 : %2f%%" % (result2*100))

#K배 교차검증
kfold = model_selection.KFold(n_splits=10, random_state=100)
model_kfold = LogisticRegression()
result_kfold = model_selection.cross_val_score(model_kfold, x, y, cv=kfold)

print("정확도 : %2f%%" % (result_kfold.mean()*100))


