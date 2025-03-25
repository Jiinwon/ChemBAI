<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>ChemBAI README 가이드 📝</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #f7f7f7;
      margin: 20px;
      padding: 20px;
      line-height: 1.6;
      color: #333;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 2px solid #ccc;
      padding-bottom: 10px;
    }
    h2 {
      color: #34495e;
      margin-top: 20px;
    }
    h3 {
      color: #3c6382;
      margin-top: 15px;
    }
    ul, ol {
      margin-left: 20px;
    }
    code {
      background-color: #eef;
      padding: 2px 4px;
      border-radius: 4px;
      font-family: Consolas, monospace;
    }
    .section {
      background-color: #fff;
      padding: 15px;
      border-left: 4px solid #27ae60;
      margin-bottom: 20px;
    }
    pre {
      background-color: #f1f1f1;
      padding: 10px;
      border-radius: 4px;
      overflow-x: auto;
    }
    hr {
      border: none;
      border-top: 1px solid #ccc;
      margin: 20px 0;
    }
  </style>
</head>
<body>
  <h1>ChemBAI 📊</h1>
  <p><strong>Data version:</strong> ToxCast_v.4.1</p>
  
  <hr>
  
  <h2>사용된 분자지문 🧬</h2>
  <ul>
    <li>MACCS</li>
    <li>Morgan</li>
    <li>RDKit</li>
    <li>Pattern</li>
    <li>Layered</li>
  </ul>
  
  <h2>사용된 알고리즘 🤖</h2>
  <ul>
    <li>Decision Tree (<code>dt.py</code>)</li>
    <li>Logistic Regression (<code>logistic.py</code>)</li>
    <li>Gradient Boost Tree (<code>gbt.py</code>)</li>
    <li>XGBoost (<code>xgb.py</code>)</li>
    <li>Random Forest (<code>rf.py</code>)</li>
  </ul>
  
  <h2>입력데이터 📥</h2>
  <div class="section">
    <p>
      <strong>입력데이터.xlsx</strong>의 데이터프레임 1행의 열 구성은 반드시 아래와 같아야 합니다.
    </p>
    <pre>
1행 : DTXSID | SMILES | assay_name1 | assay_name2 | ...
    </pre>
    <p>
      <strong>assay_name1</strong> (두번째 행)의 데이터를 이용하고자 하는 경우, <code>assay_num = 1</code>을 입력합니다.  
      코드상에서 <code>assay_num+1</code>을 사용하므로 데이터프레임의 형식이 위와 동일해야 두번째 행(assay_num+1=2)을 불러올 수 있습니다.
    </p>
  </div>
  
  <h2>분자지문 📈</h2>
  <div class="section">
    <p>
      모델 훈련 시마다 분자지문으로 변환하는 과정을 중복 실행하지 않기 위해, 최초 한 번 변환 후 저장한 파일을 불러와 사용합니다.  
      훈련하고자 하는 SMILES에 대해 <code>smiles2fing.py</code>를 통해 fingerprints를 생성합니다.
    </p>
  </div>
  
  <h2>데이터 전처리 🔍</h2>
  <div class="section">
    <p>
      ToxCast_v.4.1_v.2 데이터는 KNIME을 통한 염 제거, 무기물질 제거, 그리고 Hitcall의 개수가 5개 미만인 데이터를 제거한 결과물입니다.
    </p>
    <p>
      <strong>run</strong> 디렉토리 내 scikit-learn 기반 머신러닝 코드에서는 아래와 같이 데이터를 처리합니다.
    </p>
    <ol>
      <li>
        SMILES로부터 분자지문(mol 형식)으로 변환되지 않는 화학물질의 인덱스는 <code>dropidx.csv</code> 파일에 저장합니다.  
        (예시 코드에서 사용하는 변수: <code>drop_idx</code>)
      </li>
      <li>
        Hitcall 데이터에서 결측값이 있는 인덱스는 <code>na_idx</code> 변수로 받아 해당 행을 제거합니다.
      </li>
    </ol>
  </div>
  
  <h2>예시 코드 (dt.py, 70~77라인) 📝</h2>
  <pre>
x = pd.read_csv(file_path_fp)
df_drop_idx = pd.read_csv(f'{fp_path}/{fingerprint_type}_dropidx.csv')
drop_idx = df_drop_idx[f'{fingerprint_type}'].tolist()
df = pd.read_excel(file_path)
y = df.iloc[:, assay_num+1].drop(drop_idx).reset_index(drop=True)
na_idx = y[y.isnull()].index
y = y.drop(index=na_idx).reset_index(drop=True)
x = x.drop(index=na_idx).reset_index(drop=True)
  </pre>
  
  <h2>참고 사항 ⚙️</h2>
  <div class="section">
    <p>
      모델 코드 내 사용된 <code>assay_num</code> 변수는 ToxCast 내 다양한 assay 데이터를 반복문으로 처리하기 위해 설정되었습니다.
    </p>
  </div>
  
  <hr>
  
  <h2>파일 및 코드 수정 안내 🛠</h2>
  <p><strong>*데이터에 따라 수정하여 사용하는 파이썬 파일과 수정하는 부분*</strong></p>
  
  <h3>1. 훈련시</h3>
  <ul>
    <li>
      <strong>smiles2fing.py</strong>
      <ul>
        <li><code>input_excel_path = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"</code></li>
        <li><code>output_dir = "./data/example_data_DBPs/for_train/fingerprints"</code></li>
      </ul>
    </li>
    <li>
      <strong>ToxCast_model_training.sh</strong>
      <ul>
        <li><code>models=('dt')</code> <!-- 예시: 'xgb', 'gbt', 'rf', 'logistic' 등으로 모델 정의 --></li>
        <li><code>fingerprints=('MACCS')</code> <!-- 예시: 'Morgan', 'RDKit', 'Layered', 'Pattern' 등으로 지문 정의 --></li>
        <li><code>file_path = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"</code> (데이터 경로)</li>
        <li><code>fp_path = "./data/example_data_DBPs/for_train/fingerprint_outputs"</code> (fingerprints 경로)</li>
        <li><code>data_name = "example_DBPs_ER"</code></li>
      </ul>
    </li>
  </ul>
  
  <h3>2. 예측시</h3>
  <ul>
    <li>
      <strong>smiles2fing.py</strong>
      <ul>
        <li><code>input_excel_path = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"</code></li>
        <li><code>output_dir = "./data/example_data_DBPs/for_train/fingerprints"</code></li>
      </ul>
    </li>
    <li>
      <strong>Predict_data.py</strong>
      <ul>
        <li><code>input_excel_path = "./prediction/example_prediction_DBPs/example_assay_list_ER.xlsx"</code> <!-- 수정 필요 --></li>
        <li><code>model_path_base = "./results/example_DBPs_ER/2025-03-25/model_save_path"</code></li>
        <li><code>input_fp_path_base = "./data/example_data_DBPs/for_predict/fingerprints"</code></li>
        <li><code>SMILES_path = "./data/example_data_DBPs/for_predict/example_DBPs_for_pred.xlsx"</code></li>
      </ul>
    </li>
  </ul>
  
</body>
</html>