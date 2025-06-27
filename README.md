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
  <p>이전 버전에서는 여러 스크립트의 경로와 모델 설정을 각각 수정해야 했습니다. 이제 <code>config.py</code> 하나에서 모든 설정을 관리합니다.</p>

  <h3>1. 훈련시</h3>
  <ul>
    <li>SMILES를 분자지문으로 변환할 입력 파일 경로와 출력 디렉토리</li>
    <li>사용할 모델과 지문 종류</li>
    <li>훈련 데이터 경로 등은 모두 <code>config.py</code>에 정의됩니다.</li>
  </ul>

  <h3>2. 예측시</h3>
  <ul>
    <li>예측에 사용할 모델 경로, 분자지문 파일, SMILES 파일 역시 <code>config.py</code>에서 수정합니다.</li>
  </ul>

    <p>훈련과 예측은 <code>run_pipeline.sh</code> 스크립트 하나로 실행합니다.
      <code>config.py</code>의 <code>OBJECT</code> 값으로 동작 모드를 선택합니다.
      0: 예측, 1: 훈련, 2: 훈련 후 바로 예측을 수행합니다.</p>

  <h2>run_pipeline 사용 방법 📚</h2>
  <ol>
    <li>
      <code>experiments/년월일/PROJECT_NAME/</code> 폴더를 만들고
      동일한 이름의 엑셀 파일 <code>{PROJECT_NAME}.xlsx</code>을 그 안에 둡니다.
    </li>
    <li>
      <code>config.py</code>에서 <code>PROJECT_NAME</code>과 <code>OBJECT</code> 값만 수정합니다.
      <code>OBJECT</code>는 0(예측), 1(훈련), 2(훈련 후 예측) 중 하나를 선택합니다.
      필요하면 <code>EXPERIMENT_DATE</code>를 지정합니다.
    </li>
    <li>프로젝트 루트에서 <code>bash run_pipeline.sh</code> 명령으로 파이프라인을 실행합니다.</li>
  </ol>

  <h2>run_pipeline 사용 방법 📚</h2>
  <ol>
    <li>
      <code>experiments/년월일/PROJECT_NAME/</code> 폴더를 만들고
      동일한 이름의 엑셀 파일 <code>{PROJECT_NAME}.xlsx</code>을 그 안에 둡니다.
    </li>
    <li>
      <code>config.py</code>에서 <code>PROJECT_NAME</code>과 실행 모드 <code>OBJECT</code>
      (&nbsp;0: 예측, 1: 훈련&nbsp;)만 수정합니다. 필요하면 <code>EXPERIMENT_DATE</code>
      를 지정합니다.
    </li>
    <li>프로젝트 루트에서 <code>bash run_pipeline.sh</code> 명령으로 파이프라인을 실행합니다.</li>
  </ol>

</body>
</html>
