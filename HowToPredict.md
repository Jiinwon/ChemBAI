</head>
<body>
  <h2>⚠️ 주의사항</h2>
  <ul>
    <li>업로드된 코드 파일들은 모두 상대경로로 지정되어 있으므로 <strong>"model"</strong> 디렉토리까지 이동 후 사용해야 합니다.</li>
  </ul>
  
  <hr>
  
  <h2>📂 사용하는 파이썬 코드</h2>
  <p><code>model/prediction/Predict_data.py</code></p>
  
  <h2>🔧 수정하여 사용하는 부분</h2>
  <ul>
    <li><code>input_excel_path = "./prediction/example_prediction_DBPs/example_assay_list_ER.xlsx"</code></li>
    <li><code>model_path_base = "./results/example_DBPs_ER/2025-03-25/model_save_path"</code></li>
    <li><code>input_fp_path_base = "./data/example_data_DBPs/for_predict/fingerprints"</code></li>
    <li><code>SMILES_path = "./data/example_data_DBPs/for_predict/example_DBPs_for_pred.xlsx"</code></li>
  </ul>
  
  <hr>
  
  <h2>📑 파일 구성 및 설명</h2>
  <ol>
    <li>
      <strong>input_excel_path</strong> (예측 대상 assay 정보가 있는 데이터)
      <ul>
        <li>열의 구성: <code>assay_name</code>, <code>Model</code>, <code>MF</code></li>
      </ul>
    </li>
    <li>
      <strong>model_path_base</strong>: 사용할 모델의 (<em>model_save_path까지의</em>) 디렉토리
    </li>
    <li>
      <strong>input_fp_path_base</strong>: 예측 대상 화학물질의 분자지문 데이터 (<em>fingerprints까지의</em> 디렉토리)
      <ul>
        <li>해당 디렉토리는 <code>smiles2fing.py</code>를 통해 생성됩니다. (<em>아래 Step 1 참고</em>)</li>
      </ul>
    </li>
    <li>
      <strong>SMILES_path</strong>: 예측 대상 화학물질의 "SMILES" 열이 있는 데이터
    </li>
  </ol>
  
  <hr>
  
  <h2>🚀 실행 단계</h2>
  <div class="step">
    <h3>Step 1: fingerprints 파일 생성하기</h3>
    <ul>
      <li>fingerprints 폴더를 생성할 디렉토리를 만든 후 <code>smiles2fing.py</code>를 실행합니다.</li>
      <li>예시 경로:
        <br><code>/model/data/example_data_DBPs/for_predict</code> (fingerprints 폴더를 생성할 디렉토리)
        <br>&nbsp;&nbsp;&nbsp;└─ <code>fingerprints/</code> (<code>smiles2fing.py</code> 실행 후 생성된 폴더)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ <code>Layered_fingerprints.csv</code>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <code>MACCS_fingerprints.csv</code>
      </li>
    </ul>
  </div>
  
  <div class="step">
    <h3>Step 2: 코드 수정 후 실행</h3>
    <ul>
      <li>위 코드에서 수정하여 사용하는 부분에 명시된 파일 및 디렉토리 경로를 확인 및 수정합니다.</li>
      <li><code>Predict_data.py</code>를 실행하여 예측 작업을 수행합니다.</li>
    </ul>
  </div>
  
</body>
</html>