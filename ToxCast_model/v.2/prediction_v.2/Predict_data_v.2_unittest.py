import unittest
import pandas as pd
import joblib
import os

class TestModelPrediction(unittest.TestCase):
    def setUp(self):
        # 테스트에 필요한 파일 경로 설정
        self.training_date = '2025-04-07'
        self.training_time = '17-05-02'
        self.assay_name = 'ATG_ERE_CIS'
        self.fingerprint_type = 'MACCS'
        self.model_type = 'dt'
        dir_name = 'test_split_set'
        
        # 입력 파일 경로
        self.input_file = f"/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/v.2/data_v.2/train_test_split/20250407/ATG_ERE_CIS/MACCS_dt/17-05-02/17-05-02_MACCS_dt_x_test.csv"
        
        # 모델 파일 경로
        self.model_file = f"/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/v.2/results/example_DBPs_ER/2025-04-07/17-04-57/model_save_path/ATG_ERE_CIS/ATG_ERE_CIS_MACCS_dt/17-05-02/ATG_ERE_CIS_best_model_MACCS_dt.joblib"
        
        # 결과 저장 경로
        self.output_file = f"/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/v.2/prediction_v.2/example_prediction_DBPs/2025-04-07/17-05-02/test_prediction_ER.xlsx"

    def test_prediction(self):
        # 입력 파일 확인
        self.assertTrue(os.path.exists(self.input_file), f"Input file does not exist: {self.input_file}")
        
        # 모델 파일 확인
        self.assertTrue(os.path.exists(self.model_file), f"Model file does not exist: {self.model_file}")
        
        # 입력 데이터 로드
        x_test = pd.read_csv(self.input_file)
        self.assertFalse(x_test.empty, "Input x_test file is empty.")
        
        # 모델 로드
        model = joblib.load(self.model_file)
        
        # 예측 수행
        predictions = model.predict(x_test)
        
        # 결과 저장
        results_df = pd.DataFrame(predictions)#, columns=["Predictions"])
        results_df.to_excel(self.output_file, index=False)
        
        # 결과 파일 확인
        self.assertTrue(os.path.exists(self.output_file), f"Output file was not created: {self.output_file}")
        print(f"Predictions saved to {self.output_file}")

if __name__ == "__main__":
    unittest.main()