import os

# 기준 디렉토리 경로 설정
base_dir = '/home1/won0316/_RESEARCH/0817_Genotoxicity/Final_model/F1_model'

empty_folders = []

# os.walk를 사용하여 모든 하위 폴더 탐색
for root, dirs, files in os.walk(base_dir):
    # 현재 폴더에 파일도 없고, 서브폴더도 없는 경우
    if not dirs and not files:
        empty_folders.append(root)

# 결과 출력
if empty_folders:
    print("비어있는 폴더 목록:")
    for folder in empty_folders:
        print(folder)
else:
    print("비어있는 폴더가 없습니다.")