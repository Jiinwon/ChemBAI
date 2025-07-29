#!/usr/bin/env bash
set -euo pipefail

# 1. 대상 디렉터리 지정
target_dir="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/Final_model_save/ToxCast_model(F1)"

# 2. '&'가 들어간 항목 개수 세기
count=$(find "$target_dir" -depth -name '*&*' | wc -l)
echo "디렉터리 '$target_dir' 에서 '&'가 포함된 파일/폴더가 총 $count 개 발견되었습니다."

# 3. 0개면 종료
if [ "$count" -eq 0 ]; then
  exit 0
fi

# 4. 사용자 확인
read -p "이 항목들의 '&'를 '_'로 변경하시겠습니까? (Y/n): " answer
case "$answer" in
  [Yy]* )
    echo "→ 이름 변경을 시작합니다..."
    # 5. 깊이 우선으로 찾아서 안전하게 rename
    find "$target_dir" -depth -name '*&*' -print0 |
      while IFS= read -r -d '' file; do
        dir=$(dirname "$file")
        base=$(basename "$file")
        newbase=${base//&/_}
        mv -v "$file" "$dir/$newbase"
      done
    echo "완료: 모든 '&'가 '_'로 변경되었습니다."
    ;;
  * )
    echo "취소: 아무 작업도 수행되지 않았습니다."
    ;;
esac