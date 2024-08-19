ROOT_FOLDER_PATH="$2"
BRAT_EVAL_FILEPATH="$1"
for filename in "$ROOT_FOLDER_PATH"/RMIT-READ-BioMed-Version-2.0/results/temp/eval/*.ann; do
  BASE_NAME=$(basename "$filename")
  mv "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/temp/eval/$BASE_NAME" "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/eval/$BASE_NAME"
  mv "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/temp/gold/$BASE_NAME" "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/gold/$BASE_NAME"


  cd "$BRAT_EVAL_FILEPATH" || exit
  :
  OUTPUT=$(mvn exec:java -Dexec.mainClass=au.com.nicta.csp.brateval.CompareEntities -Dexec.args="-e $ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/eval -g $ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/gold -s exact" | grep "all|")
#  touch "$ROOT_FOLDER_PATH"/RMIT-READ-BioMed-Version-2.0/results/output.tsv
  printf "%s%s::" "$BASE_NAME" "$OUTPUT"

  rm "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/eval/$BASE_NAME"
  rm "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval/gold/$BASE_NAME"
done

rm -rf "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/brateval"
rm -rf "$ROOT_FOLDER_PATH/RMIT-READ-BioMed-Version-2.0/results/temp"