# honos
NERRE end-to-end pipeline


Add to root inside brateval inside a `.sh` script:
````
for filename in ../honos/results/temp/eval/*.ann; do
  BASE_NAME=$(basename "$filename")
  mv "../honos/results/temp/eval/$BASE_NAME" "../honos/results/brateval/eval/$BASE_NAME"
  mv "../honos/results/temp/gold/$BASE_NAME" "../honos/results/brateval/gold/$BASE_NAME"

  OUTPUT=$(mvn exec:java -Dexec.mainClass=au.com.nicta.csp.brateval.CompareEntities -Dexec.args="-e /Users/milindi/Documents/Honours/Projects/honos/results/brateval/eval -g /Users/milindi/Documents/Honours/Projects/honos/results/brateval/gold -s exact" | grep "all|")
  touch output.txt
  printf "%s | %s\n" "$BASE_NAME" "$OUTPUT" >> output.txt

  rm "../honos/results/brateval/eval/$BASE_NAME"
  rm "../honos/results/brateval/gold/$BASE_NAME"
done
````