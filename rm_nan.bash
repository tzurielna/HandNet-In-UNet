
#!/bin/bash

input="tmp.del"

python ./find_nan.py > $input

while read line
do
  loc=TrainData/$line".mat"
  rm $loc
done < $input

rm tmp.del
